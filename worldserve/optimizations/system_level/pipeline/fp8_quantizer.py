"""
DiT weight quantization for WorldServe inference.

Strategy selection based on roofline analysis (arXiv 2502.01070):
  - At batch=1, DiT-S/2 arithmetic intensity ≈ 1 FLOP/byte (memory-bandwidth-bound)
  - H100 BF16 ridge point ≈ 295 FLOP/byte → we are 295x below compute-bound regime
  - FP8 W8A8 dynamic: adds 2,600 amax reductions per generation with zero tensor-core benefit
  - Weight-only quantization captures bandwidth savings without activation overhead

Recommended mode: INT4 weight-only (int4wo)
  - 4× bandwidth reduction vs FP16 → direct latency reduction for bandwidth-bound ops
  - Dequantize INT4→BF16 in registers (~10-15% overhead), net still faster
  - group_size=128: standard per ViDiT-Q (ICLR 2025), Q-DiT (CVPR 2025)

Secondary mode: FP8 weight-only (fp8wo)
  - 2× bandwidth, near-zero overhead, safer quality than INT4
  - Correct for larger models or cases where INT4 quality is unacceptable

Boundary block exclusion (DiTAS, WACV 2025 — arXiv 2409.07756):
  - First/last N transformer blocks are most sensitive to quantization error
  - Their outputs drive input embedding and final denoised latent quality
  - Exclude them from quantization; keep middle blocks at INT4/FP8

Ablation mode:
  - ablation_mode=True: stores FP16 originals so restore() works (for benchmarks only)
  - ablation_mode=False: no copy stored → saves ~600 MB VRAM in production inference
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compatibility shim: torchao >= 0.17 calls register_constant() which is absent
# from stable PyTorch 2.6.0.
# ---------------------------------------------------------------------------
import torch.utils._pytree as _pytree
if not hasattr(_pytree, "register_constant"):
    def _register_constant(cls, flatten=None, unflatten=None):  # type: ignore[override]
        return cls
    _pytree.register_constant = _register_constant  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# torchao imports — weight-only configs are the primary path
# ---------------------------------------------------------------------------
_TORCHAO_AVAILABLE = False
_Int4WeightOnlyConfig = None
_Float8WeightOnlyConfig = None
_Int8WeightOnlyConfig = None
_Float8DynamicConfig = None
_quantize_fn = None

try:
    from torchao.quantization import quantize_ as _quantize_fn  # type: ignore[import]

    # Weight-only (recommended)
    try:
        from torchao.quantization import Int4WeightOnlyConfig as _Int4WeightOnlyConfig  # type: ignore[import]
    except ImportError:
        try:
            from torchao.quantization import Int4WeightOnlyQuantConfig as _Int4WeightOnlyConfig  # type: ignore[import]
        except ImportError:
            pass

    try:
        from torchao.quantization import Float8WeightOnlyConfig as _Float8WeightOnlyConfig  # type: ignore[import]
    except ImportError:
        pass

    try:
        from torchao.quantization import Int8WeightOnlyConfig as _Int8WeightOnlyConfig  # type: ignore[import]
    except ImportError:
        try:
            from torchao.quantization import Int8WeightOnlyQuantConfig as _Int8WeightOnlyConfig  # type: ignore[import]
        except ImportError:
            pass

    # Legacy FP8 dynamic (W8A8) — kept for ablation comparison only
    try:
        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig as _Float8DynamicConfig  # type: ignore[import]
    except ImportError:
        pass

    _TORCHAO_AVAILABLE = True
    logger.info("torchao quantization available (int4wo=%s, fp8wo=%s, int8wo=%s, fp8dq=%s)",
                _Int4WeightOnlyConfig is not None,
                _Float8WeightOnlyConfig is not None,
                _Int8WeightOnlyConfig is not None,
                _Float8DynamicConfig is not None)
except ImportError:
    logger.warning(
        "torchao not installed. DiTQuantizer will not be available. "
        "Install: pip install torchao>=0.6.0"
    )

# Layers always excluded — sensitivity confirmed by ViDiT-Q (ICLR 2025)
# and Q-DiT (CVPR 2025): RoPE, norm, and conditioning layers are high-sensitivity
# with negligible FLOPs — never worth quantizing.
DEFAULT_EXCLUDE_PATTERNS: List[str] = [
    "rope",
    "adaln",
    "action",
    "norm",
    "embed",
]

# Available quantization modes
QUANT_MODES = ("int4wo", "fp8wo", "int8wo", "fp8dq")


class DiTQuantizer:
    """
    Unified weight quantization for DiT inference at batch=1 on H100.

    Mode selection:
      "int4wo"  — INT4 weight-only, group_size=128 (default, fastest at batch=1)
      "fp8wo"   — FP8 weight-only, e4m3 (safer quality, 2× bandwidth)
      "int8wo"  — INT8 weight-only (most conservative, near-lossless)
      "fp8dq"   — FP8 W8A8 dynamic (legacy, suboptimal at batch=1 — see module docstring)

    Boundary block exclusion (DiTAS, WACV 2025):
      First and last `num_boundary_blocks` transformer blocks are excluded from
      quantization. At 12 blocks (Oasis), blocks 0,1,10,11 stay FP16.

    Ablation mode:
      ablation_mode=True  → stores FP16 originals; restore() is available
      ablation_mode=False → no copies; saves ~600 MB VRAM (production setting)
    """

    def __init__(
        self,
        ablation_mode: bool = False,
        num_boundary_blocks: int = 2,
    ) -> None:
        self._ablation_mode = ablation_mode
        self._num_boundary_blocks = num_boundary_blocks
        self._original_weights: Dict[str, torch.Tensor] = {}
        self._quantized_modules: Set[str] = set()
        self._detected_num_blocks: Optional[int] = None

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    def quantize(
        self,
        model: nn.Module,
        mode: str = "int4wo",
        group_size: int = 128,
        exclude_patterns: Optional[List[str]] = None,
    ) -> int:
        """
        Quantize DiT Linear layers.

        Args:
            model:            DiT model (or full pipeline).
            mode:             Quantization strategy (see class docstring).
            group_size:       Group size for INT4 (32/64/128/256). Ignored for FP8/INT8.
            exclude_patterns: Additional name patterns to skip (merged with defaults).

        Returns:
            Number of Linear layers quantized.

        Raises:
            RuntimeError: If torchao is not available or mode config is unavailable.
        """
        if not _TORCHAO_AVAILABLE or _quantize_fn is None:
            raise RuntimeError(
                "torchao is not installed. Install: pip install torchao>=0.6.0"
            )

        mode = mode.lower()
        if mode not in QUANT_MODES:
            raise ValueError(f"Unknown quant mode '{mode}'. Choose from: {QUANT_MODES}")

        patterns = list(DEFAULT_EXCLUDE_PATTERNS) + (exclude_patterns or [])
        self._detected_num_blocks = self._count_transformer_blocks(model)

        if self._ablation_mode:
            self._store_originals(model, patterns)

        filter_fn = self._make_filter(model, patterns)
        config = self._build_config(mode, group_size)

        # torchao 0.9.0 Int4WeightOnlyConfig stores scales as bfloat16 and requires
        # the model to be in bfloat16 before quantization. Oasis loads in float16.
        # Cast to bfloat16 for quantization; the quantized linear layers replace their
        # weights with INT4+scale tensors (bfloat16 scales), so bfloat16 stays after.
        _original_dtype: Optional[torch.dtype] = None
        if mode == "int4wo":
            _param = next(iter(model.parameters()), None)
            if _param is not None and _param.dtype == torch.float16:
                _original_dtype = torch.float16
                model.to(torch.bfloat16)
                logger.info("[DiTQuantizer] cast model float16→bfloat16 for int4wo (torchao requirement)")

        _quantize_fn(model, config, filter_fn=filter_fn)

        # Count quantized
        count = sum(
            1 for name, mod in model.named_modules()
            if isinstance(mod, nn.Linear) and filter_fn(mod, name)
        )
        self._quantized_modules = {
            name for name, mod in model.named_modules()
            if isinstance(mod, nn.Linear) and filter_fn(mod, name)
        }

        logger.info(
            "[DiTQuantizer] mode=%s group_size=%s: %d layers quantized "
            "(boundary_blocks=%d, excluded_patterns=%s, ablation_mode=%s)",
            mode, group_size if "int4" in mode else "N/A",
            count, self._num_boundary_blocks, patterns, self._ablation_mode,
        )
        print(
            f"[DiTQuantizer] mode={mode}: {count} layers quantized "
            f"(boundary_blocks excluded: ±{self._num_boundary_blocks} of "
            f"{self._detected_num_blocks or '?'} total, "
            f"ablation_mode={self._ablation_mode})"
        )
        self._quantized_count = count
        return count

    # ------------------------------------------------------------------
    # Backward-compatible entry point (called from modal_app.py)
    # ------------------------------------------------------------------

    def quantize_torchao(
        self,
        model: nn.Module,
        exclude_patterns: Optional[List[str]] = None,
        mode: str = "int4wo",
        group_size: int = 128,
    ) -> int:
        """
        Backward-compatible wrapper for modal_app.py.
        Previously used Float8DynamicActivationFloat8WeightConfig (fp8dq).
        Now defaults to INT4 weight-only (int4wo) — see module docstring for why.
        Pass mode='fp8dq' explicitly to restore old behavior.
        """
        return self.quantize(
            model,
            mode=mode,
            group_size=group_size,
            exclude_patterns=exclude_patterns,
        )

    # ------------------------------------------------------------------
    # Config builders — one per mode
    # ------------------------------------------------------------------

    def _build_config(self, mode: str, group_size: int) -> Any:
        if mode == "int4wo":
            if _Int4WeightOnlyConfig is None:
                raise RuntimeError(
                    "Int4WeightOnlyConfig not available in this torchao version. "
                    "Try mode='fp8wo' or upgrade torchao."
                )
            return _Int4WeightOnlyConfig(group_size=group_size)

        elif mode == "fp8wo":
            if _Float8WeightOnlyConfig is None:
                raise RuntimeError(
                    "Float8WeightOnlyConfig not available. Try mode='int8wo' or upgrade torchao."
                )
            return _Float8WeightOnlyConfig()

        elif mode == "int8wo":
            if _Int8WeightOnlyConfig is None:
                raise RuntimeError(
                    "Int8WeightOnlyConfig not available. Upgrade torchao."
                )
            return _Int8WeightOnlyConfig()

        elif mode == "fp8dq":
            # Legacy W8A8 dynamic — suboptimal at batch=1 (arXiv 2502.01070)
            # Kept for ablation comparison only.
            if _Float8DynamicConfig is None:
                raise RuntimeError(
                    "Float8DynamicActivationFloat8WeightConfig not available. "
                    "This is the legacy W8A8 dynamic mode. "
                    "Use mode='int4wo' or 'fp8wo' instead."
                )
            logger.warning(
                "[DiTQuantizer] mode=fp8dq (FP8 W8A8 dynamic) selected. "
                "This is the legacy path. At batch=1, dynamic activation scaling adds "
                "2,600 amax reductions with zero tensor-core benefit (arXiv 2502.01070). "
                "Use mode='int4wo' for better throughput."
            )
            return _Float8DynamicConfig()

        raise ValueError(f"Unknown mode: {mode}")

    # ------------------------------------------------------------------
    # Filter function — excludes sensitive patterns AND boundary blocks
    # ------------------------------------------------------------------

    def _make_filter(self, model: nn.Module, patterns: List[str]) -> Callable:
        num_blocks = self._detected_num_blocks or 0
        boundary = self._num_boundary_blocks

        def filter_fn(module: nn.Module, fqn: str) -> bool:
            if not isinstance(module, nn.Linear):
                return False

            fqn_lower = fqn.lower()

            # Exclude by pattern (rope, adaln, norm, embed, action, ...)
            for pat in patterns:
                if pat.lower() in fqn_lower:
                    return False

            # Exclude boundary blocks (DiTAS WACV 2025 — arXiv 2409.07756):
            # First and last `boundary` blocks are most sensitive.
            if boundary > 0 and num_blocks > 0:
                match = re.search(r'blocks?[.\[_](\d+)', fqn_lower)
                if match:
                    block_idx = int(match.group(1))
                    if block_idx < boundary or block_idx >= (num_blocks - boundary):
                        return False

            return True

        return filter_fn

    # ------------------------------------------------------------------
    # Block counting — detects actual transformer block count from model
    # ------------------------------------------------------------------

    @staticmethod
    def _count_transformer_blocks(model: nn.Module) -> int:
        """
        Detect number of transformer blocks by scanning for common attribute names.
        Works for Oasis (model.blocks) and MG2 (model.transformer_blocks).
        """
        for attr in ("blocks", "transformer_blocks", "layers", "dit_blocks"):
            candidate = getattr(model, attr, None)
            if candidate is not None and hasattr(candidate, "__len__"):
                return len(candidate)
        # Fallback: count from FQN patterns
        max_idx = -1
        for name, _ in model.named_modules():
            match = re.search(r'blocks?[.\[_](\d+)', name.lower())
            if match:
                max_idx = max(max_idx, int(match.group(1)))
        return max_idx + 1 if max_idx >= 0 else 0

    # ------------------------------------------------------------------
    # Quality measurement
    # ------------------------------------------------------------------

    def measure_quality_impact(
        self,
        model: nn.Module,
        test_input: Any,
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Compare quantized vs FP16 output.
        Only usable when ablation_mode=True (needs restore()).
        """
        if not self._ablation_mode:
            raise RuntimeError(
                "measure_quality_impact() requires ablation_mode=True. "
                "Construct DiTQuantizer(ablation_mode=True) to enable restore()."
            )

        model.eval()
        with torch.no_grad():
            quant_out = model(**test_input) if isinstance(test_input, dict) else (
                model(*test_input) if isinstance(test_input, (tuple, list)) else model(test_input)
            )
        if isinstance(quant_out, (tuple, list)):
            quant_out = quant_out[0]
        quant_f32 = quant_out.float().clone()

        self.restore(model)

        with torch.no_grad():
            fp16_out = model(**test_input) if isinstance(test_input, dict) else (
                model(*test_input) if isinstance(test_input, (tuple, list)) else model(test_input)
            )
        if isinstance(fp16_out, (tuple, list)):
            fp16_out = fp16_out[0]
        fp16_f32 = fp16_out.float()

        diff = (quant_f32 - fp16_f32).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        qf = quant_f32.flatten()
        ff = fp16_f32.flatten()
        cosine = torch.nn.functional.cosine_similarity(qf.unsqueeze(0), ff.unsqueeze(0)).item()
        rel_err = diff.flatten().norm().item() / max(ff.norm().item(), 1e-12)

        metrics = {
            "max_abs_diff": max_diff,
            "mean_abs_diff": mean_diff,
            "cosine_similarity": cosine,
            "relative_error": rel_err,
        }
        acceptable = cosine > 0.995 and rel_err < 0.05
        logger.info(
            "Quality: cosine=%.6f rel_err=%.6f max_diff=%.6f acceptable=%s",
            cosine, rel_err, max_diff, acceptable,
        )
        return acceptable, metrics

    # ------------------------------------------------------------------
    # Restore (ablation mode only)
    # ------------------------------------------------------------------

    def restore(self, model: nn.Module) -> int:
        """
        Restore quantized layers to FP16.
        Only available when ablation_mode=True.
        """
        if not self._ablation_mode:
            raise RuntimeError(
                "restore() requires ablation_mode=True. "
                "In production inference, load a fresh checkpoint instead."
            )
        count = 0
        for name, module in model.named_modules():
            if name in self._original_weights:
                module.weight.data = self._original_weights[name].to(device=module.weight.device)
                for buf in ("_fp8_weight", "_fp8_weight_scale"):
                    if hasattr(module, buf):
                        delattr(module, buf)
                count += 1
        self._original_weights.clear()
        self._quantized_modules.clear()
        logger.info("Restored %d modules to original precision.", count)
        return count

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _store_originals(self, model: nn.Module, patterns: List[str]) -> None:
        """Store FP16 weight copies. Only called when ablation_mode=True."""
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if any(p.lower() in name.lower() for p in patterns):
                continue
            if name not in self._original_weights:
                self._original_weights[name] = module.weight.data.clone()

    @property
    def quantized_module_names(self) -> Set[str]:
        return set(self._quantized_modules)

    @property
    def num_quantized(self) -> int:
        return self._quantized_count


# Backward-compatible alias
DiTFP8Quantizer = DiTQuantizer
