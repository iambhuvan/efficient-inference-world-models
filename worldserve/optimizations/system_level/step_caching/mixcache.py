"""
MixCache: Multi-granularity feature caching for Wan2.1-based models (Matrix-Game 2.0).

Benchmarked at 1.94x on Wan 14B (same architecture family as Matrix-Game 2.0).
Mixes two granularities:
  1. Step-level: reuse the full previous step's block output if Frobenius-norm
     drift is below tau_step.
  2. Block-level: per-block residual-magnitude gating (complementary to SeaCache).

Reference:
  MixCache (Aug 2025) — https://hf.co/papers/2508.12691

Architecture target:
  Matrix-Game 2.0 / Wan2.1 backbone (30 transformer blocks, 15,360 tokens/frame).
  Less beneficial for Oasis (144 tokens) — use SeaCache there instead.

Critical implementation note
-----------------------------
Post-forward hooks (register_forward_hook) fire AFTER the block computes, so they
cannot save GPU computation (only avoid downstream processing).  This implementation
uses forward-method replacement:

    original = block.forward
    block.forward = _wrapper(block_name, original)

The wrapper checks the cache BEFORE calling original.  On cache-hit steps the
block's GPU kernel is never dispatched.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MixCacheOptimizer:
    """
    Multi-granularity feature caching for Matrix-Game 2.0 / Wan2.1.

    Two complementary caching strategies:
      - Step-level: if the Frobenius-norm relative drift of a block's output
        between adjacent denoising timesteps is < tau_step, reuse last output.
      - Block-level: per-block EMA of residual magnitudes; skip blocks whose
        residual magnitude has not changed significantly.

    Integration: call attach(model) to install forward wrappers.
    The cache is automatically reset between generations via reset().
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Args:
            config: Dict with keys:
              tau_step  (float, 0.03): Frobenius drift threshold for step-level reuse.
              tau_block (float, 0.05): Residual magnitude threshold for block-level skip.
              warmup_steps (int, 1): Steps before caching starts.
              ema_alpha (float, 0.1): EMA decay for block residual magnitude.
        """
        self.tau_step     = float(config.get("tau_step",     0.03))
        self.tau_block    = float(config.get("tau_block",    0.05))
        self.warmup_steps = int(config.get("warmup_steps",   1))
        self.ema_alpha    = float(config.get("ema_alpha",    0.1))

        # block_name -> last recorded output tensor (for step-level drift)
        self._last_output: Dict[str, torch.Tensor] = {}
        # block_name -> EMA of residual magnitude
        self._ema_residual: Dict[str, float] = {}
        # block_name -> last EMA residual magnitude recorded
        self._prev_ema: Dict[str, float] = {}

        self._current_step: int = 0

        # (module, block_name, original_forward) — populated by attach()
        self._wrapped: List[Tuple[nn.Module, str, Callable]] = []

        # Stats
        self._step_hits: int = 0
        self._block_hits: int = 0
        self._total_calls: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def attach(self, model: nn.Module) -> int:
        """
        Replace forward methods on DiT blocks in *model*.

        The replacement wrapper is decorated with @torch.compiler.disable so
        Inductor does not try to capture the cache logic into a CUDA graph.

        Returns:
            Number of blocks wrapped.
        """
        self.detach()  # clean slate
        self._last_output.clear()
        self._ema_residual.clear()
        self._prev_ema.clear()

        block_names = self._discover_block_names(model)
        if not block_names:
            logger.warning("MixCache: no DiT blocks found in model.")
            return 0

        name_set = set(block_names)
        for name, module in model.named_modules():
            if name in name_set:
                original_fwd = module.forward
                wrapper = self._make_forward_wrapper(name, original_fwd)
                module.forward = wrapper
                self._wrapped.append((module, name, original_fwd))

        logger.info("MixCache: wrapped %d DiT blocks.", len(self._wrapped))
        return len(self._wrapped)

    def detach(self) -> None:
        """Restore all original forward methods."""
        for module, _name, original_fwd in self._wrapped:
            module.forward = original_fwd
        self._wrapped.clear()

    def reset(self) -> None:
        """Clear cache between generations (keep wrappers in place)."""
        self._last_output.clear()
        self._ema_residual.clear()
        self._prev_ema.clear()
        self._current_step = 0
        self._step_hits = 0
        self._block_hits = 0
        self._total_calls = 0

    def pre_step(self, step_idx: int, total_steps: int, latents: Any) -> bool:
        """Called before each denoising step by OptimizationStack."""
        self._current_step = step_idx
        return False

    def post_step(self, step_idx: int, total_steps: int, latents: Any) -> None:
        pass

    def get_stats(self) -> Dict[str, Any]:
        total = max(self._total_calls, 1)
        return {
            "step_hit_rate":       self._step_hits  / total,
            "block_hit_rate":      self._block_hits / total,
            "total_block_calls":   self._total_calls,
            "total_blocks_wrapped": len(self._wrapped),
        }

    def wrap_model(self, model: nn.Module) -> None:
        """Alias for attach() to match OptimizationStack convention."""
        self.attach(model)

    # ------------------------------------------------------------------
    # Forward-wrapper factory
    # ------------------------------------------------------------------

    def _make_forward_wrapper(self, block_name: str, original_fwd: Callable) -> Callable:
        """
        Return a replacement for block.forward.

        Key design decisions (from HW5 and course content):

        1. Self-advancing step counter: instead of relying on an external pre_step()
           call (which the MG2 denoising loop doesn't make), we compute the step from
           the running total call count divided by the number of wrapped blocks.
           After N blocks execute = one denoising step completes.

        2. Step-level check uses DIRECT INPUT DRIFT vs the cached output, not
           EMA-to-prevEMA comparison.  In autoregressive generation x_{t} is the
           previous denoised output, so if x_{t} ≈ cached_{t-1} the block is stable.
           This works for any step count, including the 3-step MG2 schedule.

        3. Warmup seeds the EMA residual so step 1 has data to compare against.

        4. Block-level EMA check runs AFTER forward (can't skip compute here) but
           returns cached output when residual EMA is low, saving downstream work.
        """
        optimizer = self

        @torch.compiler.disable
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # ── Self-advancing step counter (HW5 insight: use call count) ───
            optimizer._total_calls += 1
            num_blocks = max(len(optimizer._wrapped), 1)
            step = (optimizer._total_calls - 1) // num_blocks
            optimizer._current_step = step

            # ── Warmup: full forward, seed EMA for future comparisons ───────
            if step < optimizer.warmup_steps:
                output = original_fwd(*args, **kwargs)
                optimizer._store(block_name, output)
                # Seed EMA residual during warmup so step 1 can compare
                x_in = args[0] if args else None
                out_tensor = _primary(output)
                if (x_in is not None
                        and isinstance(x_in, torch.Tensor)
                        and isinstance(out_tensor, torch.Tensor)
                        and x_in.shape == out_tensor.shape):
                    with torch.no_grad():
                        residual_mag = float((out_tensor - x_in).norm().item())
                    optimizer._ema_residual[block_name] = residual_mag
                    optimizer._prev_ema[block_name] = residual_mag
                return output

            # ── Step-level: direct input-vs-cached-output drift ─────────────
            # In autoregressive video generation the input at step t is the
            # denoised output from step t-1.  If it hasn't drifted much from
            # the cached value, the block is stable and we skip its forward.
            cached = optimizer._last_output.get(block_name)
            if cached is not None:
                x_in = args[0] if args else None
                cached_tensor = _primary(cached) if isinstance(cached, (tuple, list)) else cached
                if (x_in is not None
                        and isinstance(x_in, torch.Tensor)
                        and isinstance(cached_tensor, torch.Tensor)
                        and x_in.shape == cached_tensor.shape):
                    with torch.no_grad():
                        norm_cached = float(cached_tensor.norm().item())
                        drift = float(
                            ((x_in - cached_tensor).norm() / (norm_cached + 1e-8)).item()
                        )
                    if drift < optimizer.tau_step:
                        optimizer._step_hits += 1
                        return cached

            # ── Block-level EMA predictive skip (BEFORE forward) ────────────
            # Use the EMA change recorded in the PREVIOUS step as a predictor:
            # if the residual magnitude was stable then, it will likely be
            # stable now too — skip the forward entirely and return cached.
            # This is the key fix: checking AFTER forward (old code) computed
            # the block anyway, saving nothing; checking BEFORE saves GPU work.
            if (cached is not None
                    and block_name in optimizer._prev_ema
                    and block_name in optimizer._ema_residual
                    and optimizer._prev_ema[block_name] > 1e-8):
                prev_change = abs(
                    optimizer._ema_residual[block_name] - optimizer._prev_ema[block_name]
                ) / (optimizer._prev_ema[block_name] + 1e-8)
                if prev_change < optimizer.tau_block:
                    optimizer._block_hits += 1
                    return cached

            # ── Compute: run block normally ──────────────────────────────────
            output = original_fwd(*args, **kwargs)

            # ── Update EMA residual for future predictions ───────────────────
            x_in = args[0] if args else None
            out_tensor = _primary(output)
            if (x_in is not None
                    and isinstance(x_in, torch.Tensor)
                    and isinstance(out_tensor, torch.Tensor)
                    and x_in.shape == out_tensor.shape):
                with torch.no_grad():
                    residual_mag = float((out_tensor - x_in).norm().item())

                prev_ema_val = optimizer._ema_residual.get(block_name, residual_mag)
                new_ema = (
                    (1.0 - optimizer.ema_alpha) * prev_ema_val
                    + optimizer.ema_alpha * residual_mag
                )
                optimizer._prev_ema[block_name] = optimizer._ema_residual.get(block_name, new_ema)
                optimizer._ema_residual[block_name] = new_ema

            optimizer._store(block_name, output)
            return output

        return wrapper

    def _store(self, block_name: str, output: Any) -> None:
        """Store output (or first tensor of a tuple) for future comparisons."""
        if isinstance(output, (tuple, list)):
            t = output[0]
        else:
            t = output
        if isinstance(t, torch.Tensor):
            self._last_output[block_name] = output  # store full output for return

    # ------------------------------------------------------------------
    # Block discovery
    # ------------------------------------------------------------------

    def _discover_block_names(self, model: nn.Module) -> List[str]:
        """Return list of module names that look like DiT transformer blocks.

        Mirrors TaylorSeerOptimizer._discover_block_names() — uses class-name
        matching first, then falls back to standard attribute paths, then to the
        largest uniform nn.ModuleList.
        """
        _BLOCK_CLASS_NAMES = frozenset({
            "DiTBlock", "Block",
            "WanBlock",
            "TransformerBlock", "BasicTransformerBlock",
            "JointTransformerBlock", "MMDiTBlock",
            "FluxTransformerBlock", "SD3TransformerBlock",
        })
        found = []
        for name, module in model.named_modules():
            if type(module).__name__ in _BLOCK_CLASS_NAMES:
                found.append(name)
        if found:
            logger.info(
                "MixCache: discovered %d blocks by class name (%s).",
                len(found),
                type(next(
                    m for n, m in model.named_modules() if n == found[0]
                )).__name__,
            )
            return found

        # Fallback 1: standard attribute paths
        for attr_path in ("blocks", "layers", "transformer_blocks",
                          "model.blocks", "model.layers"):
            obj = model
            ok = True
            for part in attr_path.split("."):
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    ok = False
                    break
            if ok and isinstance(obj, (nn.ModuleList, nn.Sequential)) and len(obj) > 0:
                logger.info(
                    "MixCache: discovered %d blocks via attribute path '%s' (class: %s).",
                    len(obj), attr_path, type(obj[0]).__name__,
                )
                return [f"{attr_path}.{i}" for i in range(len(obj))]

        # Fallback 2: largest uniform nn.ModuleList
        best_name, best_len = "", 0
        for name, module in model.named_modules():
            if isinstance(module, (nn.ModuleList, nn.Sequential)) and len(module) >= 4:
                child_classes = {type(c).__name__ for c in module}
                if len(child_classes) == 1 and len(module) > best_len:
                    best_len = len(module)
                    best_name = name
        if best_name:
            logger.info(
                "MixCache: using largest uniform ModuleList '%s' (%d blocks).",
                best_name, best_len,
            )
            return [f"{best_name}.{i}" for i in range(best_len)]

        logger.warning(
            "MixCache: could not discover DiT blocks. "
            "Top-level children: %s",
            {type(c).__name__ for _, c in model.named_children()},
        )
        return []


# ------------------------------------------------------------------
# Module-level helper
# ------------------------------------------------------------------

def _primary(output: Any) -> Any:
    """Return the first tensor from a tuple/list output, or the output itself."""
    if isinstance(output, (tuple, list)):
        return output[0]
    return output
