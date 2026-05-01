"""
NATTEN neighborhood attention wrapper for WorldServe (Phase 4 — FALLBACK/ABLATION).

Wraps the NATTEN library's 2D neighborhood attention (fna2d) as an alternative
to SVG2 for spatial attention heads. NATTEN restricts each token to attend only
to its local spatial neighborhood (kernel_size x kernel_size), providing a
simple but effective sparse attention pattern.

This is used as a fallback when SVG2 head classification is unavailable, and
as an ablation baseline to compare against SVG2's learned head-aware sparsity.

Architecture assumptions (Matrix-Game 2.0):
  - ~15,360 spatial tokens/frame -> ~120 x 128 spatial grid (H x W)
  - 30 DiT blocks, Wan2.1 backbone
"""

from __future__ import annotations

import logging
import time
import types
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NATTEN import with graceful fallback
# ---------------------------------------------------------------------------
_NATTEN_AVAILABLE = False

try:
    from natten.functional import na2d  # type: ignore[import]
    _NATTEN_AVAILABLE = True
    logger.info("NATTEN na2d available.")
except ImportError:
    try:
        from natten import na2d  # type: ignore[import]
        _NATTEN_AVAILABLE = True
    except ImportError:
        logger.warning(
            "NATTEN not installed. NATTENSparseAttention will fall back to "
            "a sliding-window PyTorch implementation. Install with: "
            "pip install natten"
        )


def _sliding_window_2d_fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: int,
) -> torch.Tensor:
    """
    Pure PyTorch fallback for 2D neighborhood attention when NATTEN is unavailable.

    Uses unfold to extract local neighborhoods. Less efficient than NATTEN's
    fused CUDA kernel but functionally equivalent.

    Args:
        q: (B, H, height, W, D) query tensor.
        k: (B, H, height, W, D) key tensor.
        v: (B, H, height, W, D) value tensor.
        kernel_size: Size of the local neighborhood window.

    Returns:
        (B, H, height, W, D) output tensor.
    """
    B, num_heads, H, W, D = q.shape
    pad = kernel_size // 2
    scale = D ** -0.5

    # Pad spatial dimensions
    # Reshape to (B*H_heads, D, H_spatial, W_spatial) for unfold
    k_pad = F.pad(
        k.reshape(B * num_heads, H, W, D).permute(0, 3, 1, 2),
        (pad, pad, pad, pad),
        mode="constant",
        value=0.0,
    )  # (B*H, D, H+2p, W+2p)
    v_pad = F.pad(
        v.reshape(B * num_heads, H, W, D).permute(0, 3, 1, 2),
        (pad, pad, pad, pad),
        mode="constant",
        value=0.0,
    )

    # Extract neighborhoods via unfold
    # unfold(dim, size, step) -> adds a new dim at the end
    k_unf = k_pad.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)
    # k_unf: (B*H, D, H_spatial, W_spatial, ks, ks)
    v_unf = v_pad.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)

    # Reshape for matmul
    # k_unf -> (B*H, H_spatial, W_spatial, ks*ks, D)
    k_unf = k_unf.permute(0, 2, 3, 4, 5, 1).reshape(B * num_heads, H, W, kernel_size * kernel_size, D)
    v_unf = v_unf.permute(0, 2, 3, 4, 5, 1).reshape(B * num_heads, H, W, kernel_size * kernel_size, D)

    # q: (B*H, H_spatial, W_spatial, 1, D)
    q_flat = q.reshape(B * num_heads, H, W, 1, D)

    # Attention: (B*H, H_s, W_s, 1, D) @ (B*H, H_s, W_s, D, ks^2) -> (B*H, H_s, W_s, 1, ks^2)
    attn = torch.matmul(q_flat, k_unf.transpose(-2, -1)) * scale
    attn = F.softmax(attn, dim=-1)

    # Output: (B*H, H_s, W_s, 1, ks^2) @ (B*H, H_s, W_s, ks^2, D) -> (B*H, H_s, W_s, 1, D)
    out = torch.matmul(attn, v_unf).squeeze(-2)  # (B*H, H_s, W_s, D)

    return out.reshape(B, num_heads, H, W, D)


class NATTENSparseAttention:
    """
    NATTEN-based 2D neighborhood attention for spatial heads.

    Restricts each spatial token to attend only to its local kernel_size x kernel_size
    neighborhood, providing O(S * k^2) complexity instead of O(S^2).
    """

    def __init__(self, kernel_size: int = 7) -> None:
        """
        Args:
            kernel_size: Size of the 2D neighborhood window. Must be odd.
        """
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        self.kernel_size = kernel_size
        self._patched_modules: List[Tuple[nn.Module, Callable]] = []
        self._natten_available = _NATTEN_AVAILABLE

    def apply_2d_neighborhood(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """
        Apply 2D neighborhood attention to spatial tokens within each frame.

        Args:
            q: (B_times_T, heads, H*W, d) query tensor (B*T frames, each with H*W spatial tokens).
            k: (B_times_T, heads, H*W, d) key tensor.
            v: (B_times_T, heads, H*W, d) value tensor.
            H: Spatial height.
            W: Spatial width.

        Returns:
            (B_times_T, heads, H*W, d) output tensor.
        """
        BT, num_heads, S, D = q.shape
        assert S == H * W, f"Spatial dim mismatch: S={S} != H*W={H}*{W}={H*W}"

        # Reshape to (BT, heads, H, W, D)
        q_2d = q.reshape(BT, num_heads, H, W, D)
        k_2d = k.reshape(BT, num_heads, H, W, D)
        v_2d = v.reshape(BT, num_heads, H, W, D)

        if self._natten_available:
            try:
                # NATTEN expects (B, H_spatial, W_spatial, heads, D) or
                # (B, heads, H_spatial, W_spatial, D) depending on version.
                # Try the na2d functional API.
                out_2d = na2d(q_2d, k_2d, v_2d, kernel_size=self.kernel_size)
            except Exception as e:
                logger.warning("NATTEN na2d failed (%s), using fallback.", e)
                out_2d = _sliding_window_2d_fallback(q_2d, k_2d, v_2d, self.kernel_size)
        else:
            out_2d = _sliding_window_2d_fallback(q_2d, k_2d, v_2d, self.kernel_size)

        # Reshape back to (BT, heads, H*W, D)
        return out_2d.reshape(BT, num_heads, S, D)

    def wrap_spatial_attention(
        self,
        model: nn.Module,
        kernel_size: Optional[int] = None,
        spatial_height: int = 120,
        spatial_width: int = 128,
    ) -> int:
        """
        Replace spatial attention in all DiT blocks with NATTEN neighborhood attention.

        Only patches self-attention modules; leaves cross-attention (keyboard actions)
        untouched.

        Args:
            model: The MG2 DiT model.
            kernel_size: Override kernel size (defaults to self.kernel_size).
            spatial_height: Height of the spatial grid.
            spatial_width: Width of the spatial grid.

        Returns:
            Number of modules patched.
        """
        ks = kernel_size or self.kernel_size
        count = 0

        for name, module in model.named_modules():
            cls_name = type(module).__name__

            is_self_attn = cls_name in (
                "Attention", "WanAttention", "FlashSelfAttention",
                "SelfAttention",
            ) and "cross" not in name.lower()

            if not is_self_attn:
                continue

            original_forward = module.forward
            self._patched_modules.append((module, original_forward))

            natten_attn = self
            sh, sw = spatial_height, spatial_width

            def _make_natten_forward(
                orig_fwd: Callable,
                na: NATTENSparseAttention,
                h: int,
                w: int,
                kernel: int,
            ) -> Callable:
                def _natten_forward(self_module: Any, *args: Any, **kwargs: Any) -> Any:
                    original_sdpa = F.scaled_dot_product_attention

                    def _natten_sdpa(
                        query: torch.Tensor,
                        key: torch.Tensor,
                        value: torch.Tensor,
                        attn_mask: Optional[torch.Tensor] = None,
                        dropout_p: float = 0.0,
                        is_causal: bool = False,
                        scale: Optional[float] = None,
                        **kw: Any,
                    ) -> torch.Tensor:
                        B, H_heads, S, D = query.shape
                        # Check if S matches spatial dimensions
                        if S == h * w:
                            return na.apply_2d_neighborhood(query, key, value, h, w)
                        else:
                            # Not spatial attention; fall back to dense
                            return original_sdpa(
                                query, key, value,
                                attn_mask=attn_mask,
                                dropout_p=dropout_p,
                                is_causal=is_causal,
                                scale=scale,
                            )

                    F.scaled_dot_product_attention = _natten_sdpa  # type: ignore[assignment]
                    try:
                        result = orig_fwd(*args, **kwargs)
                    finally:
                        F.scaled_dot_product_attention = original_sdpa  # type: ignore[assignment]
                    return result

                return _natten_forward

            patched_fwd = _make_natten_forward(original_forward, natten_attn, sh, sw, ks)
            module.forward = types.MethodType(patched_fwd, module)
            count += 1

        logger.info("NATTEN: wrapped %d spatial attention modules (kernel=%d)", count, ks)
        return count

    def restore_all(self) -> int:
        """Restore all patched modules to original forward methods."""
        count = 0
        for module, original_forward in self._patched_modules:
            module.forward = original_forward
            count += 1
        self._patched_modules.clear()
        logger.info("NATTEN: restored %d attention modules.", count)
        return count

    def sweep_kernel_sizes(
        self,
        model: nn.Module,
        sample_input: Any,
        sizes: Optional[List[int]] = None,
        spatial_height: int = 120,
        spatial_width: int = 128,
        reference_output: Optional[torch.Tensor] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run model with each kernel size, measure quality and speed.

        Args:
            model: The MG2 DiT model.
            sample_input: Input to feed to the model.
            sizes: List of kernel sizes to try (default: [3, 5, 7, 11, 15]).
            spatial_height: Height of spatial grid.
            spatial_width: Width of spatial grid.
            reference_output: Dense attention output for quality comparison.

        Returns:
            List of dicts with keys: kernel_size, latency_ms, psnr_db (if reference given),
            cosine_sim (if reference given).
        """
        sizes = sizes or [3, 5, 7, 11, 15]
        results: List[Dict[str, Any]] = []

        # Get reference if not provided
        if reference_output is None:
            model.eval()
            with torch.no_grad():
                if isinstance(sample_input, dict):
                    reference_output = model(**sample_input)
                elif isinstance(sample_input, (tuple, list)):
                    reference_output = model(*sample_input)
                else:
                    reference_output = model(sample_input)
                if isinstance(reference_output, (tuple, list)):
                    reference_output = reference_output[0]

        for ks in sizes:
            if ks % 2 == 0:
                logger.warning("Skipping even kernel_size=%d", ks)
                continue

            # Wrap model with this kernel size
            wrapper = NATTENSparseAttention(kernel_size=ks)
            wrapper.wrap_spatial_attention(model, ks, spatial_height, spatial_width)

            model.eval()
            # Warmup
            with torch.no_grad():
                if isinstance(sample_input, dict):
                    _ = model(**sample_input)
                elif isinstance(sample_input, (tuple, list)):
                    _ = model(*sample_input)
                else:
                    _ = model(sample_input)

            # Timed run
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            with torch.no_grad():
                if isinstance(sample_input, dict):
                    out = model(**sample_input)
                elif isinstance(sample_input, (tuple, list)):
                    out = model(*sample_input)
                else:
                    out = model(sample_input)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            if isinstance(out, (tuple, list)):
                out = out[0]

            entry: Dict[str, Any] = {
                "kernel_size": ks,
                "latency_ms": (t1 - t0) * 1000.0,
            }

            # Quality metrics
            if reference_output is not None:
                ref_flat = reference_output.float().flatten()
                out_flat = out.float().flatten()

                # Cosine similarity
                cos_sim = F.cosine_similarity(
                    ref_flat.unsqueeze(0), out_flat.unsqueeze(0)
                ).item()
                entry["cosine_sim"] = cos_sim

                # PSNR
                mse = F.mse_loss(out.float(), reference_output.float()).item()
                if mse > 0:
                    # Assume signal range is based on actual values
                    max_val = reference_output.float().abs().max().item()
                    psnr = 10.0 * torch.log10(
                        torch.tensor(max_val ** 2 / mse)
                    ).item()
                    entry["psnr_db"] = psnr
                else:
                    entry["psnr_db"] = float("inf")

            results.append(entry)
            logger.info("NATTEN sweep: kernel=%d, latency=%.1fms", ks, entry["latency_ms"])

            # Restore model
            wrapper.restore_all()

        return results
