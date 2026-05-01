"""
worldserve/kernels/__init__.py — WorldServe custom CUDA kernel exports.

Active Oasis kernels (after pruning unused RoPE / topk / gather-scatter):
    FusedAdaLN         — nn.Module wrapper for fused AdaLN-zero
    quantize_int4      — FP16 → packed INT4 + per-group scales
    dequantize_int4    — packed INT4 + scales → FP16
    quantize_fp8       — FP16 → FP8 E4M3 (uint8) + block scales
    dequantize_fp8     — FP8 E4M3 + scales → FP16

Triton kernels live in `worldserve.kernels.triton`:
    flash_attention, int4_fp8_attention (sage), tempache, sliding_tile_attention
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

# Lazy import: kernels compiled on first use
_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        from .load import load_kernels
        _ext = load_kernels()
    return _ext


# ─── FusedAdaLN (nn.Module) ──────────────────────────────────────────────────

class FusedAdaLN(nn.Module):
    """
    Fused AdaLN-zero: LayerNorm + scale + shift + gate in one CUDA kernel.

    y = alpha * ((x - mean) / sqrt(var + eps)) * (1 + gamma) + beta

    Args:
        hidden_dim: Feature dimension D. Must be one of {128, 256, 512, 1024}.
        eps:        Epsilon for numerical stability (default 1e-5).

    Forward:
        x:     [B, T, D]  fp16 — input activations
        gamma: [B, D]     fp16 — conditioning scale from timestep MLP
        beta:  [B, D]     fp16 — conditioning shift from timestep MLP
        alpha: [B, D]     fp16 — conditioning gate from timestep MLP

    Returns:
        out:   [B, T, D]  fp16
    """

    def __init__(self, hidden_dim: int, eps: float = 1e-5):
        super().__init__()
        assert hidden_dim in (128, 256, 512, 1024), \
            f"FusedAdaLN: hidden_dim must be in {{128,256,512,1024}}, got {hidden_dim}"
        self.hidden_dim = hidden_dim
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        assert x.is_cuda and x.dtype == torch.float16, \
            "FusedAdaLN expects CUDA fp16 input"
        return _get_ext().fused_adaln(x, gamma, beta, alpha, self.eps)


# ─── INT4 quantization ────────────────────────────────────────────────────────

def quantize_int4(
    input: torch.Tensor,
    group_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize fp16 tensor to packed INT4 with per-group scales.

    Args:
        input:      [N] or any shape fp16 CUDA tensor (flattened internally)
        group_size: number of elements per quantization group (default 128)

    Returns:
        packed: [(N+1)//2] uint8 — two INT4 values per byte
        scales: [ceil(N/group_size)] fp16 — per-group absmax/7 scales
    """
    flat = input.reshape(-1).contiguous()
    return _get_ext().quantize_int4(flat, group_size)


def dequantize_int4(
    packed: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    N: int,
    out_shape: Optional[torch.Size] = None,
) -> torch.Tensor:
    """
    Dequantize packed INT4 back to fp16.
    """
    out = _get_ext().dequantize_int4(packed, scales, group_size, N)
    if out_shape is not None:
        out = out.reshape(out_shape)
    return out


# ─── FP8 quantization ────────────────────────────────────────────────────────

def quantize_fp8(
    input: torch.Tensor,
    scale: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor:
    """
    Quantize fp16 tensor to FP8 E4M3 (stored as uint8).
    """
    flat = input.reshape(-1).contiguous()
    return _get_ext().quantize_fp8(flat, scale.contiguous(), block_size)


def dequantize_fp8(
    packed: torch.Tensor,
    scale: torch.Tensor,
    block_size: int = 128,
    out_shape: Optional[torch.Size] = None,
) -> torch.Tensor:
    """
    Dequantize FP8 E4M3 (uint8) back to fp16.
    """
    out = _get_ext().dequantize_fp8(packed, scale.contiguous(), block_size)
    if out_shape is not None:
        out = out.reshape(out_shape)
    return out


__all__ = [
    "FusedAdaLN",
    "quantize_int4",
    "dequantize_int4",
    "quantize_fp8",
    "dequantize_fp8",
]
