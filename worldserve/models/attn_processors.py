"""
Custom diffusers AttnProcessor classes that inject WorldServe Triton kernels.

Three processors are provided:

SageAttnProcessor  — SageAttention2 INT8/FP8 quantized attention (D1/K2).
STAAttnProcessor   — Sliding Tile Attention for video DiTs (A1).
TempCacheAttnProcessor — TempCache KV deduplication via LSH (A3).

All processors follow the diffusers AttnProcessor protocol:
  __call__(attn, hidden_states, encoder_hidden_states=None,
           attention_mask=None, image_rotary_emb=None, **kwargs) -> Tensor

Apply via::

    pipe.transformer.set_attn_processor(SageAttnProcessor())

Requires: torch >= 2.1, diffusers >= 0.28, triton >= 2.2
"""

import math
import torch
import torch.nn.functional as F
from typing import Optional


def _infer_3d_dims(seq_len: int, T_hint: int, H_hint: int, W_hint: int):
    """
    Infer actual (T, H, W) token dims from runtime sequence length.

    T is typically correct (determined by frames + VAE temporal factor).
    H and W may be off by a DiT patchification factor (e.g. 2×2).
    Returns (T, H, W) such that T*H*W == seq_len, or (seq_len, 1, 1) as fallback.
    """
    if seq_len == T_hint * H_hint * W_hint:
        return T_hint, H_hint, W_hint
    if T_hint > 0 and seq_len % T_hint == 0:
        hw = seq_len // T_hint
        ratio = H_hint / max(W_hint, 1)
        W_f = max(1, round(math.sqrt(hw / ratio)))
        H_f = hw // W_f
        if H_f * W_f == hw:
            return T_hint, H_f, W_f
        for delta in range(1, W_f + 1):
            for w in (W_f - delta, W_f + delta):
                if w > 0 and hw % w == 0:
                    return T_hint, hw // w, w
    return seq_len, 1, 1

# diffusers import — present at runtime; guarded to allow offline import.
try:
    from diffusers.models.attention_processor import Attention
except ImportError:  # pragma: no cover
    Attention = object  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_heads(
    tensor: torch.Tensor, n_heads: int
) -> torch.Tensor:
    """Reshape (B, S, H*D) → (B, H, S, D)."""
    B, S, _ = tensor.shape
    head_dim = tensor.shape[-1] // n_heads
    return tensor.view(B, S, n_heads, head_dim).transpose(1, 2)


def _apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    image_rotary_emb,
) -> tuple:
    """Apply RoPE embeddings to Q and K using diffusers utility."""
    from diffusers.models.embeddings import apply_rotary_emb

    q = apply_rotary_emb(q, image_rotary_emb)
    k = apply_rotary_emb(k, image_rotary_emb)
    return q, k


def _merge_heads(tensor: torch.Tensor) -> torch.Tensor:
    """Reshape (B, H, S, D) → (B, S, H*D)."""
    B, H, S, D = tensor.shape
    return tensor.transpose(1, 2).reshape(B, S, H * D)


# ---------------------------------------------------------------------------
# SageAttnProcessor
# ---------------------------------------------------------------------------

class SageAttnProcessor:
    """Drop-in AttnProcessor using SageAttention2 INT8/FP8 on H100.

    Implements the D1 / K2 kernel from advanced_kernels.md:
      - Q, K: per-thread INT4 quantization before QK^T GEMM.
      - P, V: FP8 E4M3 with two-level FP22 accumulator fix.

    Falls back to standard scaled dot-product attention when the Triton
    kernel is unavailable (e.g. CPU, non-H100 hardware, import failure).

    Usage::

        pipe.transformer.set_attn_processor(SageAttnProcessor())
    """

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb=None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        kv_src = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

        q = attn.to_q(hidden_states)
        k = attn.to_k(kv_src)
        v = attn.to_v(kv_src)

        q = _split_heads(q, attn.heads)   # (B, H, S_q, D)
        k = _split_heads(k, attn.heads)   # (B, H, S_kv, D)
        v = _split_heads(v, attn.heads)

        if image_rotary_emb is not None:
            q, k = _apply_rope(q, k, image_rotary_emb)

        try:
            from ..kernels.triton.int4_fp8_attention import sage_attention_h100
            out = sage_attention_h100(q, k, v, softmax_scale=attn.scale)
        except (ImportError, AttributeError):
            # Fallback: standard scaled dot-product attention (torch >= 2.0)
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                scale=attn.scale,
            )

        out = _merge_heads(out)   # (B, S, H*D)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        return out


# ---------------------------------------------------------------------------
# STAAttnProcessor
# ---------------------------------------------------------------------------

class STAAttnProcessor:
    """AttnProcessor using Sliding Tile Attention for video DiTs.

    Implements the A1 kernel (arXiv 2502.04507, ICML 2025):
      - Tile boundaries are aligned with the attention window boundaries so
        every Q-K tile is either 100% attend or 100% skip (zero mixed tiles).
      - Supports 1D, 2D, and 3D spatial layouts.

    Falls back to full scaled dot-product attention when the Triton kernel
    is unavailable.

    Args:
        window_size: Number of tiles in the sliding window.
        layout: Token layout type: ``'1d'``, ``'2d'``, or ``'3d'``.
        temporal_size: Number of temporal frames (required for 3D layout).
        spatial_h: Spatial height in tokens (required for 2D/3D layout).
        spatial_w: Spatial width in tokens (required for 2D/3D layout).
    """

    def __init__(
        self,
        window_size: int = 16,
        layout: str = "3d",
        temporal_size: Optional[int] = None,
        spatial_h: Optional[int] = None,
        spatial_w: Optional[int] = None,
    ) -> None:
        self.window_size = window_size
        self.layout = layout
        self.T = temporal_size
        self.H = spatial_h
        self.W = spatial_w
        self._cached_dims = None  # lazily set on first forward pass

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb=None,
        **kwargs,
    ) -> torch.Tensor:
        kv_src = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

        q = attn.to_q(hidden_states)
        k = attn.to_k(kv_src)
        v = attn.to_v(kv_src)

        q = _split_heads(q, attn.heads)
        k = _split_heads(k, attn.heads)
        v = _split_heads(v, attn.heads)

        if image_rotary_emb is not None:
            q, k = _apply_rope(q, k, image_rotary_emb)

        try:
            from ..kernels.triton.sliding_tile_attention import sliding_tile_attention

            # Resolve spatial dims lazily from actual sequence length on first call.
            if self.layout == "3d" and self.T is not None:
                seq_len = hidden_states.shape[1]
                if self._cached_dims is None or self._cached_dims[0] * self._cached_dims[1] * self._cached_dims[2] != seq_len:
                    self._cached_dims = _infer_3d_dims(
                        seq_len,
                        self.T,
                        self.H or 1,
                        self.W or 1,
                    )
                T_tok, H_tok, W_tok = self._cached_dims
                spatial_dims = (T_tok, H_tok, W_tok)
            else:
                spatial_dims = None

            out = sliding_tile_attention(
                q, k, v,
                window_size=self.window_size,
                layout=self.layout,
                spatial_dims=spatial_dims,
            )
        except (ImportError, AttributeError):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                scale=attn.scale,
            )

        out = _merge_heads(out)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        return out


# ---------------------------------------------------------------------------
# TempCacheAttnProcessor
# ---------------------------------------------------------------------------

class TempCacheAttnProcessor:
    """AttnProcessor with TempCache KV deduplication (A3 from advanced_kernels.md).

    Implements the Duplicate Attention Lemma (arXiv 2602.01801):
      - Merges duplicate KV entries via LSH bucketing.
      - V_merged = mean of V in each bucket.
      - log(bucket_count) bias added in softmax (mathematically exact).

    Falls back to full scaled dot-product attention when the Triton kernel
    is unavailable.

    Args:
        threshold: L1 distance threshold for considering two K vectors
            identical (used by the Python-level fallback; the Triton kernel
            uses LSH sign-bits for bucketing).
    """

    def __init__(self, threshold: float = 1e-3) -> None:
        self.threshold = threshold

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb=None,
        **kwargs,
    ) -> torch.Tensor:
        kv_src = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

        q = attn.to_q(hidden_states)
        k = attn.to_k(kv_src)
        v = attn.to_v(kv_src)

        q = _split_heads(q, attn.heads)
        k = _split_heads(k, attn.heads)
        v = _split_heads(v, attn.heads)

        try:
            from ..kernels.triton.tempache import tempache_attention
            out = tempache_attention(q, k, v, threshold=self.threshold)
        except (ImportError, AttributeError):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                scale=attn.scale,
            )

        out = _merge_heads(out)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        return out
