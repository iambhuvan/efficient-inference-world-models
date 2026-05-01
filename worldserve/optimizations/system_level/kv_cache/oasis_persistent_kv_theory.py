"""
Persistent past-frame KV cache prototype for Oasis-style attention.

This module is intentionally isolated from existing model code. It implements
the tensor-level logic needed to reuse past-frame K/V while recomputing only
current-frame K/V.

Assumptions:
- Attention tensors are shaped (B, H, S, D) where S = S_past + S_current.
- Past-frame tokens are frozen within a frame's denoising loop.
- Current-frame tokens change each denoising step and must be recomputed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class LayerKV:
    key: torch.Tensor
    value: torch.Tensor


class PersistentPastKVCache:
    """
    Cache only past-frame K/V per layer, keyed by frame index.

    The caller owns how layer indices map to concrete attention modules.
    """

    def __init__(self) -> None:
        self._cache: Dict[int, LayerKV] = {}
        self._frame_idx: int = -1

    def clear(self) -> None:
        self._cache.clear()
        self._frame_idx = -1

    def start_frame(self, frame_idx: int) -> None:
        """
        Move to a new target frame.

        Cached past K/V remains valid across denoising steps for this frame.
        """
        self._frame_idx = frame_idx

    def has_layer(self, layer_idx: int) -> bool:
        return layer_idx in self._cache

    def set_past_kv(self, layer_idx: int, key_past: torch.Tensor, value_past: torch.Tensor) -> None:
        self._cache[layer_idx] = LayerKV(
            key=key_past.detach().clone(),
            value=value_past.detach().clone(),
        )

    def get_past_kv(self, layer_idx: int) -> LayerKV:
        if layer_idx not in self._cache:
            raise KeyError(f"Layer {layer_idx} not found in past-KV cache")
        return self._cache[layer_idx]


def split_past_current_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    num_past_tokens: int,
) -> tuple[LayerKV, LayerKV]:
    """
    Split full K/V into past and current token ranges.
    """
    if key.dim() != 4 or value.dim() != 4:
        raise ValueError("Expected key/value tensors with shape (B, H, S, D)")
    if key.shape != value.shape:
        raise ValueError("key/value shapes must match")
    if not (0 <= num_past_tokens <= key.shape[2]):
        raise ValueError("num_past_tokens must be within [0, S]")

    past = LayerKV(key=key[:, :, :num_past_tokens, :], value=value[:, :, :num_past_tokens, :])
    current = LayerKV(key=key[:, :, num_past_tokens:, :], value=value[:, :, num_past_tokens:, :])
    return past, current


def compose_cached_past_with_current(
    cached_past: LayerKV,
    current: LayerKV,
) -> LayerKV:
    """
    Reconstruct full K/V by concatenating cached past with live current.
    """
    if cached_past.key.dim() != 4 or current.key.dim() != 4:
        raise ValueError("Expected (B, H, S, D) tensors")
    if cached_past.key.shape[:2] != current.key.shape[:2]:
        raise ValueError("Batch/head dimensions must match")
    if cached_past.key.shape[3] != current.key.shape[3]:
        raise ValueError("Head dimension D must match")

    key = torch.cat([cached_past.key, current.key], dim=2)
    value = torch.cat([cached_past.value, current.value], dim=2)
    return LayerKV(key=key, value=value)


def attention_with_cached_past(
    query: torch.Tensor,
    full_key: torch.Tensor,
    full_value: torch.Tensor,
    num_past_tokens: int,
    cache: PersistentPastKVCache,
    layer_idx: int,
) -> torch.Tensor:
    """
    Reference attention path:
    - First call for a layer caches past K/V from full_key/full_value.
    - Subsequent calls reuse cached past K/V and only consume live current K/V.
    """
    past, current = split_past_current_kv(full_key, full_value, num_past_tokens)
    if not cache.has_layer(layer_idx):
        cache.set_past_kv(layer_idx, past.key, past.value)
    cached_past = cache.get_past_kv(layer_idx)
    merged = compose_cached_past_with_current(cached_past, current)
    return torch.nn.functional.scaled_dot_product_attention(query, merged.key, merged.value)

