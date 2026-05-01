"""
Isolated experimental wrapper for Oasis TemporalAxialAttention.

Goal:
- Reuse projected past-frame temporal K/V across denoising steps within a frame.
- Recompute projections only for the current frame token slice.
- Keep implementation separate from upstream/open-oasis sources.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F


@dataclass
class TemporalRuntimeState:
    frame_idx: int = -1
    step_idx: int = 0
    enabled: bool = False


class TemporalPersistentKVRuntime:
    """
    Runtime controller and cache for all wrapped temporal attention layers.
    """

    def __init__(self) -> None:
        self.state = TemporalRuntimeState()
        # layer_idx -> {"q_past": Tensor, "k_past": Tensor, "v_past": Tensor}
        self.cache: Dict[int, Dict[str, torch.Tensor]] = {}

    def start_frame(self, frame_idx: int) -> None:
        self.state.frame_idx = int(frame_idx)
        self.state.step_idx = 0
        self.cache.clear()

    def set_step(self, step_idx: int) -> None:
        self.state.step_idx = int(step_idx)

    def enable(self, on: bool) -> None:
        self.state.enabled = bool(on)


class PersistentTemporalAxialAttention(nn.Module):
    """
    Drop-in replacement for open-oasis TemporalAxialAttention.

    It matches output shape/semantics but can reuse past-frame projected Q/K/V
    (prefix along T axis) from step 0 for later denoising steps.
    """

    def __init__(
        self,
        base_attn: nn.Module,
        layer_idx: int,
        runtime: TemporalPersistentKVRuntime,
    ) -> None:
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.runtime = runtime

        # Copy architecture fields
        self.inner_dim = int(base_attn.inner_dim)
        self.heads = int(base_attn.heads)
        self.head_dim = int(base_attn.head_dim)
        self.rotary_emb = base_attn.rotary_emb
        self.is_causal = bool(base_attn.is_causal)

        # Clone weights/bias exactly
        self.to_qkv = nn.Linear(base_attn.to_qkv.in_features, base_attn.to_qkv.out_features, bias=False)
        self.to_out = nn.Linear(base_attn.to_out.in_features, base_attn.to_out.out_features, bias=True)
        self.to_qkv.load_state_dict(base_attn.to_qkv.state_dict())
        self.to_out.load_state_dict(base_attn.to_out.state_dict())
        # Keep wrapper parameters on the same device/dtype as the source module.
        self.to(
            device=base_attn.to_qkv.weight.device,
            dtype=base_attn.to_qkv.weight.dtype,
        )

    def _full_projection(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        return q, k, v

    def _mixed_projection_with_cache(self, x: torch.Tensor) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Reuse projected prefix [0:T-1) from cache and project only current T-1:T.
        Returns None if cache miss/shape mismatch; caller should fallback to full.
        """
        B, T, H, W, _ = x.shape
        if T <= 1:
            return None
        entry = self.runtime.cache.get(self.layer_idx)
        if entry is None:
            return None

        q_past = entry["q_past"]
        k_past = entry["k_past"]
        v_past = entry["v_past"]
        # Expected past prefix shape is (B, T-1, H, W, inner_dim)
        expected = (B, T - 1, H, W, self.inner_dim)
        if tuple(q_past.shape) != expected:
            return None

        q_cur, k_cur, v_cur = self.to_qkv(x[:, -1:, :, :, :]).chunk(3, dim=-1)
        q = torch.cat([q_past, q_cur], dim=1)
        k = torch.cat([k_past, k_cur], dim=1)
        v = torch.cat([v_past, v_cur], dim=1)
        return q, k, v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H, W, _ = x.shape
        use_cache = self.runtime.state.enabled and T > 1 and self.runtime.state.step_idx > 0

        q: torch.Tensor
        k: torch.Tensor
        v: torch.Tensor
        if use_cache:
            mixed = self._mixed_projection_with_cache(x)
            if mixed is None:
                q, k, v = self._full_projection(x)
            else:
                q, k, v = mixed
        else:
            q, k, v = self._full_projection(x)
            # Cache past prefix from step 0 (or when disabled, skip)
            if self.runtime.state.enabled and T > 1:
                self.runtime.cache[self.layer_idx] = {
                    "q_past": q[:, :-1, :, :, :].detach(),
                    "k_past": k[:, :-1, :, :, :].detach(),
                    "v_past": v[:, :-1, :, :, :].detach(),
                }

        q = rearrange(q, "B T H W (h d) -> (B H W) h T d", h=self.heads)
        k = rearrange(k, "B T H W (h d) -> (B H W) h T d", h=self.heads)
        v = rearrange(v, "B T H W (h d) -> (B H W) h T d", h=self.heads)

        q = self.rotary_emb.rotate_queries_or_keys(q, self.rotary_emb.freqs)
        k = self.rotary_emb.rotate_queries_or_keys(k, self.rotary_emb.freqs)
        q, k, v = map(lambda t: t.contiguous(), (q, k, v))

        out = F.scaled_dot_product_attention(query=q, key=k, value=v, is_causal=self.is_causal)
        out = rearrange(out, "(B H W) h T d -> B T H W (h d)", B=B, H=H, W=W)
        out = out.to(q.dtype)
        return self.to_out(out)


def wrap_model_temporal_attention(model: nn.Module, runtime: TemporalPersistentKVRuntime) -> int:
    """
    Replace all block.t_attn modules with PersistentTemporalAxialAttention.

    Returns number of wrapped temporal layers.
    """
    wrapped = 0
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        return 0
    for idx, block in enumerate(blocks):
        t_attn = getattr(block, "t_attn", None)
        if t_attn is None:
            continue
        block.t_attn = PersistentTemporalAxialAttention(t_attn, layer_idx=idx, runtime=runtime)
        wrapped += 1
    return wrapped
