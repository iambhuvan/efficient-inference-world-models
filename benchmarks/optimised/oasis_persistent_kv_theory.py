"""
Synthetic benchmark for persistent past-frame KV reuse.

This does not modify any existing Oasis files. It validates the tensor-level
theory: if past tokens are frozen, cached-past attention matches baseline
attention exactly while removing repeated "past projection" work in principle.

Run:
  python benchmarks/optimised/oasis_persistent_kv_theory.py
"""

from __future__ import annotations

import time

import torch

from worldserve.optimizations.system_level.kv_cache.oasis_persistent_kv_theory import (
    PersistentPastKVCache,
    attention_with_cached_past,
)


def _simulate(
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    batch: int = 1,
    heads: int = 8,
    d_head: int = 64,
    q_tokens: int = 144,
    past_tokens: int = 288,
    current_tokens: int = 144,
    steps: int = 10,
    layers: int = 32,
) -> dict:
    """
    Compare baseline vs cached-past attention outputs and timing.

    Baseline:
      recompute + use full K/V every step.
    Cached:
      first step captures past K/V, later steps reuse cached past + live current.
    """
    torch.manual_seed(0)
    q = torch.randn(batch, heads, q_tokens, d_head, device=device, dtype=dtype)

    # Past is frozen within a frame's denoising loop.
    k_past = torch.randn(batch, heads, past_tokens, d_head, device=device, dtype=dtype)
    v_past = torch.randn(batch, heads, past_tokens, d_head, device=device, dtype=dtype)

    # Current changes per step.
    k_curr = [torch.randn(batch, heads, current_tokens, d_head, device=device, dtype=dtype) for _ in range(steps)]
    v_curr = [torch.randn(batch, heads, current_tokens, d_head, device=device, dtype=dtype) for _ in range(steps)]

    # Baseline
    t0 = time.perf_counter()
    baseline_out = []
    for s in range(steps):
        k_full = torch.cat([k_past, k_curr[s]], dim=2)
        v_full = torch.cat([v_past, v_curr[s]], dim=2)
        # mimic per-layer repeated attention calls
        o = None
        for _ in range(layers):
            o = torch.nn.functional.scaled_dot_product_attention(q, k_full, v_full)
        baseline_out.append(o)
    baseline_ms = (time.perf_counter() - t0) * 1000.0

    # Cached past
    cache = PersistentPastKVCache()
    cache.start_frame(frame_idx=1)
    t1 = time.perf_counter()
    cached_out = []
    for s in range(steps):
        k_full = torch.cat([k_past, k_curr[s]], dim=2)
        v_full = torch.cat([v_past, v_curr[s]], dim=2)
        o = None
        for l in range(layers):
            o = attention_with_cached_past(
                query=q,
                full_key=k_full,
                full_value=v_full,
                num_past_tokens=past_tokens,
                cache=cache,
                layer_idx=l,
            )
        cached_out.append(o)
    cached_ms = (time.perf_counter() - t1) * 1000.0

    max_abs = 0.0
    for a, b in zip(baseline_out, cached_out):
        max_abs = max(max_abs, float((a - b).abs().max().item()))

    # Theoretical projection savings (not measured directly here).
    # Baseline recomputes past + current projections every step.
    # Cached path recomputes current every step + past once.
    baseline_proj_units = steps * (past_tokens + current_tokens)
    cached_proj_units = (past_tokens + steps * current_tokens)
    proj_reduction = 1.0 - (cached_proj_units / baseline_proj_units)

    return {
        "steps": steps,
        "layers": layers,
        "past_tokens": past_tokens,
        "current_tokens": current_tokens,
        "max_abs_diff": max_abs,
        "baseline_ms": round(baseline_ms, 3),
        "cached_ms": round(cached_ms, 3),
        "runtime_ratio_baseline_over_cached": round(
            (baseline_ms / cached_ms) if cached_ms > 0 else 0.0, 3
        ),
        "theoretical_projection_reduction_pct": round(proj_reduction * 100.0, 2),
    }


if __name__ == "__main__":
    out = _simulate()
    print("Persistent past-KV theory check")
    for k, v in out.items():
        print(f"{k}: {v}")

