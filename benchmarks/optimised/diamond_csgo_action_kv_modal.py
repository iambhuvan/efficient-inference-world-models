"""
DIAMOND CSGO + Action-Aware Cross-Frame Self-Attention K/V Reuse (speculative).

Caveat upfront: DIAMOND is a FiLM-conditioned UNet, NOT a transformer.  Action
enters as a single conditioning vector that scale-shifts feature maps; there
is no cross-attention to action and therefore no action-derived K/V tensors
in the canonical sense.  This script implements the closest analog as a
research probe.

What this does
--------------
Each `SelfAttention2d` layer in the UNet computes Q, K, V from a Conv1x1.
We hook every `SelfAttention2d.forward`:
  • Always recompute Q from the current x_t (frame-specific).
  • Cache (K, V) keyed by `(layer_id, denoise_step)` after a *full* frame.
  • On the next frame, when
        cosine(act_t, act_{t-1}) > τ_a   AND
        ||obs_t − obs_{t-1}||_F / ||obs_{t-1}||_F < δ_o
    pull (K, V) from the cache instead of computing them.
  • Q × K_cache^T then attends as usual.

The save is the QKV-projection compute (1/3 of the conv per attn layer)
plus the V projection — both K and V are skipped on cache-hit frames.  On
DIAMOND CSGO this is small per frame (UNet attn is at 5 depths max), so
the headline number is more about correctness/quality than raw speedup.
The bookkeeping reports `kv_hit_rate`, per-mode latency, and FPS.

Run:
    PYTHONPATH=benchmarks modal run benchmarks/optimised/diamond_csgo_action_kv_modal.py
"""

import sys
sys.path.insert(0, "/root")
sys.path.insert(0, "/root/benchmarks")
sys.path.insert(0, "/root/benchmarks/optimised")

from _diamond_common import (
    app, diamond_image, hf_secret, model_volume, MODEL_CACHE,
    load_diamond, make_result,
)


@app.function(
    gpu="H100",
    image=diamond_image,
    secrets=[hf_secret],
    volumes={MODEL_CACHE: model_volume},
    timeout=3600,
    memory=32768,
)
def run_action_kv(
    num_frames: int = 64,
    action_threshold: float = 0.95,    # τ_a: cosine sim on action vector
    obs_threshold: float = 0.05,       # δ_o: relative L2 on conditioning obs
    num_warmup: int = 4,
    num_iters: int = 3,
) -> dict:
    """Action-aware self-attention K/V reuse on DIAMOND's UNet.

    Per-frame flow:
        - reset step counter
        - decide should_reuse based on action+obs similarity to last frame
        - run sampler.sample → triggers many attn forward calls
        - patched forward consults cache when should_reuse=True
        - at end of frame, if it was a "fresh" frame, K/V cache is now
          populated and ready for reuse next frame
    """
    import math
    import statistics
    from typing import Optional

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    bundle = load_diamond()
    wm_env = bundle["wm_env"]
    agent = bundle["agent"]

    # --------------------------------------------------------------- discover
    # Walk both denoiser and upsampler, label each SelfAttention2d with a
    # unique integer id, and remember the original forward.
    from models.blocks import SelfAttention2d  # type: ignore

    attn_layers: list[SelfAttention2d] = []
    for m in agent.modules():
        if isinstance(m, SelfAttention2d):
            attn_layers.append(m)
    for i, m in enumerate(attn_layers):
        m._akv_layer_id = i
    print(f"  Hooking {len(attn_layers)} SelfAttention2d layers.")

    # --------------------------------------------------------------- controller
    class Ctrl:
        def __init__(self) -> None:
            self.cache: dict[tuple[int, int], tuple[Tensor, Tensor]] = {}
            self.step_idx = 0
            self.should_reuse = False
            self.kv_hits = 0
            self.kv_misses = 0
            self.last_act: Optional[Tensor] = None
            self.last_obs: Optional[Tensor] = None
            self.last_sim_act: float = 0.0
            self.last_obs_diff: float = 0.0
            self.fresh_frame_ms: list[float] = []
            self.reuse_frame_ms: list[float] = []

    ctrl = Ctrl()

    # --------------------------------------------------------------- patched fwd
    _ATTN_HEAD_DIM = 64  # matches blocks.py default

    def patched_attn_forward(self: SelfAttention2d, x: Tensor) -> Tensor:
        n, c, h, w = x.shape
        x_norm = self.norm(x)
        layer_id = getattr(self, "_akv_layer_id", -1)
        key = (layer_id, ctrl.step_idx)

        if ctrl.should_reuse and key in ctrl.cache:
            # Reuse path: compute only Q from current x; K, V from cache.
            qkv = self.qkv_proj(x_norm)
            qkv = qkv.view(n, self.n_head * 3, c // self.n_head, h * w).transpose(2, 3).contiguous()
            q, _, _ = qkv.chunk(3, dim=1)
            k, v = ctrl.cache[key]
            ctrl.kv_hits += 1
        else:
            # Fresh path: compute Q, K, V; store K, V.
            qkv = self.qkv_proj(x_norm)
            qkv = qkv.view(n, self.n_head * 3, c // self.n_head, h * w).transpose(2, 3).contiguous()
            q, k, v = qkv.chunk(3, dim=1)
            ctrl.cache[key] = (k.detach(), v.detach())
            ctrl.kv_misses += 1

        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(2, 3).reshape(n, c, h, w)
        return x + self.out_proj(y)

    SelfAttention2d.forward = patched_attn_forward    # global swap

    # --------------------------------------------------------------- step ctr
    # Wrap denoiser.denoise / upsampler.denoise so step_idx advances on
    # each model call. step_idx encodes which sigma we're at within the
    # current frame; sigmas are shared across frames so the cache key is
    # well-defined.
    base_denoise = agent.denoiser.denoise

    def patched_denoise(noisy_next_obs, sigma, sigma_cond, obs, act):
        out = base_denoise(noisy_next_obs, sigma, sigma_cond, obs, act)
        ctrl.step_idx += 1
        return out

    agent.denoiser.denoise = patched_denoise

    if agent.upsampler is not None:
        base_up = agent.upsampler.denoise

        def patched_up_denoise(noisy_next_obs, sigma, sigma_cond, obs, act):
            out = base_up(noisy_next_obs, sigma, sigma_cond, obs, act)
            ctrl.step_idx += 1
            return out

        agent.upsampler.denoise = patched_up_denoise

    # --------------------------------------------------------------- gate hook
    # Decide should_reuse just before each wm_env.step. Compare the latest
    # action and conditioning obs against the previous frame's.
    base_step = wm_env.step

    def gated_step(act):
        # Read what the buffer will contain *for this step*.
        # act_buffer[:, -1] is overwritten with `act` inside step(); the
        # rest of the buffer is the past 3 frames + the new one we just
        # got. For gating purposes we use `act` and the latest obs.
        with torch.no_grad():
            curr_act = act.reshape(act.shape[0], -1).float()
            curr_obs = wm_env.obs_buffer[:, -1].reshape(act.shape[0], -1).float()

            decision = False
            sim_a = 0.0
            obs_diff = 1.0
            if ctrl.last_act is not None and ctrl.last_obs is not None:
                sim_a = F.cosine_similarity(
                    curr_act, ctrl.last_act, dim=-1,
                ).mean().item()
                # relative L2 on the conditioning obs latent
                num = (curr_obs - ctrl.last_obs).norm()
                den = ctrl.last_obs.norm().clamp_min(1e-6)
                obs_diff = (num / den).item()
                decision = (sim_a > action_threshold) and (obs_diff < obs_threshold)

            ctrl.should_reuse = bool(decision and len(ctrl.cache) > 0)
            ctrl.step_idx = 0
            ctrl.last_sim_act = float(sim_a)
            ctrl.last_obs_diff = float(obs_diff)

        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        result = base_step(act)
        e.record()
        torch.cuda.synchronize()
        ms = s.elapsed_time(e)
        if ctrl.should_reuse:
            ctrl.reuse_frame_ms.append(ms)
        else:
            ctrl.fresh_frame_ms.append(ms)

        # Update last_* AFTER the step using the now-committed obs/act.
        with torch.no_grad():
            ctrl.last_act = curr_act.detach().clone()
            ctrl.last_obs = curr_obs.detach().clone()
        return result

    wm_env.step = gated_step

    # --------------------------------------------------------------- timing
    @torch.inference_mode()
    def _run_once() -> tuple[float, int]:
        wm_env.reset()
        future_acts = wm_env.next_act
        n_future = future_acts.shape[0]
        torch.cuda.synchronize()
        t_start = torch.cuda.Event(enable_timing=True)
        t_end = torch.cuda.Event(enable_timing=True)
        t_start.record()
        for t in range(num_frames):
            act = future_acts[t % n_future].unsqueeze(0)
            wm_env.step(act)
        t_end.record()
        torch.cuda.synchronize()
        return t_start.elapsed_time(t_end), num_frames

    print(f"Warmup ({num_warmup} iters) ...")
    for i in range(num_warmup):
        # reset bookkeeping each warmup
        ctrl.kv_hits = ctrl.kv_misses = 0
        ctrl.fresh_frame_ms.clear(); ctrl.reuse_frame_ms.clear()
        ms, _ = _run_once()
        n_calls = ctrl.kv_hits + ctrl.kv_misses
        hit_rate = ctrl.kv_hits / max(1, n_calls)
        print(f"  warmup {i + 1}: {ms:.0f} ms total, kv_hit_rate={hit_rate:.2%}, "
              f"fresh_frames={len(ctrl.fresh_frame_ms)}, "
              f"reuse_frames={len(ctrl.reuse_frame_ms)}")

    # Steady-state
    ctrl.kv_hits = ctrl.kv_misses = 0
    ctrl.fresh_frame_ms.clear(); ctrl.reuse_frame_ms.clear()
    torch.cuda.reset_peak_memory_stats()
    iter_totals: list[float] = []
    print(f"Timed runs ({num_iters} iters) ...")
    for i in range(num_iters):
        ms, n = _run_once()
        iter_totals.append(ms)
        n_calls = ctrl.kv_hits + ctrl.kv_misses
        hit_rate = ctrl.kv_hits / max(1, n_calls)
        fps = n / (ms / 1000.0)
        print(f"  iter {i + 1}: {ms:.0f} ms ({n} frames, {fps:.2f} fps, "
              f"kv_hit_rate={hit_rate:.2%}, sim_a={ctrl.last_sim_act:.3f}, "
              f"obs_diff={ctrl.last_obs_diff:.3f})")

    if not iter_totals:
        return {"error": "all iters failed"}

    mean_total = statistics.mean(iter_totals)
    fps = num_frames / (mean_total / 1000.0)
    n_calls = ctrl.kv_hits + ctrl.kv_misses
    hit_rate = ctrl.kv_hits / max(1, n_calls)
    avg_fresh = statistics.mean(ctrl.fresh_frame_ms) if ctrl.fresh_frame_ms else 0.0
    avg_reuse = statistics.mean(ctrl.reuse_frame_ms) if ctrl.reuse_frame_ms else 0.0
    frame_reuse_rate = (
        len(ctrl.reuse_frame_ms)
        / max(1, len(ctrl.reuse_frame_ms) + len(ctrl.fresh_frame_ms))
    )

    timing = {
        "num_frames": num_frames,
        "latency_ms_mean_total": round(mean_total, 2),
        "latency_per_frame_ms_mean": round(mean_total / num_frames, 3),
        "frames_per_sec": round(fps, 3),
        "vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "raw_iter_totals_ms": iter_totals,
    }
    return make_result(
        kernel="diamond_action_kv_reuse",
        bundle=bundle,
        timing=timing,
        extras={
            "action_threshold": action_threshold,
            "obs_threshold": obs_threshold,
            "n_attn_layers_hooked": len(attn_layers),
            "kv_hit_rate_per_attn_call": round(hit_rate, 4),
            "frame_reuse_rate": round(frame_reuse_rate, 4),
            "avg_frame_ms_fresh": round(avg_fresh, 3),
            "avg_frame_ms_reuse": round(avg_reuse, 3),
            "last_action_similarity": round(ctrl.last_sim_act, 4),
            "last_obs_relative_diff": round(ctrl.last_obs_diff, 4),
        },
    )


@app.local_entrypoint(name="diamond_csgo_action_kv_modal")
def main(
    num_frames: int = 64,
    action_threshold: float = 0.95,
    obs_threshold: float = 0.05,
    num_warmup: int = 4,
    num_iters: int = 3,
):
    result = run_action_kv.remote(
        num_frames=num_frames,
        action_threshold=action_threshold,
        obs_threshold=obs_threshold,
        num_warmup=num_warmup,
        num_iters=num_iters,
    )
    import json
    print("\nDIAMOND CSGO + action-aware K/V reuse result:")
    print(json.dumps(result, indent=2))

    sys.path.insert(0, "/root/benchmarks")
    from result_store import save_result
    save_result(
        result, "optimised_kernels",
        f"diamond_csgo_action_kv_a{int(action_threshold * 100)}_o{int(obs_threshold * 100)}_f{num_frames}",
    )
