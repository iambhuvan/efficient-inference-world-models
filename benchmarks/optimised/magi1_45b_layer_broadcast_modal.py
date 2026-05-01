"""
MAGI-1 4.5B-base + Layer-Conditioned Attention-Output Broadcast.

Schedule derivation
-------------------
Source: benchmarks/runs/profiling/magi1_45b_attn_profile_f96_s64.json
(11,424 K-projection records, 34 layers × 336 calls).

For each layer we computed mean `cos_prev` (cosine sim between consecutive
K projections of the same layer).  Three observed regimes:

  cos_prev > 0.92   : K nearly static → attn output is a near-constant
                      function of the step → broadcast every 4 steps
  cos_prev 0.85-0.92: moderate drift → broadcast every 2 steps
  cos_prev < 0.85   : volatile → recompute every step

Layers per bucket (locked from the profile):
  very_stable: [1, 4, 17, 26, 31, 32, 33]
  moderate   : [2, 3, 5, 6, 7, 8, 9, 15, 21, 25, 27, 29]
  volatile   : [0, 10, 11, 12, 13, 14, 16, 18, 19, 20, 22, 23, 24, 28, 30]

Implementation
--------------
We wrap each `TransformerLayer.self_attention.forward` with a thin shim
that:
  1. Tracks an integer step counter per layer.
  2. If `(step - last_compute_step) < horizon` AND the input shape matches
     the cached shape, return the cached attention output.
  3. Else run the original forward, store the output (cloned) + shape,
     advance.

The shape check guards against chunk boundaries where past-K context
length changes — broadcast resets automatically there.

Run:
    PYTHONPATH=benchmarks modal run benchmarks/optimised/magi1_45b_layer_broadcast_modal.py
"""

import sys
sys.path.insert(0, "/root")
sys.path.insert(0, "/root/benchmarks")

import os

from baseline.magi1_45b_baseline_modal import (    # type: ignore
    MAGI_REPO,
    MAGI_LOAD_REL,
    T5_PRETRAINED_REL,
    VAE_PRETRAINED_REL,
    MAGI_DOWNLOAD_REL,
    T5_DOWNLOAD_REL,
    VAE_DOWNLOAD_REL,
    MAGI_CONFIG_REL,
    DEFAULT_PROMPT,
    image,
)
from modal_common import app, hf_secret, model_volume, MODEL_CACHE


# Locked schedule from the profile.
LAYER_BUCKET = {}
for lid in [1, 4, 17, 26, 31, 32, 33]:
    LAYER_BUCKET[lid] = ("very_stable", 4)
for lid in [2, 3, 5, 6, 7, 8, 9, 15, 21, 25, 27, 29]:
    LAYER_BUCKET[lid] = ("moderate", 2)
for lid in [0, 10, 11, 12, 13, 14, 16, 18, 19, 20, 22, 23, 24, 28, 30]:
    LAYER_BUCKET[lid] = ("volatile", 1)


@app.function(
    gpu="H100",
    image=image,
    secrets=[hf_secret],
    volumes={MODEL_CACHE: model_volume},
    timeout=10800,
    memory=131072,
)
def run_magi1_layer_broadcast(
    num_frames: int = 96,
    num_steps: int = 64,
    width: int = 720,
    height: int = 720,
    seed: int = 42,
    num_warmup: int = 1,
    num_iters: int = 1,
    prompt: str = DEFAULT_PROMPT,
    horizon_very_stable: int = 4,
    horizon_moderate: int = 2,
    horizon_volatile: int = 1,
) -> dict:
    """One T2V generation with per-layer attention-output broadcast."""
    import json
    import statistics

    import torch
    from huggingface_hub import snapshot_download

    hf_token = os.environ.get("HF_TOKEN", "").strip() or None
    print(f"Downloading {MAGI_REPO} ...")
    local_dir = snapshot_download(
        MAGI_REPO,
        cache_dir=MODEL_CACHE,
        token=hf_token,
        allow_patterns=[
            f"{MAGI_DOWNLOAD_REL}/**",
            f"{T5_DOWNLOAD_REL}/**",
            f"{VAE_DOWNLOAD_REL}/**",
            "README.md",
        ],
    )

    load_dir = os.path.join(local_dir, MAGI_LOAD_REL)
    t5_dir = os.path.join(local_dir, T5_PRETRAINED_REL)
    vae_dir = os.path.join(local_dir, VAE_PRETRAINED_REL)

    src_cfg = os.path.join("/root/MAGI-1", MAGI_CONFIG_REL)
    with open(src_cfg) as fh:
        cfg = json.load(fh)
    cfg["runtime_config"]["load"] = load_dir
    cfg["runtime_config"]["t5_pretrained"] = t5_dir
    cfg["runtime_config"]["vae_pretrained"] = vae_dir
    cfg["runtime_config"]["num_frames"] = num_frames
    cfg["runtime_config"]["num_steps"] = num_steps
    cfg["runtime_config"]["video_size_h"] = height
    cfg["runtime_config"]["video_size_w"] = width
    cfg["runtime_config"]["seed"] = seed
    patched = "/tmp/magi1_layer_broadcast.json"
    with open(patched, "w") as fh:
        json.dump(cfg, fh, indent=2)

    os.chdir("/root/MAGI-1")
    sys.path.insert(0, "/root/MAGI-1")
    from inference.pipeline import MagiPipeline  # type: ignore

    print("Constructing MagiPipeline ...")
    pipeline = MagiPipeline(patched)

    # Override the schedule horizons if the user passed non-default values.
    custom_horizons = {
        "very_stable": horizon_very_stable,
        "moderate":    horizon_moderate,
        "volatile":    horizon_volatile,
    }
    layer_horizon = {
        lid: custom_horizons[label] for lid, (label, _) in LAYER_BUCKET.items()
    }

    # ─── broadcast bookkeeping ────────────────────────────────────────────
    state = {
        "compute_calls": 0,
        "broadcast_calls": 0,
        "shape_mismatch_resets": 0,
        "per_layer_compute": {lid: 0 for lid in LAYER_BUCKET},
        "per_layer_broadcast": {lid: 0 for lid in LAYER_BUCKET},
    }

    def _make_wrapped(layer_id: int, original):
        horizon = layer_horizon[layer_id]
        cache = {"out": None, "in_shape": None, "step_at_compute": -(10**9)}
        local = {"step_counter": 0}

        def wrapped(*args, **kwargs):
            # Identify input shape signature (first positional tensor).
            signature = None
            for arg in args:
                if hasattr(arg, "shape"):
                    signature = tuple(arg.shape)
                    break
            if signature is None:
                for v in kwargs.values():
                    if hasattr(v, "shape"):
                        signature = tuple(v.shape)
                        break

            elapsed = local["step_counter"] - cache["step_at_compute"]
            shape_match = cache["in_shape"] == signature
            can_broadcast = (
                cache["out"] is not None
                and shape_match
                and elapsed < horizon
                and horizon > 1
            )
            if can_broadcast:
                state["broadcast_calls"] += 1
                state["per_layer_broadcast"][layer_id] += 1
                local["step_counter"] += 1
                return cache["out"]

            if cache["in_shape"] is not None and not shape_match:
                state["shape_mismatch_resets"] += 1

            out = original(*args, **kwargs)
            # Store a clone so downstream in-place ops can't corrupt the cache.
            if isinstance(out, tuple):
                cached = tuple(
                    o.detach().clone() if hasattr(o, "detach") else o for o in out
                )
            elif hasattr(out, "detach"):
                cached = out.detach().clone()
            else:
                cached = out

            cache["out"] = cached
            cache["in_shape"] = signature
            cache["step_at_compute"] = local["step_counter"]

            state["compute_calls"] += 1
            state["per_layer_compute"][layer_id] += 1
            local["step_counter"] += 1
            return out
        return wrapped

    # Hook each transformer layer's self_attention.forward.
    from inference.model.dit import get_dit as _orig_get_dit  # type: ignore

    def _patched_get_dit(*args, **kwargs):
        dit = _orig_get_dit(*args, **kwargs)
        # Walk and find TransformerLayer-like modules. MAGI-1's layers each
        # have a `self_attention` attribute (FullyParallelAttention).
        layer_id = 0
        n_hooked = 0
        for name, mod in dit.named_modules():
            if hasattr(mod, "self_attention") and hasattr(mod.self_attention, "forward"):
                # Only wrap actual transformer layers (avoid re-wrapping the
                # outer container that itself has a self_attention attr).
                # Heuristic: name ends with a numeric index in a `.layers.N`
                # path, or matches the per-layer naming pattern.
                if ".layers." in name and name.split(".")[-1].isdigit():
                    if layer_id in layer_horizon:
                        attn = mod.self_attention
                        attn.forward = _make_wrapped(layer_id, attn.forward)
                        n_hooked += 1
                    layer_id += 1
        print(f"  Wrapped {n_hooked}/{len(layer_horizon)} layer self-attentions.")
        return dit

    import inference.pipeline.pipeline as _pl_mod   # type: ignore
    _pl_mod.get_dit = _patched_get_dit

    # ─── timing ────────────────────────────────────────────────────────────
    @torch.inference_mode()
    def _generate(seed_val: int, output_path: str) -> tuple[float, int]:
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        pipeline.run_text_to_video(prompt=prompt, output_path=output_path)
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e), num_frames

    print(f"Warmup ({num_warmup} iters) ...")
    n_obs = num_frames
    for i in range(num_warmup):
        ms, n_obs = _generate(seed + i + 1000, f"/tmp/magi_lb_warmup_{i}.mp4")
        print(f"  warmup {i + 1}: {ms:.0f} ms")
        # zero out counters between iters so steady-state numbers are clean
        state["compute_calls"] = 0
        state["broadcast_calls"] = 0
        for lid in state["per_layer_compute"]:
            state["per_layer_compute"][lid] = 0
            state["per_layer_broadcast"][lid] = 0

    torch.cuda.reset_peak_memory_stats()
    latencies: list[float] = []
    print(f"Timed runs ({num_iters} iters) ...")
    for i in range(num_iters):
        try:
            ms, n_obs = _generate(seed + i, f"/tmp/magi_lb_timed_{i}.mp4")
            latencies.append(ms)
            fps = n_obs / (ms / 1000.0)
            tot_calls = state["compute_calls"] + state["broadcast_calls"]
            br_rate = state["broadcast_calls"] / max(1, tot_calls)
            print(f"  iter {i + 1}: {ms:.0f} ms ({fps:.3f} fps), "
                  f"broadcast_rate={br_rate:.2%} "
                  f"({state['broadcast_calls']:,}/{tot_calls:,})")
        except torch.cuda.OutOfMemoryError as oom:
            print(f"  iter {i + 1}: OOM — {oom}")
            torch.cuda.empty_cache()

    if not latencies:
        return {"error": "all OOM"}

    mean_ms = statistics.mean(latencies)
    fps = n_obs / (mean_ms / 1000.0)
    tot = state["compute_calls"] + state["broadcast_calls"]
    br_rate = state["broadcast_calls"] / max(1, tot)

    return {
        "model": MAGI_REPO,
        "variant": "4.5B_base",
        "kernel": "magi1_layer_conditioned_broadcast",
        "num_frames": num_frames,
        "num_steps": num_steps,
        "horizons": custom_horizons,
        "schedule": {
            lid: {"bucket": LAYER_BUCKET[lid][0], "horizon": layer_horizon[lid]}
            for lid in sorted(LAYER_BUCKET)
        },
        "compute_calls": state["compute_calls"],
        "broadcast_calls": state["broadcast_calls"],
        "broadcast_rate": round(br_rate, 4),
        "shape_mismatch_resets": state["shape_mismatch_resets"],
        "per_layer_compute": state["per_layer_compute"],
        "per_layer_broadcast": state["per_layer_broadcast"],
        "latency_ms_mean": round(mean_ms, 2),
        "latency_per_frame_ms": round(mean_ms / n_obs, 2),
        "frames_per_sec": round(fps, 3),
        "vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "gpu": "H100",
        "raw_latencies_ms": latencies,
    }


@app.local_entrypoint(name="magi1_45b_layer_broadcast_modal")
def main(
    num_frames: int = 96,
    num_steps: int = 64,
    horizon_very_stable: int = 4,
    horizon_moderate: int = 2,
    horizon_volatile: int = 1,
):
    result = run_magi1_layer_broadcast.remote(
        num_frames=num_frames, num_steps=num_steps,
        horizon_very_stable=horizon_very_stable,
        horizon_moderate=horizon_moderate,
        horizon_volatile=horizon_volatile,
    )
    import json
    print("\nMAGI-1 layer-broadcast result:")
    print(json.dumps(
        {k: v for k, v in result.items() if k not in ("schedule", "per_layer_compute", "per_layer_broadcast")},
        indent=2,
    ))

    sys.path.insert(0, "/root/benchmarks")
    from result_store import save_result
    save_result(
        result, "optimised_kernels",
        f"magi1_45b_layer_broadcast_h{horizon_very_stable}_{horizon_moderate}_{horizon_volatile}",
    )
