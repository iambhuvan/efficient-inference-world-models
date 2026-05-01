"""
Helios-Distilled + TeaCache step caching — Optimization Layer 8.

Uses diffusers' built-in TaylorSeerCacheConfig (the diffusers analogue of
TeaCache). Caches transformer outputs across denoising steps when residual
similarity exceeds threshold.

NOTE: Helios-Distilled is DMD-distilled to few steps — step-caching has
limited amortization headroom. This script is for RIGOR (negative result
documentation) and to confirm step-caching is correctly disqualified for
distilled models.

Expected speedup: 0.9×–1.2× on Distilled (likely no win, possibly small
regression from cache management overhead).

For Helios-Base (50 steps), this same script with `--variant base` would
likely give 2×–2.5× — that's the reason we keep this layer in the stack
for the Base variant.

Run:
    PYTHONPATH=benchmarks modal run benchmarks/optimised/helios_distilled_teacache_modal.py
"""

import sys
sys.path.insert(0, "/root")
sys.path.insert(0, "/root/benchmarks")

import modal
from modal_common import app, image_cuda_devel, hf_secret, model_volume, MODEL_CACHE
image = image_cuda_devel

HELIOS_REPO = "BestWishYsh/Helios-Distilled"
PAPER_FPS = 19.5
BASELINE_FPS = 0.94


@app.function(
    gpu="H100",
    image=image,
    secrets=[hf_secret],
    volumes={MODEL_CACHE: model_volume},
    timeout=3600,
    memory=65536,
)
def run_helios_distilled_teacache(
    cache_interval: int = 5,
    max_order: int = 1,
    disable_cache_before_step: int = 3,
    num_warmup: int = 1,
    num_iters: int = 3,
    seed: int = 42,
    prompt: str = (
        "A camera slowly pans through a sunlit forest clearing, golden light "
        "filtering through tall trees, soft wind moving the leaves."
    ),
) -> dict:
    """Enable TaylorSeer step caching on the transformer."""
    import os
    import statistics

    import torch
    from huggingface_hub import snapshot_download
    from diffusers import DiffusionPipeline

    hf_token = os.environ.get("HF_TOKEN", "").strip() or None

    print(f"Downloading {HELIOS_REPO} ...")
    local_dir = snapshot_download(HELIOS_REPO, cache_dir=MODEL_CACHE, token=hf_token)

    print("Loading HeliosPyramidPipeline (bf16, trust_remote_code=True) ...")
    pipe = DiffusionPipeline.from_pretrained(
        local_dir, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    # ── OPTIMIZATION: TaylorSeerCacheConfig step caching ────────────────
    cache_enabled = False
    try:
        from diffusers import TaylorSeerCacheConfig
        cfg = TaylorSeerCacheConfig(
            cache_interval=cache_interval,
            max_order=max_order,
            disable_cache_before_step=disable_cache_before_step,
        )
        if hasattr(pipe.transformer, "enable_cache"):
            pipe.transformer.enable_cache(cfg)
            cache_enabled = True
            print(f"TaylorSeerCacheConfig enabled: {cfg}")
        else:
            print("transformer.enable_cache not available; TeaCache may not be supported.")
    except ImportError as exc:
        print(f"TaylorSeerCacheConfig not in diffusers ({exc}). Skipping cache.")
    # ────────────────────────────────────────────────────────────────────

    n_params = sum(
        p.numel() for m in pipe.components.values()
        if isinstance(m, torch.nn.Module) for p in m.parameters()
    )

    @torch.inference_mode()
    def _run_once(seed_val: int) -> tuple[float, int]:
        gen = torch.Generator(device="cuda").manual_seed(seed_val)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        out = pipe(prompt=prompt, generator=gen, return_dict=True)
        e.record()
        torch.cuda.synchronize()

        frames = getattr(out, "frames", None)
        if frames is None:
            frames = getattr(out, "videos", None)
        return s.elapsed_time(e), _get_frame_count(frames)

    print(f"Warmup ({num_warmup} iters) ...")
    n_observed = 1
    for i in range(num_warmup):
        ms, n_observed = _run_once(seed_val=seed + i + 1000)
        print(f"  warmup {i + 1}: {ms:.0f} ms ({n_observed} frames)")

    torch.cuda.reset_peak_memory_stats()
    latencies: list[float] = []
    print(f"Timed runs ({num_iters} iters) ...")
    for i in range(num_iters):
        ms, n_observed = _run_once(seed_val=seed + i)
        latencies.append(ms)
        fps = n_observed / (ms / 1000.0)
        print(f"  iter {i + 1}: {ms:.0f} ms ({fps:.2f} fps)")

    mean_ms = statistics.mean(latencies)
    fps = n_observed / (mean_ms / 1000.0)

    return {
        "model": HELIOS_REPO,
        "kernel": "teacache_taylorseer",
        "cache_interval": cache_interval,
        "max_order": max_order,
        "disable_cache_before_step": disable_cache_before_step,
        "cache_enabled": cache_enabled,
        "num_frames_observed": n_observed,
        "n_params_B": round(n_params / 1e9, 3),
        "latency_ms_mean": round(mean_ms, 2),
        "latency_ms_std": round(statistics.stdev(latencies) if len(latencies) > 1 else 0.0, 2),
        "latency_per_frame_ms": round(mean_ms / n_observed, 2),
        "frames_per_sec": round(fps, 3),
        "speedup_vs_baseline": round(fps / BASELINE_FPS, 3),
        "baseline_fps": BASELINE_FPS,
        "paper_fps_target": PAPER_FPS,
        "vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "gpu": "H100",
        "raw_latencies_ms": latencies,
    }


def _get_frame_count(frames) -> int:
    if frames is None:
        return 1
    shape = getattr(frames, "shape", None)
    if shape is not None and len(shape) >= 4:
        return shape[0] if len(shape) == 4 else shape[1]
    if isinstance(frames, list):
        first = frames[0] if frames else None
        return len(first) if isinstance(first, list) else len(frames)
    return 1


@app.local_entrypoint(name="helios_distilled_teacache_modal")
def main(cache_interval: int = 5, max_order: int = 1, num_iters: int = 3):
    result = run_helios_distilled_teacache.remote(
        cache_interval=cache_interval, max_order=max_order, num_iters=num_iters,
    )
    import json
    print("\n" + json.dumps(result, indent=2))
    print(f"\nSpeedup vs baseline: {result.get('speedup_vs_baseline', 'N/A')}×")

    sys.path.insert(0, "/root/benchmarks")
    from result_store import save_result
    save_result(result, "optimised_kernels", "helios_distilled_teacache")
