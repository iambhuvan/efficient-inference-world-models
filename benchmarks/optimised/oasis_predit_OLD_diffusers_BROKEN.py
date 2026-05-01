"""
Modal H100 benchmark: Open-Oasis 500M + PrediT AB-2 extrapolation only.

Applies PrediT with predict_every=2 (AB-2 alpha=1.5).  The step callback
advances the cache counter at the end of each denoising step via
callback_on_step_end.  cache.reset() is called between timed iterations.
Hooks are removed after timing.

Run:
    modal run modal/optimised/oasis_prediT_modal.py
"""

import sys
sys.path.insert(0, "/root")
sys.path.insert(0, "/root/benchmarks")
sys.path.insert(0, "/root/worldserve/optimizations")

import modal
from modal_common import app, image, hf_secret, model_volume, MODEL_CACHE


@app.function(
    gpu="H100",
    image=image,
    secrets=[hf_secret],
    volumes={MODEL_CACHE: model_volume},
    timeout=3600,
    memory=32768,
)
def run_oasis_prediT(
    num_frames: int = 16,
    height: int = 360,
    width: int = 640,
    num_steps: int = 20,
    num_warmup: int = 1,
    num_iters: int = 1,
    predict_every: int = 2,
    ab_alpha: float = 1.5,
    prompt: str = "A first-person view of walking through a Minecraft forest",
) -> dict:
    """
    Run Open-Oasis 500M with PrediT AB-2 feature extrapolation and return
    timing results.
    """
    import json
    import os
    import statistics

    import torch

    from prediT import PrediTCache, apply_prediT, make_prediT_step_callback

    hf_token = os.environ.get("HF_TOKEN", "").strip() or None
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN is required for the gated Etched/oasis-500m model."
        )

    print(f"Loading Etched/oasis-500m (cache={MODEL_CACHE}) ...")
    try:
        from diffusers import DiffusionPipeline
        pipe = DiffusionPipeline.from_pretrained(
            "Etched/oasis-500m",
            torch_dtype=torch.bfloat16,
            cache_dir=MODEL_CACHE,
            token=hf_token,
        ).to("cuda")
    except Exception as exc:
        print(f"ERROR loading pipeline: {exc}")
        raise

    if not hasattr(pipe, "transformer"):
        raise RuntimeError("Pipeline has no .transformer attribute; cannot apply PrediT.")

    cache = PrediTCache(ab_alpha=ab_alpha)
    cache, hooks = apply_prediT(pipe.transformer, cache=cache, predict_every=predict_every)
    step_callback = make_prediT_step_callback(cache)
    print(
        f"PrediT attached: {len(hooks) // 2} blocks hooked, "
        f"predict_every={predict_every}, alpha={ab_alpha}"
    )

    generator = torch.Generator(device="cuda").manual_seed(42)

    def _run_once() -> float:
        cache.reset()
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        pipe(
            prompt=prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=num_steps,
            generator=generator,
            output_type="latent",
            callback_on_step_end=lambda *_a, **_kw: step_callback() or {},
        )
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end)

    print(f"Warmup ({num_warmup} iters) ...")
    for i in range(num_warmup):
        try:
            _run_once()
            print(f"  warmup {i + 1}/{num_warmup} done")
        except torch.cuda.OutOfMemoryError as oom:
            print(f"  warmup {i + 1}: OOM — {oom}; skipping")
            torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()
    latencies = []
    print(f"Timed runs ({num_iters} iters) ...")
    for i in range(num_iters):
        try:
            ms = _run_once()
            latencies.append(ms)
            fps = num_frames / (ms / 1000.0)
            print(f"  iter {i + 1}: {ms:.1f} ms  ({fps:.2f} fps)")
        except torch.cuda.OutOfMemoryError as oom:
            print(f"  iter {i + 1}: OOM — {oom}; skipping")
            torch.cuda.empty_cache()

    for h in hooks:
        h.remove()

    if not latencies:
        return {
            "error": "All iterations OOM",
            "model": "Etched/oasis-500m",
            "kernel": "PrediT AB-2",
        }

    mean_ms = statistics.mean(latencies)
    std_ms  = statistics.stdev(latencies) if len(latencies) > 1 else 0.0

    speedup_vs_baseline = None
    baseline_path = "/root/benchmarks/baseline/results/oasis_baseline.json"
    try:
        with open(baseline_path) as fh:
            bl = json.load(fh)
        bl_mean = bl.get("latency_ms", {}).get("total_mean")
        if bl_mean and mean_ms > 0:
            speedup_vs_baseline = round(bl_mean / mean_ms, 4)
    except Exception:
        pass

    results = {
        "model": "Etched/oasis-500m",
        "kernel": "PrediT AB-2",
        "prediT": {
            "predict_every": predict_every,
            "ab_alpha": ab_alpha,
            "blocks_hooked": len(hooks) // 2,
        },
        "num_frames": num_frames,
        "height": height,
        "width": width,
        "steps": num_steps,
        "mean_ms": round(mean_ms, 2),
        "std_ms": round(std_ms, 2),
        "frames_per_sec": round(num_frames / (mean_ms / 1000.0), 3),
        "vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "speedup_vs_baseline": speedup_vs_baseline,
        "gpu": "H100",
        "raw_latencies_ms": latencies,
    }
    print(json.dumps(results, indent=2))
    return results


@app.local_entrypoint(name="oasis_prediT_modal")
def main():
    result = run_oasis_prediT.remote()
    print("\nOpen-Oasis 500M (PrediT AB-2 only) result:")
    import json
    print(json.dumps(result, indent=2))
