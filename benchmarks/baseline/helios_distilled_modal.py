"""
Modal H100 benchmark: Helios-Distilled 14B baseline (no optimizations).

Model    : BestWishYsh/Helios-Distilled — DMD-distilled Wan2.1-T2V-14B finetune
Pipeline : HeliosPyramidPipeline (custom diffusers pipeline, trust_remote_code=True)
Task     : Text-to-video, 832x480, 81 frames (~5 s @ 16 fps)
Paper    : arXiv 2603.04379 (Yuan et al., Mar 2026) — published 19.5 FPS H100

`num_inference_steps` is left unset so the pipeline uses its built-in
distilled schedule (which is what produces the paper's 19.5 FPS). Override
explicitly only if you know the distilled step count.

Run:
    PYTHONPATH=benchmarks modal run benchmarks/baseline/helios_distilled_modal.py
"""

import sys
sys.path.insert(0, "/root")
sys.path.insert(0, "/root/benchmarks")

import modal
from modal_common import app, image_cuda_devel, hf_secret, model_volume, MODEL_CACHE

# Use the cuda-devel image so flash-attn can build (debian_slim image has no
# nvcc → flash-attn install fails → PyTorch SDPA fallback → ~10× slower).
image = image_cuda_devel

HELIOS_REPO = "BestWishYsh/Helios-Distilled"
PAPER_FPS = 19.5  # Yuan et al., arXiv 2603.04379


@app.function(
    gpu="H100",
    image=image,
    secrets=[hf_secret],
    volumes={MODEL_CACHE: model_volume},
    timeout=3600,
    memory=65536,  # 64 GB host RAM — 14B model + diffusers overhead
)
def run_helios_distilled_baseline(
    num_frames: int | None = None,   # None = pipeline default
    num_steps: int | None = None,    # None = pipeline default (paper config)
    width: int | None = None,        # None = pipeline default
    height: int | None = None,       # None = pipeline default
    guidance_scale: float = 5.0,     # auto-ignored on DMD scheduler
    seed: int = 42,
    num_warmup: int = 1,
    num_iters: int = 3,              # Distilled is fast — 3 timed runs
    prompt: str = (
        "A camera slowly pans through a sunlit forest clearing, golden light "
        "filtering through tall trees, soft wind moving the leaves."
    ),
) -> dict:
    """
    Pull weights once into the Modal volume (~30 GB), load via diffusers
    with trust_remote_code=True (HeliosPyramidPipeline lives in the repo
    pipeline.py), and time end-to-end generation with torch.cuda.Event.
    """
    import json
    import os
    import statistics

    import torch
    from huggingface_hub import snapshot_download

    hf_token = os.environ.get("HF_TOKEN", "").strip() or None

    # -----------------------------------------------------------------------
    # Snapshot the full diffusers-format repo into the persistent volume.
    # Subsequent calls hit the cache instantly.
    # -----------------------------------------------------------------------
    print(f"Downloading {HELIOS_REPO} ...")
    local_dir = snapshot_download(
        HELIOS_REPO,
        cache_dir=MODEL_CACHE,
        token=hf_token,
    )
    print(f"Helios-Distilled local_dir: {local_dir}")

    # -----------------------------------------------------------------------
    # Load HeliosPyramidPipeline (custom — trust_remote_code).
    # -----------------------------------------------------------------------
    from diffusers import DiffusionPipeline

    print("Loading HeliosPyramidPipeline (trust_remote_code=True) in bf16 ...")
    pipe = DiffusionPipeline.from_pretrained(
        local_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)

    n_params = sum(
        p.numel()
        for module in pipe.components.values()
        if isinstance(module, torch.nn.Module)
        for p in module.parameters()
    )
    print(f"HeliosPyramidPipeline loaded: {n_params / 1e9:.2f}B params total.")

    # -----------------------------------------------------------------------
    # Single timed run — one full T2V generation.
    # -----------------------------------------------------------------------
    _last_n_frames = num_frames or 0  # populated inside _run_once after first call

    @torch.inference_mode()
    def _run_once(seed_val: int) -> float:
        gen = torch.Generator(device="cuda").manual_seed(seed_val)

        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        # Only pass kwargs that were explicitly set — let HeliosPyramidPipeline
        # use its own defaults for everything else. The pyramid sampler has
        # tight constraints on latent dimensions; the safest reproduction of
        # the paper's 19.5 FPS is the pipeline's own default config.
        call_kwargs: dict = {
            "prompt": prompt,
            "guidance_scale": guidance_scale,
            "generator": gen,
            "return_dict": True,
        }
        if num_frames is not None:
            call_kwargs["num_frames"] = num_frames
        if num_steps is not None:
            call_kwargs["num_inference_steps"] = num_steps
        if width is not None:
            call_kwargs["width"] = width
        if height is not None:
            call_kwargs["height"] = height

        out = pipe(**call_kwargs)
        # Capture actual frame count produced (pipeline may use its own default).
        # Avoid `or` here because numpy/tensor truth-values raise ValueError.
        nonlocal _last_n_frames
        frames_attr = getattr(out, "frames", None)
        if frames_attr is None:
            frames_attr = getattr(out, "videos", None)
        if frames_attr is not None:
            shape = getattr(frames_attr, "shape", None)
            if shape is not None and len(shape) >= 4:
                # Likely (T,C,H,W) or (B,T,C,H,W) or (T,H,W,C)
                _last_n_frames = shape[0] if len(shape) == 4 else shape[1]
            elif isinstance(frames_attr, list):
                first = frames_attr[0] if frames_attr else None
                _last_n_frames = (
                    len(first) if isinstance(first, list) else len(frames_attr)
                )

        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end)

    # -----------------------------------------------------------------------
    # Warmup (compiles kernels, populates caches) — not timed.
    # -----------------------------------------------------------------------
    print(f"Warmup ({num_warmup} iters) ...")
    for i in range(num_warmup):
        _ = _run_once(seed_val=seed + i + 1000)
        print(f"  warmup {i + 1}/{num_warmup} done")

    # -----------------------------------------------------------------------
    # Timed runs.
    # -----------------------------------------------------------------------
    torch.cuda.reset_peak_memory_stats()
    latencies: list[float] = []
    print(f"Timed runs ({num_iters} iters) ...")
    for i in range(num_iters):
        try:
            ms = _run_once(seed_val=seed + i)
            latencies.append(ms)
            n = _last_n_frames if _last_n_frames > 0 else 1
            fps = n / (ms / 1000.0)
            print(
                f"  iter {i + 1}: {ms:.1f} ms  "
                f"({n} frames, {fps:.2f} fps, {ms / n:.1f} ms/frame)"
            )
        except torch.cuda.OutOfMemoryError as oom:
            print(f"  iter {i + 1}: OOM — {oom}; skipping")
            torch.cuda.empty_cache()

    if not latencies:
        return {"error": "All iterations OOM", "model": HELIOS_REPO}

    # Use observed frame count if pipeline picked its own default
    n_frames_observed = _last_n_frames if _last_n_frames > 0 else (num_frames or 1)

    mean_ms = statistics.mean(latencies)
    fps = n_frames_observed / (mean_ms / 1000.0)

    results = {
        "model": HELIOS_REPO,
        "kernel": "baseline_helios_pyramid_pipeline_bf16",
        "num_frames_requested": num_frames,
        "num_frames_observed": n_frames_observed,
        "num_steps_override": num_steps,
        "width": width,
        "height": height,
        "guidance_scale": guidance_scale,
        "n_params_B": round(n_params / 1e9, 3),
        "latency_ms_mean": round(mean_ms, 2),
        "latency_ms_std": round(
            statistics.stdev(latencies) if len(latencies) > 1 else 0.0, 2
        ),
        "latency_per_frame_ms": round(mean_ms / n_frames_observed, 2),
        "frames_per_sec": round(fps, 3),
        "paper_fps_target": PAPER_FPS,
        "paper_fps_match_pct": round(100.0 * fps / PAPER_FPS, 2),
        "vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "gpu": "H100",
        "raw_latencies_ms": latencies,
    }
    print("\n" + json.dumps(results, indent=2))

    if "frames_per_sec" in results:
        delta = results["frames_per_sec"] - PAPER_FPS
        sign = "+" if delta >= 0 else ""
        print(
            f"\nVs paper: {results['frames_per_sec']:.2f} FPS "
            f"({sign}{delta:.2f} vs paper's {PAPER_FPS} FPS, "
            f"{results['paper_fps_match_pct']:.1f}% match)"
        )

    return results


@app.local_entrypoint(name="helios_distilled_modal")
def main(
    num_frames: int = -1,          # -1 sentinel → pipeline default
    num_steps: int = -1,           # -1 sentinel → pipeline default
    width: int = -1,
    height: int = -1,
    num_warmup: int = 1,
    num_iters: int = 3,
):
    frames_arg: int | None = None if num_frames < 0 else num_frames
    steps_arg: int | None = None if num_steps < 0 else num_steps
    width_arg: int | None = None if width < 0 else width
    height_arg: int | None = None if height < 0 else height
    result = run_helios_distilled_baseline.remote(
        num_frames=frames_arg,
        num_steps=steps_arg,
        width=width_arg,
        height=height_arg,
        num_warmup=num_warmup,
        num_iters=num_iters,
    )

    print("\nHelios-Distilled baseline result:")
    import json
    print(json.dumps(result, indent=2))

    sys.path.insert(0, "/root/benchmarks")
    from result_store import save_result
    step_tag = f"s{num_steps}" if num_steps >= 0 else "sdefault"
    frames_tag = f"f{num_frames}" if num_frames >= 0 else "fdefault"
    save_result(
        result,
        "baseline",
        f"helios_distilled_baseline_{frames_tag}_{step_tag}",
    )
