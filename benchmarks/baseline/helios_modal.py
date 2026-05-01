"""
Modal H100 benchmark: Helios-Base 14B world model baseline (no optimizations).

Model    : BestWishYsh/Helios-Base — Wan2.1-T2V-14B finetune
Pipeline : HeliosPipeline (custom diffusers pipeline, trust_remote_code=True)
Task     : Text-to-video, 50-step diffusion, 832x480, 81 frames (~5 s @ 16 fps)
Paper    : arXiv 2603.04379 (Yuan et al., Mar 2026)

The published 19.5 FPS H100 figure is for the *distilled* variant
(BestWishYsh/Helios-Distilled). Helios-Base is the full-quality 50-step model
— expect ~2-5 FPS on the bare baseline. The larger denoising-step count is
why we chose Base for the WorldServe optimization study: every per-step
acceleration (sage attn, KV quant, sparse attn) compounds 50× here.

Run:
    PYTHONPATH=benchmarks modal run benchmarks/baseline/helios_modal.py
"""

import sys
sys.path.insert(0, "/root")
sys.path.insert(0, "/root/benchmarks")

import modal
from modal_common import app, image, hf_secret, model_volume, MODEL_CACHE

HELIOS_REPO = "BestWishYsh/Helios-Base"


@app.function(
    gpu="H100",
    image=image,
    secrets=[hf_secret],
    volumes={MODEL_CACHE: model_volume},
    timeout=3600,
    memory=65536,  # 64 GB host RAM — 14B model + diffusers overhead
)
def run_helios_baseline(
    num_frames: int = 81,
    num_steps: int = 50,
    width: int = 832,
    height: int = 480,
    guidance_scale: float = 5.0,
    seed: int = 42,
    num_warmup: int = 1,
    num_iters: int = 1,
    prompt: str = (
        "A camera slowly pans through a sunlit forest clearing, golden light "
        "filtering through tall trees, soft wind moving the leaves."
    ),
) -> dict:
    """
    Run Helios-Base text-to-video generation and report timing.

    Pulls weights once into the Modal volume (~28 GB FP16 weights + VAE).
    Timed runs use torch.cuda.Event for GPU-side timing only — no Python
    dispatch in the measured window.
    """
    import json
    import os
    import statistics

    import torch
    from huggingface_hub import snapshot_download

    hf_token = os.environ.get("HF_TOKEN", "").strip() or None

    # -----------------------------------------------------------------------
    # Snapshot the full diffusers-format repo into the persistent volume.
    # On second run this is a no-op; subsequent calls are instant.
    # -----------------------------------------------------------------------
    print(f"Downloading {HELIOS_REPO} ...")
    local_dir = snapshot_download(
        HELIOS_REPO,
        cache_dir=MODEL_CACHE,
        token=hf_token,
        # Diffusers needs the full layout — config.json, model_index.json,
        # transformer/, vae/, text_encoder/, scheduler/, etc.
    )
    print(f"Helios-Base local_dir: {local_dir}")

    # -----------------------------------------------------------------------
    # Load via custom HeliosPipeline (trust_remote_code is required because
    # the pipeline class lives in the repo's pipeline.py, not in diffusers
    # core).
    # -----------------------------------------------------------------------
    from diffusers import DiffusionPipeline

    print("Loading HeliosPipeline (trust_remote_code=True) in bf16 ...")
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
    print(f"HeliosPipeline loaded: {n_params / 1e9:.2f}B params total.")

    # -----------------------------------------------------------------------
    # Single timed run = one full text-to-video generation.
    # -----------------------------------------------------------------------
    @torch.inference_mode()
    def _run_once(seed_val: int) -> float:
        gen = torch.Generator(device="cuda").manual_seed(seed_val)

        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        _ = pipe(
            prompt=prompt,
            num_frames=num_frames,
            num_inference_steps=num_steps,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            generator=gen,
            return_dict=True,
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
            fps = num_frames / (ms / 1000.0)
            print(
                f"  iter {i + 1}: {ms:.1f} ms  "
                f"({fps:.2f} fps,  {ms / num_frames:.1f} ms/frame)"
            )
        except torch.cuda.OutOfMemoryError as oom:
            print(f"  iter {i + 1}: OOM — {oom}; skipping")
            torch.cuda.empty_cache()

    if not latencies:
        return {"error": "All iterations OOM", "model": HELIOS_REPO}

    mean_ms = statistics.mean(latencies)
    fps = num_frames / (mean_ms / 1000.0)

    results = {
        "model": HELIOS_REPO,
        "kernel": "baseline_helios_pipeline_bf16",
        "num_frames": num_frames,
        "num_steps": num_steps,
        "width": width,
        "height": height,
        "guidance_scale": guidance_scale,
        "n_params_B": round(n_params / 1e9, 3),
        "latency_ms_mean": round(mean_ms, 2),
        "latency_ms_std": round(
            statistics.stdev(latencies) if len(latencies) > 1 else 0.0, 2
        ),
        "latency_per_frame_ms": round(mean_ms / num_frames, 2),
        "frames_per_sec": round(fps, 3),
        "vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "gpu": "H100",
        "raw_latencies_ms": latencies,
    }
    print("\n" + json.dumps(results, indent=2))
    return results


@app.local_entrypoint(name="helios_modal")
def main(
    num_frames: int = 81,
    num_steps: int = 50,
    num_warmup: int = 1,
    num_iters: int = 1,
):
    result = run_helios_baseline.remote(
        num_frames=num_frames,
        num_steps=num_steps,
        num_warmup=num_warmup,
        num_iters=num_iters,
    )

    print("\nHelios-Base baseline result:")
    import json
    print(json.dumps(result, indent=2))

    sys.path.insert(0, "/root/benchmarks")
    from result_store import save_result
    save_result(
        result,
        "baseline",
        f"helios_baseline_f{num_frames}_s{num_steps}",
    )
