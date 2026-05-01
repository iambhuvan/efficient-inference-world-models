"""
Modal H100 benchmark: HunyuanVideo 13B baseline (no optimizations).

Model    : tencent/HunyuanVideo
Pipeline : HunyuanVideoPipeline (diffusers native — no trust_remote_code)
Task     : Text-to-video, 50-step diffusion, 720×1280 × 129 frames
License  : Tencent (gated free; academic OK, commercial restricted), 49.7K dl

Published benchmarks at this configuration:
  - TeaCache:   1.95× (arXiv 2411.19108 — direct HunyuanVideo measurement)
  - SageAttn2:  2–3× (FP8 P·V, arXiv 2411.10958 table 4)
  - FlashAttention-3: 2.0× over FA2
  - STA sparse 3D: 1.5–2× (arXiv 2502.13070)
  - Stack ceiling (paper-validated): 4–6×

This is the **highest absolute speedup target** in the open ecosystem
because it's >80 % attention-bound at 115K tokens. But it's also the
slowest baseline (~30 min/gen), so we run num_iters=1 by default.

Run:
    PYTHONPATH=benchmarks modal run benchmarks/baseline/hunyuanvideo_baseline_modal.py
"""

import sys
sys.path.insert(0, "/root")
sys.path.insert(0, "/root/benchmarks")

import modal
from modal_common import app, image_cuda_devel, hf_secret, model_volume, MODEL_CACHE
image = image_cuda_devel

HUNYUAN_REPO = "tencent/HunyuanVideo"
DEFAULT_PROMPT = (
    "A cat walks on the grass, realistic style. Cinematic shot, "
    "warm sunlight, smooth camera motion, 4k detail."
)


@app.function(
    gpu="H100",
    image=image,
    secrets=[hf_secret],
    volumes={MODEL_CACHE: model_volume},
    timeout=10800,               # 3 hr — HunyuanVideo gens can take 30 min each
    memory=65536,
)
def run_hunyuanvideo_baseline(
    num_frames: int = 129,
    num_steps: int = 50,
    width: int = 1280,
    height: int = 720,
    guidance_scale: float = 6.0,
    seed: int = 42,
    num_warmup: int = 1,
    num_iters: int = 1,           # ~30 min/gen baseline — keep iters minimal
    prompt: str = DEFAULT_PROMPT,
) -> dict:
    """HunyuanVideo 13B T2V baseline. Highest-token-count open video DiT."""
    import json
    import os
    import statistics

    import torch
    from huggingface_hub import snapshot_download

    hf_token = os.environ.get("HF_TOKEN", "").strip() or None
    if hf_token is None:
        raise RuntimeError(
            "HUGGINGFACE_TOKEN required: HunyuanVideo is gated. "
            "Set HF_TOKEN in the Modal `huggingface` secret with read access "
            "to tencent/HunyuanVideo (accept the license on HF first)."
        )

    print(f"Downloading {HUNYUAN_REPO} ...")
    local_dir = snapshot_download(HUNYUAN_REPO, cache_dir=MODEL_CACHE, token=hf_token)

    from diffusers import HunyuanVideoPipeline

    print("Loading HunyuanVideoPipeline in bf16 ...")
    pipe = HunyuanVideoPipeline.from_pretrained(
        local_dir, torch_dtype=torch.bfloat16,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    # HunyuanVideo at 720p × 129 frames pushes 80 GB VRAM hard.
    # Enable VAE slicing / tiling to reduce decode memory pressure.
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
        print("VAE tiling enabled (memory pressure reduction).")

    n_params = sum(
        p.numel() for m in pipe.components.values()
        if isinstance(m, torch.nn.Module) for p in m.parameters()
    )
    print(f"HunyuanVideoPipeline loaded: {n_params / 1e9:.2f}B params total.")

    @torch.inference_mode()
    def _run_once(seed_val: int) -> tuple[float, int]:
        gen = torch.Generator(device="cuda").manual_seed(seed_val)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()

        out = pipe(
            prompt=prompt,
            num_frames=num_frames,
            num_inference_steps=num_steps,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            generator=gen,
            return_dict=True,
        )

        e.record()
        torch.cuda.synchronize()

        frames = getattr(out, "frames", None)
        n_observed = _get_frame_count(frames)
        return s.elapsed_time(e), n_observed

    print(f"Warmup ({num_warmup} iters) ...")
    n_observed = num_frames
    for i in range(num_warmup):
        ms, n_observed = _run_once(seed_val=seed + i + 1000)
        print(f"  warmup {i + 1}: {ms:.0f} ms ({n_observed} frames)")

    torch.cuda.reset_peak_memory_stats()
    latencies: list[float] = []
    print(f"Timed runs ({num_iters} iters) ...")
    for i in range(num_iters):
        try:
            ms, n_observed = _run_once(seed_val=seed + i)
            latencies.append(ms)
            fps = n_observed / (ms / 1000.0)
            print(
                f"  iter {i + 1}: {ms:.0f} ms "
                f"({n_observed} frames, {fps:.3f} fps, {ms / n_observed:.1f} ms/frame)"
            )
        except torch.cuda.OutOfMemoryError as oom:
            print(f"  iter {i + 1}: OOM — {oom}")
            torch.cuda.empty_cache()

    if not latencies:
        return {"error": "All iterations OOM", "model": HUNYUAN_REPO}

    mean_ms = statistics.mean(latencies)
    fps = n_observed / (mean_ms / 1000.0)

    results = {
        "model": HUNYUAN_REPO,
        "kernel": "baseline_hunyuanvideo_pipeline_bf16",
        "num_frames_requested": num_frames,
        "num_frames_observed": n_observed,
        "num_steps": num_steps,
        "width": width,
        "height": height,
        "guidance_scale": guidance_scale,
        "n_params_B": round(n_params / 1e9, 3),
        "latency_ms_mean": round(mean_ms, 2),
        "latency_ms_std": round(
            statistics.stdev(latencies) if len(latencies) > 1 else 0.0, 2
        ),
        "latency_per_frame_ms": round(mean_ms / n_observed, 2),
        "frames_per_sec": round(fps, 3),
        "vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "gpu": "H100",
        "raw_latencies_ms": latencies,
    }
    print("\n" + json.dumps(results, indent=2))
    return results


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


@app.local_entrypoint(name="hunyuanvideo_baseline_modal")
def main(
    num_frames: int = 129,
    num_steps: int = 50,
    num_warmup: int = 1,
    num_iters: int = 1,
):
    result = run_hunyuanvideo_baseline.remote(
        num_frames=num_frames,
        num_steps=num_steps,
        num_warmup=num_warmup,
        num_iters=num_iters,
    )

    print("\nHunyuanVideo 13B baseline result:")
    import json
    print(json.dumps(result, indent=2))

    sys.path.insert(0, "/root/benchmarks")
    from result_store import save_result
    save_result(
        result, "baseline",
        f"hunyuanvideo_baseline_f{num_frames}_s{num_steps}",
    )
