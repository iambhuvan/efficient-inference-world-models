"""
Run all 3 video DiT baselines (CogVideoX-5B, Wan2.1-T2V-14B, HunyuanVideo) on
ONE H100 container, sequentially. Saves three result files into
benchmarks/runs/baseline/.

Why one container:
  - Single H100 lifetime → no per-container startup tax × 3
  - Image build cost paid once (not three times)
  - HuggingFace volume mount shared across all three downloads
  - Sequential execution = ~50–60 min total wall-clock vs 3 × parallel
    containers (~30 min wall-clock but 3× GPU-hours billed)

Run:
    PYTHONPATH=benchmarks modal run benchmarks/baseline/run_all_baselines_modal.py
"""

import sys
sys.path.insert(0, "/root")
sys.path.insert(0, "/root/benchmarks")

import modal
from modal_common import app, image_cuda_devel, hf_secret, model_volume, MODEL_CACHE
image = image_cuda_devel

DEFAULT_PROMPT_VIDEO = (
    "A cat walks on the grass, realistic, 4k, cinematic, smooth motion, "
    "natural lighting"
)


@app.function(
    gpu="H100",
    image=image,
    secrets=[hf_secret],
    volumes={MODEL_CACHE: model_volume},
    timeout=14400,                # 4 hr — covers all three sequentially
    memory=65536,
)
def run_all_baselines(
    seed: int = 42,
    skip_cogvideox: bool = False,
    skip_wan21: bool = False,
    skip_hunyuanvideo: bool = False,
    cogvideox_num_frames: int = 49,
    cogvideox_num_steps: int = 50,
    wan21_num_frames: int = 81,
    wan21_num_steps: int = 50,
    hunyuanvideo_num_frames: int = 129,
    hunyuanvideo_num_steps: int = 50,
) -> dict:
    """
    Run CogVideoX-5B, Wan2.1-T2V-14B, HunyuanVideo baselines back-to-back on
    one H100. Each section: download → load → 1 warmup → 1 timed → record.
    """
    import gc
    import json
    import os
    import statistics
    import time

    import torch
    from huggingface_hub import snapshot_download

    hf_token = os.environ.get("HF_TOKEN", "").strip() or None

    print("=" * 70)
    print("WORLDSERVE — All-Baselines Run (single H100, sequential)")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Quick pre-flight: confirm flash-attn + sageattention if available.
    try:
        import flash_attn
        print(f"flash_attn: {flash_attn.__version__}")
    except ImportError:
        print("flash_attn: NOT INSTALLED")
    try:
        import sageattention as _sa
        print(f"sageattention: {getattr(_sa, '__version__', 'unknown')}")
    except ImportError:
        print("sageattention: NOT INSTALLED")
    print()

    results: dict[str, dict] = {}

    # ─────────────────────────────────────────────────────────────────────
    # Helper: free GPU between models so each run starts from a clean slate
    # ─────────────────────────────────────────────────────────────────────
    def _release(*objects):
        for o in objects:
            try:
                del o
            except Exception:
                pass
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # ─────────────────────────────────────────────────────────────────────
    # 1. CogVideoX-5B
    # ─────────────────────────────────────────────────────────────────────
    if not skip_cogvideox:
        print("\n" + "─" * 70)
        print("[1/3] CogVideoX-5B  (zai-org/CogVideoX-5b)")
        print("─" * 70)
        t0 = time.time()
        try:
            from diffusers import CogVideoXPipeline
            local_dir = snapshot_download(
                "zai-org/CogVideoX-5b", cache_dir=MODEL_CACHE, token=hf_token,
            )
            pipe = CogVideoXPipeline.from_pretrained(
                local_dir, torch_dtype=torch.bfloat16,
            ).to("cuda")
            pipe.set_progress_bar_config(disable=True)

            results["cogvideox_5b"] = _bench_pipeline(
                pipe=pipe,
                model_name="zai-org/CogVideoX-5b",
                kernel="baseline_cogvideox_pipeline_bf16",
                prompt=DEFAULT_PROMPT_VIDEO,
                num_frames=cogvideox_num_frames,
                num_steps=cogvideox_num_steps,
                width=720, height=480, guidance_scale=6.0, seed=seed,
            )
            _release(pipe)
        except Exception as exc:
            print(f"[CogVideoX-5B] FAILED: {exc}")
            results["cogvideox_5b"] = {"error": str(exc), "model": "zai-org/CogVideoX-5b"}
            _release()
        print(f"[CogVideoX-5B] section wall: {time.time() - t0:.1f}s")

    # ─────────────────────────────────────────────────────────────────────
    # 2. Wan2.1-T2V-14B
    # ─────────────────────────────────────────────────────────────────────
    if not skip_wan21:
        print("\n" + "─" * 70)
        print("[2/3] Wan2.1-T2V-14B  (Wan-AI/Wan2.1-T2V-14B-Diffusers)")
        print("─" * 70)
        t0 = time.time()
        try:
            from diffusers import WanPipeline
            local_dir = snapshot_download(
                "Wan-AI/Wan2.1-T2V-14B-Diffusers", cache_dir=MODEL_CACHE, token=hf_token,
            )
            pipe = WanPipeline.from_pretrained(
                local_dir, torch_dtype=torch.bfloat16,
            ).to("cuda")
            pipe.set_progress_bar_config(disable=True)

            results["wan21_14b"] = _bench_pipeline(
                pipe=pipe,
                model_name="Wan-AI/Wan2.1-T2V-14B-Diffusers",
                kernel="baseline_wan_pipeline_bf16",
                prompt=DEFAULT_PROMPT_VIDEO,
                num_frames=wan21_num_frames,
                num_steps=wan21_num_steps,
                width=832, height=480, guidance_scale=5.0, seed=seed,
            )
            _release(pipe)
        except Exception as exc:
            print(f"[Wan2.1-14B] FAILED: {exc}")
            results["wan21_14b"] = {"error": str(exc), "model": "Wan-AI/Wan2.1-T2V-14B-Diffusers"}
            _release()
        print(f"[Wan2.1-14B] section wall: {time.time() - t0:.1f}s")

    # ─────────────────────────────────────────────────────────────────────
    # 3. HunyuanVideo 13B (gated — requires accepted Tencent license)
    # ─────────────────────────────────────────────────────────────────────
    if not skip_hunyuanvideo:
        print("\n" + "─" * 70)
        print("[3/3] HunyuanVideo  (tencent/HunyuanVideo)")
        print("─" * 70)
        t0 = time.time()
        try:
            from diffusers import HunyuanVideoPipeline
            local_dir = snapshot_download(
                "tencent/HunyuanVideo", cache_dir=MODEL_CACHE, token=hf_token,
            )
            pipe = HunyuanVideoPipeline.from_pretrained(
                local_dir, torch_dtype=torch.bfloat16,
            ).to("cuda")
            pipe.set_progress_bar_config(disable=True)
            # 720p × 129f decode pushes 80 GB hard — VAE tiling required
            if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
                pipe.vae.enable_tiling()

            results["hunyuanvideo"] = _bench_pipeline(
                pipe=pipe,
                model_name="tencent/HunyuanVideo",
                kernel="baseline_hunyuanvideo_pipeline_bf16",
                prompt=DEFAULT_PROMPT_VIDEO,
                num_frames=hunyuanvideo_num_frames,
                num_steps=hunyuanvideo_num_steps,
                width=1280, height=720, guidance_scale=6.0, seed=seed,
            )
            _release(pipe)
        except Exception as exc:
            print(f"[HunyuanVideo] FAILED: {exc}")
            results["hunyuanvideo"] = {"error": str(exc), "model": "tencent/HunyuanVideo"}
            _release()
        print(f"[HunyuanVideo] section wall: {time.time() - t0:.1f}s")

    print("\n" + "=" * 70)
    print("ALL DONE")
    print("=" * 70)
    print(json.dumps(
        {k: {"frames_per_sec": v.get("frames_per_sec", "N/A"),
             "vram_gb": v.get("vram_gb", "N/A"),
             "error": v.get("error", None)}
         for k, v in results.items()},
        indent=2,
    ))
    return results


def _bench_pipeline(
    pipe,
    model_name: str,
    kernel: str,
    prompt: str,
    num_frames: int,
    num_steps: int,
    width: int,
    height: int,
    guidance_scale: float,
    seed: int,
    num_warmup: int = 1,
    num_iters: int = 1,
) -> dict:
    """Common bench loop — warmup + timed runs, returns result dict."""
    import statistics
    import torch

    n_params = sum(
        p.numel() for m in pipe.components.values()
        if isinstance(m, torch.nn.Module) for p in m.parameters()
    )
    print(f"  Loaded: {n_params / 1e9:.2f}B params")

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
            width=width, height=height,
            guidance_scale=guidance_scale,
            generator=gen,
            return_dict=True,
        )
        e.record()
        torch.cuda.synchronize()
        frames = getattr(out, "frames", None)
        return s.elapsed_time(e), _get_frame_count(frames)

    print(f"  Warmup ({num_warmup}) ...")
    n_observed = num_frames
    for i in range(num_warmup):
        ms, n_observed = _run_once(seed_val=seed + i + 1000)
        print(f"    warmup {i + 1}: {ms:.0f} ms ({n_observed} frames)")

    torch.cuda.reset_peak_memory_stats()
    latencies: list[float] = []
    print(f"  Timed ({num_iters}) ...")
    for i in range(num_iters):
        try:
            ms, n_observed = _run_once(seed_val=seed + i)
            latencies.append(ms)
            fps = n_observed / (ms / 1000.0)
            print(f"    iter {i + 1}: {ms:.0f} ms ({fps:.3f} fps, {ms / n_observed:.1f} ms/frame)")
        except torch.cuda.OutOfMemoryError as oom:
            print(f"    iter {i + 1}: OOM — {oom}")
            torch.cuda.empty_cache()

    if not latencies:
        return {"error": "All iterations OOM", "model": model_name, "kernel": kernel}

    mean_ms = statistics.mean(latencies)
    fps = n_observed / (mean_ms / 1000.0)
    return {
        "model": model_name,
        "kernel": kernel,
        "num_frames_requested": num_frames,
        "num_frames_observed": n_observed,
        "num_steps": num_steps,
        "width": width, "height": height, "guidance_scale": guidance_scale,
        "n_params_B": round(n_params / 1e9, 3),
        "latency_ms_mean": round(mean_ms, 2),
        "latency_per_frame_ms": round(mean_ms / n_observed, 2),
        "frames_per_sec": round(fps, 3),
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


@app.local_entrypoint(name="run_all_baselines_modal")
def main(
    skip_cogvideox: bool = False,
    skip_wan21: bool = False,
    skip_hunyuanvideo: bool = False,
):
    result = run_all_baselines.remote(
        skip_cogvideox=skip_cogvideox,
        skip_wan21=skip_wan21,
        skip_hunyuanvideo=skip_hunyuanvideo,
    )
    import json
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(json.dumps(result, indent=2))

    sys.path.insert(0, "/root/benchmarks")
    from result_store import save_result
    for key, value in result.items():
        if "error" in value:
            print(f"\n  [skip] {key}: {value['error']}")
            continue
        save_result(value, "baseline", f"{key}_baseline")
        print(f"  [saved] {key}_baseline")
