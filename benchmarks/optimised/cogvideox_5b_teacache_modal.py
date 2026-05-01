"""CogVideoX-5B + TeaCache (TaylorSeerCacheConfig). Published 1.9× on this exact model."""

import sys
sys.path.insert(0, "/root"); sys.path.insert(0, "/root/benchmarks")

import modal
from modal_common import app, image_cuda_devel, hf_secret, model_volume, MODEL_CACHE
image = image_cuda_devel

REPO = "zai-org/CogVideoX-5b"
BASELINE_FPS = 0.442
PROMPT = ("A panda, dressed in a small, red jacket and a tiny hat, sits on a "
          "wooden stool in a serene bamboo forest. The panda's fluffy paws "
          "strum a miniature acoustic guitar.")


@app.function(gpu="H100", image=image, secrets=[hf_secret],
              volumes={MODEL_CACHE: model_volume}, timeout=3600, memory=65536)
def run_cogvideox_teacache(cache_interval: int = 5, max_order: int = 1,
                           disable_cache_before_step: int = 3,
                           num_warmup: int = 1, num_iters: int = 1) -> dict:
    import os, statistics, torch
    from huggingface_hub import snapshot_download
    from diffusers import CogVideoXPipeline

    hf_token = os.environ.get("HF_TOKEN", "").strip() or None
    local_dir = snapshot_download(REPO, cache_dir=MODEL_CACHE, token=hf_token)
    pipe = CogVideoXPipeline.from_pretrained(local_dir, torch_dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    # ── OPTIMIZATION: TaylorSeerCacheConfig step-caching ────────────────
    cache_enabled = False
    try:
        from diffusers import TaylorSeerCacheConfig
        cfg = TaylorSeerCacheConfig(cache_interval=cache_interval, max_order=max_order,
                                    disable_cache_before_step=disable_cache_before_step)
        if hasattr(pipe.transformer, "enable_cache"):
            pipe.transformer.enable_cache(cfg); cache_enabled = True
            print(f"  TaylorSeerCacheConfig enabled: {cfg}")
        else:
            print(f"  transformer.enable_cache not available")
    except ImportError as exc:
        print(f"  TaylorSeerCacheConfig not in this diffusers version: {exc}")

    n_params = sum(p.numel() for m in pipe.components.values()
                   if isinstance(m, torch.nn.Module) for p in m.parameters())

    @torch.inference_mode()
    def _run(seed_val):
        gen = torch.Generator(device="cuda").manual_seed(seed_val)
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        out = pipe(prompt=PROMPT, num_frames=49, num_inference_steps=50,
                   width=720, height=480, guidance_scale=6.0, generator=gen, return_dict=True)
        e.record(); torch.cuda.synchronize()
        return s.elapsed_time(e), _frames(getattr(out, "frames", None))

    print(f"Warmup ({num_warmup}) ...")
    n = 49
    for i in range(num_warmup):
        ms, n = _run(42 + i + 1000); print(f"  warmup {i+1}: {ms:.0f} ms ({n} frames)")
    torch.cuda.reset_peak_memory_stats()
    lat = []
    for i in range(num_iters):
        ms, n = _run(42 + i); lat.append(ms)
        print(f"  iter {i+1}: {ms:.0f} ms ({n/(ms/1000):.3f} fps)")
    mean_ms = statistics.mean(lat); fps = n / (mean_ms / 1000)
    return {"model": REPO, "kernel": "teacache_taylorseer",
            "cache_enabled": cache_enabled, "cache_interval": cache_interval,
            "max_order": max_order, "disable_cache_before_step": disable_cache_before_step,
            "num_frames_observed": n, "n_params_B": round(n_params/1e9, 3),
            "latency_ms_mean": round(mean_ms, 2), "latency_per_frame_ms": round(mean_ms/n, 2),
            "frames_per_sec": round(fps, 3),
            "speedup_vs_baseline": round(fps/BASELINE_FPS, 3), "baseline_fps": BASELINE_FPS,
            "vram_gb": round(torch.cuda.max_memory_allocated()/1e9, 3),
            "gpu": "H100", "raw_latencies_ms": lat}


def _frames(f):
    if f is None: return 1
    s = getattr(f, "shape", None)
    if s and len(s) >= 4: return s[0] if len(s) == 4 else s[1]
    if isinstance(f, list):
        return len(f[0]) if f and isinstance(f[0], list) else len(f)
    return 1


@app.local_entrypoint(name="cogvideox_5b_teacache_modal")
def main(num_iters: int = 1):
    r = run_cogvideox_teacache.remote(num_iters=num_iters)
    import json; print(json.dumps(r, indent=2))
    sys.path.insert(0, "/root/benchmarks")
    from result_store import save_result
    save_result(r, "optimised_kernels", "cogvideox_5b_teacache")
