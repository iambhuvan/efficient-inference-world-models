"""Wan2.1-T2V-14B + torch.compile(max-autotune). Expected ~1.4× over 0.171 FPS."""

import sys
sys.path.insert(0, "/root"); sys.path.insert(0, "/root/benchmarks")

import modal
from modal_common import app, image_cuda_devel, hf_secret, model_volume, MODEL_CACHE
image = image_cuda_devel

REPO = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
BASELINE_FPS = 0.171
PROMPT = ("A cat walks on the grass, realistic, 4k, cinematic, smooth motion, "
          "natural lighting")


@app.function(gpu="H100", image=image, secrets=[hf_secret],
              volumes={MODEL_CACHE: model_volume}, timeout=7200, memory=65536)
def run_wan_compile(num_warmup: int = 2, num_iters: int = 1) -> dict:
    import os, statistics, torch
    from huggingface_hub import snapshot_download
    from diffusers import WanPipeline

    hf_token = os.environ.get("HF_TOKEN", "").strip() or None
    local_dir = snapshot_download(REPO, cache_dir=MODEL_CACHE, token=hf_token)
    pipe = WanPipeline.from_pretrained(local_dir, torch_dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    print("Compiling pipe.transformer with mode='max-autotune' ...")
    pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune")
    print("First warmup will trigger ~2-3 min JIT.")

    n_params = sum(p.numel() for m in pipe.components.values()
                   if isinstance(m, torch.nn.Module) for p in m.parameters())

    @torch.inference_mode()
    def _run(seed_val):
        gen = torch.Generator(device="cuda").manual_seed(seed_val)
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        out = pipe(prompt=PROMPT, num_frames=81, num_inference_steps=50,
                   width=832, height=480, guidance_scale=5.0, generator=gen, return_dict=True)
        e.record(); torch.cuda.synchronize()
        return s.elapsed_time(e), _frames(getattr(out, "frames", None))

    print(f"Warmup ({num_warmup}) ...")
    n = 81
    for i in range(num_warmup):
        ms, n = _run(42 + i + 1000); print(f"  warmup {i+1}: {ms:.0f} ms ({n} frames)")
    torch.cuda.reset_peak_memory_stats()
    lat = []
    for i in range(num_iters):
        ms, n = _run(42 + i); lat.append(ms)
        print(f"  iter {i+1}: {ms:.0f} ms ({n/(ms/1000):.3f} fps)")
    mean_ms = statistics.mean(lat); fps = n / (mean_ms / 1000)
    return {"model": REPO, "kernel": "torch_compile_max_autotune",
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


@app.local_entrypoint(name="wan21_14b_compile_modal")
def main(num_iters: int = 1):
    r = run_wan_compile.remote(num_iters=num_iters)
    import json; print(json.dumps(r, indent=2))
    sys.path.insert(0, "/root/benchmarks")
    from result_store import save_result
    save_result(r, "optimised_kernels", "wan21_14b_compile")
