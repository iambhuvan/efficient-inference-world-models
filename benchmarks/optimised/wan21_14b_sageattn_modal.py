"""Wan2.1-T2V-14B + SageAttention2 backend (after fixed image rebuild)."""

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
def run_wan_sageattn(num_warmup: int = 1, num_iters: int = 1) -> dict:
    import os, statistics, torch
    from huggingface_hub import snapshot_download

    # CRITICAL: setting DIFFUSERS_ATTN_BACKEND to an invalid name makes
    # diffusers REFUSE TO IMPORT AT ALL ('not a valid AttentionBackendName').
    # We must enumerate valid names FIRST before setting the env var.
    chosen = "native"
    valid_backends = []
    try:
        from diffusers.models.attention_dispatch import AttentionBackendName
        valid_backends = [b.value for b in AttentionBackendName]
        print(f"  valid backends in this diffusers: {valid_backends[:10]}{'...' if len(valid_backends)>10 else ''}")
    except Exception as exc:
        print(f"  could not enumerate AttentionBackendName: {exc}")

    # SageAttention 2.x SM90: only choose if BOTH the diffusers backend name
    # exists AND the sageattention kernel is callable.
    sa_target = "_sage_qk_int8_pv_fp8_cuda_sm90"
    if sa_target in valid_backends:
        try:
            from sageattention import sageattn_qk_int8_pv_fp8_cuda_sm90 as _sa_kernel
            if callable(_sa_kernel):
                chosen = sa_target
                print(f"  ✓ {sa_target}: sageattention kernel callable, selected")
        except ImportError:
            print(f"  ✗ {sa_target}: backend name valid but sageattention not importable")

    # FA3: find the actual valid name (could be _flash_attention_3, _flash_3, _fa3_hub, etc.)
    if chosen == "native":
        fa_candidates = [n for n in valid_backends
                         if "flash" in n.lower() and ("3" in n or "fa3" in n.lower())]
        try:
            from flash_attn import flash_attn_func as _fa_kernel
            fa_kernel_callable = callable(_fa_kernel)
        except ImportError:
            fa_kernel_callable = False
        if fa_kernel_callable and fa_candidates:
            chosen = fa_candidates[0]
            print(f"  ✓ {chosen}: flash_attn_func callable, selected (from {fa_candidates})")
        else:
            print(f"  ✗ no FA3 backend selectable (flash_attn={fa_kernel_callable}, "
                  f"valid_FA3_names={fa_candidates})")

    if chosen == "native":
        print(f"  → falling back to native SDPA (no accelerated kernel installed)")
    candidates = valid_backends or ["native"]
    os.environ["DIFFUSERS_ATTN_BACKEND"] = chosen
    print(f"  DIFFUSERS_ATTN_BACKEND={chosen}")

    try:
        import sageattention as _sa
        print(f"  sageattention: {getattr(_sa, '__version__', 'unknown')}")
    except ImportError: print("  sageattention: NOT INSTALLED")
    try:
        import flash_attn
        print(f"  flash_attn: {flash_attn.__version__}")
    except ImportError: print("  flash_attn: NOT INSTALLED")

    hf_token = os.environ.get("HF_TOKEN", "").strip() or None
    local_dir = snapshot_download(REPO, cache_dir=MODEL_CACHE, token=hf_token)
    from diffusers import WanPipeline
    pipe = WanPipeline.from_pretrained(local_dir, torch_dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    n_patched = 0
    try:
        from diffusers.models.attention import AttentionModuleMixin
        for mod in pipe.transformer.modules():
            if isinstance(mod, AttentionModuleMixin) and hasattr(mod, "set_attention_backend"):
                mod.set_attention_backend(chosen); n_patched += 1
        print(f"  per-module backend applied to {n_patched} modules")
    except Exception as e:
        print(f"  per-module skip: {e}")

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
    return {"model": REPO, "kernel": "sageattn_or_fallback", "backend_used": chosen,
            "candidates": candidates, "n_attention_modules_patched": n_patched,
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


@app.local_entrypoint(name="wan21_14b_sageattn_modal")
def main(num_iters: int = 1):
    r = run_wan_sageattn.remote(num_iters=num_iters)
    import json; print(json.dumps(r, indent=2))
    sys.path.insert(0, "/root/benchmarks")
    from result_store import save_result
    save_result(r, "optimised_kernels", "wan21_14b_sageattn")
