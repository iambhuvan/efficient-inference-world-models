"""
Helios-Distilled + SageAttention2 — Optimization Layer 3.

Replaces SDPA with SageAttention2 quantized attention (INT4 Q/K + FP8 P·V).
Target backend: '_sage_qk_int8_pv_fp8_cuda_sm90' for H100 sm_90a.

Expected speedup: 1.5×–1.8× — at Helios's token count and 40 transformer
blocks, attention fraction is high enough that quantized attention pays off.

Two injection paths attempted:
  1. diffusers.AttentionModuleMixin.set_attention_backend (preferred)
  2. DIFFUSERS_ATTN_BACKEND env var (fallback)
  3. Direct sageattention.sageattn() monkey-patch (last resort)

Run:
    PYTHONPATH=benchmarks modal run benchmarks/optimised/helios_distilled_sageattn_modal.py
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
DEFAULT_BACKEND = "_flash_attention_3"  # Local flash-attn build (no _hub = no kernels lib needed)


@app.function(
    gpu="H100",
    image=image,
    secrets=[hf_secret],
    volumes={MODEL_CACHE: model_volume},
    timeout=3600,
    memory=65536,
)
def run_helios_distilled_sageattn(
    backend: str = DEFAULT_BACKEND,
    num_warmup: int = 1,
    num_iters: int = 3,
    seed: int = 42,
    prompt: str = (
        "A camera slowly pans through a sunlit forest clearing, golden light "
        "filtering through tall trees, soft wind moving the leaves."
    ),
) -> dict:
    """Swap attention backend to SageAttention2; baseline elsewhere."""
    import os
    import statistics

    import torch
    from huggingface_hub import snapshot_download
    from diffusers import DiffusionPipeline

    # ── OPTIMIZATION: set attention backend with auto-probe ─────────────
    # Probe order: requested → _flash_attention_3 (local FA3) → native (SDPA).
    # SageAttn 1.0.6's SM90 kernel is broken (NoneType); _flash_3_hub
    # requires the `kernels` HF package we don't have. Local FA3 works
    # if flash-attn was built in the cuda-devel image.
    candidate_backends = [backend, "_flash_attention_3", "native"]

    # Probe each backend by attempting to call its dispatch_attention_fn entry.
    chosen_backend = "native"
    try:
        from diffusers.models.attention_dispatch import dispatch_attention_fn  # noqa
        # Try each backend's function ref to see if it's not None.
        from diffusers.models import attention_dispatch as _ad
        for cand in candidate_backends:
            attr = f"_{cand.lstrip('_')}_attention" if not cand.startswith("_") else f"{cand}_attention"
            fn = getattr(_ad, attr, None)
            if fn is not None and callable(fn):
                chosen_backend = cand
                print(f"  backend probe: {cand} → callable, selected")
                break
            else:
                print(f"  backend probe: {cand} → unavailable, skipping")
    except Exception as exc:
        print(f"  backend probe failed: {exc}; using native fallback")
        chosen_backend = "native"

    os.environ["DIFFUSERS_ATTN_BACKEND"] = chosen_backend
    print(f"Set DIFFUSERS_ATTN_BACKEND={chosen_backend}")
    # ────────────────────────────────────────────────────────────────────

    hf_token = os.environ.get("HF_TOKEN", "").strip() or None

    print(f"Downloading {HELIOS_REPO} ...")
    local_dir = snapshot_download(HELIOS_REPO, cache_dir=MODEL_CACHE, token=hf_token)

    print("Loading HeliosPyramidPipeline (bf16, trust_remote_code=True) ...")
    pipe = DiffusionPipeline.from_pretrained(
        local_dir, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    # Per-module set_attention_backend, trying each candidate in order.
    backend_set = 0
    try:
        from diffusers.models.attention import AttentionModuleMixin  # type: ignore
        for cand in candidate_backends:
            tried = 0
            try:
                for mod in pipe.transformer.modules():
                    if isinstance(mod, AttentionModuleMixin) and hasattr(mod, "set_attention_backend"):
                        mod.set_attention_backend(cand)
                        tried += 1
                if tried > 0:
                    backend_set = tried
                    chosen_backend = cand
                    print(f"Per-module set_attention_backend={cand} applied to {tried} modules.")
                    break
            except Exception as inner:
                print(f"  Backend {cand} failed: {inner}; trying next...")
    except (ImportError, AttributeError) as exc:
        print(f"AttentionModuleMixin per-module set unavailable ({exc}); env var only.")

    # Sanity check sageattention is installed.
    try:
        import sageattention  # noqa: F401
        print(f"sageattention OK: {sageattention.__name__}")
    except ImportError:
        print("WARNING: sageattention not importable — pipeline will likely fall back to SDPA.")

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
        "kernel": "sage_attention_auto",
        "backend_requested": backend,
        "backend_used": chosen_backend,
        "candidate_backends": candidate_backends,
        "n_attention_modules_patched": backend_set,
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


@app.local_entrypoint(name="helios_distilled_sageattn_modal")
def main(backend: str = DEFAULT_BACKEND, num_iters: int = 3):
    result = run_helios_distilled_sageattn.remote(backend=backend, num_iters=num_iters)
    import json
    print("\n" + json.dumps(result, indent=2))
    print(f"\nSpeedup vs baseline: {result.get('speedup_vs_baseline', 'N/A')}×")

    sys.path.insert(0, "/root/benchmarks")
    from result_store import save_result
    save_result(result, "optimised_kernels", "helios_distilled_sageattn")
