"""
Helios-Distilled + transformer_ode drop — Optimization Layer H1 (free win).

The Helios-Distilled checkpoint ships with TWO transformers:
  - `transformer`     (HeliosTransformer3DModel, 14B) — used at inference
  - `transformer_ode` (HeliosTransformer3DModel, 14B) — DMD teacher, ONLY
                       used during distillation training, NEVER called at
                       inference (verified: HeliosPyramidPipeline.__call__
                       in diffusers makes no reference to transformer_ode)

Dropping it after pipeline load:
  - 0% latency change (it wasn't being run)
  - ~28 GB VRAM freed (14B BF16 weights)
  - Headroom for larger batches, longer history, cheaper cache eviction

Source: BestWishYsh/Helios-Distilled/transformer_ode/config.json is identical
to transformer/config.json; the diffusers pipeline source code shows zero
references to it.

Run:
    PYTHONPATH=benchmarks modal run benchmarks/optimised/helios_distilled_ode_drop_modal.py
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
def run_helios_distilled_ode_drop(
    num_warmup: int = 1,
    num_iters: int = 3,
    seed: int = 42,
    prompt: str = (
        "A camera slowly pans through a sunlit forest clearing, golden light "
        "filtering through tall trees, soft wind moving the leaves."
    ),
) -> dict:
    """Drop transformer_ode after load; otherwise identical to baseline."""
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

    # ── Measure VRAM before drop ────────────────────────────────────────
    vram_before_gb = torch.cuda.memory_allocated() / 1e9
    n_params_before = sum(
        p.numel() for m in pipe.components.values()
        if isinstance(m, torch.nn.Module) for p in m.parameters()
    )
    print(f"VRAM before drop: {vram_before_gb:.2f} GB | params: {n_params_before / 1e9:.2f}B")

    # ── OPTIMIZATION: drop transformer_ode ──────────────────────────────
    dropped_params_B = 0.0
    if hasattr(pipe, "transformer_ode") and pipe.transformer_ode is not None:
        ode_params = sum(p.numel() for p in pipe.transformer_ode.parameters())
        dropped_params_B = ode_params / 1e9
        print(f"Dropping transformer_ode ({dropped_params_B:.2f}B params)")
        pipe.transformer_ode = None
        # Also remove from components dict if present
        if hasattr(pipe, "components") and "transformer_ode" in pipe.components:
            try:
                del pipe.components["transformer_ode"]
            except (TypeError, KeyError):
                pass
        # Force garbage collection
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print("No transformer_ode found on pipeline — already absent (no-op).")
    # ────────────────────────────────────────────────────────────────────

    vram_after_gb = torch.cuda.memory_allocated() / 1e9
    n_params_after = sum(
        p.numel() for m in pipe.components.values()
        if isinstance(m, torch.nn.Module) for p in m.parameters()
    )
    print(
        f"VRAM after drop:  {vram_after_gb:.2f} GB | "
        f"params: {n_params_after / 1e9:.2f}B | freed: {vram_before_gb - vram_after_gb:.2f} GB"
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
        "kernel": "drop_transformer_ode",
        "vram_before_drop_gb": round(vram_before_gb, 2),
        "vram_after_drop_gb": round(vram_after_gb, 2),
        "vram_freed_gb": round(vram_before_gb - vram_after_gb, 2),
        "params_before_B": round(n_params_before / 1e9, 3),
        "params_after_B": round(n_params_after / 1e9, 3),
        "dropped_params_B": round(dropped_params_B, 3),
        "num_frames_observed": n_observed,
        "latency_ms_mean": round(mean_ms, 2),
        "latency_ms_std": round(statistics.stdev(latencies) if len(latencies) > 1 else 0.0, 2),
        "latency_per_frame_ms": round(mean_ms / n_observed, 2),
        "frames_per_sec": round(fps, 3),
        "speedup_vs_baseline": round(fps / BASELINE_FPS, 3),
        "baseline_fps": BASELINE_FPS,
        "paper_fps_target": PAPER_FPS,
        "vram_peak_during_inference_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
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


@app.local_entrypoint(name="helios_distilled_ode_drop_modal")
def main(num_iters: int = 3):
    result = run_helios_distilled_ode_drop.remote(num_iters=num_iters)
    import json
    print("\n" + json.dumps(result, indent=2))
    print(f"\nVRAM freed: {result.get('vram_freed_gb', 0)} GB | "
          f"Speedup vs baseline: {result.get('speedup_vs_baseline', 'N/A')}×")

    sys.path.insert(0, "/root/benchmarks")
    from result_store import save_result
    save_result(result, "optimised_kernels", "helios_distilled_ode_drop")
