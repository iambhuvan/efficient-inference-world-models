"""
Helios-Distilled + INT4 weight-only quantization (torchao) — Layer 2.

Applies Int4WeightOnlyConfig(group_size=128) to the 14B transformer.
NOT applied to T5 text encoder (would corrupt prompt embeddings) or
VAE (small, FP16 already efficient there).

Expected speedup: 1.4×–1.7× — Helios's 14B at batch=1 is firmly
memory-bandwidth-bound (AI ≈ 1 FLOP/byte ≪ H100 590 ridge).

Run:
    PYTHONPATH=benchmarks modal run benchmarks/optimised/helios_distilled_int4_modal.py
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
def run_helios_distilled_int4(
    group_size: int = 128,
    num_warmup: int = 1,
    num_iters: int = 3,
    seed: int = 42,
    prompt: str = (
        "A camera slowly pans through a sunlit forest clearing, golden light "
        "filtering through tall trees, soft wind moving the leaves."
    ),
) -> dict:
    """Apply INT4 weight-only quant to transformer; baseline elsewhere."""
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

    # ── OPTIMIZATION: weight-only quant on transformer ──────────────────
    # torchao 0.5.0 int4_weight_only has a bf16-scale dtype bug with bf16
    # models (scales come out fp32, packing expects bf16). Try INT4 first;
    # fall back to INT8 weight-only which is bf16-safe.
    quant_kernel = "unknown"
    print(f"Quantizing transformer (target: INT4 g={group_size}, fallback: INT8) ...")
    try:
        from torchao.quantization import quantize_, int4_weight_only
        quantize_(pipe.transformer, int4_weight_only(group_size=group_size))
        quant_kernel = f"int4_weight_only_g{group_size}"
        print(f"  Applied: {quant_kernel}")
    except (ValueError, TypeError, RuntimeError) as int4_exc:
        print(f"  INT4 failed: {int4_exc}; falling back to INT8 weight-only.")
        from torchao.quantization import quantize_, int8_weight_only
        quantize_(pipe.transformer, int8_weight_only())
        quant_kernel = "int8_weight_only"
        print(f"  Applied: {quant_kernel}")

    n_params = sum(
        p.numel() for m in pipe.components.values()
        if isinstance(m, torch.nn.Module) for p in m.parameters()
    )
    # ────────────────────────────────────────────────────────────────────

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
        "kernel": quant_kernel,
        "quant_kernel_used": quant_kernel,
        "group_size_requested": group_size,
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


@app.local_entrypoint(name="helios_distilled_int4_modal")
def main(group_size: int = 128, num_iters: int = 3):
    result = run_helios_distilled_int4.remote(group_size=group_size, num_iters=num_iters)
    import json
    print("\n" + json.dumps(result, indent=2))
    print(f"\nSpeedup vs baseline: {result.get('speedup_vs_baseline', 'N/A')}×")

    sys.path.insert(0, "/root/benchmarks")
    from result_store import save_result
    save_result(result, "optimised_kernels", f"helios_distilled_int4_g{group_size}")
