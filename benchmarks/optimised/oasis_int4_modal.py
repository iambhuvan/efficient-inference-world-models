"""
Modal H100 benchmark: Open-Oasis 500M + INT4 weight-only quantization only.

Applies torchao `int4_weight_only()` to all Linear layers in the transformer.
Reports both speedup and VRAM reduction alongside timing.

Run:
    modal run modal/optimised/oasis_int4_modal.py
"""

import sys
sys.path.insert(0, "/root")
sys.path.insert(0, "/root/benchmarks")
sys.path.insert(0, "/root/worldserve/optimizations")

import modal
from modal_common import app, image, hf_secret, model_volume, MODEL_CACHE


@app.function(
    gpu="H100",
    image=image,
    secrets=[hf_secret],
    volumes={MODEL_CACHE: model_volume},
    timeout=3600,
    memory=32768,
)
def run_oasis_int4(
    num_frames: int = 16,
    height: int = 360,
    width: int = 640,
    num_steps: int = 20,
    num_warmup: int = 1,
    num_iters: int = 1,
    prompt: str = "A first-person view of walking through a Minecraft forest",
) -> dict:
    """
    Run Open-Oasis 500M with INT4 weight-only quantization (torchao) and return
    timing results including VRAM reduction.
    """
    import json
    import os
    import statistics

    import torch

    hf_token = os.environ.get("HF_TOKEN", "").strip() or None
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN is required for the gated Etched/oasis-500m model."
        )

    print(f"Loading Etched/oasis-500m (cache={MODEL_CACHE}) ...")
    try:
        from diffusers import DiffusionPipeline
        pipe = DiffusionPipeline.from_pretrained(
            "Etched/oasis-500m",
            torch_dtype=torch.bfloat16,
            cache_dir=MODEL_CACHE,
            token=hf_token,
        ).to("cuda")
    except Exception as exc:
        print(f"ERROR loading pipeline: {exc}")
        raise

    if not hasattr(pipe, "transformer"):
        raise RuntimeError("Pipeline has no .transformer attribute; cannot apply INT4.")

    torch.cuda.synchronize()
    vram_before_gb = torch.cuda.memory_allocated() / 1e9

    int4_applied = False
    try:
        from torchao.quantization import quantize_, int4_weight_only
        quantize_(pipe.transformer, int4_weight_only())
        int4_applied = True
        print("INT4 weight-only quantization applied via torchao.")
    except ImportError:
        print("WARNING: torchao not installed — INT4 quantization skipped.")
    except Exception as exc:
        print(f"WARNING: INT4 quantization failed ({exc}) — skipping.")

    torch.cuda.synchronize()
    vram_after_gb = torch.cuda.memory_allocated() / 1e9
    if int4_applied:
        print(
            f"VRAM: {vram_before_gb:.2f} GB → {vram_after_gb:.2f} GB "
            f"(reduction: {vram_before_gb - vram_after_gb:.2f} GB)"
        )

    generator = torch.Generator(device="cuda").manual_seed(42)

    def _run_once() -> float:
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        pipe(
            prompt=prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=num_steps,
            generator=generator,
            output_type="latent",
        )
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end)

    print(f"Warmup ({num_warmup} iters) ...")
    for i in range(num_warmup):
        try:
            _run_once()
            print(f"  warmup {i + 1}/{num_warmup} done")
        except torch.cuda.OutOfMemoryError as oom:
            print(f"  warmup {i + 1}: OOM — {oom}; skipping")
            torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()
    latencies = []
    print(f"Timed runs ({num_iters} iters) ...")
    for i in range(num_iters):
        try:
            ms = _run_once()
            latencies.append(ms)
            fps = num_frames / (ms / 1000.0)
            print(f"  iter {i + 1}: {ms:.1f} ms  ({fps:.2f} fps)")
        except torch.cuda.OutOfMemoryError as oom:
            print(f"  iter {i + 1}: OOM — {oom}; skipping")
            torch.cuda.empty_cache()

    if not latencies:
        return {
            "error": "All iterations OOM",
            "model": "Etched/oasis-500m",
            "kernel": "INT4 weight-only (torchao)",
        }

    mean_ms = statistics.mean(latencies)
    std_ms  = statistics.stdev(latencies) if len(latencies) > 1 else 0.0

    speedup_vs_baseline = None
    baseline_path = "/root/benchmarks/baseline/results/oasis_baseline.json"
    try:
        with open(baseline_path) as fh:
            bl = json.load(fh)
        bl_mean = bl.get("latency_ms", {}).get("total_mean")
        if bl_mean and mean_ms > 0:
            speedup_vs_baseline = round(bl_mean / mean_ms, 4)
    except Exception:
        pass

    results = {
        "model": "Etched/oasis-500m",
        "kernel": "INT4 weight-only (torchao)",
        "int4": {
            "applied": int4_applied,
            "vram_before_gb": round(vram_before_gb, 3),
            "vram_after_gb": round(vram_after_gb, 3),
            "vram_reduction_gb": round(vram_before_gb - vram_after_gb, 3),
        },
        "num_frames": num_frames,
        "height": height,
        "width": width,
        "steps": num_steps,
        "mean_ms": round(mean_ms, 2),
        "std_ms": round(std_ms, 2),
        "frames_per_sec": round(num_frames / (mean_ms / 1000.0), 3),
        "vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "speedup_vs_baseline": speedup_vs_baseline,
        "gpu": "H100",
        "raw_latencies_ms": latencies,
    }
    print(json.dumps(results, indent=2))
    return results


@app.local_entrypoint(name="oasis_int4_modal")
def main():
    result = run_oasis_int4.remote()
    print("\nOpen-Oasis 500M (INT4 weight-only only) result:")
    import json
    print(json.dumps(result, indent=2))
