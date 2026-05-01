"""
Helios-Distilled 14B baseline benchmark — GCP H100 runnable (no Modal).

Run on the GCE VM via:
    PYTHONPATH=. python benchmarks/baseline/helios_distilled_baseline.py

Or via the wrapper:
    bash scripts/gcp/run.sh helios_distilled_baseline

Helios-Distilled is the few-step DMD-distilled variant whose published H100
single-GPU baseline is 19.5 FPS at 832×480 × 81 frames (Yuan et al., arXiv
2603.04379, Mar 2026). The pipeline class HeliosPyramidPipeline performs
multi-resolution pyramidal sampling — by default we let the pipeline use
its built-in step count so we reproduce the paper's number exactly.

This is the model whose baseline we are committed to BEAT with the
WorldServe stack (INT4 weight-only + Quant VideoGen 2-bit KV + STA sparse
+ LayerSkip self-spec). Target: 19.5 FPS → 80+ FPS = ~4-5×.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

HELIOS_REPO = "BestWishYsh/Helios-Distilled"
DEFAULT_PROMPT = (
    "A camera slowly pans through a sunlit forest clearing, golden light "
    "filtering through tall trees, soft wind moving the leaves."
)


def run(
    num_frames: int = 81,
    num_steps: int | None = None,
    width: int = 832,
    height: int = 480,
    guidance_scale: float = 5.0,
    seed: int = 42,
    num_warmup: int = 1,
    num_iters: int = 3,
    prompt: str = DEFAULT_PROMPT,
    cache_dir: str | None = None,
) -> dict:
    """
    Run Helios-Distilled T2V baseline. Returns a dict for json.dump.

    num_steps=None → use HeliosPyramidPipeline's built-in default (paper config).
    Override only if you know the distilled checkpoint's step schedule.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available — Helios-Distilled needs an H100."
        )

    hf_token = os.environ.get("HF_TOKEN", "").strip() or None
    cache_dir = cache_dir or os.environ.get(
        "HF_HOME", os.path.expanduser("~/.cache/huggingface")
    )

    # ---------------------------------------------------------------- weights
    print(f"==> snapshot_download({HELIOS_REPO})")
    local_dir = snapshot_download(
        HELIOS_REPO,
        cache_dir=cache_dir,
        token=hf_token,
    )
    print(f"   local_dir: {local_dir}")

    # ---------------------------------------------------------------- pipeline
    from diffusers import DiffusionPipeline

    print(
        "==> Loading HeliosPyramidPipeline (trust_remote_code=True) in bf16"
    )
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
    print(f"   total params: {n_params / 1e9:.2f}B")

    # ---------------------------------------------------------------- timed run
    @torch.inference_mode()
    def _run_once(seed_val: int) -> float:
        gen = torch.Generator(device="cuda").manual_seed(seed_val)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()

        call_kwargs: dict = {
            "prompt": prompt,
            "num_frames": num_frames,
            "width": width,
            "height": height,
            "guidance_scale": guidance_scale,
            "generator": gen,
            "return_dict": True,
        }
        # Only pass num_inference_steps if the user explicitly set it —
        # otherwise let HeliosPyramidPipeline use its own default schedule
        # (so we reproduce the paper's 19.5 FPS).
        if num_steps is not None:
            call_kwargs["num_inference_steps"] = num_steps

        _ = pipe(**call_kwargs)

        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e)

    # ---------------------------------------------------------------- warmup
    print(f"==> Warmup ({num_warmup} iters)")
    for i in range(num_warmup):
        _ = _run_once(seed_val=seed + i + 1000)
        print(f"   warmup {i + 1}/{num_warmup} done")

    # ---------------------------------------------------------------- timed
    torch.cuda.reset_peak_memory_stats()
    latencies: list[float] = []
    print(f"==> Timed runs ({num_iters} iters)")
    for i in range(num_iters):
        try:
            ms = _run_once(seed_val=seed + i)
            latencies.append(ms)
            fps = num_frames / (ms / 1000.0)
            print(
                f"   iter {i + 1}: {ms:.1f} ms  "
                f"({fps:.2f} fps, {ms / num_frames:.1f} ms/frame)"
            )
        except torch.cuda.OutOfMemoryError as oom:
            print(f"   iter {i + 1}: OOM — {oom}")
            torch.cuda.empty_cache()

    if not latencies:
        return {"error": "All iterations OOM", "model": HELIOS_REPO}

    mean_ms = statistics.mean(latencies)
    fps = num_frames / (mean_ms / 1000.0)
    paper_fps = 19.5  # Yuan et al., arXiv 2603.04379

    return {
        "model": HELIOS_REPO,
        "kernel": "baseline_helios_pyramid_pipeline_bf16",
        "num_frames": num_frames,
        "num_steps_override": num_steps,  # None = pipeline default
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
        "paper_fps_target": paper_fps,
        "paper_fps_match_pct": round(100.0 * fps / paper_fps, 2),
        "vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "gpu": torch.cuda.get_device_name(0),
        "raw_latencies_ms": latencies,
    }


def _save(result: dict, num_frames: int, num_steps: int | None) -> Path:
    """Write result to benchmarks/runs/baseline/<name>.json."""
    project_root = Path(__file__).resolve().parents[2]
    out_dir = project_root / "benchmarks" / "runs" / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    step_tag = f"s{num_steps}" if num_steps is not None else "sdefault"
    out_path = out_dir / f"helios_distilled_baseline_f{num_frames}_{step_tag}.json"
    out_path.write_text(json.dumps(result, indent=2))
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="Helios-Distilled baseline benchmark")
    p.add_argument("--num-frames", type=int, default=81)
    p.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Override pipeline default. None = paper config (recommended).",
    )
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--guidance-scale", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-warmup", type=int, default=1)
    p.add_argument(
        "--num-iters",
        type=int,
        default=3,
        help="3 timed runs — Distilled is fast enough to afford the variance.",
    )
    p.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    args = p.parse_args()

    t0 = time.time()
    result = run(
        num_frames=args.num_frames,
        num_steps=args.num_steps,
        width=args.width,
        height=args.height,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        num_warmup=args.num_warmup,
        num_iters=args.num_iters,
        prompt=args.prompt,
    )
    wall = time.time() - t0
    result["wall_seconds_total"] = round(wall, 2)

    print("\n" + "=" * 60)
    print("Helios-Distilled baseline result:")
    print(json.dumps(result, indent=2))

    if "frames_per_sec" in result:
        delta = result["frames_per_sec"] - result["paper_fps_target"]
        sign = "+" if delta >= 0 else ""
        print(
            f"\nVs paper: {result['frames_per_sec']:.2f} FPS "
            f"({sign}{delta:.2f} vs paper's {result['paper_fps_target']} FPS, "
            f"{result['paper_fps_match_pct']:.1f}% match)"
        )

    out_path = _save(result, args.num_frames, args.num_steps)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    sys.exit(main())
