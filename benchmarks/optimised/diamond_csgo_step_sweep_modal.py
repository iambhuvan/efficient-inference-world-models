"""
DIAMOND CSGO + DDIM step-count sweep.

DIAMOND's CSGO release uses 10 denoising steps for the next-obs sampler
and (typically) 10 for the upsampler. Both are configurable via
`cfg.world_model_env.diffusion_sampler_next_obs.num_steps_denoising`.

This script runs the same per-frame timing loop at multiple step counts
in ONE Modal container so we get a clean speed/quality curve in a single
H100 hour. We sweep both the next-obs sampler AND the upsampler — they
are independent denoising loops that compose multiplicatively.

Expected: 10 → 5 steps gives ~1.8× on the next-obs path; the upsampler
is independent and adds ~1.5× more if it also drops to 5. Quality
deteriorates below ~5 — the curve is the headline.

Run:
    PYTHONPATH=benchmarks modal run benchmarks/optimised/diamond_csgo_step_sweep_modal.py
"""

import sys
sys.path.insert(0, "/root")
sys.path.insert(0, "/root/benchmarks")
sys.path.insert(0, "/root/benchmarks/optimised")

from _diamond_common import (
    app, diamond_image, hf_secret, model_volume, MODEL_CACHE,
    load_diamond, time_world_model, make_result,
)


@app.function(
    gpu="H100",
    image=diamond_image,
    secrets=[hf_secret],
    volumes={MODEL_CACHE: model_volume},
    timeout=7200,                 # full sweep — 4 step counts × warm+timed
    memory=32768,
)
def run_step_sweep(
    step_counts: list[int] | None = None,
    sweep_upsampler: bool = True,
    num_frames: int = 64,
    num_warmup: int = 3,
    num_iters: int = 3,
) -> dict:
    """Sweep the DDIM step count for the next-obs sampler.

    If `sweep_upsampler` is True, the upsampler step count is set to the
    same value (mirrors a real deployment where you'd halve both).
    """
    import gc
    import torch

    if step_counts is None:
        step_counts = [3, 5, 7, 10, 15]

    sweep: list[dict] = []

    for k in step_counts:
        print("\n" + "─" * 70)
        print(f"DDIM steps = {k}")
        print("─" * 70)

        bundle = load_diamond(
            num_steps_denoising=k,
            num_steps_upsampling=k if sweep_upsampler else None,
        )
        timing = time_world_model(
            bundle["wm_env"], num_frames, num_warmup, num_iters,
        )
        entry = make_result(
            kernel="diamond_step_sweep",
            bundle=bundle,
            timing=timing,
            extras={
                "num_steps_denoising": k,
                "num_steps_upsampling": k if sweep_upsampler else "default",
            },
        )
        sweep.append(entry)

        # Free GPU before next variant — reuses the same container.
        del bundle
        gc.collect()
        torch.cuda.empty_cache()

    return {
        "model": "eloialonso/diamond",
        "variant": "csgo",
        "kernel": "diamond_step_sweep",
        "gpu": "H100",
        "sweep": sweep,
    }


@app.local_entrypoint(name="diamond_csgo_step_sweep_modal")
def main(
    num_frames: int = 64,
    num_warmup: int = 3,
    num_iters: int = 3,
):
    result = run_step_sweep.remote(
        num_frames=num_frames, num_warmup=num_warmup, num_iters=num_iters,
    )
    import json
    print("\nDIAMOND CSGO step-sweep result:")
    print(json.dumps(result, indent=2))

    # Persist a summary line per step count so the ablation table is
    # readable directly out of the master file.
    sys.path.insert(0, "/root/benchmarks")
    from result_store import save_result
    for entry in result.get("sweep", []):
        k = entry.get("num_steps_denoising", "?")
        save_result(
            entry, "optimised_kernels",
            f"diamond_csgo_steps_{k}_f{num_frames}",
        )
