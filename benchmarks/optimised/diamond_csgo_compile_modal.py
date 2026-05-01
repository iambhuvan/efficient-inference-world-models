"""
DIAMOND CSGO + torch.compile (reduce-overhead) on the world-model fwd path.

Mirrors what `src/play.py --compile` does: wraps `wm_env.predict_next_obs`
and `wm_env.upsample_next_obs` with `torch.compile(mode='reduce-overhead')`.
With reduce-overhead, the compiler additionally captures CUDA graphs for
the steady-state shape, eliminating per-call kernel-launch overhead — the
dominant cost on a small UNet at batch=1.

Expected: 1.5–2.5× over baseline (25 ms/frame → ~10–17 ms/frame).

Run:
    PYTHONPATH=benchmarks modal run benchmarks/optimised/diamond_csgo_compile_modal.py
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
    timeout=3600,
    memory=32768,
)
def run_compile(
    num_frames: int = 64,
    num_warmup: int = 4,         # extra warmup so compile cost is amortised
    num_iters: int = 3,
) -> dict:
    import torch

    bundle = load_diamond()
    wm_env = bundle["wm_env"]

    print("Wrapping wm_env.predict_next_obs / upsample_next_obs with torch.compile ...")
    wm_env.predict_next_obs = torch.compile(
        wm_env.predict_next_obs, mode="reduce-overhead",
    )
    if wm_env.sampler_upsampling is not None:
        wm_env.upsample_next_obs = torch.compile(
            wm_env.upsample_next_obs, mode="reduce-overhead",
        )

    timing = time_world_model(wm_env, num_frames, num_warmup, num_iters)
    return make_result(
        kernel="diamond_compile_reduce_overhead",
        bundle=bundle,
        timing=timing,
    )


@app.local_entrypoint(name="diamond_csgo_compile_modal")
def main(
    num_frames: int = 64,
    num_warmup: int = 4,
    num_iters: int = 3,
):
    result = run_compile.remote(
        num_frames=num_frames, num_warmup=num_warmup, num_iters=num_iters,
    )
    import json
    print("\nDIAMOND CSGO + torch.compile result:")
    print(json.dumps(result, indent=2))

    sys.path.insert(0, "/root/benchmarks")
    from result_store import save_result
    save_result(
        result, "optimised_kernels",
        f"diamond_csgo_compile_f{num_frames}",
    )
