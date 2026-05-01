"""
DIAMOND CSGO + BF16 autocast on the inner_model forward.

DIAMOND's pretrained checkpoint is FP32. The inner UNet is the dominant
cost in `denoiser.compute_model_output`. Wrapping the UNet call in
`torch.autocast(device_type='cuda', dtype=torch.bfloat16)` halves bandwidth
without touching weights — safest possible BF16 win. The downstream
quantisation step in `wrap_model_output` clamps & quantises to byte and
back, so BF16 noise inside the UNet is irrelevant past that boundary.

Expected: 1.3–1.6× over baseline.

Run:
    PYTHONPATH=benchmarks modal run benchmarks/optimised/diamond_csgo_bf16_modal.py
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
def run_bf16(
    num_frames: int = 64,
    num_warmup: int = 4,
    num_iters: int = 3,
) -> dict:
    import torch

    bundle = load_diamond()
    agent = bundle["agent"]

    # Monkey-patch `compute_model_output` (the UNet call site) to enter
    # BF16 autocast. Doing it at this granularity means `wrap_model_output`
    # (which quantises to byte) still runs at FP32, preserving determinism
    # of the post-step state.
    def _wrap(model_obj):
        original = model_obj.compute_model_output

        def patched(noisy_next_obs, obs, act, cs):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                return original(noisy_next_obs, obs, act, cs)

        model_obj.compute_model_output = patched

    _wrap(agent.denoiser)
    if agent.upsampler is not None:
        _wrap(agent.upsampler)

    timing = time_world_model(bundle["wm_env"], num_frames, num_warmup, num_iters)
    return make_result(
        kernel="diamond_bf16_autocast",
        bundle=bundle,
        timing=timing,
        extras={"autocast_dtype": "bfloat16"},
    )


@app.local_entrypoint(name="diamond_csgo_bf16_modal")
def main(
    num_frames: int = 64,
    num_warmup: int = 4,
    num_iters: int = 3,
):
    result = run_bf16.remote(
        num_frames=num_frames, num_warmup=num_warmup, num_iters=num_iters,
    )
    import json
    print("\nDIAMOND CSGO + BF16 autocast result:")
    print(json.dumps(result, indent=2))

    sys.path.insert(0, "/root/benchmarks")
    from result_store import save_result
    save_result(
        result, "optimised_kernels",
        f"diamond_csgo_bf16_f{num_frames}",
    )
