"""
Modal H100 benchmark for true projected-past KV reuse via wrapped temporal attention.

Ablations:
- baseline: original Oasis model
- wrapped_no_cache: wrapped temporal attention but cache disabled (wrapper overhead)
- wrapped_cache: wrapped temporal attention with projected past reuse enabled
"""

from __future__ import annotations

import copy
import sys
sys.path.insert(0, "/root")
sys.path.insert(0, "/root/benchmarks")

import modal
from modal_common import app, image, hf_secret, model_volume, MODEL_CACHE

SCALING_FACTOR = 0.07843137255


@app.function(
    gpu="H100",
    image=image,
    secrets=[hf_secret],
    volumes={MODEL_CACHE: model_volume},
    timeout=3600,
    memory=32768,
)
def run_oasis_persistent_kv_wrapped(
    num_frames: int = 32,
    n_prompt: int = 1,
    ddim_steps: int = 10,
    max_noise_level: int = 1000,
    stabilization_level: int = 15,
    noise_abs_max: float = 20.0,
    seed: int = 42,
    prompt_file: str = "sample_image_0.png",
    actions_file: str = "sample_actions_0.one_hot_actions.pt",
) -> dict:
    import os
    import statistics
    from typing import Dict, Tuple

    import torch
    from einops import rearrange

    sys.path.insert(0, "/root/open-oasis")
    from huggingface_hub import hf_hub_download
    from dit import DiT_models
    from vae import VAE_models
    from safetensors.torch import load_file as safetensors_load
    from utils import sigmoid_beta_schedule, load_prompt, load_actions
    from worldserve.optimizations.system_level.kv_cache.oasis_temporal_persistent_wrapper import (
        TemporalPersistentKVRuntime,
        wrap_model_temporal_attention,
    )

    hf_token = os.environ.get("HF_TOKEN", "").strip() or None

    # Load base weights
    dit_ckpt = hf_hub_download("Etched/oasis-500m", "oasis500m.safetensors", cache_dir=MODEL_CACHE, token=hf_token)
    vae_ckpt = hf_hub_download("Etched/oasis-500m", "vit-l-20.safetensors", cache_dir=MODEL_CACHE, token=hf_token)
    base_model = DiT_models["DiT-S/2"]()
    base_model.load_state_dict(safetensors_load(dit_ckpt), strict=False)
    base_model = base_model.to("cuda").eval()

    # Build wrapped model from deep copy so baseline remains untouched
    wrapped_model = copy.deepcopy(base_model).eval()
    runtime = TemporalPersistentKVRuntime()
    n_wrapped = wrap_model_temporal_attention(wrapped_model, runtime)

    vae = VAE_models["vit-l-20-shallow-encoder"]()
    vae.load_state_dict(safetensors_load(vae_ckpt), strict=False)
    vae = vae.to("cuda").eval()

    prompt_path = f"/root/open-oasis/sample_data/{prompt_file}"
    actions_path = f"/root/open-oasis/sample_data/{actions_file}"
    total_frames = n_prompt + num_frames

    x_pixels = load_prompt(prompt_path, n_prompt_frames=n_prompt).to("cuda")
    actions = load_actions(actions_path).to("cuda")[:, :total_frames]
    H_pix, W_pix = x_pixels.shape[-2:]
    flat = rearrange(x_pixels, "b t c h w -> (b t) c h w")
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            z = vae.encode(flat * 2 - 1).mean * SCALING_FACTOR
    H_lat = H_pix // vae.patch_size
    W_lat = W_pix // vae.patch_size
    prompt_latent = rearrange(z, "(b t) (h w) c -> b t c h w", t=n_prompt, h=H_lat, w=W_lat)

    betas = sigmoid_beta_schedule(max_noise_level).float().to("cuda")
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    alphas_cumprod = rearrange(alphas_cumprod, "T -> T 1 1 1")
    noise_range = torch.linspace(-1, max_noise_level - 1, ddim_steps + 1)
    model_max_frames = getattr(base_model, "max_frames", total_frames + 1)

    def _run_once(mode: str) -> Tuple[torch.Tensor, float]:
        # mode in {"baseline", "wrapped_no_cache", "wrapped_cache"}
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        model = base_model if mode == "baseline" else wrapped_model
        runtime.enable(mode == "wrapped_cache")

        B = 1
        x = prompt_latent.clone()

        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()

        for i in range(n_prompt, total_frames):
            if mode != "baseline":
                runtime.start_frame(i)

            chunk = torch.randn((B, 1, *x.shape[-3:]), device="cuda")
            chunk = torch.clamp(chunk, -noise_abs_max, +noise_abs_max)
            x = torch.cat([x, chunk], dim=1)
            start_frame = max(0, i + 1 - model_max_frames)

            for step_ctr, noise_idx in enumerate(reversed(range(1, ddim_steps + 1))):
                if mode != "baseline":
                    runtime.set_step(step_ctr)

                t_ctx = torch.full((B, i), stabilization_level - 1, dtype=torch.long, device="cuda")
                t = torch.full((B, 1), int(noise_range[noise_idx].item()), dtype=torch.long, device="cuda")
                t_next = torch.full((B, 1), int(noise_range[noise_idx - 1].item()), dtype=torch.long, device="cuda")
                t_next = torch.where(t_next < 0, t, t_next)
                t = torch.cat([t_ctx, t], dim=1)
                t_next = torch.cat([t_ctx, t_next], dim=1)
                x_curr = x.clone()[:, start_frame:]
                t = t[:, start_frame:]
                t_next = t_next[:, start_frame:]

                with torch.no_grad():
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        v = model(x_curr, t, actions[:, start_frame : i + 1])

                ab_t = alphas_cumprod[t]
                ab_next = alphas_cumprod[t_next].clone()
                ab_next[:, :-1] = torch.ones_like(ab_next[:, :-1])
                if noise_idx == 1:
                    ab_next[:, -1:] = torch.ones_like(ab_next[:, -1:])
                x_start = ab_t.sqrt() * x_curr - (1 - ab_t).sqrt() * v
                eps_t = (1 - ab_t).sqrt() * x_curr + ab_t.sqrt() * v
                x_pred = ab_next.sqrt() * x_start + (1 - ab_next).sqrt() * eps_t
                x[:, -1:] = x_pred[:, -1:]

        e.record()
        torch.cuda.synchronize()
        return x, s.elapsed_time(e)

    modes = ["baseline", "wrapped_no_cache", "wrapped_cache"]
    lat: Dict[str, list[float]] = {m: [] for m in modes}
    outs: Dict[str, torch.Tensor] = {}
    for _ in range(2):
        for m in modes:
            out, ms = _run_once(m)
            lat[m].append(ms)
            outs[m] = out

    mean = {k: statistics.mean(v) for k, v in lat.items()}
    base = mean["baseline"]
    rel = {k: round(base / v, 3) if v > 0 else 0.0 for k, v in mean.items()}
    diffs = {
        "wrapped_no_cache_vs_baseline": float((outs["wrapped_no_cache"] - outs["baseline"]).abs().max().item()),
        "wrapped_cache_vs_baseline": float((outs["wrapped_cache"] - outs["baseline"]).abs().max().item()),
    }

    return {
        "model": "Etched/oasis-500m",
        "kernel": "persistent_temporal_wrapper",
        "num_frames": num_frames,
        "ddim_steps": ddim_steps,
        "n_temporal_wrapped": n_wrapped,
        "latency_ms_mean_by_mode": {k: round(v, 2) for k, v in mean.items()},
        "relative_speed_vs_baseline": rel,
        "max_abs_diff_vs_baseline": diffs,
        "raw_latencies_ms_by_mode": lat,
        "gpu": "H100",
    }


@app.local_entrypoint(name="oasis_persistent_kv_wrapped_modal")
def main(
    num_frames: int = 32,
    ddim_steps: int = 10,
    seed: int = 42,
):
    result = run_oasis_persistent_kv_wrapped.remote(
        num_frames=num_frames,
        ddim_steps=ddim_steps,
        seed=seed,
    )
    print("\n=== Persistent temporal wrapper run ===")
    print("latency_ms_mean_by_mode:", result.get("latency_ms_mean_by_mode"))
    print("relative_speed_vs_baseline:", result.get("relative_speed_vs_baseline"))
    print("max_abs_diff_vs_baseline:", result.get("max_abs_diff_vs_baseline"))
    print("n_temporal_wrapped:", result.get("n_temporal_wrapped"))

    sys.path.insert(0, "/root/benchmarks")
    from result_store import save_result
    save_result(result, "optimised_kernels", f"oasis_persistent_wrapper_f{num_frames}")

