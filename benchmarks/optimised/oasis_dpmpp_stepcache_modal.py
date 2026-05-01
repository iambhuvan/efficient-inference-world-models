"""
Modal H100 benchmark: Open-Oasis 500M + DPM-Solver++ 2M (5 steps) + stepcache.

Stacks two orthogonal optimizations:
  * DPM-Solver++ 2M cuts DDIM 10 steps -> 5 solver steps (~2x)
  * Step caching skips a fraction of the 5 solver-call forwards (~1.67x)

Expected combined speedup vs baseline DDIM-10: ~2x * 1.67x ~= 3.3x.

Step-cache pattern semantics on 5 solver steps:
    alternate / interval_2  -> compute on noise_idx [5, 3, 1] (3 forwards / 5 = 60% compute)
    interval_3              -> compute on [5, 2] (2 forwards / 5 = 40% compute)
"""

import sys
sys.path.insert(0, "/root")
sys.path.insert(0, "/root/benchmarks")

import modal
from modal_common import app, image, hf_secret, model_volume, MODEL_CACHE

SCALING_FACTOR = 0.07843137255


def _build_compute_set(pattern: str, n: int) -> set:
    if pattern == "alternate":
        return set(range(n, 0, -2))
    if pattern.startswith("interval_"):
        k = int(pattern.split("_", 1)[1])
        return set(range(n, 0, -k))
    if pattern == "none":
        return set(range(1, n + 1))
    raise ValueError(f"unknown pattern {pattern!r}")


@app.function(
    gpu="H100",
    image=image,
    secrets=[hf_secret],
    volumes={MODEL_CACHE: model_volume},
    timeout=3600,
    memory=32768,
)
def run_oasis_dpmpp_stepcache(
    num_frames: int = 32,
    n_prompt: int = 1,
    solver_steps: int = 5,
    skip_pattern: str = "alternate",
    max_noise_level: int = 1000,
    stabilization_level: int = 15,
    noise_abs_max: float = 20.0,
    seed: int = 42,
    num_warmup: int = 1,
    num_iters: int = 2,
    prompt_file: str = "sample_image_0.png",
    actions_file: str = "sample_actions_0.one_hot_actions.pt",
) -> dict:
    import statistics
    import os
    import torch
    sys.path.insert(0, "/root/open-oasis")
    from einops import rearrange
    from huggingface_hub import hf_hub_download
    from dit import DiT_models
    from vae import VAE_models
    from safetensors.torch import load_file as safetensors_load
    from utils import sigmoid_beta_schedule, load_prompt, load_actions

    hf_token = os.environ.get("HF_TOKEN", "").strip() or None

    print("Loading Oasis ...")
    dit_ckpt = hf_hub_download("Etched/oasis-500m", "oasis500m.safetensors",
                               cache_dir=MODEL_CACHE, token=hf_token)
    vae_ckpt = hf_hub_download("Etched/oasis-500m", "vit-l-20.safetensors",
                               cache_dir=MODEL_CACHE, token=hf_token)
    model = DiT_models["DiT-S/2"]()
    model.load_state_dict(safetensors_load(dit_ckpt), strict=False)
    model = model.to("cuda").eval()
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
    prompt_latent = rearrange(z, "(b t) (h w) c -> b t c h w",
                              t=n_prompt, h=H_lat, w=W_lat)

    betas = sigmoid_beta_schedule(max_noise_level).float().to("cuda")
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    alphas_cumprod = rearrange(alphas_cumprod, "T -> T 1 1 1")
    noise_range = torch.linspace(-1, max_noise_level - 1, solver_steps + 1)
    model_max_frames = getattr(model, "max_frames", total_frames + 1)

    compute_steps = _build_compute_set(skip_pattern, solver_steps)
    print(f"\nDPM-Solver++ 2M @ {solver_steps} steps + skip_pattern={skip_pattern!r}, "
          f"compute on {sorted(compute_steps)}")

    forward_count = 0
    skip_count = 0

    def _run_once(seed_val: int = 42) -> float:
        nonlocal forward_count, skip_count
        torch.manual_seed(seed_val); torch.cuda.manual_seed_all(seed_val)

        x = prompt_latent.clone()
        B = x.shape[0]

        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()

        for i in range(n_prompt, total_frames):
            chunk = torch.randn((B, 1, *x.shape[-3:]), device="cuda")
            chunk = torch.clamp(chunk, -noise_abs_max, +noise_abs_max)
            x = torch.cat([x, chunk], dim=1)
            start_frame = max(0, i + 1 - model_max_frames)

            cached_v = None
            eps_prev = None

            for noise_idx in reversed(range(1, solver_steps + 1)):
                t_ctx = torch.full((B, i), stabilization_level - 1,
                                   dtype=torch.long, device="cuda")
                t = torch.full((B, 1), int(noise_range[noise_idx].item()),
                               dtype=torch.long, device="cuda")
                t_next = torch.full((B, 1), int(noise_range[noise_idx - 1].item()),
                                    dtype=torch.long, device="cuda")
                t_next = torch.where(t_next < 0, t, t_next)
                t = torch.cat([t_ctx, t], dim=1); t_next = torch.cat([t_ctx, t_next], dim=1)

                x_curr = x.clone()[:, start_frame:]
                t = t[:, start_frame:]; t_next = t_next[:, start_frame:]

                if (noise_idx in compute_steps) or (cached_v is None):
                    with torch.no_grad():
                        with torch.cuda.amp.autocast(dtype=torch.float16):
                            v = model(x_curr, t, actions[:, start_frame: i + 1])
                    cached_v = v
                    forward_count += 1
                else:
                    v = cached_v
                    skip_count += 1

                ab_t = alphas_cumprod[t]
                ab_next = alphas_cumprod[t_next].clone()
                ab_next[:, :-1] = torch.ones_like(ab_next[:, :-1])
                if noise_idx == 1:
                    ab_next[:, -1:] = torch.ones_like(ab_next[:, -1:])

                x_start = ab_t.sqrt() * x_curr - (1 - ab_t).sqrt() * v
                eps_t   = (1 - ab_t).sqrt() * x_curr + ab_t.sqrt() * v

                # DPM-Solver++ 2M multistep correction
                if eps_prev is None or eps_prev.shape != eps_t.shape:
                    D = eps_t
                else:
                    D = (3.0 * eps_t - eps_prev) / 2.0
                eps_prev = eps_t

                x_pred = ab_next.sqrt() * x_start + (1 - ab_next).sqrt() * D
                x[:, -1:] = x_pred[:, -1:]

        e.record(); torch.cuda.synchronize()
        return s.elapsed_time(e)

    print(f"\nWarmup ({num_warmup} iters) ...")
    for i in range(num_warmup):
        ms = _run_once(seed_val=seed)
        print(f"  warmup {i+1}: {ms:.1f} ms ({num_frames/(ms/1000):.2f} fps)")

    forward_count = 0; skip_count = 0
    torch.cuda.reset_peak_memory_stats()
    latencies: list = []
    print(f"\nTimed runs ({num_iters} iters) ...")
    for i in range(num_iters):
        try:
            ms = _run_once(seed_val=seed)
            latencies.append(ms)
            print(f"  iter {i+1}: {ms:.1f} ms ({num_frames/(ms/1000):.2f} fps)")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()

    if not latencies:
        return {"error": "OOM"}

    mean_ms = statistics.mean(latencies)
    return {
        "model": "Etched/oasis-500m",
        "kernel": f"dpmpp2m_{solver_steps}step_+_stepcache_{skip_pattern}",
        "solver": "DPM-Solver++ 2M",
        "solver_steps": solver_steps,
        "skip_pattern": skip_pattern,
        "compute_set": sorted(compute_steps),
        "skip_rate": round(skip_count / max(forward_count + skip_count, 1), 4),
        "forwards_per_iter": forward_count // max(num_iters, 1),
        "skips_per_iter": skip_count // max(num_iters, 1),
        "num_frames": num_frames,
        "latency_ms_mean": round(mean_ms, 2),
        "frames_per_sec": round(num_frames / (mean_ms / 1000.0), 3),
        "vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "raw_latencies_ms": latencies,
        "gpu": "H100",
    }


@app.local_entrypoint(name="oasis_dpmpp_stepcache_modal")
def main(
    num_frames: int = 32,
    solver_steps: int = 5,
    skip_pattern: str = "alternate",
    num_iters: int = 2,
    num_warmup: int = 1,
):
    result = run_oasis_dpmpp_stepcache.remote(
        num_frames=num_frames, solver_steps=solver_steps,
        skip_pattern=skip_pattern, num_iters=num_iters, num_warmup=num_warmup,
    )
    print(f"\n=== DPM-Solver++ + stepcache result ===")
    print(f"  config:        DPM-Solver++ 2M {solver_steps}step + stepcache {skip_pattern}")
    print(f"  forwards/iter: {result.get('forwards_per_iter')} (vs {solver_steps * num_frames} unstacked)")
    print(f"  latency_mean:  {result.get('latency_ms_mean')}")
    print(f"  frames_per_sec:{result.get('frames_per_sec')}")
    sys.path.insert(0, "/root/benchmarks")
    from result_store import save_result
    save_result(result, "optimised_kernels",
                f"oasis_dpmpp_stepcache_{skip_pattern}_s{solver_steps}_f{num_frames}")
