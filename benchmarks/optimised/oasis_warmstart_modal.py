"""
Modal H100: Cross-Frame Step Warm-Start for Oasis-500M.

Reuse frame T's intermediate denoising state at step k as the initialization
for frame T+1's step k. Adjacent frames share most structure, so we start
much closer to the manifold and can run fewer DDIM steps.

For each frame, we cache the latent at every reverse-DDIM step. When the next
frame starts:
    if warm_start_enabled and warm_history available:
        chunk = warm_history[noise_idx_init]  (instead of pure random noise)
    Then we still iterate DDIM but starting from a better init.

Optionally, we ALSO reduce the step count when warming because we start
closer to the answer.
"""

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
def run_oasis_warmstart(
    num_frames: int = 32,
    n_prompt: int = 1,
    ddim_steps: int = 10,
    warm_start_steps: int = 6,  # use last K steps when warming (skip first 10-K steps)
    blend_alpha: float = 0.5,    # 1.0 = pure warm start, 0.0 = pure noise
    max_noise_level: int = 1000,
    stabilization_level: int = 15,
    noise_abs_max: float = 20.0,
    seed: int = 42,
    prompt_file: str = "sample_image_0.png",
    actions_file: str = "sample_actions_0.one_hot_actions.pt",
) -> dict:
    import math, os, torch
    sys.path.insert(0, "/root/open-oasis")
    from einops import rearrange
    from huggingface_hub import hf_hub_download
    from dit import DiT_models
    from vae import VAE_models
    from safetensors.torch import load_file as safetensors_load
    from utils import sigmoid_beta_schedule, load_prompt, load_actions
    import imageio.v2 as imageio

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
    H_lat = H_pix // vae.patch_size; W_lat = W_pix // vae.patch_size
    prompt_latent = rearrange(z, "(b t) (h w) c -> b t c h w",
                              t=n_prompt, h=H_lat, w=W_lat)

    betas = sigmoid_beta_schedule(max_noise_level).float().to("cuda")
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    alphas_cumprod = rearrange(alphas_cumprod, "T -> T 1 1 1")
    noise_range = torch.linspace(-1, max_noise_level - 1, ddim_steps + 1)
    model_max_frames = getattr(model, "max_frames", total_frames + 1)

    def _generate(use_warmstart: bool, seed_val: int = 42) -> torch.Tensor:
        torch.manual_seed(seed_val); torch.cuda.manual_seed_all(seed_val)
        x = prompt_latent.clone()
        B = x.shape[0]
        # Cache of last frame's intermediate latents per step (key = noise_idx)
        # so frame T+1 can warm-start at the same noise level
        last_frame_step_cache: dict = {}

        for i in range(n_prompt, total_frames):
            # Choose the starting noise level. With warm start, we skip the
            # first (ddim_steps - warm_start_steps) high-noise steps and start
            # from the cached latent at noise_idx = warm_start_steps.
            if use_warmstart and last_frame_step_cache:
                start_noise_idx = warm_start_steps
                # Initialize chunk from last frame's cached latent at this step
                # Blend with fresh noise to avoid pure copy
                noise = torch.randn((B, 1, *x.shape[-3:]), device="cuda")
                noise = torch.clamp(noise, -noise_abs_max, +noise_abs_max)
                chunk = blend_alpha * last_frame_step_cache[warm_start_steps] + (1.0 - blend_alpha) * noise
            else:
                start_noise_idx = ddim_steps
                chunk = torch.randn((B, 1, *x.shape[-3:]), device="cuda")
                chunk = torch.clamp(chunk, -noise_abs_max, +noise_abs_max)

            x = torch.cat([x, chunk], dim=1)
            start_frame = max(0, i + 1 - model_max_frames)
            this_frame_cache: dict = {}

            for noise_idx in reversed(range(1, start_noise_idx + 1)):
                t_ctx = torch.full((B, i), stabilization_level - 1, dtype=torch.long, device="cuda")
                t = torch.full((B, 1), int(noise_range[noise_idx].item()), dtype=torch.long, device="cuda")
                t_next = torch.full((B, 1), int(noise_range[noise_idx - 1].item()), dtype=torch.long, device="cuda")
                t_next = torch.where(t_next < 0, t, t_next)
                t = torch.cat([t_ctx, t], dim=1); t_next = torch.cat([t_ctx, t_next], dim=1)
                x_curr = x.clone()[:, start_frame:]
                t = t[:, start_frame:]; t_next = t_next[:, start_frame:]

                with torch.no_grad():
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        v = model(x_curr, t, actions[:, start_frame: i + 1])

                ab_t = alphas_cumprod[t]
                ab_next = alphas_cumprod[t_next].clone()
                ab_next[:, :-1] = torch.ones_like(ab_next[:, :-1])
                if noise_idx == 1:
                    ab_next[:, -1:] = torch.ones_like(ab_next[:, -1:])
                x_start = ab_t.sqrt() * x_curr - (1 - ab_t).sqrt() * v
                eps_t = (1 - ab_t).sqrt() * x_curr + ab_t.sqrt() * v
                x_pred = ab_next.sqrt() * x_start + (1 - ab_next).sqrt() * eps_t
                x[:, -1:] = x_pred[:, -1:]
                # Cache the latest chunk at this noise_idx for next frame to warm-start
                this_frame_cache[noise_idx] = x[:, -1:].clone().detach()

            last_frame_step_cache = this_frame_cache
        return x

    def _time_run(fn) -> tuple:
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        out = fn()
        e.record(); torch.cuda.synchronize()
        return out, s.elapsed_time(e)

    print(f"\n=== Pass 1: DDIM-{ddim_steps} baseline ===")
    bl_lat, bl_ms = _time_run(lambda: _generate(use_warmstart=False, seed_val=seed))
    print(f"Baseline: {bl_ms:.1f} ms")
    print(f"\n=== Pass 2: warm start (skip first {ddim_steps-warm_start_steps} steps, blend={blend_alpha}) ===")
    sk_lat, sk_ms = _time_run(lambda: _generate(use_warmstart=True, seed_val=seed))
    print(f"WarmStart: {sk_ms:.1f} ms")

    @torch.no_grad()
    def _decode(latents):
        z = rearrange(latents, "b t c h w -> (b t) (h w) c")
        with torch.cuda.amp.autocast(dtype=torch.float16):
            decoded = (vae.decode(z / SCALING_FACTOR) + 1.0) / 2.0
        return decoded.float().clamp(0.0, 1.0)
    bl_pix = _decode(bl_lat.float()).cpu()
    sc_pix = _decode(sk_lat.float()).cpu()

    bl_lat_cpu = bl_lat[0].float().cpu(); sc_lat_cpu = sk_lat[0].float().cpu()
    per_frame: list = []
    for f in range(bl_lat_cpu.shape[0]):
        latent_mse = ((bl_lat_cpu[f] - sc_lat_cpu[f]) ** 2).mean().item()
        pixel_mse = ((bl_pix[f] - sc_pix[f]) ** 2).mean().item()
        pixel_psnr = 10.0 * math.log10(1.0 / max(pixel_mse, 1e-12))
        per_frame.append({"frame": f, "latent_mse": round(latent_mse, 6),
                          "pixel_psnr": round(pixel_psnr, 3)})
    psnr_vals = [p["pixel_psnr"] for p in per_frame[n_prompt:]]
    aggregate = {"psnr_mean_db": round(sum(psnr_vals) / len(psnr_vals), 3),
                 "psnr_min_db": round(min(psnr_vals), 3),
                 "latent_mse_max": round(max(p["latent_mse"] for p in per_frame[n_prompt:]), 6)}
    def _self_coh(pix):
        prevs = []
        for i in range(1, pix.shape[0]):
            mse = ((pix[i] - pix[i-1]) ** 2).mean().item()
            prevs.append(10.0 * math.log10(1.0 / max(mse, 1e-12)))
        return {"vs_prev_psnr_mean": round(sum(prevs) / len(prevs), 3),
                "vs_prev_psnr_min": round(min(prevs), 3)}
    coh_bl = _self_coh(bl_pix); coh_sk = _self_coh(sc_pix)
    print(f"\nCross-PSNR: {aggregate}")
    print(f"Self-coh baseline: {coh_bl}")
    print(f"Self-coh warmstart: {coh_sk}")

    out_dir = os.path.join(MODEL_CACHE, "eval_outputs"); os.makedirs(out_dir, exist_ok=True)
    def _save(pix, fname):
        path = os.path.join(out_dir, fname)
        arr = (pix * 255.0).round().clamp(0, 255).byte()
        arr = rearrange(arr, "t c h w -> t h w c").cpu().numpy()
        try:
            imageio.mimwrite(path, arr, fps=24, codec="libx264", quality=8)
            print(f"  saved {path}")
        except Exception as e: print(f"  fail {path}: {e}")
        return path
    bl_mp4 = _save(bl_pix, f"baseline_warmstart_f{num_frames}_seed{seed}.mp4")
    sk_mp4 = _save(sc_pix, f"warmstart_k{warm_start_steps}_b{blend_alpha}_f{num_frames}_seed{seed}.mp4")
    sxs_mp4 = os.path.join(out_dir, f"sidebyside_warmstart_f{num_frames}_seed{seed}.mp4")
    try:
        sxs = torch.cat([bl_pix, sc_pix], dim=-1)
        sxs_arr = (sxs * 255.0).round().clamp(0, 255).byte()
        sxs_arr = rearrange(sxs_arr, "t c h w -> t h w c").cpu().numpy()
        imageio.mimwrite(sxs_mp4, sxs_arr, fps=24, codec="libx264", quality=8)
    except Exception: pass
    try: model_volume.commit()
    except Exception: pass

    return {
        "model": "Etched/oasis-500m",
        "kernel": f"warmstart_k{warm_start_steps}_b{blend_alpha}",
        "warm_start_steps": warm_start_steps, "blend_alpha": blend_alpha,
        "baseline_latency_ms": round(bl_ms, 2),
        "warmstart_latency_ms": round(sk_ms, 2),
        "speedup": round(bl_ms / sk_ms, 3),
        "quality_aggregate": aggregate,
        "self_coherence_baseline": coh_bl,
        "self_coherence_warmstart": coh_sk,
        "videos": {"baseline_mp4": bl_mp4, "warmstart_mp4": sk_mp4, "sidebyside_mp4": sxs_mp4},
        "num_frames": num_frames, "seed": seed, "gpu": "H100",
    }


@app.local_entrypoint(name="oasis_warmstart_modal")
def main(num_frames: int = 32, seed: int = 42,
         warm_start_steps: int = 6, blend_alpha: float = 0.5):
    result = run_oasis_warmstart.remote(
        num_frames=num_frames, seed=seed,
        warm_start_steps=warm_start_steps, blend_alpha=blend_alpha,
    )
    print(f"\n=== Cross-Frame Warm-Start ===")
    print(f"  speedup:        {result.get('speedup')}x")
    print(f"  vs_prev (base): {result['self_coherence_baseline']['vs_prev_psnr_mean']} dB")
    print(f"  vs_prev (warm): {result['self_coherence_warmstart']['vs_prev_psnr_mean']} dB")
    sys.path.insert(0, "/root/benchmarks")
    from result_store import save_result
    save_result(result, "optimised_kernels", f"oasis_warmstart_k{warm_start_steps}_f{num_frames}")
