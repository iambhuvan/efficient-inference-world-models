"""
Modal H100 quality eval: baseline DDIM-10 vs DPM-Solver++ 2M @ 5 steps.

Same canonical sampler infrastructure as oasis_quality_eval_modal.py.
Two passes at the same seed:
  Pass 1: DDIM-10 baseline
  Pass 2: DPM-Solver++ 2M with 5 solver steps
Reports: latency, cross-PSNR, self-coherence, saves baseline / dpmpp / sxs MP4s.
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
def run_oasis_dpmpp_quality(
    num_frames: int = 32,
    n_prompt: int = 1,
    ddim_steps: int = 10,
    solver_steps: int = 5,
    max_noise_level: int = 1000,
    stabilization_level: int = 15,
    noise_abs_max: float = 20.0,
    seed: int = 42,
    prompt_file: str = "sample_image_0.png",
    actions_file: str = "sample_actions_0.one_hot_actions.pt",
) -> dict:
    import math
    import os
    import torch
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
    H_lat = H_pix // vae.patch_size
    W_lat = W_pix // vae.patch_size
    prompt_latent = rearrange(z, "(b t) (h w) c -> b t c h w",
                              t=n_prompt, h=H_lat, w=W_lat)

    betas = sigmoid_beta_schedule(max_noise_level).float().to("cuda")
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    alphas_cumprod = rearrange(alphas_cumprod, "T -> T 1 1 1")
    model_max_frames = getattr(model, "max_frames", total_frames + 1)

    def _generate(use_dpmpp: bool, seed_val: int = 42) -> torch.Tensor:
        torch.manual_seed(seed_val); torch.cuda.manual_seed_all(seed_val)
        x = prompt_latent.clone()
        B = x.shape[0]
        steps = solver_steps if use_dpmpp else ddim_steps
        noise_range = torch.linspace(-1, max_noise_level - 1, steps + 1)

        for i in range(n_prompt, total_frames):
            chunk = torch.randn((B, 1, *x.shape[-3:]), device="cuda")
            chunk = torch.clamp(chunk, -noise_abs_max, +noise_abs_max)
            x = torch.cat([x, chunk], dim=1)
            start_frame = max(0, i + 1 - model_max_frames)

            eps_prev = None
            for noise_idx in reversed(range(1, steps + 1)):
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

                if use_dpmpp:
                    # DPM-Solver++ 2M multistep
                    if eps_prev is None or eps_prev.shape != eps_t.shape:
                        D = eps_t
                    else:
                        D = (3.0 * eps_t - eps_prev) / 2.0
                    eps_prev = eps_t
                    x_pred = ab_next.sqrt() * x_start + (1 - ab_next).sqrt() * D
                else:
                    # Plain DDIM v-prediction update
                    x_pred = ab_next.sqrt() * x_start + (1 - ab_next).sqrt() * eps_t
                x[:, -1:] = x_pred[:, -1:]

        return x

    print(f"\n=== Pass 1: DDIM-{ddim_steps} baseline ===")
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    baseline_lat = _generate(use_dpmpp=False, seed_val=seed)
    e.record(); torch.cuda.synchronize()
    baseline_ms = s.elapsed_time(e)
    print(f"DDIM-{ddim_steps}: {baseline_ms:.1f} ms ({num_frames/(baseline_ms/1000):.2f} fps)")

    print(f"\n=== Pass 2: DPM-Solver++ 2M @ {solver_steps} steps ===")
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    dpmpp_lat = _generate(use_dpmpp=True, seed_val=seed)
    e.record(); torch.cuda.synchronize()
    dpmpp_ms = s.elapsed_time(e)
    print(f"DPM-Solver++ 2M: {dpmpp_ms:.1f} ms ({num_frames/(dpmpp_ms/1000):.2f} fps)")

    @torch.no_grad()
    def _decode(latents: torch.Tensor) -> torch.Tensor:
        z = rearrange(latents, "b t c h w -> (b t) (h w) c")
        with torch.cuda.amp.autocast(dtype=torch.float16):
            decoded = (vae.decode(z / SCALING_FACTOR) + 1.0) / 2.0
        return decoded.float().clamp(0.0, 1.0)

    bl_pix = _decode(baseline_lat.float()).cpu()
    sc_pix = _decode(dpmpp_lat.float()).cpu()

    # Cross + self-coherence metrics
    per_frame: list = []
    bl_lat = baseline_lat[0].float().cpu()
    sc_lat = dpmpp_lat[0].float().cpu()
    for f in range(bl_lat.shape[0]):
        latent_mse = ((bl_lat[f] - sc_lat[f]) ** 2).mean().item()
        pixel_mse = ((bl_pix[f] - sc_pix[f]) ** 2).mean().item()
        pixel_psnr = 10.0 * math.log10(1.0 / max(pixel_mse, 1e-12))
        per_frame.append({
            "frame": f, "latent_mse": round(latent_mse, 6),
            "pixel_psnr": round(pixel_psnr, 3),
        })
        if f in [0, 1, 5, 10, 15, 20, 25, num_frames]:
            print(f"  frame {f:3d}  lat_mse={latent_mse:.5f}  pix_psnr={pixel_psnr:6.2f} dB")

    psnr_vals = [p["pixel_psnr"] for p in per_frame[n_prompt:]]
    aggregate = {
        "psnr_mean_db": round(sum(psnr_vals) / len(psnr_vals), 3),
        "psnr_min_db": round(min(psnr_vals), 3),
        "latent_mse_max": round(max(p["latent_mse"] for p in per_frame[n_prompt:]), 6),
    }
    print(f"\nCross-PSNR aggregate: {aggregate}")

    def _self_coh(pix):
        prev_psnrs = []
        for i in range(1, pix.shape[0]):
            mse = ((pix[i] - pix[i - 1]) ** 2).mean().item()
            prev_psnrs.append(10.0 * math.log10(1.0 / max(mse, 1e-12)))
        return {
            "vs_prev_psnr_mean": round(sum(prev_psnrs) / len(prev_psnrs), 3),
            "vs_prev_psnr_min": round(min(prev_psnrs), 3),
        }
    coh_baseline = _self_coh(bl_pix)
    coh_dpmpp = _self_coh(sc_pix)
    print(f"\nSelf-coherence:")
    print(f"  ddim:   {coh_baseline}")
    print(f"  dpmpp:  {coh_dpmpp}")

    # Save MP4s
    out_dir = os.path.join(MODEL_CACHE, "eval_outputs")
    os.makedirs(out_dir, exist_ok=True)
    def _save(pix, fname):
        path = os.path.join(out_dir, fname)
        arr = (pix * 255.0).round().clamp(0, 255).byte()
        arr = rearrange(arr, "t c h w -> t h w c").cpu().numpy()
        try:
            imageio.mimwrite(path, arr, fps=24, codec="libx264", quality=8)
            print(f"  saved {path}")
        except Exception as ex:
            print(f"  FAILED {path}: {ex}")
        return path

    bl_mp4 = _save(bl_pix, f"baseline_dpmpp_f{num_frames}_seed{seed}.mp4")
    sc_mp4 = _save(sc_pix, f"dpmpp_{solver_steps}step_f{num_frames}_seed{seed}.mp4")
    sxs_mp4 = os.path.join(out_dir, f"sidebyside_dpmpp_f{num_frames}_seed{seed}.mp4")
    try:
        sxs = torch.cat([bl_pix, sc_pix], dim=-1)
        sxs_arr = (sxs * 255.0).round().clamp(0, 255).byte()
        sxs_arr = rearrange(sxs_arr, "t c h w -> t h w c").cpu().numpy()
        imageio.mimwrite(sxs_mp4, sxs_arr, fps=24, codec="libx264", quality=8)
        print(f"  saved {sxs_mp4}")
    except Exception as ex:
        print(f"  side-by-side failed: {ex}")
    try:
        model_volume.commit()
    except Exception:
        pass

    return {
        "model": "Etched/oasis-500m",
        "kernel": f"dpmpp_quality_eval_{solver_steps}step",
        "ddim_steps": ddim_steps,
        "solver_steps": solver_steps,
        "baseline_latency_ms": round(baseline_ms, 2),
        "dpmpp_latency_ms": round(dpmpp_ms, 2),
        "speedup": round(baseline_ms / dpmpp_ms, 3),
        "quality_aggregate": aggregate,
        "self_coherence_baseline": coh_baseline,
        "self_coherence_dpmpp": coh_dpmpp,
        "videos": {"baseline_mp4": bl_mp4, "dpmpp_mp4": sc_mp4, "sidebyside_mp4": sxs_mp4},
        "num_frames": num_frames,
        "seed": seed,
        "gpu": "H100",
    }


@app.local_entrypoint(name="oasis_dpmpp_quality_modal")
def main(num_frames: int = 32, seed: int = 42, solver_steps: int = 5):
    result = run_oasis_dpmpp_quality.remote(
        num_frames=num_frames, seed=seed, solver_steps=solver_steps,
    )
    print(f"\n=== DPM-Solver++ quality eval ===")
    print(f"  Speedup:    {result.get('speedup')}x")
    print(f"  PSNR mean:  {result['quality_aggregate']['psnr_mean_db']} dB")
    print(f"  PSNR min:   {result['quality_aggregate']['psnr_min_db']} dB")
    print(f"  vs_prev (ddim):   {result['self_coherence_baseline']['vs_prev_psnr_mean']} dB")
    print(f"  vs_prev (dpmpp):  {result['self_coherence_dpmpp']['vs_prev_psnr_mean']} dB")
    sys.path.insert(0, "/root/benchmarks")
    from result_store import save_result
    save_result(result, "optimised_kernels", f"oasis_dpmpp_quality_s{solver_steps}_f{num_frames}")
