"""
Modal H100: Difficulty-Aware Step Budget with smooth (non-bucketed) formulas.

Tests whether continuous mappings from action magnitude to step count beat
the 3-bucket step function (which gave 2.44× / +0.76 dB at 32f).

Formulas (--formula flag):
    "bucket"      : 3 thresholds (idle/mod/active → reduced/mid/full)
                    [matches existing oasis_difficulty_steps_modal.py]
    "linear"      : steps = clip(min_steps + slope * mag, min_steps, max_steps)
    "exponential" : steps = clip(min_steps + (max_steps - min_steps)
                                   * (1 - exp(-decay * mag)), min_steps, max_steps)
    "log"         : steps = clip(min_steps + scale * log(1 + mag), min_steps, max_steps)

All clipped to [min_steps, max_steps] and rounded to int.
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
def run_oasis_difficulty_smooth(
    num_frames: int = 32,
    n_prompt: int = 1,
    formula: str = "linear",
    min_steps: int = 2,
    max_steps: int = 10,
    slope: float = 2.5,        # for linear: steps grows by `slope` per unit of mag
    decay: float = 1.0,         # for exponential: 1-exp(-decay*mag)
    scale: float = 4.0,         # for log
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
    model_max_frames = getattr(model, "max_frames", total_frames + 1)

    action_mags = actions[0].abs().sum(dim=-1).cpu().tolist()

    def steps_for_frame(i: int, use_smooth: bool) -> int:
        if not use_smooth:
            return max_steps
        m = action_mags[i]
        if formula == "linear":
            s = min_steps + slope * m
        elif formula == "exponential":
            s = min_steps + (max_steps - min_steps) * (1.0 - math.exp(-decay * m))
        elif formula == "log":
            s = min_steps + scale * math.log(1.0 + m)
        else:
            return max_steps
        s = max(min_steps, min(max_steps, int(round(s))))
        return s

    step_log: list = []

    def _generate(use_smooth: bool, seed_val: int = 42) -> torch.Tensor:
        torch.manual_seed(seed_val); torch.cuda.manual_seed_all(seed_val)
        x = prompt_latent.clone()
        B = x.shape[0]
        for i in range(n_prompt, total_frames):
            steps_i = steps_for_frame(i, use_smooth)
            noise_range = torch.linspace(-1, max_noise_level - 1, steps_i + 1)
            if use_smooth:
                step_log.append({"frame": i, "mag": round(action_mags[i], 3), "steps": steps_i})

            chunk = torch.randn((B, 1, *x.shape[-3:]), device="cuda")
            chunk = torch.clamp(chunk, -noise_abs_max, +noise_abs_max)
            x = torch.cat([x, chunk], dim=1)
            start_frame = max(0, i + 1 - model_max_frames)

            for noise_idx in reversed(range(1, steps_i + 1)):
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
        return x

    def _time_run(fn) -> tuple:
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); out = fn()
        e.record(); torch.cuda.synchronize()
        return out, s.elapsed_time(e)

    print(f"\n=== Pass 1: DDIM-{max_steps} baseline ===")
    bl_lat, bl_ms = _time_run(lambda: _generate(use_smooth=False, seed_val=seed))
    print(f"Baseline: {bl_ms:.1f} ms")
    step_log = []
    print(f"\n=== Pass 2: smooth difficulty ({formula}, "
          f"min={min_steps} max={max_steps}, slope={slope}, decay={decay}, scale={scale}) ===")
    sk_lat, sk_ms = _time_run(lambda: _generate(use_smooth=True, seed_val=seed))
    print(f"Smooth: {sk_ms:.1f} ms")

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
        per_frame.append({"frame": f, "latent_mse": round(latent_mse, 6), "pixel_psnr": round(pixel_psnr, 3)})
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

    total_steps_baseline = num_frames * max_steps
    total_steps_smooth = sum(s["steps"] for s in step_log)
    step_savings = 1.0 - total_steps_smooth / total_steps_baseline
    print(f"\nCross-PSNR: {aggregate}")
    print(f"Self-coh baseline: {coh_bl}")
    print(f"Self-coh smooth: {coh_sk}")
    print(f"Step savings: {step_savings*100:.1f}%")

    out_dir = os.path.join(MODEL_CACHE, "eval_outputs"); os.makedirs(out_dir, exist_ok=True)
    def _save(pix, fname):
        path = os.path.join(out_dir, fname)
        arr = (pix * 255.0).round().clamp(0, 255).byte()
        arr = rearrange(arr, "t c h w -> t h w c").cpu().numpy()
        try: imageio.mimwrite(path, arr, fps=24, codec="libx264", quality=8); print(f"  saved {path}")
        except Exception as e: print(f"  fail {path}: {e}")
        return path
    bl_mp4 = _save(bl_pix, f"baseline_diff_smooth_f{num_frames}_seed{seed}.mp4")
    sk_mp4 = _save(sc_pix, f"diff_smooth_{formula}_f{num_frames}_seed{seed}.mp4")
    sxs_mp4 = os.path.join(out_dir, f"sidebyside_diff_smooth_{formula}_f{num_frames}_seed{seed}.mp4")
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
        "kernel": f"diff_smooth_{formula}",
        "formula": formula,
        "params": {"min_steps": min_steps, "max_steps": max_steps,
                   "slope": slope, "decay": decay, "scale": scale},
        "baseline_latency_ms": round(bl_ms, 2),
        "smooth_latency_ms": round(sk_ms, 2),
        "speedup": round(bl_ms / sk_ms, 3),
        "step_savings_pct": round(step_savings * 100.0, 2),
        "total_steps_baseline": total_steps_baseline,
        "total_steps_smooth": total_steps_smooth,
        "quality_aggregate": aggregate,
        "self_coherence_baseline": coh_bl,
        "self_coherence_smooth": coh_sk,
        "videos": {"baseline_mp4": bl_mp4, "smooth_mp4": sk_mp4, "sidebyside_mp4": sxs_mp4},
        "num_frames": num_frames, "seed": seed, "gpu": "H100",
    }


@app.local_entrypoint(name="oasis_difficulty_smooth_modal")
def main(num_frames: int = 32, seed: int = 42, formula: str = "linear",
         min_steps: int = 2, max_steps: int = 10,
         slope: float = 2.5, decay: float = 1.0, scale: float = 4.0):
    result = run_oasis_difficulty_smooth.remote(
        num_frames=num_frames, seed=seed, formula=formula,
        min_steps=min_steps, max_steps=max_steps,
        slope=slope, decay=decay, scale=scale,
    )
    print(f"\n=== Smooth Difficulty ({formula}) ===")
    print(f"  speedup:        {result.get('speedup')}x")
    print(f"  step_savings:   {result.get('step_savings_pct')}%")
    print(f"  vs_prev (base): {result['self_coherence_baseline']['vs_prev_psnr_mean']}")
    print(f"  vs_prev (smt):  {result['self_coherence_smooth']['vs_prev_psnr_mean']}")
    sys.path.insert(0, "/root/benchmarks")
    from result_store import save_result
    save_result(result, "optimised_kernels", f"oasis_diff_smooth_{formula}_f{num_frames}")
