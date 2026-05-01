"""
Modal H100: Action-Aware Cross-Frame V Reuse for Oasis-500M.

When consecutive frames have near-identical action vectors (e.g. player
holding 'forward'), the model's v predictions at the same DDIM step are
also near-identical. We exploit this by caching v across frames.

Per-frame:
  if action[i] cosine-sim with action[i-1] > sim_threshold:
      reuse v_predictions from prior frame at matching steps
  else:
      compute v fresh as in baseline

Most novel item — almost no published work on action-conditional KV/V reuse
for autoregressive video world models.
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
def run_oasis_action_kv(
    num_frames: int = 32,
    n_prompt: int = 1,
    ddim_steps: int = 10,
    sim_threshold: float = 0.95,    # cosine similarity above this → reuse
    max_reuse_steps: int = 5,        # max DDIM steps within a frame to reuse
    max_noise_level: int = 1000,
    stabilization_level: int = 15,
    noise_abs_max: float = 20.0,
    seed: int = 42,
    prompt_file: str = "sample_image_0.png",
    actions_file: str = "sample_actions_0.one_hot_actions.pt",
) -> dict:
    import math, os, torch, torch.nn.functional as F
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

    # Precompute action similarity to previous frame
    action_sim = []
    for i in range(total_frames):
        if i == 0:
            action_sim.append(0.0)
        else:
            a = actions[0, i].float().reshape(-1)
            b = actions[0, i - 1].float().reshape(-1)
            sim = float(F.cosine_similarity(a, b, dim=0)) if (a.norm() > 1e-6 and b.norm() > 1e-6) else 1.0
            action_sim.append(sim)

    forward_count = 0; reuse_count = 0
    reuse_log: list = []

    def _generate(use_kv: bool, seed_val: int = 42) -> torch.Tensor:
        nonlocal forward_count, reuse_count
        torch.manual_seed(seed_val); torch.cuda.manual_seed_all(seed_val)
        x = prompt_latent.clone()
        B = x.shape[0]
        # Cache only the LAST-frame slice of v at each noise_idx.
        # When we reuse, we apply DDIM's slice-math directly on x[:, -1:]
        # without ever invoking the model for the new frame at this step.
        last_frame_v_last_slice: dict = {}

        for i in range(n_prompt, total_frames):
            this_action_sim = action_sim[i]
            should_reuse = use_kv and (this_action_sim > sim_threshold) and bool(last_frame_v_last_slice)
            reuse_log.append({"frame": i, "action_sim": round(this_action_sim, 4),
                              "reuse": bool(should_reuse)})

            chunk = torch.randn((B, 1, *x.shape[-3:]), device="cuda")
            chunk = torch.clamp(chunk, -noise_abs_max, +noise_abs_max)
            x = torch.cat([x, chunk], dim=1)
            start_frame = max(0, i + 1 - model_max_frames)
            this_frame_v_last_slice: dict = {}

            reuse_remaining = max_reuse_steps if should_reuse else 0

            for noise_idx in reversed(range(1, ddim_steps + 1)):
                t_ctx = torch.full((B, i), stabilization_level - 1, dtype=torch.long, device="cuda")
                t = torch.full((B, 1), int(noise_range[noise_idx].item()), dtype=torch.long, device="cuda")
                t_next = torch.full((B, 1), int(noise_range[noise_idx - 1].item()), dtype=torch.long, device="cuda")
                t_next = torch.where(t_next < 0, t, t_next)
                t = torch.cat([t_ctx, t], dim=1); t_next = torch.cat([t_ctx, t_next], dim=1)
                x_curr = x.clone()[:, start_frame:]
                t = t[:, start_frame:]; t_next = t_next[:, start_frame:]

                ab_t = alphas_cumprod[t]
                ab_next = alphas_cumprod[t_next].clone()
                ab_next[:, :-1] = torch.ones_like(ab_next[:, :-1])
                if noise_idx == 1:
                    ab_next[:, -1:] = torch.ones_like(ab_next[:, -1:])

                cached_last = last_frame_v_last_slice.get(noise_idx)
                shape_ok = (cached_last is not None and cached_last.shape == (B, 1, *x.shape[-3:]))
                if reuse_remaining > 0 and shape_ok:
                    # SLICE-MATH: compute DDIM update on last slice only,
                    # using v_last from prior frame's cache (no model forward).
                    v_last = cached_last
                    ab_t_last = ab_t[:, -1:]
                    ab_next_last = ab_next[:, -1:]
                    x_curr_last = x_curr[:, -1:]
                    x_start_last = ab_t_last.sqrt() * x_curr_last - (1 - ab_t_last).sqrt() * v_last
                    eps_t_last = (1 - ab_t_last).sqrt() * x_curr_last + ab_t_last.sqrt() * v_last
                    x_pred_last = ab_next_last.sqrt() * x_start_last + (1 - ab_next_last).sqrt() * eps_t_last
                    x[:, -1:] = x_pred_last
                    reuse_count += 1
                    reuse_remaining -= 1
                    # Update cache with the same slice (will overwrite when fresh)
                    this_frame_v_last_slice[noise_idx] = v_last.detach().clone()
                else:
                    with torch.no_grad():
                        with torch.cuda.amp.autocast(dtype=torch.float16):
                            v = model(x_curr, t, actions[:, start_frame: i + 1])
                    forward_count += 1
                    x_start = ab_t.sqrt() * x_curr - (1 - ab_t).sqrt() * v
                    eps_t = (1 - ab_t).sqrt() * x_curr + ab_t.sqrt() * v
                    x_pred = ab_next.sqrt() * x_start + (1 - ab_next).sqrt() * eps_t
                    x[:, -1:] = x_pred[:, -1:]
                    this_frame_v_last_slice[noise_idx] = v[:, -1:].detach().clone()

            last_frame_v_last_slice = this_frame_v_last_slice
        return x

    def _time_run(fn) -> tuple:
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); out = fn()
        e.record(); torch.cuda.synchronize()
        return out, s.elapsed_time(e)

    print(f"\n=== Pass 1: DDIM-{ddim_steps} baseline ===")
    bl_lat, bl_ms = _time_run(lambda: _generate(use_kv=False, seed_val=seed))
    print(f"Baseline: {bl_ms:.1f} ms")
    forward_count = 0; reuse_count = 0
    print(f"\n=== Pass 2: action-aware V reuse (sim>{sim_threshold}, "
          f"max_reuse={max_reuse_steps}) ===")
    sk_lat, sk_ms = _time_run(lambda: _generate(use_kv=True, seed_val=seed))
    print(f"Action-KV: {sk_ms:.1f} ms")

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

    out_dir = os.path.join(MODEL_CACHE, "eval_outputs"); os.makedirs(out_dir, exist_ok=True)
    def _save(pix, fname):
        path = os.path.join(out_dir, fname)
        arr = (pix * 255.0).round().clamp(0, 255).byte()
        arr = rearrange(arr, "t c h w -> t h w c").cpu().numpy()
        try: imageio.mimwrite(path, arr, fps=24, codec="libx264", quality=8); print(f"  saved {path}")
        except Exception as e: print(f"  fail {path}: {e}")
        return path
    bl_mp4 = _save(bl_pix, f"baseline_action_kv_f{num_frames}_seed{seed}.mp4")
    sk_mp4 = _save(sc_pix, f"action_kv_sim{sim_threshold}_f{num_frames}_seed{seed}.mp4")
    sxs_mp4 = os.path.join(out_dir, f"sidebyside_action_kv_f{num_frames}_seed{seed}.mp4")
    try:
        sxs = torch.cat([bl_pix, sc_pix], dim=-1)
        sxs_arr = (sxs * 255.0).round().clamp(0, 255).byte()
        sxs_arr = rearrange(sxs_arr, "t c h w -> t h w c").cpu().numpy()
        imageio.mimwrite(sxs_mp4, sxs_arr, fps=24, codec="libx264", quality=8)
    except Exception: pass
    try: model_volume.commit()
    except Exception: pass

    reuse_rate = reuse_count / max(forward_count + reuse_count, 1)
    return {
        "model": "Etched/oasis-500m",
        "kernel": f"action_kv_sim{sim_threshold}_max{max_reuse_steps}",
        "sim_threshold": sim_threshold, "max_reuse_steps": max_reuse_steps,
        "baseline_latency_ms": round(bl_ms, 2),
        "action_kv_latency_ms": round(sk_ms, 2),
        "speedup": round(bl_ms / sk_ms, 3),
        "forward_count": forward_count, "reuse_count": reuse_count,
        "reuse_rate": round(reuse_rate, 4),
        "frames_with_reuse": sum(1 for r in reuse_log if r["reuse"]),
        "quality_aggregate": aggregate,
        "self_coherence_baseline": coh_bl,
        "self_coherence_action_kv": coh_sk,
        "videos": {"baseline_mp4": bl_mp4, "action_kv_mp4": sk_mp4, "sidebyside_mp4": sxs_mp4},
        "num_frames": num_frames, "seed": seed, "gpu": "H100",
    }


@app.local_entrypoint(name="oasis_action_kv_modal")
def main(num_frames: int = 32, seed: int = 42,
         sim_threshold: float = 0.95, max_reuse_steps: int = 5):
    result = run_oasis_action_kv.remote(
        num_frames=num_frames, seed=seed,
        sim_threshold=sim_threshold, max_reuse_steps=max_reuse_steps,
    )
    print(f"\n=== Action-Aware V Reuse ===")
    print(f"  speedup:        {result.get('speedup')}x")
    print(f"  reuse_rate:     {result.get('reuse_rate')}")
    print(f"  frames_reused:  {result.get('frames_with_reuse')}")
    print(f"  vs_prev (base): {result['self_coherence_baseline']['vs_prev_psnr_mean']} dB")
    print(f"  vs_prev (kv):   {result['self_coherence_action_kv']['vs_prev_psnr_mean']} dB")
    sys.path.insert(0, "/root/benchmarks")
    from result_store import save_result
    save_result(result, "optimised_kernels",
                f"oasis_action_kv_sim{sim_threshold}_f{num_frames}")
