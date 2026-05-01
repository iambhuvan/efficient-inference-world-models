"""
Modal H100 benchmark: pure-baseline Oasis generation on real data, save video
+ self-coherence metrics so we can verify baseline output is meaningful at the
generation lengths we benchmark KV compression against.

Concretely:
  * Loads sample_image_0.png as the real prompt frame, sample_actions_0 as the
    real keyboard sequence.
  * Runs the canonical generate.py sampler (stabilization_level=15, sliding
    window, fp16 autocast).
  * Decodes every frame to RGB.
  * Saves the full sequence as an MP4 (and frame snapshots as PNG) to the
    persistent worldserve-models volume so we can inspect them locally via
    `modal volume get worldserve-models eval_outputs/...`.
  * Computes baseline self-coherence metrics:
      - L1 / MSE / PSNR vs the prompt frame  (drift over time)
      - L1 / MSE / PSNR vs the previous frame (per-step delta)
    These tell us whether baseline is collapsing (per-step delta -> 0),
    drifting wildly (per-step delta exploding), or staying coherent.

Run:
    PYTHONPATH=benchmarks modal run benchmarks/baseline/oasis_baseline_inspection_modal.py
After the run:
    modal volume get worldserve-models eval_outputs/baseline_75f.mp4 ./
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
def run_baseline_inspection(
    num_frames: int = 75,
    n_prompt: int = 1,
    ddim_steps: int = 10,
    max_noise_level: int = 1000,
    stabilization_level: int = 15,
    noise_abs_max: float = 20.0,
    seed: int = 42,
    prompt_file: str = "sample_image_0.png",
    actions_file: str = "sample_actions_0.one_hot_actions.pt",
    save_dir: str = "eval_outputs",
    fps: int = 24,
) -> dict:
    """Run pure baseline at *num_frames*, save the video, return self-coherence metrics."""
    import json
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

    # -------------------------------------------------------------------------
    # Load DiT + VAE
    # -------------------------------------------------------------------------
    print("Downloading Etched/oasis-500m checkpoints ...")
    dit_ckpt = hf_hub_download("Etched/oasis-500m", "oasis500m.safetensors",
                               cache_dir=MODEL_CACHE, token=hf_token)
    vae_ckpt = hf_hub_download("Etched/oasis-500m", "vit-l-20.safetensors",
                               cache_dir=MODEL_CACHE, token=hf_token)

    print("Loading DiT-S/2 ...")
    model = DiT_models["DiT-S/2"]()
    model.load_state_dict(safetensors_load(dit_ckpt), strict=False)
    model = model.to("cuda").eval()

    print("Loading ViT-L VAE ...")
    vae = VAE_models["vit-l-20-shallow-encoder"]()
    vae.load_state_dict(safetensors_load(vae_ckpt), strict=False)
    vae = vae.to("cuda").eval()

    # -------------------------------------------------------------------------
    # Real prompt + actions
    # -------------------------------------------------------------------------
    prompt_path = f"/root/open-oasis/sample_data/{prompt_file}"
    actions_path = f"/root/open-oasis/sample_data/{actions_file}"
    total_frames = n_prompt + num_frames

    x_pixels = load_prompt(prompt_path, n_prompt_frames=n_prompt).to("cuda")
    actions = load_actions(actions_path).to("cuda")[:, :total_frames]
    print(f"Prompt RGB: {tuple(x_pixels.shape)} | Actions: {tuple(actions.shape)}")

    H_pix, W_pix = x_pixels.shape[-2:]
    flat = rearrange(x_pixels, "b t c h w -> (b t) c h w")
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            z = vae.encode(flat * 2 - 1).mean * SCALING_FACTOR
    H_lat = H_pix // vae.patch_size
    W_lat = W_pix // vae.patch_size
    prompt_latent = rearrange(z, "(b t) (h w) c -> b t c h w",
                              t=n_prompt, h=H_lat, w=W_lat)

    # -------------------------------------------------------------------------
    # Canonical generate.py sampling loop
    # -------------------------------------------------------------------------
    betas = sigmoid_beta_schedule(max_noise_level).float().to("cuda")
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    alphas_cumprod = rearrange(alphas_cumprod, "T -> T 1 1 1")
    noise_range = torch.linspace(-1, max_noise_level - 1, ddim_steps + 1)

    model_max_frames = getattr(model, "max_frames", total_frames + 1)

    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    x = prompt_latent.clone()
    B = x.shape[0]

    print("\n=== Baseline generation ===")
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()

    for i in range(n_prompt, total_frames):
        chunk = torch.randn((B, 1, *x.shape[-3:]), device="cuda")
        chunk = torch.clamp(chunk, -noise_abs_max, +noise_abs_max)
        x = torch.cat([x, chunk], dim=1)
        start_frame = max(0, i + 1 - model_max_frames)

        for noise_idx in reversed(range(1, ddim_steps + 1)):
            t_ctx = torch.full((B, i), stabilization_level - 1,
                               dtype=torch.long, device="cuda")
            t = torch.full((B, 1), int(noise_range[noise_idx].item()),
                           dtype=torch.long, device="cuda")
            t_next = torch.full((B, 1), int(noise_range[noise_idx - 1].item()),
                                dtype=torch.long, device="cuda")
            t_next = torch.where(t_next < 0, t, t_next)
            t = torch.cat([t_ctx, t], dim=1)
            t_next = torch.cat([t_ctx, t_next], dim=1)

            x_curr = x.clone()[:, start_frame:]
            t = t[:, start_frame:]
            t_next = t_next[:, start_frame:]

            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    v = model(x_curr, t, actions[:, start_frame: i + 1])

            x_start = (alphas_cumprod[t].sqrt() * x_curr
                       - (1 - alphas_cumprod[t]).sqrt() * v)
            x_noise = (((1 / alphas_cumprod[t]).sqrt() * x_curr - x_start)
                       / (1 / alphas_cumprod[t] - 1).sqrt())
            alpha_next = alphas_cumprod[t_next].clone()
            alpha_next[:, :-1] = torch.ones_like(alpha_next[:, :-1])
            if noise_idx == 1:
                alpha_next[:, -1:] = torch.ones_like(alpha_next[:, -1:])
            x_pred = alpha_next.sqrt() * x_start + x_noise * (1 - alpha_next).sqrt()
            x[:, -1:] = x_pred[:, -1:]

    e.record(); torch.cuda.synchronize()
    baseline_ms = s.elapsed_time(e)
    print(f"Baseline: {baseline_ms:.1f} ms ({num_frames / (baseline_ms / 1000):.2f} fps)")

    # -------------------------------------------------------------------------
    # VAE decode → RGB in [0, 1]
    # -------------------------------------------------------------------------
    print("\n=== Decoding ===")
    z_flat = rearrange(x.float(), "b t c h w -> (b t) (h w) c")
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            decoded = (vae.decode(z_flat / SCALING_FACTOR) + 1.0) / 2.0
    pixels = decoded.float().clamp(0, 1)  # (T, 3, H, W)
    print(f"Decoded: {tuple(pixels.shape)}")

    # -------------------------------------------------------------------------
    # Compute metrics FIRST so we don't lose them if video write fails
    # -------------------------------------------------------------------------
    print("\n=== Baseline self-coherence ===")
    pixels_cpu = pixels.cpu()
    prompt_pix = pixels_cpu[0]   # (3, H, W) — frame 0 (the encoded-then-decoded prompt)
    per_frame: list = []
    for i in range(total_frames):
        f = pixels_cpu[i]

        # vs prompt (frame 0): how much has the scene drifted?
        diff_prompt = f - prompt_pix
        l1_prompt = diff_prompt.abs().mean().item()
        mse_prompt = (diff_prompt ** 2).mean().item()
        psnr_prompt = 10.0 * math.log10(1.0 / max(mse_prompt, 1e-12))

        # vs previous frame: per-step delta. Tells us if model collapses
        # (delta ≈ 0) or drifts wildly (delta huge).
        if i == 0:
            l1_prev = mse_prev = psnr_prev = float("nan")
        else:
            diff_prev = f - pixels_cpu[i - 1]
            l1_prev = diff_prev.abs().mean().item()
            mse_prev = (diff_prev ** 2).mean().item()
            psnr_prev = 10.0 * math.log10(1.0 / max(mse_prev, 1e-12))

        per_frame.append({
            "frame": i,
            "vs_prompt_l1": round(l1_prompt, 5),
            "vs_prompt_psnr_db": round(psnr_prompt, 3),
            "vs_prev_l1": round(l1_prev, 5) if not math.isnan(l1_prev) else None,
            "vs_prev_psnr_db": round(psnr_prev, 3) if not math.isnan(psnr_prev) else None,
        })
        if i in [0, 1, 5, 10, 20, 30, 40, 50, 60, 70, total_frames - 1]:
            prev_str = (f"vs_prev_psnr={psnr_prev:6.2f} dB" if not math.isnan(psnr_prev)
                        else "                     ")
            print(f"  frame {i:3d}: vs_prompt_psnr={psnr_prompt:6.2f} dB | {prev_str}")

    aggregate = {
        "num_frames": total_frames,
        "vs_prompt_psnr_mean": round(
            sum(p["vs_prompt_psnr_db"] for p in per_frame) / len(per_frame), 3),
        "vs_prompt_psnr_min":  round(min(p["vs_prompt_psnr_db"] for p in per_frame), 3),
        "vs_prev_psnr_mean": round(
            sum(p["vs_prev_psnr_db"] for p in per_frame[1:]) / (len(per_frame) - 1), 3),
        "vs_prev_psnr_min":  round(min(p["vs_prev_psnr_db"] for p in per_frame[1:]), 3),
        "vs_prev_l1_mean":   round(
            sum(p["vs_prev_l1"] for p in per_frame[1:]) / (len(per_frame) - 1), 5),
    }
    print(f"\nAggregate: {aggregate}")

    # -------------------------------------------------------------------------
    # Save MP4 + snapshot PNGs (after metrics so failures here don't lose data)
    # -------------------------------------------------------------------------
    out_dir = os.path.join(MODEL_CACHE, save_dir)
    os.makedirs(out_dir, exist_ok=True)
    mp4_path = os.path.join(out_dir, f"baseline_f{num_frames}_seed{seed}.mp4")
    snapshot_paths: list = []

    try:
        # imageio expects (T, H, W, C) uint8 in [0, 255]
        video_arr = (pixels * 255.0).round().clamp(0, 255).byte()
        video_arr = rearrange(video_arr, "t c h w -> t h w c").cpu().numpy()
        imageio.mimwrite(mp4_path, video_arr, fps=fps, codec="libx264", quality=8)
        print(f"Saved video → {mp4_path}")
    except Exception as exc:
        print(f"video write failed: {exc}")

    try:
        from PIL import Image
        snapshot_frames = sorted(set(
            [0, n_prompt] + list(range(0, total_frames, 10)) + [total_frames - 1]
        ))
        for f_idx in snapshot_frames:
            if 0 <= f_idx < total_frames:
                arr = (pixels[f_idx].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
                png_path = os.path.join(
                    out_dir,
                    f"baseline_f{num_frames}_seed{seed}_frame{f_idx:02d}.png",
                )
                Image.fromarray(arr).save(png_path)
                snapshot_paths.append(png_path)
        print(f"Saved {len(snapshot_paths)} snapshot PNGs")
    except Exception as exc:
        print(f"snapshot write failed: {exc}")

    try:
        model_volume.commit()
        print("Volume committed")
    except Exception as exc:
        print(f"volume commit failed: {exc}")

    return {
        "model": "Etched/oasis-500m",
        "kernel": "baseline_inspection",
        "data_source": {
            "prompt_file": prompt_file,
            "actions_file": actions_file,
            "n_prompt": n_prompt,
            "num_frames": num_frames,
            "ddim_steps": ddim_steps,
            "stabilization_level": stabilization_level,
            "seed": seed,
        },
        "baseline_latency_ms": round(baseline_ms, 2),
        "baseline_fps": round(num_frames / (baseline_ms / 1000), 3),
        "saved_video": mp4_path,
        "saved_snapshots": snapshot_paths,
        "self_coherence_aggregate": aggregate,
        "self_coherence_per_frame": per_frame,
        "gpu": "H100",
    }


@app.local_entrypoint(name="oasis_baseline_inspection_modal")
def main(
    num_frames: int = 75,
    seed: int = 42,
):
    result = run_baseline_inspection.remote(num_frames=num_frames, seed=seed)
    print("\n=== Baseline inspection summary ===")
    print(f"Latency:  {result['baseline_latency_ms']} ms ({result['baseline_fps']} fps)")
    print(f"Video:    {result['saved_video']}")
    print(f"Coherence aggregate: {result['self_coherence_aggregate']}")
    print(
        f"\nDownload locally with:\n"
        f"  modal volume get worldserve-models "
        f"{result['saved_video'].replace('/models/', '')} ./"
    )

    sys.path.insert(0, "/root/benchmarks")
    from result_store import save_result
    save_result(result, "baseline", f"oasis_baseline_inspection_f{num_frames}")
