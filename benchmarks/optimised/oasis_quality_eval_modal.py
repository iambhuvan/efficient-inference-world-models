"""
Modal H100 quality eval: real Open-Oasis prompt + real Minecraft action
sequence, run baseline AND KVCache+substitution at the SAME seed under the
canonical open-oasis generate.py sampler, then compute per-frame divergence
metrics (latent MSE, pixel PSNR, pixel L1).

Inputs (shipped at /root/open-oasis/sample_data/):
  * sample_image_0.png                              — 1 real Minecraft frame
  * sample_actions_0.one_hot_actions.pt             — real 25-dim action stream

Loaded via open-oasis/utils.py: load_prompt() and load_actions().

Sampler matches open-oasis/generate.py exactly (stabilization_level=15 for
context frames, sliding-window over model.max_frames, fp16 autocast,
v-prediction DDIM with the noise_abs_max=20 clip on initial chunk noise).

Run:
    PYTHONPATH=benchmarks modal run benchmarks/optimised/oasis_quality_eval_modal.py
"""

import sys
sys.path.insert(0, "/root")
sys.path.insert(0, "/root/benchmarks")

import modal
from modal_common import app, image, hf_secret, model_volume, MODEL_CACHE


SCALING_FACTOR = 0.07843137255  # open-oasis VAE scaling factor


@app.function(
    gpu="H100",
    image=image,
    secrets=[hf_secret],
    volumes={MODEL_CACHE: model_volume},
    timeout=3600,
    memory=32768,
)
def run_oasis_quality_eval(
    num_frames: int = 32,
    n_prompt: int = 1,
    ddim_steps: int = 10,
    max_noise_level: int = 1000,
    stabilization_level: int = 15,
    noise_abs_max: float = 20.0,
    seed: int = 42,
    prompt_file: str = "sample_image_0.png",
    actions_file: str = "sample_actions_0.one_hot_actions.pt",
    # KVCache config
    enable_quant_substitution: bool = True,
    use_temporal_merge: bool = True,
    use_progressive_quant: bool = True,
    fp8_age_threshold: int = 3,
    int4_age_threshold: int = 10,
    similarity_threshold: float = 0.95,
    # Action-magnitude gate: skip the substitution roundtrip on frames
    # whose action vector exceeds this L1 magnitude. 0 = gate off (always
    # substitute). Typical action magnitudes are 0-3 for the open-oasis
    # one_hot stream. Default 0.0 keeps prior behavior.
    action_gate_threshold: float = 0.0,
) -> dict:
    """Run baseline + KVCache at the same seed on real Oasis data; compute metrics."""
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

    hf_token = os.environ.get("HF_TOKEN", "").strip() or None

    # -------------------------------------------------------------------------
    # Load DiT + VAE
    # -------------------------------------------------------------------------
    print("Downloading Etched/oasis-500m checkpoints ...")
    dit_ckpt = hf_hub_download(
        "Etched/oasis-500m", "oasis500m.safetensors",
        cache_dir=MODEL_CACHE, token=hf_token,
    )
    vae_ckpt = hf_hub_download(
        "Etched/oasis-500m", "vit-l-20.safetensors",
        cache_dir=MODEL_CACHE, token=hf_token,
    )

    print("Loading DiT-S/2 ...")
    model = DiT_models["DiT-S/2"]()
    model.load_state_dict(safetensors_load(dit_ckpt), strict=False)
    model = model.to("cuda").eval()

    print("Loading ViT-L VAE ...")
    vae = VAE_models["vit-l-20-shallow-encoder"]()
    vae.load_state_dict(safetensors_load(vae_ckpt), strict=False)
    vae = vae.to("cuda").eval()

    # -------------------------------------------------------------------------
    # Real Oasis inputs
    # -------------------------------------------------------------------------
    prompt_path = f"/root/open-oasis/sample_data/{prompt_file}"
    actions_path = f"/root/open-oasis/sample_data/{actions_file}"
    print(f"Loading prompt:  {prompt_path}")
    print(f"Loading actions: {actions_path}")

    total_frames = n_prompt + num_frames

    # Real prompt frame → (1, n_prompt, 3, 360, 640) in [0, 1]
    x_pixels = load_prompt(prompt_path, n_prompt_frames=n_prompt).to("cuda")
    # Real action sequence → (1, T+1, 25); load_actions prepends a zero action
    actions = load_actions(actions_path).to("cuda")[:, :total_frames]
    print(f"Prompt RGB shape: {tuple(x_pixels.shape)}, dtype={x_pixels.dtype}")
    print(f"Actions shape:    {tuple(actions.shape)}, "
          f"active-key-count={int(actions.sum().item())}")

    # VAE-encode the prompt frame to a latent (matches generate.py exactly)
    B = x_pixels.shape[0]
    H_pix, W_pix = x_pixels.shape[-2:]
    flat = rearrange(x_pixels, "b t c h w -> (b t) c h w")
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            z = vae.encode(flat * 2 - 1).mean * SCALING_FACTOR  # (B*T, S, 16)
    H_lat = H_pix // vae.patch_size
    W_lat = W_pix // vae.patch_size
    prompt_latent = rearrange(
        z, "(b t) (h w) c -> b t c h w", t=n_prompt, h=H_lat, w=W_lat
    )
    print(f"Prompt latent: {tuple(prompt_latent.shape)}  "
          f"(patch_size={vae.patch_size}, latent grid {H_lat}x{W_lat})")

    # -------------------------------------------------------------------------
    # DDPM sigmoid schedule
    # -------------------------------------------------------------------------
    betas = sigmoid_beta_schedule(max_noise_level).float().to("cuda")
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    alphas_cumprod = rearrange(alphas_cumprod, "T -> T 1 1 1")
    noise_range = torch.linspace(-1, max_noise_level - 1, ddim_steps + 1)

    model_max_frames = getattr(model, "max_frames", total_frames + 1)

    # Per-frame action magnitude (for the action gate). actions[0, i] is the
    # 25-dim one_hot vector for frame i. We use L1 magnitude as a coarse
    # "is this frame busy" signal — high mag = camera moving / keys pressed.
    action_magnitudes = actions[0].abs().sum(dim=-1).cpu().tolist()  # length total_frames

    gate_log: list = []  # populated only when manager is wired

    # -------------------------------------------------------------------------
    # Generation helper — verbatim port of open-oasis/generate.py main loop
    # -------------------------------------------------------------------------
    def _generate(manager=None, seed_val: int = 42) -> torch.Tensor:
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        if manager is not None:
            manager.create_cache(num_layers=16, max_frames=total_frames + 1)

        x = prompt_latent.clone()  # (B, n_prompt, 16, H_lat, W_lat)

        for i in range(n_prompt, total_frames):
            if manager is not None:
                manager.set_current_frame_idx(i)
                # Action-magnitude gate: skip substitution on busy frames.
                if action_gate_threshold > 0.0:
                    am = float(action_magnitudes[i])
                    substitute = am < action_gate_threshold
                    manager.set_substitution_active(substitute)
                    gate_log.append({
                        "frame": i,
                        "action_mag": round(am, 4),
                        "substitute": substitute,
                    })
                else:
                    manager.set_substitution_active(True)

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

        return x  # (B, total_frames, 16, 18, 32)

    # -------------------------------------------------------------------------
    # Pass 1: baseline (no manager)
    # -------------------------------------------------------------------------
    print("\n=== Pass 1: baseline ===")
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    baseline_latents = _generate(manager=None, seed_val=seed)
    e.record(); torch.cuda.synchronize()
    baseline_ms = s.elapsed_time(e)
    print(f"Baseline: {baseline_ms:.1f} ms  ({num_frames / (baseline_ms / 1000):.2f} fps)")

    # -------------------------------------------------------------------------
    # Pass 2: KVCache + substitution
    # -------------------------------------------------------------------------
    print("\n=== Pass 2: KVCache + substitution ===")
    config = {
        "num_layers": 16,
        "max_frames": total_frames + 1,
        "enable_quant_substitution": enable_quant_substitution,
    }
    if use_temporal_merge:
        config["temporal_merge"] = {
            "similarity_threshold": similarity_threshold,
            "merge_strategy": "weighted_average",
        }
    if use_progressive_quant:
        config["progressive_quant"] = {
            "fp8_age_threshold": fp8_age_threshold,
            "int4_age_threshold": int4_age_threshold,
            "key_value_asymmetry": True,
        }
    from worldserve.optimizations.system_level.kv_cache.manager import KVCacheManager

    manager = KVCacheManager(config=config)
    n_wired = manager.wire_to_model(model)
    print(f"KVCacheManager wired into {n_wired} attention modules")

    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    kvcache_latents = _generate(manager=manager, seed_val=seed)
    e.record(); torch.cuda.synchronize()
    kvcache_ms = s.elapsed_time(e)
    print(f"KVCache + sub: {kvcache_ms:.1f} ms  ({num_frames / (kvcache_ms / 1000):.2f} fps)")

    mem_report = manager.get_memory_report()

    # -------------------------------------------------------------------------
    # VAE decode both latent traces to pixels in [0, 1]
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def _decode(latents: torch.Tensor) -> torch.Tensor:
        """(1, T, 16, H_lat, W_lat) → (T, 3, 360, 640) in [0, 1]."""
        T = latents.shape[1]
        z = rearrange(latents, "b t c h w -> (b t) (h w) c")
        with torch.cuda.amp.autocast(dtype=torch.float16):
            decoded = (vae.decode(z / SCALING_FACTOR) + 1.0) / 2.0  # (B*T, 3, H, W)
        decoded = decoded.float().clamp(0.0, 1.0)
        return decoded  # (T, 3, 360, 640)

    print("\n=== Decoding ===")
    baseline_pixels = _decode(baseline_latents.float())
    kvcache_pixels = _decode(kvcache_latents.float())
    print(f"Decoded shapes: baseline={tuple(baseline_pixels.shape)}, "
          f"kvcache={tuple(kvcache_pixels.shape)}")

    # -------------------------------------------------------------------------
    # Per-frame divergence metrics
    # -------------------------------------------------------------------------
    print("\n=== Per-frame divergence metrics ===")
    per_frame: list = []
    bl_lat = baseline_latents[0].float().cpu()
    kv_lat = kvcache_latents[0].float().cpu()
    bl_pix = baseline_pixels.cpu()
    kv_pix = kvcache_pixels.cpu()

    for f in range(bl_lat.shape[0]):
        latent_mse = ((bl_lat[f] - kv_lat[f]) ** 2).mean().item()
        latent_l1 = (bl_lat[f] - kv_lat[f]).abs().mean().item()
        pixel_mse = ((bl_pix[f] - kv_pix[f]) ** 2).mean().item()
        pixel_psnr = 10.0 * math.log10(1.0 / max(pixel_mse, 1e-12))
        pixel_l1 = (bl_pix[f] - kv_pix[f]).abs().mean().item()

        if f < n_prompt:
            tier = "(prompt)"
        elif f >= int4_age_threshold:
            tier = "int4"
        elif f >= fp8_age_threshold:
            tier = "fp8"
        else:
            tier = "fp16"

        per_frame.append({
            "frame": f,
            "tier": tier,
            "latent_mse": round(latent_mse, 6),
            "latent_l1": round(latent_l1, 6),
            "pixel_mse": round(pixel_mse, 6),
            "pixel_psnr": round(pixel_psnr, 3),
            "pixel_l1": round(pixel_l1, 6),
        })
        print(
            f"  frame {f:2d}  tier={tier:9s}  "
            f"lat_mse={latent_mse:.5f}  pix_psnr={pixel_psnr:6.2f} dB  "
            f"pix_l1={pixel_l1:.5f}"
        )

    psnr_vals = [p["pixel_psnr"] for p in per_frame[n_prompt:]]
    lat_mse_vals = [p["latent_mse"] for p in per_frame[n_prompt:]]
    aggregate = {
        "psnr_mean_db": round(sum(psnr_vals) / len(psnr_vals), 3),
        "psnr_min_db": round(min(psnr_vals), 3),
        "psnr_max_db": round(max(psnr_vals), 3),
        "latent_mse_mean": round(sum(lat_mse_vals) / len(lat_mse_vals), 6),
        "latent_mse_max": round(max(lat_mse_vals), 6),
    }
    print(f"\nAggregate over {len(psnr_vals)} generated frames: {aggregate}")

    # -------------------------------------------------------------------------
    # Self-coherence: does each run produce coherent video on its own?
    # vs_prompt = drift from start (lower = drift, higher = static)
    # vs_prev   = per-step delta (~23 dB = coherent motion; near 0 = chaos;
    #             near 60+ = frozen)
    # -------------------------------------------------------------------------
    def _self_coherence(pix: torch.Tensor) -> dict:
        """pix: (T, 3, H, W) cpu fp32 in [0, 1]."""
        per: list = []
        prompt = pix[0]
        for i in range(pix.shape[0]):
            f = pix[i]
            mse_p = ((f - prompt) ** 2).mean().item()
            psnr_p = 10.0 * math.log10(1.0 / max(mse_p, 1e-12))
            if i == 0:
                psnr_v = float("nan")
                l1_v = float("nan")
            else:
                d = f - pix[i - 1]
                mse_v = (d ** 2).mean().item()
                psnr_v = 10.0 * math.log10(1.0 / max(mse_v, 1e-12))
                l1_v = d.abs().mean().item()
            per.append({
                "frame": i,
                "vs_prompt_psnr_db": round(psnr_p, 3),
                "vs_prev_psnr_db": round(psnr_v, 3) if not math.isnan(psnr_v) else None,
                "vs_prev_l1": round(l1_v, 5) if not math.isnan(l1_v) else None,
            })
        prev_vals = [p["vs_prev_psnr_db"] for p in per[1:]]
        prompt_vals = [p["vs_prompt_psnr_db"] for p in per]
        return {
            "vs_prompt_psnr_mean": round(sum(prompt_vals) / len(prompt_vals), 3),
            "vs_prompt_psnr_min":  round(min(prompt_vals), 3),
            "vs_prev_psnr_mean":   round(sum(prev_vals) / len(prev_vals), 3),
            "vs_prev_psnr_min":    round(min(prev_vals), 3),
            "per_frame": per,
        }

    print("\n=== Self-coherence ===")
    coh_baseline = _self_coherence(bl_pix)
    coh_kvcache  = _self_coherence(kv_pix)
    print(f"baseline:      vs_prompt mean={coh_baseline['vs_prompt_psnr_mean']:.2f} dB  "
          f"vs_prev mean={coh_baseline['vs_prev_psnr_mean']:.2f} dB  "
          f"vs_prev min={coh_baseline['vs_prev_psnr_min']:.2f} dB")
    print(f"kvcache+sub:   vs_prompt mean={coh_kvcache['vs_prompt_psnr_mean']:.2f} dB  "
          f"vs_prev mean={coh_kvcache['vs_prev_psnr_mean']:.2f} dB  "
          f"vs_prev min={coh_kvcache['vs_prev_psnr_min']:.2f} dB")

    # -------------------------------------------------------------------------
    # How MUCH quantization actually fired
    # -------------------------------------------------------------------------
    quant_stats = manager.get_quantization_stats()
    print(f"\n=== Quantization volume ===\n{json.dumps({k:v for k,v in quant_stats.items() if k != 'per_frame'}, indent=2)}")

    # -------------------------------------------------------------------------
    # Save BOTH MP4s + snapshots so the user can play side by side
    # -------------------------------------------------------------------------
    print("\n=== Saving videos ===")
    save_dir = "eval_outputs"
    out_dir = os.path.join(MODEL_CACHE, save_dir)
    os.makedirs(out_dir, exist_ok=True)
    import imageio.v2 as imageio

    def _save_video(pix: torch.Tensor, fname: str) -> str:
        path = os.path.join(out_dir, fname)
        arr = (pix * 255.0).round().clamp(0, 255).byte()
        arr = rearrange(arr, "t c h w -> t h w c").cpu().numpy()
        try:
            imageio.mimwrite(path, arr, fps=24, codec="libx264", quality=8)
            print(f"  saved {path}")
        except Exception as exc:
            print(f"  video write failed for {path}: {exc}")
        return path

    base_tag = "fp8only" if int4_age_threshold > num_frames else "fp8int4"
    if action_gate_threshold > 0.0:
        config_tag = f"{base_tag}_gated_t{action_gate_threshold:g}"
    else:
        config_tag = base_tag
    bl_mp4 = _save_video(bl_pix, f"baseline_f{num_frames}_seed{seed}.mp4")
    kv_mp4 = _save_video(kv_pix, f"kvcache_{config_tag}_f{num_frames}_seed{seed}.mp4")

    # Side-by-side composite (left: baseline, right: KVCache)
    sxs_mp4 = os.path.join(out_dir, f"sidebyside_{config_tag}_f{num_frames}_seed{seed}.mp4")
    try:
        sxs_pix = torch.cat([bl_pix, kv_pix], dim=-1)  # concat along W
        sxs_arr = (sxs_pix * 255.0).round().clamp(0, 255).byte()
        sxs_arr = rearrange(sxs_arr, "t c h w -> t h w c").cpu().numpy()
        imageio.mimwrite(sxs_mp4, sxs_arr, fps=24, codec="libx264", quality=8)
        print(f"  saved {sxs_mp4}")
    except Exception as exc:
        print(f"  side-by-side write failed: {exc}")

    try:
        model_volume.commit()
        print("  volume committed")
    except Exception as exc:
        print(f"  volume commit failed: {exc}")

    fps_baseline = num_frames / (baseline_ms / 1000.0)
    fps_kvcache = num_frames / (kvcache_ms / 1000.0)
    results = {
        "model": "Etched/oasis-500m",
        "kernel": "kvcache_quality_eval_real_oasis_data",
        "data_source": {
            "prompt_file": prompt_file,
            "actions_file": actions_file,
            "n_prompt": n_prompt,
            "num_frames": num_frames,
            "ddim_steps": ddim_steps,
            "stabilization_level": stabilization_level,
            "seed": seed,
        },
        "baseline": {"latency_ms": round(baseline_ms, 2), "fps": round(fps_baseline, 3)},
        "kvcache":  {"latency_ms": round(kvcache_ms, 2),  "fps": round(fps_kvcache, 3)},
        "kvcache_overhead_pct": round(100.0 * (kvcache_ms - baseline_ms) / baseline_ms, 2),
        "kvcache_config": config,
        "kvcache_n_attn_modules_wired": n_wired,
        "kvcache_memory_report": {
            "total_fp16_bytes": mem_report.get("total_fp16_bytes", 0),
            "total_actual_bytes": mem_report.get("total_actual_bytes", 0),
            "savings_ratio": round(mem_report.get("savings_ratio", 0.0), 4),
            "per_technique": mem_report.get("per_technique", {}),
        },
        "quantization_stats": quant_stats,
        "self_coherence": {
            "baseline": {k: v for k, v in coh_baseline.items() if k != "per_frame"},
            "kvcache":  {k: v for k, v in coh_kvcache.items()  if k != "per_frame"},
            "baseline_per_frame": coh_baseline["per_frame"],
            "kvcache_per_frame":  coh_kvcache["per_frame"],
        },
        "videos": {
            "baseline_mp4": bl_mp4,
            "kvcache_mp4":  kv_mp4,
            "sidebyside_mp4": sxs_mp4,
        },
        "action_gating": {
            "threshold": action_gate_threshold,
            "frames_substituted": sum(1 for g in gate_log if g["substitute"]),
            "frames_skipped":     sum(1 for g in gate_log if not g["substitute"]),
            "per_frame_log": gate_log,
            "action_magnitudes": [round(m, 4) for m in action_magnitudes],
        },
        "quality_metrics": {
            "aggregate": aggregate,
            "per_frame": per_frame,
        },
        "gpu": "H100",
    }
    print("\n" + json.dumps({"baseline": results["baseline"],
                              "kvcache": results["kvcache"],
                              "overhead_pct": results["kvcache_overhead_pct"],
                              "aggregate": aggregate}, indent=2))
    return results


@app.local_entrypoint(name="oasis_quality_eval_modal")
def main(
    num_frames: int = 32,
    seed: int = 42,
    enable_quant_substitution: bool = True,
    fp8_age_threshold: int = 3,
    int4_age_threshold: int = 10,
    action_gate_threshold: float = 0.0,
):
    result = run_oasis_quality_eval.remote(
        num_frames=num_frames,
        seed=seed,
        enable_quant_substitution=enable_quant_substitution,
        fp8_age_threshold=fp8_age_threshold,
        int4_age_threshold=int4_age_threshold,
        action_gate_threshold=action_gate_threshold,
    )
    print("\n=== Quality eval summary ===")
    print(f"Baseline: {result['baseline']}")
    print(f"KVCache:  {result['kvcache']}")
    print(f"Overhead: {result['kvcache_overhead_pct']}%")
    print(f"Aggregate quality: {result['quality_metrics']['aggregate']}")

    sys.path.insert(0, "/root/benchmarks")
    from result_store import save_result
    save_result(result, "optimised_kernels", f"oasis_quality_eval_f{num_frames}")
