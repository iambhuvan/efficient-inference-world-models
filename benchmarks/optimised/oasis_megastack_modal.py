"""
Modal H100 benchmark: Open-Oasis 500M MEGA STACK
   DPM-Solver++ 2M  +  Difficulty step budget  +  TaylorSeer block prediction.

Three orthogonal step-count / compute-skip optimizations layered:

  1. DPM-Solver++ 2M base solver (5-step ceiling vs DDIM-10)
  2. Difficulty schedule (idle=2, moderate=3, active=5 steps based on action mag)
  3. TaylorSeer block-feature prediction (skips block compute on history-replay)

Constraints:
  - TaylorSeer order=2 needs 3 history entries before predicting.  On the r=2
    and m=3 difficulty buckets, TaylorSeer cannot activate (history too short).
    Only on the f=5 (active motion) bucket does TaylorSeer get 2 predictable
    steps after a 3-step warmup.
  - The seer state resets per frame (each frame has its own DDIM loop).

Run:
    PYTHONPATH=benchmarks modal run benchmarks/optimised/oasis_megastack_modal.py \
        --num-frames 32 --num-iters 1 \
        --solver-full-steps 5 --solver-mid-steps 3 --solver-reduced-steps 2 \
        --taylor-order 2 --taylor-threshold 0.15
"""

import math
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
    timeout=7200,
    memory=32768,
)
def run_oasis_megastack(
    num_frames: int = 32,
    n_prompt: int = 1,
    ddim_steps: int = 10,
    max_noise_level: int = 1000,
    stabilization_level: int = 15,
    noise_abs_max: float = 20.0,
    seed: int = 42,
    solver_full_steps: int = 5,
    solver_mid_steps: int = 3,
    solver_reduced_steps: int = 2,
    lo_threshold: float = 0.5,
    hi_threshold: float = 1.5,
    taylor_order: int = 2,
    taylor_threshold: float = 0.15,
    prompt_file: str = "sample_image_0.png",
    actions_file: str = "sample_actions_0.one_hot_actions.pt",
) -> dict:
    """DPM-Solver++ + Difficulty + TaylorSeer stacked, with same-session baseline."""
    import json
    import os
    import statistics

    import imageio.v2 as imageio
    import torch

    sys.path.insert(0, "/root/open-oasis")

    from einops import rearrange
    from huggingface_hub import hf_hub_download
    from dit import DiT_models
    from vae import VAE_models
    from safetensors.torch import load_file as safetensors_load
    from utils import sigmoid_beta_schedule, load_prompt, load_actions

    from worldserve.optimizations.model_level.feature_caching.taylor_seer import (
        TaylorSeer,
        _TaylorMetrics,
    )

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

    n_blocks = len(model.blocks)
    print(f"DiT has {n_blocks} blocks")

    # ── prompt + actions ────────────────────────────────────────────────
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
    action_mags = actions[0].abs().sum(dim=-1).cpu().tolist()

    def steps_for_frame(i: int) -> int:
        m = action_mags[i]
        if m < lo_threshold:
            return solver_reduced_steps
        if m < hi_threshold:
            return solver_mid_steps
        return solver_full_steps

    # ── Generation loop, parameterised by stack mode + seer ─────────────
    def _generate(use_stack: bool, seer=None, seed_val: int = 42) -> torch.Tensor:
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)
        x = prompt_latent.clone()
        B = x.shape[0]

        for i in range(n_prompt, total_frames):
            steps_i = steps_for_frame(i) if use_stack else ddim_steps
            noise_range = torch.linspace(-1, max_noise_level - 1, steps_i + 1)

            chunk = torch.randn((B, 1, *x.shape[-3:]), device="cuda")
            chunk = torch.clamp(chunk, -noise_abs_max, +noise_abs_max)
            x = torch.cat([x, chunk], dim=1)
            start_frame = max(0, i + 1 - model_max_frames)

            if seer is not None:
                seer.reset_cache()

            local_step = 0
            eps_prev = None
            for noise_idx in reversed(range(1, steps_i + 1)):
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

                if seer is not None:
                    seer.pre_step(local_step, steps_i, x_curr)

                with torch.no_grad():
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        v = model(x_curr, t, actions[:, start_frame: i + 1])

                if seer is not None:
                    seer.post_step(local_step, steps_i, x_curr)

                ab_t = alphas_cumprod[t]
                ab_next = alphas_cumprod[t_next].clone()
                ab_next[:, :-1] = torch.ones_like(ab_next[:, :-1])
                if noise_idx == 1:
                    ab_next[:, -1:] = torch.ones_like(ab_next[:, -1:])
                x_start = ab_t.sqrt() * x_curr - (1 - ab_t).sqrt() * v
                eps_t = (1 - ab_t).sqrt() * x_curr + ab_t.sqrt() * v

                if use_stack:
                    if eps_prev is None or eps_prev.shape != eps_t.shape:
                        D = eps_t
                    else:
                        D = (3.0 * eps_t - eps_prev) / 2.0
                    eps_prev = eps_t
                    x_pred = ab_next.sqrt() * x_start + (1 - ab_next).sqrt() * D
                else:
                    x_pred = ab_next.sqrt() * x_start + (1 - ab_next).sqrt() * eps_t
                x[:, -1:] = x_pred[:, -1:]
                local_step += 1
        return x

    def _time_run(fn) -> tuple:
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        out = fn()
        e.record()
        torch.cuda.synchronize()
        return out, s.elapsed_time(e)

    # ── Pass 1: DDIM-10 baseline (canonical) ──────────────────────────
    print(f"\n=== Pass 1: DDIM-{ddim_steps} baseline ===")
    bl_lat, bl_ms = _time_run(lambda: _generate(use_stack=False, seed_val=seed))
    print(f"Baseline: {bl_ms:.1f} ms ({num_frames / (bl_ms / 1000):.2f} fps)")

    # ── Pass 2: DPM++ + Difficulty (no TaylorSeer)  — reference midstack
    print(f"\n=== Pass 2: DPM++ {solver_full_steps}/{solver_mid_steps}/{solver_reduced_steps} + Difficulty ===")
    dd_lat, dd_ms = _time_run(lambda: _generate(use_stack=True, seer=None, seed_val=seed))
    print(f"DPM++ + Difficulty: {dd_ms:.1f} ms ({num_frames / (dd_ms / 1000):.2f} fps)")

    # ── Pass 3: full mega-stack with TaylorSeer ───────────────────────
    print(f"\n=== Pass 3: MEGASTACK (DPM++ + Difficulty + TaylorSeer order={taylor_order}) ===")
    seer = TaylorSeer(
        num_layers=n_blocks,
        num_steps=solver_full_steps,  # use the maximum bucket count for capacity
        order=taylor_order,
        prediction_threshold=taylor_threshold,
    )
    seer.wrap_model(model)
    # Reset metrics after wrap (wrap may have logged adjustments)
    seer.metrics = _TaylorMetrics()
    mega_lat, mega_ms = _time_run(lambda: _generate(use_stack=True, seer=seer, seed_val=seed))
    metrics = seer.get_metrics()
    seer.remove_hooks()
    print(f"MegaStack: {mega_ms:.1f} ms ({num_frames / (mega_ms / 1000):.2f} fps) "
          f"predict_rate={metrics['prediction_rate']:.3f}")

    # ── Quality eval: cross-PSNR vs baseline + self-coherence ────────
    @torch.no_grad()
    def _decode(latents):
        z2 = rearrange(latents, "b t c h w -> (b t) (h w) c")
        with torch.cuda.amp.autocast(dtype=torch.float16):
            decoded = (vae.decode(z2 / SCALING_FACTOR) + 1.0) / 2.0
        return decoded.float().clamp(0.0, 1.0)

    bl_pix = _decode(bl_lat.float()).cpu()
    dd_pix = _decode(dd_lat.float()).cpu()
    mega_pix = _decode(mega_lat.float()).cpu()

    def _cross_psnr(a_pix, b_pix):
        a_lat = a_pix; b_lat = b_pix
        per_frame = []
        for f in range(a_lat.shape[0]):
            mse = ((a_lat[f] - b_lat[f]) ** 2).mean().item()
            psnr = 10.0 * math.log10(1.0 / max(mse, 1e-12))
            per_frame.append(psnr)
        vals = per_frame[n_prompt:]
        return {"mean": round(sum(vals) / len(vals), 3),
                "min": round(min(vals), 3)}

    def _self_coh(pix):
        ps = []
        for k in range(1, pix.shape[0]):
            mse = ((pix[k] - pix[k - 1]) ** 2).mean().item()
            ps.append(10.0 * math.log10(1.0 / max(mse, 1e-12)))
        return {"vs_prev_psnr_mean": round(sum(ps) / len(ps), 3),
                "vs_prev_psnr_min": round(min(ps), 3)}

    cross_dd = _cross_psnr(bl_pix, dd_pix)
    cross_mega = _cross_psnr(bl_pix, mega_pix)
    coh_bl = _self_coh(bl_pix)
    coh_dd = _self_coh(dd_pix)
    coh_mega = _self_coh(mega_pix)

    # ── Save side-by-side videos ─────────────────────────────────────
    out_dir = os.path.join(MODEL_CACHE, "eval_outputs")
    os.makedirs(out_dir, exist_ok=True)

    def _save(pix, fname):
        path = os.path.join(out_dir, fname)
        arr = (pix * 255.0).round().clamp(0, 255).byte()
        arr = rearrange(arr, "t c h w -> t h w c").cpu().numpy()
        try:
            imageio.mimwrite(path, arr, fps=24, codec="libx264", quality=8)
            print(f"  saved {path}")
        except Exception as e:
            print(f"  fail {path}: {e}")
        return path

    bl_mp4 = _save(bl_pix, f"baseline_megastack_f{num_frames}_seed{seed}.mp4")
    mega_mp4 = _save(mega_pix, f"megastack_f{num_frames}_seed{seed}.mp4")
    sxs_mp4 = os.path.join(out_dir, f"sidebyside_megastack_f{num_frames}_seed{seed}.mp4")
    try:
        sxs = torch.cat([bl_pix, mega_pix], dim=-1)
        sxs_arr = (sxs * 255.0).round().clamp(0, 255).byte()
        sxs_arr = rearrange(sxs_arr, "t c h w -> t h w c").cpu().numpy()
        imageio.mimwrite(sxs_mp4, sxs_arr, fps=24, codec="libx264", quality=8)
    except Exception:
        pass
    try:
        model_volume.commit()
    except Exception:
        pass

    results = {
        "model": "Etched/oasis-500m",
        "kernel": (f"megastack_dpmpp{solver_full_steps}_diff_taylor_o{taylor_order}"),
        "num_frames": num_frames,
        "ddim_steps": ddim_steps,
        "solver_steps": {"full": solver_full_steps, "mid": solver_mid_steps,
                         "reduced": solver_reduced_steps},
        "taylor": {"order": taylor_order, "threshold": taylor_threshold,
                   "n_blocks": n_blocks,
                   "prediction_rate": round(metrics["prediction_rate"], 4),
                   "predictions_used": metrics["predictions_used"],
                   "full_forwards": metrics["full_forwards"],
                   "validation_failures": metrics["validation_failures"]},
        "baseline_latency_ms": round(bl_ms, 2),
        "dpmpp_diff_latency_ms": round(dd_ms, 2),
        "megastack_latency_ms": round(mega_ms, 2),
        "speedup_dpmpp_diff": round(bl_ms / dd_ms, 3),
        "speedup_megastack": round(bl_ms / mega_ms, 3),
        "extra_speedup_from_taylor": round(dd_ms / mega_ms, 3),
        "cross_psnr_dpmpp_diff": cross_dd,
        "cross_psnr_megastack": cross_mega,
        "self_coherence_baseline": coh_bl,
        "self_coherence_dpmpp_diff": coh_dd,
        "self_coherence_megastack": coh_mega,
        "delta_vs_prev_megastack": round(
            coh_mega["vs_prev_psnr_mean"] - coh_bl["vs_prev_psnr_mean"], 3,
        ),
        "videos": {"baseline_mp4": bl_mp4, "megastack_mp4": mega_mp4,
                   "sidebyside_mp4": sxs_mp4},
        "seed": seed, "gpu": "H100",
    }

    print(f"\n=== SUMMARY ===")
    print(f"Baseline DDIM-10:        {bl_ms:.1f} ms ({num_frames / (bl_ms / 1000):.2f} fps)")
    print(f"DPM++ + Difficulty:      {dd_ms:.1f} ms ({num_frames / (dd_ms / 1000):.2f} fps)  speedup={bl_ms / dd_ms:.3f}x")
    print(f"MEGA STACK:              {mega_ms:.1f} ms ({num_frames / (mega_ms / 1000):.2f} fps)  speedup={bl_ms / mega_ms:.3f}x")
    print(f"  extra from TaylorSeer: {dd_ms / mega_ms:.3f}x  (predict_rate={metrics['prediction_rate']:.1%})")
    print(f"  delta vs_prev:         {results['delta_vs_prev_megastack']:+.2f} dB")
    print("\n" + json.dumps(results, indent=2, default=str))
    return results


@app.local_entrypoint(name="oasis_megastack_modal")
def main(
    num_frames: int = 32,
    seed: int = 42,
    solver_full_steps: int = 5,
    solver_mid_steps: int = 3,
    solver_reduced_steps: int = 2,
    taylor_order: int = 2,
    taylor_threshold: float = 0.15,
):
    result = run_oasis_megastack.remote(
        num_frames=num_frames, seed=seed,
        solver_full_steps=solver_full_steps,
        solver_mid_steps=solver_mid_steps,
        solver_reduced_steps=solver_reduced_steps,
        taylor_order=taylor_order, taylor_threshold=taylor_threshold,
    )
    print("\n=== MEGA STACK result ===")
    print(f"speedup_megastack:    {result.get('speedup_megastack')}x")
    print(f"speedup vs DPM++Diff: {result.get('extra_speedup_from_taylor')}x")
    print(f"predict_rate:         {result['taylor']['prediction_rate']}")
    print(f"delta vs_prev:        {result.get('delta_vs_prev_megastack'):+.2f} dB")

    sys.path.insert(0, "/root/benchmarks")
    from result_store import save_result
    save_result(
        result, "optimised_kernels",
        f"oasis_megastack_f{num_frames}",
    )
