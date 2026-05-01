"""
Modal H100 benchmark: Open-Oasis 500M + TaylorSeer feature prediction.

TaylorSeer (arXiv 2503.06923) attaches forward hooks to each DiT block and
predicts that block's output across DDIM steps via a truncated Taylor series:

    f_l(t + h) ≈ f_l(t) + h·f'_l(t) + (h²/2)·f''_l(t)

History accumulates during the first `order+1` steps of each frame.  After
that, predicted block outputs replace the real forward (block call skipped
entirely).  Validation on a 3-layer subset checks relative error against
`prediction_threshold`; if the average exceeds the threshold the step falls
back to a full forward.

Per Oasis canonical sampler (10 DDIM steps per frame):
  - order=2 (default for >=5 steps) needs 3 history entries
  - steps 0,1,2 always full forward
  - steps 3..9 are prediction candidates (max 70% predict cap)
  - validation may push some steps back to full forward

The autoregressive frame loop calls `seer.reset_cache()` per new frame so
history doesn't carry across frames (each frame's noise schedule starts at
the same noise level).

Run:
    PYTHONPATH=benchmarks modal run benchmarks/optimised/oasis_taylorseer_modal.py \
        --num-frames 32 --num-iters 2 --num-warmup 1 \
        --order 2 --threshold 0.15
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
def run_oasis_taylorseer(
    num_frames: int = 32,
    n_prompt: int = 1,
    ddim_steps: int = 10,
    max_noise_level: int = 1000,
    stabilization_level: int = 15,
    noise_abs_max: float = 20.0,
    seed: int = 42,
    num_warmup: int = 1,
    num_iters: int = 2,
    order: int = 2,
    threshold: float = 0.15,
    prompt_file: str = "sample_image_0.png",
    actions_file: str = "sample_actions_0.one_hot_actions.pt",
) -> dict:
    """Canonical Oasis DDIM with TaylorSeer block-feature prediction."""
    import json
    import os
    import statistics

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

    # ─────────────────────────────────────────────────────────────────────
    # Real prompt + actions
    # ─────────────────────────────────────────────────────────────────────
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
    noise_range = torch.linspace(-1, max_noise_level - 1, ddim_steps + 1)
    model_max_frames = getattr(model, "max_frames", total_frames + 1)

    # ─────────────────────────────────────────────────────────────────────
    # Canonical loop, parameterized by an optional seer
    # ─────────────────────────────────────────────────────────────────────
    def _run_once(seer=None, seed_val: int = 42) -> float:
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        x = prompt_latent.clone()
        B = x.shape[0]

        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()

        for i in range(n_prompt, total_frames):
            chunk = torch.randn((B, 1, *x.shape[-3:]), device="cuda")
            chunk = torch.clamp(chunk, -noise_abs_max, +noise_abs_max)
            x = torch.cat([x, chunk], dim=1)
            start_frame = max(0, i + 1 - model_max_frames)

            # Reset seer history per frame: each frame restarts at high noise,
            # so prior frame's block features are not a useful Taylor history.
            if seer is not None:
                seer.reset_cache()

            local_step = 0
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

                if seer is not None:
                    seer.pre_step(local_step, ddim_steps, x_curr)

                with torch.no_grad():
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        v = model(x_curr, t, actions[:, start_frame: i + 1])

                if seer is not None:
                    seer.post_step(local_step, ddim_steps, x_curr)

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
                local_step += 1

        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e)

    # ─────────────────────────────────────────────────────────────────────
    # Same-session BASELINE eager
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n=== BASELINE eager ({num_warmup}+{num_iters} runs) ===")
    for i in range(num_warmup):
        ms = _run_once(seer=None, seed_val=seed)
        print(f"  warmup {i+1}: {ms:.1f} ms ({num_frames/(ms/1000):.2f} fps)")
    baseline_lats: list = []
    for i in range(num_iters):
        ms = _run_once(seer=None, seed_val=seed)
        baseline_lats.append(ms)
        print(f"  iter {i+1}: {ms:.1f} ms ({num_frames/(ms/1000):.2f} fps)")
    baseline_mean = statistics.mean(baseline_lats)

    # ─────────────────────────────────────────────────────────────────────
    # Wrap model with TaylorSeer
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n=== TaylorSeer order={order}, threshold={threshold} ===")
    seer = TaylorSeer(
        num_layers=n_blocks,
        num_steps=ddim_steps,
        order=order,
        prediction_threshold=threshold,
    )
    seer.wrap_model(model)

    print(f"Warmup ({num_warmup} iters) ...")
    for i in range(num_warmup):
        ms = _run_once(seer=seer, seed_val=seed)
        m = seer.get_metrics()
        print(f"  warmup {i+1}: {ms:.1f} ms  predict_rate={m['prediction_rate']:.3f}")

    # Reset metrics for clean timed measurement
    from worldserve.optimizations.model_level.feature_caching.taylor_seer import (
        _TaylorMetrics,
    )
    seer.metrics = _TaylorMetrics()

    torch.cuda.reset_peak_memory_stats()
    ts_lats: list = []
    print(f"Timed runs ({num_iters} iters) ...")
    for i in range(num_iters):
        try:
            ms = _run_once(seer=seer, seed_val=seed)
            ts_lats.append(ms)
            m = seer.get_metrics()
            print(f"  iter {i+1}: {ms:.1f} ms ({num_frames/(ms/1000):.2f} fps) "
                  f"predict_rate={m['prediction_rate']:.3f} "
                  f"val_failures={m['validation_failures']}")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()

    seer.remove_hooks()

    if not ts_lats:
        return {"error": "All TS iterations OOM"}

    ts_mean = statistics.mean(ts_lats)
    speedup = baseline_mean / ts_mean
    metrics = seer.get_metrics()

    results = {
        "model": "Etched/oasis-500m",
        "kernel": f"taylorseer_order{order}_thr{threshold}",
        "order": order,
        "threshold": threshold,
        "n_blocks": n_blocks,
        "num_frames": num_frames,
        "ddim_steps": ddim_steps,
        "baseline_latency_ms_mean": round(baseline_mean, 2),
        "baseline_fps": round(num_frames / (baseline_mean / 1000.0), 3),
        "latency_ms_mean": round(ts_mean, 2),
        "frames_per_sec": round(num_frames / (ts_mean / 1000.0), 3),
        "speedup": round(speedup, 3),
        "prediction_rate": round(metrics["prediction_rate"], 4),
        "predictions_used": metrics["predictions_used"],
        "full_forwards": metrics["full_forwards"],
        "validation_failures": metrics["validation_failures"],
        "estimated_speedup_from_skip": round(metrics["estimated_speedup"], 3),
        "vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "raw_baseline_latencies_ms": baseline_lats,
        "raw_latencies_ms": ts_lats,
        "gpu": "H100",
    }

    print(f"\n=== SUMMARY ===")
    print(f"Baseline:        {baseline_mean:.1f} ms ({results['baseline_fps']:.2f} fps)")
    print(f"TaylorSeer:      {ts_mean:.1f} ms ({results['frames_per_sec']:.2f} fps)")
    print(f"Speedup:         {speedup:.3f}x")
    print(f"Prediction rate: {metrics['prediction_rate']:.1%}  "
          f"({metrics['predictions_used']} predictions, "
          f"{metrics['full_forwards']} full forwards)")
    print(f"Val failures:    {metrics['validation_failures']}")
    print("\n" + json.dumps(results, indent=2, default=str))
    return results


@app.local_entrypoint(name="oasis_taylorseer_modal")
def main(
    num_frames: int = 32,
    num_iters: int = 2,
    num_warmup: int = 1,
    order: int = 2,
    threshold: float = 0.15,
):
    result = run_oasis_taylorseer.remote(
        num_frames=num_frames,
        num_iters=num_iters,
        num_warmup=num_warmup,
        order=order,
        threshold=threshold,
    )
    print("\n=== TaylorSeer result ===")
    print(f"speedup:         {result.get('speedup')}x")
    print(f"prediction_rate: {result.get('prediction_rate')}")

    sys.path.insert(0, "/root/benchmarks")
    from result_store import save_result
    save_result(
        result, "optimised_kernels",
        f"oasis_taylorseer_o{order}_thr{threshold}_f{num_frames}",
    )
