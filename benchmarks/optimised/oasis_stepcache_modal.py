"""
Modal H100 benchmark: Open-Oasis 500M baseline + DDIM step caching.

Cleaner replacement for the broken (diffusers-style) `oasis_teacache_modal.py`.
This one operates directly on Oasis's custom DDIM loop (canonical generate.py
sampler).

Method (per-frame step caching):
  Each frame runs `ddim_steps=10` DDIM iterations with v-prediction. For each
  iteration we either:
    (a) compute `v = model(x_curr, t, actions)` fresh, or
    (b) reuse the v from the previous iteration.
  Decision is governed by `skip_pattern`:
    "alternate"        — skip every-other step (5 forwards / 10 steps)  ← 1.5x expected
    "skip_late"        — only compute on steps 1,2,3,5,7  (5 forwards)
    "skip_early"       — only compute on steps 6,7,8,9,10 (5 forwards)
    "interval_<N>"     — compute every Nth step
  Within a single frame the v from step N approximates the v at step N-1 because
  consecutive DDIM steps walk through similar local gradient regions —
  reusing v is the v-prediction analog of TeaCache's temb-distance reuse, but
  applied to the model output directly so we skip the entire forward pass
  rather than just the late blocks.

Caveat: this is `PrediT(alpha=0)` essentially — predict v_t = v_{t-1}.
Quality depends on whether v changes smoothly between DDIM steps, which is
generally true for v-prediction diffusion.

Run:
    PYTHONPATH=benchmarks modal run benchmarks/optimised/oasis_stepcache_modal.py \
        --num-frames 75 --skip-pattern alternate
"""

import sys
sys.path.insert(0, "/root")
sys.path.insert(0, "/root/benchmarks")

import modal
from modal_common import app, image, hf_secret, model_volume, MODEL_CACHE

SCALING_FACTOR = 0.07843137255


def _build_skip_set(pattern: str, ddim_steps: int) -> set:
    """Return the SET of noise_idx values where we COMPUTE (rest are skipped)."""
    if pattern == "alternate":
        # noise_idx iterates [10, 9, 8, ..., 1] in reversed order.
        # Compute on odd noise_idx (10, 8, 6, 4, 2 if iterating reversed range(1,11)).
        return set(range(ddim_steps, 0, -2))
    if pattern == "skip_late":
        # Compute on highest noise (early in reversed loop) — those matter most
        return set(range(ddim_steps, ddim_steps // 2, -1))  # e.g. 10..6
    if pattern == "skip_early":
        # Compute on lowest noise (late in reversed loop) — fine details
        return set(range(ddim_steps // 2, 0, -1))  # e.g. 5..1
    if pattern.startswith("interval_"):
        n = int(pattern.split("_", 1)[1])
        return set(range(ddim_steps, 0, -n))
    if pattern == "none":
        return set(range(1, ddim_steps + 1))  # compute every step (== baseline)
    raise ValueError(f"unknown skip pattern {pattern!r}")


@app.function(
    gpu="H100",
    image=image,
    secrets=[hf_secret],
    volumes={MODEL_CACHE: model_volume},
    timeout=3600,
    memory=32768,
)
def run_oasis_stepcache(
    num_frames: int = 75,
    n_prompt: int = 1,
    ddim_steps: int = 10,
    max_noise_level: int = 1000,
    stabilization_level: int = 15,
    noise_abs_max: float = 20.0,
    seed: int = 42,
    num_warmup: int = 1,
    num_iters: int = 2,
    skip_pattern: str = "alternate",
    prompt_file: str = "sample_image_0.png",
    actions_file: str = "sample_actions_0.one_hot_actions.pt",
) -> dict:
    """Baseline canonical-sampler Oasis + step caching."""
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

    hf_token = os.environ.get("HF_TOKEN", "").strip() or None

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

    # Real prompt + actions (latency benchmark — we don't decode here, just time)
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

    compute_steps = _build_skip_set(skip_pattern, ddim_steps)
    print(f"\nskip_pattern={skip_pattern!r}; computing on noise_idx values "
          f"{sorted(compute_steps)} (skipping the rest, total {ddim_steps} steps)")

    # Track how often we actually skipped vs computed
    forward_count = 0
    skip_count = 0

    def _run_once(seed_val: int = 42) -> float:
        nonlocal forward_count, skip_count
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

            cached_v = None  # reset per-frame

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

                if noise_idx in compute_steps or cached_v is None:
                    with torch.no_grad():
                        with torch.cuda.amp.autocast(dtype=torch.float16):
                            v = model(x_curr, t, actions[:, start_frame: i + 1])
                    cached_v = v
                    forward_count += 1
                else:
                    # REUSE cached v — same shape required (it is, since x_curr shape is constant per frame)
                    v = cached_v
                    skip_count += 1

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

        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e)

    print(f"\nWarmup ({num_warmup} iters) ...")
    for i in range(num_warmup):
        ms = _run_once(seed_val=seed)
        fps = num_frames / (ms / 1000.0)
        print(f"  warmup {i + 1}/{num_warmup}: {ms:.1f} ms ({fps:.2f} fps)")

    # Reset counters before timed runs
    forward_count = 0
    skip_count = 0
    torch.cuda.reset_peak_memory_stats()
    latencies: list = []
    print(f"\nTimed runs ({num_iters} iters) ...")
    for i in range(num_iters):
        try:
            ms = _run_once(seed_val=seed)
            latencies.append(ms)
            fps = num_frames / (ms / 1000.0)
            print(f"  iter {i + 1}: {ms:.1f} ms  ({fps:.2f} fps)")
        except torch.cuda.OutOfMemoryError as oom:
            print(f"  iter {i + 1}: OOM — {oom}; skipping")
            torch.cuda.empty_cache()

    if not latencies:
        return {"error": "All iterations OOM", "model": "Etched/oasis-500m"}

    mean_ms = statistics.mean(latencies)
    skip_rate = skip_count / max(forward_count + skip_count, 1)
    results = {
        "model": "Etched/oasis-500m",
        "kernel": f"stepcache_{skip_pattern}",
        "skip_pattern": skip_pattern,
        "computed_steps_per_frame": sorted(compute_steps),
        "num_frames": num_frames,
        "n_prompt": n_prompt,
        "ddim_steps": ddim_steps,
        "stabilization_level": stabilization_level,
        "total_dit_fwd_passes_no_cache": num_frames * ddim_steps,
        "total_dit_fwd_passes_actual": forward_count // num_iters,
        "skip_count_per_iter": skip_count // num_iters,
        "skip_rate": round(skip_rate, 4),
        "latency_ms_mean": round(mean_ms, 2),
        "latency_ms_std": round(
            statistics.stdev(latencies) if len(latencies) > 1 else 0.0, 2
        ),
        "latency_per_frame_ms": round(mean_ms / num_frames, 2),
        "frames_per_sec": round(num_frames / (mean_ms / 1000.0), 3),
        "vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "raw_latencies_ms": latencies,
        "gpu": "H100",
    }
    print("\n" + json.dumps(results, indent=2, default=str))
    return results


@app.local_entrypoint(name="oasis_stepcache_modal")
def main(
    num_frames: int = 75,
    num_iters: int = 2,
    num_warmup: int = 1,
    skip_pattern: str = "alternate",
):
    result = run_oasis_stepcache.remote(
        num_frames=num_frames,
        num_iters=num_iters,
        num_warmup=num_warmup,
        skip_pattern=skip_pattern,
    )
    print("\n=== Step cache result ===")
    print(f"skip_pattern: {result.get('skip_pattern')}")
    print(f"skip_rate:    {result.get('skip_rate')}")
    print(f"latency_ms_mean: {result.get('latency_ms_mean')}")
    print(f"frames_per_sec:  {result.get('frames_per_sec')}")

    sys.path.insert(0, "/root/benchmarks")
    from result_store import save_result
    save_result(result, "optimised_kernels", f"oasis_stepcache_{skip_pattern}_f{num_frames}")
