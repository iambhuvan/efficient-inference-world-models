"""
Modal H100 benchmark: Open-Oasis 500M world model baseline (no custom kernels).

Model  : Etched/oasis-500m  (gated — requires HF_TOKEN with access granted)
Backend: native SDPA (F.scaled_dot_product_attention inside axial attention)
Task   : Autoregressive video generation — 16 new frames from a random latent prompt

Architecture (NOT diffusers):
  - DiT-S/2: SpatioTemporalDiTBlock with SpatialAxialAttention + TemporalAxialAttention
  - VAE: ViT-L-20-shallow-encoder (360x640 → 18x32 latent patches, 16 channels)
  - Latent shape per frame: (B, 1, 16, 18, 32)
  - Attention tokens: 144 spatial (9x16) + T temporal per location
  - DDIM steps: 10 per frame

Run:
    modal run modal/baseline/oasis_modal.py
"""

import sys
sys.path.insert(0, "/root")
sys.path.insert(0, "/root/benchmarks")

import modal
from modal_common import app, image, hf_secret, model_volume, MODEL_CACHE

# Oasis latent dimensions: VAE encodes 360x640 → 18x32 patches, 16 channels
LATENT_C = 16
LATENT_H = 18
LATENT_W = 32


@app.function(
    gpu="H100",
    image=image,
    secrets=[hf_secret],
    volumes={MODEL_CACHE: model_volume},
    timeout=3600,
    memory=32768,
)
def run_oasis_baseline(
    num_frames: int = 16,
    n_prompt: int = 1,
    ddim_steps: int = 10,
    max_noise_level: int = 1000,
    num_warmup: int = 1,
    num_iters: int = 1,
) -> dict:
    """
    Run Open-Oasis 500M DDIM generation and return timing results.

    Uses random latent as prompt (no real image needed for benchmarking).
    Actions are all-zero (no-op) for reproducibility.
    Times num_frames * ddim_steps DiT forward passes end-to-end.
    """
    import json
    import os
    import statistics

    import torch

    sys.path.insert(0, "/root/open-oasis")

    hf_token = os.environ.get("HF_TOKEN", "").strip() or None

    # -----------------------------------------------------------------------
    # Load model weights from HF volume cache
    # -----------------------------------------------------------------------
    from huggingface_hub import hf_hub_download

    print("Downloading Etched/oasis-500m checkpoints ...")
    dit_ckpt = hf_hub_download(
        "Etched/oasis-500m", "oasis500m.safetensors",
        cache_dir=MODEL_CACHE, token=hf_token,
    )
    vae_ckpt = hf_hub_download(
        "Etched/oasis-500m", "vit-l-20.safetensors",
        cache_dir=MODEL_CACHE, token=hf_token,
    )

    from dit import DiT_models
    from safetensors.torch import load_file as safetensors_load

    print("Loading DiT-S/2 ...")
    model = DiT_models["DiT-S/2"]().to("cuda").eval()
    state = safetensors_load(dit_ckpt)
    model.load_state_dict(state, strict=False)
    print(f"DiT loaded: {sum(p.numel() for p in model.parameters())/1e6:.0f}M params")

    # -----------------------------------------------------------------------
    # Noise schedule (sigmoid, same as generate.py)
    # -----------------------------------------------------------------------
    from utils import sigmoid_beta_schedule

    betas = sigmoid_beta_schedule(max_noise_level).float().to("cuda")
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    noise_range = torch.linspace(-1, max_noise_level - 1, ddim_steps + 1)

    # -----------------------------------------------------------------------
    # Timed generation function
    # -----------------------------------------------------------------------
    def _run_once() -> float:
        B = 1
        total_frames = n_prompt + num_frames

        # Random prompt latent — no real video needed for benchmarking
        x = torch.randn(B, n_prompt, LATENT_C, LATENT_H, LATENT_W, device="cuda")
        # external_cond is nn.Linear(25, hidden_size) — needs float (B, T, 25)
        actions = torch.zeros(B, total_frames, 25, device="cuda", dtype=torch.float32)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()

        for frame_i in range(n_prompt, total_frames):
            # Start new frame from noise
            chunk = torch.randn(B, 1, LATENT_C, LATENT_H, LATENT_W, device="cuda")

            for noise_idx in reversed(range(1, ddim_steps + 1)):
                t_val = noise_range[noise_idx].long().clamp(0, max_noise_level - 1)

                # Timestep tensor: zeros for past frames, t_val for current
                t_ctx = torch.zeros(B, frame_i, device="cuda", dtype=torch.long)
                t_cur = t_val.to("cuda").view(1, 1).expand(B, 1)
                t_in = torch.cat([t_ctx, t_cur], dim=1)

                x_in = torch.cat([x, chunk], dim=1)  # (B, frame_i+1, C, H, W)

                with torch.no_grad():
                    v = model(x_in, t_in, actions[:, :frame_i + 1])

                # v-prediction DDIM update on the current chunk
                t_idx = t_val.item()
                alpha_t = alphas_cumprod[t_idx]
                t_prev_idx = max(int(noise_range[noise_idx - 1].item()), 0)
                alpha_t_prev = alphas_cumprod[t_prev_idx]

                v_cur = v[:, -1:]
                x0_pred = alpha_t.sqrt() * chunk - (1 - alpha_t).sqrt() * v_cur
                eps_pred = (1 - alpha_t).sqrt() * chunk + alpha_t.sqrt() * v_cur
                chunk = alpha_t_prev.sqrt() * x0_pred + (1 - alpha_t_prev).sqrt() * eps_pred

            x = torch.cat([x, chunk], dim=1)

        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end)

    # -----------------------------------------------------------------------
    # Warmup + timed runs
    # -----------------------------------------------------------------------
    print(f"Warmup ({num_warmup} iters) ...")
    for i in range(num_warmup):
        _run_once()
        print(f"  warmup {i + 1}/{num_warmup} done")

    torch.cuda.reset_peak_memory_stats()
    latencies = []
    print(f"Timed runs ({num_iters} iters) ...")
    for i in range(num_iters):
        try:
            ms = _run_once()
            latencies.append(ms)
            fps = num_frames / (ms / 1000.0)
            print(f"  iter {i + 1}: {ms:.1f} ms  ({fps:.2f} fps)")
        except torch.cuda.OutOfMemoryError as oom:
            print(f"  iter {i + 1}: OOM — {oom}; skipping")
            torch.cuda.empty_cache()

    if not latencies:
        return {"error": "All iterations OOM", "model": "Etched/oasis-500m"}

    mean_ms = statistics.mean(latencies)
    results = {
        "model": "Etched/oasis-500m",
        "kernel": "baseline_native_sdpa",
        "num_frames": num_frames,
        "n_prompt": n_prompt,
        "ddim_steps": ddim_steps,
        "latent_shape": [LATENT_C, LATENT_H, LATENT_W],
        "total_dit_fwd_passes": num_frames * ddim_steps,
        "latency_ms_mean": round(mean_ms, 2),
        "latency_ms_std": round(statistics.stdev(latencies) if len(latencies) > 1 else 0.0, 2),
        "latency_per_frame_ms": round(mean_ms / num_frames, 2),
        "frames_per_sec": round(num_frames / (mean_ms / 1000.0), 3),
        "vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "gpu": "H100",
        "raw_latencies_ms": latencies,
    }
    print(json.dumps(results, indent=2))
    return results


@app.local_entrypoint(name="oasis_modal")
def main(
    num_frames: int = 16,
    num_iters: int = 1,
    num_warmup: int = 1,
):
    result = run_oasis_baseline.remote(
        num_frames=num_frames,
        num_iters=num_iters,
        num_warmup=num_warmup,
    )
    print("\nOpen-Oasis 500M baseline result:")
    import json
    print(json.dumps(result, indent=2))
    sys.path.insert(0, "/root/benchmarks")
    from result_store import save_result
    save_result(result, "baseline", f"oasis_baseline_f{num_frames}")
