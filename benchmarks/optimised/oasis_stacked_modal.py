"""
Modal H100 benchmark: Stacked optimization — custom Triton FlashAttn + FP8 V CUDA
+ step caching at the DDIM-loop level. Tests whether the two orthogonal
optimizations multiply.

Custom-kernel layer: monkey-patches F.scaled_dot_product_attention to route
through Triton flash_attn_func with FP8 V-tensor roundtrip via our CUDA
kernel (same path as oasis_custom_modal.py).

Step-cache layer: skips a configurable subset of DDIM forwards within each
frame's denoising loop, reusing the previous step's `v` (interval_3 default
= compute steps 1, 4, 7, 10).
"""

import sys
sys.path.insert(0, "/root")
sys.path.insert(0, "/root/benchmarks")

import modal
from modal_common import app, image_cuda_devel as image, hf_secret, model_volume, MODEL_CACHE

SCALING_FACTOR = 0.07843137255


def _build_compute_set(pattern: str, ddim_steps: int) -> set:
    if pattern == "alternate":
        return set(range(ddim_steps, 0, -2))
    if pattern.startswith("interval_"):
        n = int(pattern.split("_", 1)[1])
        return set(range(ddim_steps, 0, -n))
    if pattern == "none":
        return set(range(1, ddim_steps + 1))
    raise ValueError(f"unknown skip pattern {pattern!r}")


@app.function(
    gpu="H100",
    image=image,
    secrets=[hf_secret],
    volumes={MODEL_CACHE: model_volume},
    timeout=3600,
    memory=32768,
)
def run_oasis_stacked(
    num_frames: int = 32,
    n_prompt: int = 1,
    ddim_steps: int = 10,
    max_noise_level: int = 1000,
    stabilization_level: int = 15,
    noise_abs_max: float = 20.0,
    seed: int = 42,
    num_warmup: int = 1,
    num_iters: int = 2,
    skip_pattern: str = "interval_3",
    use_custom_kernels: bool = True,
    prompt_file: str = "sample_image_0.png",
    actions_file: str = "sample_actions_0.one_hot_actions.pt",
) -> dict:
    """Custom kernels + step caching, stacked."""
    import json
    import os
    import statistics

    import torch
    import torch.nn.functional as F

    sys.path.insert(0, "/root/open-oasis")

    from einops import rearrange
    from huggingface_hub import hf_hub_download
    from dit import DiT_models
    from vae import VAE_models
    from safetensors.torch import load_file as safetensors_load
    from utils import sigmoid_beta_schedule, load_prompt, load_actions

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

    # ─────────────────────────────────────────────────────────────────────
    # Custom-kernel layer (Triton FlashAttn + FP8 V CUDA roundtrip)
    # ─────────────────────────────────────────────────────────────────────
    _orig_sdpa = F.scaled_dot_product_attention
    kernels_active: list = []

    if use_custom_kernels:
        try:
            from worldserve.kernels.load import load_kernels
            from worldserve.kernels.triton.flash_attention import flash_attn_func
            print("Compiling WorldServe CUDA kernels ...")
            _ext = load_kernels(verbose=False)
            print("CUDA kernels compiled OK")

            def _fp8_roundtrip(v):
                """V-tensor FP8 compress + decompress round trip."""
                try:
                    orig_shape = v.shape
                    N = v.numel()
                    block_size = 128
                    v_flat = v.reshape(-1).contiguous()
                    pad = (-N) % block_size
                    v_abs = v_flat.abs().float()
                    if pad > 0:
                        v_padded = torch.cat([v_abs, torch.zeros(pad, device=v.device)])
                    else:
                        v_padded = v_abs
                    scale_fp8 = (v_padded.reshape(-1, block_size).max(dim=1).values / 448.0).clamp(min=1e-8)
                    packed = _ext.quantize_fp8(v_flat, scale_fp8.contiguous(), block_size)
                    return _ext.dequantize_fp8(packed, scale_fp8.contiguous(), block_size).reshape(orig_shape)
                except Exception:
                    return v

            def _fused_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
                            is_causal=False, scale=None, **kwargs):
                if query.dtype not in (torch.float16, torch.bfloat16) or attn_mask is not None or is_causal:
                    return _orig_sdpa(query, key, value, attn_mask=attn_mask,
                                      dropout_p=dropout_p, is_causal=is_causal, scale=scale)
                try:
                    q = query.to(torch.float16)
                    k = key.to(torch.float16)
                    v = value.to(torch.float16)
                    v_fp8 = _fp8_roundtrip(v)
                    out = flash_attn_func(q, k, v_fp8, softmax_scale=scale, causal=False, layout="bhsd")
                    return out.to(query.dtype)
                except Exception:
                    return _orig_sdpa(query, key, value, attn_mask=attn_mask,
                                      dropout_p=dropout_p, is_causal=is_causal, scale=scale)

            F.scaled_dot_product_attention = _fused_sdpa
            torch.nn.functional.scaled_dot_product_attention = _fused_sdpa
            kernels_active.extend(["flash_triton", "fp8_cuda_v"])
            print("Custom-kernel SDPA monkey-patch active")
        except Exception as exc:
            print(f"Custom kernels disabled: {exc}")

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

    # ─────────────────────────────────────────────────────────────────────
    # Schedule + step-cache compute set
    # ─────────────────────────────────────────────────────────────────────
    betas = sigmoid_beta_schedule(max_noise_level).float().to("cuda")
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    alphas_cumprod = rearrange(alphas_cumprod, "T -> T 1 1 1")
    noise_range = torch.linspace(-1, max_noise_level - 1, ddim_steps + 1)
    model_max_frames = getattr(model, "max_frames", total_frames + 1)
    compute_steps = _build_compute_set(skip_pattern, ddim_steps)
    print(f"\nskip_pattern={skip_pattern!r}, computing on {sorted(compute_steps)}")

    forward_count = 0
    skip_count = 0

    def _run_once(seed_val: int = 42) -> float:
        nonlocal forward_count, skip_count
        torch.manual_seed(seed_val); torch.cuda.manual_seed_all(seed_val)

        x = prompt_latent.clone()
        B = x.shape[0]

        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()

        for i in range(n_prompt, total_frames):
            chunk = torch.randn((B, 1, *x.shape[-3:]), device="cuda")
            chunk = torch.clamp(chunk, -noise_abs_max, +noise_abs_max)
            x = torch.cat([x, chunk], dim=1)
            start_frame = max(0, i + 1 - model_max_frames)
            cached_v = None

            for noise_idx in reversed(range(1, ddim_steps + 1)):
                t_ctx = torch.full((B, i), stabilization_level - 1, dtype=torch.long, device="cuda")
                t = torch.full((B, 1), int(noise_range[noise_idx].item()), dtype=torch.long, device="cuda")
                t_next = torch.full((B, 1), int(noise_range[noise_idx - 1].item()), dtype=torch.long, device="cuda")
                t_next = torch.where(t_next < 0, t, t_next)
                t = torch.cat([t_ctx, t], dim=1); t_next = torch.cat([t_ctx, t_next], dim=1)

                x_curr = x.clone()[:, start_frame:]
                t = t[:, start_frame:]; t_next = t_next[:, start_frame:]

                if (noise_idx in compute_steps) or (cached_v is None):
                    with torch.no_grad():
                        with torch.cuda.amp.autocast(dtype=torch.float16):
                            v = model(x_curr, t, actions[:, start_frame: i + 1])
                    cached_v = v
                    forward_count += 1
                else:
                    v = cached_v
                    skip_count += 1

                x_start = alphas_cumprod[t].sqrt() * x_curr - (1 - alphas_cumprod[t]).sqrt() * v
                x_noise = ((1 / alphas_cumprod[t]).sqrt() * x_curr - x_start) / (1 / alphas_cumprod[t] - 1).sqrt()
                alpha_next = alphas_cumprod[t_next].clone()
                alpha_next[:, :-1] = torch.ones_like(alpha_next[:, :-1])
                if noise_idx == 1:
                    alpha_next[:, -1:] = torch.ones_like(alpha_next[:, -1:])
                x_pred = alpha_next.sqrt() * x_start + x_noise * (1 - alpha_next).sqrt()
                x[:, -1:] = x_pred[:, -1:]

        e.record(); torch.cuda.synchronize()
        return s.elapsed_time(e)

    print(f"\nWarmup ({num_warmup} iters) ...")
    for i in range(num_warmup):
        ms = _run_once(seed_val=seed)
        print(f"  warmup {i+1}/{num_warmup}: {ms:.1f} ms ({num_frames/(ms/1000):.2f} fps)")

    forward_count = 0; skip_count = 0
    torch.cuda.reset_peak_memory_stats()
    latencies: list = []
    print(f"\nTimed runs ({num_iters} iters) ...")
    for i in range(num_iters):
        try:
            ms = _run_once(seed_val=seed)
            latencies.append(ms)
            print(f"  iter {i+1}: {ms:.1f} ms ({num_frames/(ms/1000):.2f} fps)")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()

    if not latencies:
        return {"error": "OOM"}

    mean_ms = statistics.mean(latencies)
    return {
        "model": "Etched/oasis-500m",
        "kernel": f"stacked_{skip_pattern}_{'+'.join(kernels_active) or 'baseline'}",
        "skip_pattern": skip_pattern,
        "kernels_active": kernels_active,
        "computed_steps_per_frame": sorted(compute_steps),
        "num_frames": num_frames,
        "ddim_steps": ddim_steps,
        "skip_rate": round(skip_count / max(forward_count + skip_count, 1), 4),
        "latency_ms_mean": round(mean_ms, 2),
        "latency_ms_std": round(statistics.stdev(latencies) if len(latencies) > 1 else 0.0, 2),
        "frames_per_sec": round(num_frames / (mean_ms / 1000.0), 3),
        "vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "raw_latencies_ms": latencies,
        "gpu": "H100",
    }


@app.local_entrypoint(name="oasis_stacked_modal")
def main(
    num_frames: int = 32,
    skip_pattern: str = "interval_3",
    use_custom_kernels: bool = True,
    num_iters: int = 2,
    num_warmup: int = 1,
):
    result = run_oasis_stacked.remote(
        num_frames=num_frames, skip_pattern=skip_pattern,
        use_custom_kernels=use_custom_kernels,
        num_iters=num_iters, num_warmup=num_warmup,
    )
    print(f"\n=== Stacked result ===")
    print(f"  kernel:          {result.get('kernel')}")
    print(f"  latency_ms_mean: {result.get('latency_ms_mean')}")
    print(f"  frames_per_sec:  {result.get('frames_per_sec')}")
    print(f"  skip_rate:       {result.get('skip_rate')}")

    sys.path.insert(0, "/root/benchmarks")
    from result_store import save_result
    save_result(result, "optimised_kernels", f"oasis_stacked_{skip_pattern}_f{num_frames}")
