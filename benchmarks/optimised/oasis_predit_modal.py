"""
Modal H100 benchmark: Open-Oasis 500M + PrediT AB-2 extrapolation
(true Adams-Bashforth, not v=v_prev).

In the per-frame DDIM loop:
    on compute step:  v_t = model(x_curr, t, actions);   history.append(v_t)
    on skip step:     v_t ≈ v_{t-1} + alpha * (v_{t-1} - v_{t-2})    (AB-2)

Differs from oasis_stepcache `alternate` (which sets v_t = v_{t-1}, i.e.
PrediT(alpha=0)). True PrediT uses alpha > 0 to extrapolate forward.
arxiv 2602.18093 reports alpha ~= 1.5 best for FLUX-class DiTs.
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
def run_oasis_predit(
    num_frames: int = 32,
    n_prompt: int = 1,
    ddim_steps: int = 10,
    max_noise_level: int = 1000,
    stabilization_level: int = 15,
    noise_abs_max: float = 20.0,
    seed: int = 42,
    num_warmup: int = 1,
    num_iters: int = 2,
    predict_every: int = 2,
    predict_alpha: float = 1.5,
    prompt_file: str = "sample_image_0.png",
    actions_file: str = "sample_actions_0.one_hot_actions.pt",
) -> dict:
    """PrediT AB-2 extrapolation in canonical Oasis DDIM loop."""
    import statistics
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
    noise_range = torch.linspace(-1, max_noise_level - 1, ddim_steps + 1)
    model_max_frames = getattr(model, "max_frames", total_frames + 1)

    compute_count = 0
    predict_count = 0

    def _run_once(seed_val: int = 42) -> float:
        nonlocal compute_count, predict_count
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
            v_hist: list = []
            local_step = 0

            for noise_idx in reversed(range(1, ddim_steps + 1)):
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

                use_predict = (
                    predict_every > 1
                    and local_step >= 2
                    and (local_step % predict_every == 0)
                    and len(v_hist) >= 2
                )

                if use_predict:
                    v = v_hist[-1] + predict_alpha * (v_hist[-1] - v_hist[-2])
                    predict_count += 1
                else:
                    with torch.no_grad():
                        with torch.cuda.amp.autocast(dtype=torch.float16):
                            v = model(x_curr, t, actions[:, start_frame: i + 1])
                    compute_count += 1
                    v_hist.append(v)
                    if len(v_hist) > 2:
                        v_hist.pop(0)

                x_start = alphas_cumprod[t].sqrt() * x_curr - (1 - alphas_cumprod[t]).sqrt() * v
                x_noise = ((1 / alphas_cumprod[t]).sqrt() * x_curr - x_start) / (1 / alphas_cumprod[t] - 1).sqrt()
                alpha_next = alphas_cumprod[t_next].clone()
                alpha_next[:, :-1] = torch.ones_like(alpha_next[:, :-1])
                if noise_idx == 1:
                    alpha_next[:, -1:] = torch.ones_like(alpha_next[:, -1:])
                x_pred = alpha_next.sqrt() * x_start + x_noise * (1 - alpha_next).sqrt()
                x[:, -1:] = x_pred[:, -1:]
                local_step += 1

        e.record(); torch.cuda.synchronize()
        return s.elapsed_time(e)

    print(f"\nWarmup ({num_warmup} iters) ...")
    for i in range(num_warmup):
        ms = _run_once(seed_val=seed)
        print(f"  warmup {i+1}: {ms:.1f} ms ({num_frames/(ms/1000):.2f} fps)")

    compute_count = 0; predict_count = 0
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
        "kernel": f"predit_alpha{predict_alpha}_every{predict_every}",
        "predict_every": predict_every,
        "predict_alpha": predict_alpha,
        "compute_steps_per_iter": compute_count // max(num_iters, 1),
        "predict_steps_per_iter": predict_count // max(num_iters, 1),
        "predict_rate": round(predict_count / max(compute_count + predict_count, 1), 4),
        "num_frames": num_frames,
        "ddim_steps": ddim_steps,
        "latency_ms_mean": round(mean_ms, 2),
        "frames_per_sec": round(num_frames / (mean_ms / 1000.0), 3),
        "vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "raw_latencies_ms": latencies,
        "gpu": "H100",
    }


@app.local_entrypoint(name="oasis_predit_modal")
def main(
    num_frames: int = 32,
    predict_every: int = 2,
    predict_alpha: float = 1.5,
    num_iters: int = 2,
    num_warmup: int = 1,
):
    result = run_oasis_predit.remote(
        num_frames=num_frames, predict_every=predict_every,
        predict_alpha=predict_alpha, num_iters=num_iters, num_warmup=num_warmup,
    )
    print(f"\n=== PrediT AB-2 result ===")
    print(f"  kernel:        {result.get('kernel')}")
    print(f"  predict_rate:  {result.get('predict_rate')}")
    print(f"  latency_mean:  {result.get('latency_ms_mean')}")
    print(f"  frames_per_sec:{result.get('frames_per_sec')}")
    sys.path.insert(0, "/root/benchmarks")
    from result_store import save_result
    save_result(result, "optimised_kernels",
                f"oasis_predit_a{predict_alpha}_e{predict_every}_f{num_frames}")
