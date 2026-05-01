"""
Open-Oasis (Etched/oasis-500m) model backend for WorldServe.

Wraps the upstream open-oasis code and provides a uniform load_model / generate
interface.

Critical Oasis sampling semantics (verified against open-oasis/generate.py and
our working H100 baseline):

  * Schedule        : sigmoid_beta_schedule (NOT linear).
                      utils.sigmoid_beta_schedule(1000) → β_t.
  * Diffusion T     : 1000 steps.
  * DDIM steps      : 10 (canonical Oasis sampling, not 20 or 50).
  * Parameterisation: v-prediction (DiT outputs v, NOT eps).
                          x0_pred = √ᾱ_t · x_t − √(1−ᾱ_t) · v
                          eps_pred = √(1−ᾱ_t) · x_t + √ᾱ_t · v
                          x_{t-1} = √ᾱ_{t-1} · x0_pred + √(1−ᾱ_{t-1}) · eps_pred
  * Generation      : Autoregressive per-frame. For frame i:
                          chunk_i ∼ 𝒩(0,I)
                          for noise_idx in reversed(1..ddim_steps):
                              x_in = concat([prior_frames, chunk_i])
                              t_in = concat([zeros(prior), t_val])  ← past = clean
                              v    = dit(x_in, t_in, actions[:i+1])
                              chunk_i ← DDIM_v_step(chunk_i, v[:,-1:], t_val, t_prev)
                          frames ← concat([frames, chunk_i])
                      The first n_prompt frames come pre-noised (or random
                      latent for benchmarking) and stay fixed at t=0.
  * Action vector   : (B, total_frames, 25) Minecraft keyboard state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


# Oasis DiT-S/2 latent geometry (decodes via ViT-L VAE to 360×640 RGB).
LATENT_C = 16
LATENT_H = 18
LATENT_W = 32
ACTION_DIM = 25
DDPM_T = 1000
DEFAULT_DDIM_STEPS = 10
N_PROMPT = 1
VAE_SCALE = 0.07843137255


def load_model(
    checkpoint_dir: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> dict[str, Any]:
    """Load DiT-S/2 + ViT-L VAE from *checkpoint_dir*."""
    from safetensors.torch import load_file

    ckpt = Path(checkpoint_dir)
    dit_path = ckpt / "oasis500m.safetensors"
    vae_path = ckpt / "vit-l-20.safetensors"
    if not dit_path.exists():
        raise FileNotFoundError(f"DiT checkpoint not found: {dit_path}")
    if not vae_path.exists():
        raise FileNotFoundError(f"VAE checkpoint not found: {vae_path}")

    try:
        from dit import DiT_models  # type: ignore
        from vae import VAE_models  # type: ignore
    except ImportError as e:
        raise ImportError(
            "Could not import open-oasis. Add the open-oasis repo root to sys.path."
        ) from e

    print(f"[oasis] Loading DiT from {dit_path}")
    dit = DiT_models["DiT-S/2"]()
    dit.load_state_dict(load_file(str(dit_path)), strict=False)
    dit = dit.to(device=device, dtype=dtype).eval()

    print(f"[oasis] Loading VAE from {vae_path}")
    vae = VAE_models["vit-l-20-shallow-encoder"]()
    vae.load_state_dict(load_file(str(vae_path)), strict=False)
    vae = vae.to(device=device, dtype=dtype).eval()

    print(f"[oasis] DiT params: {sum(p.numel() for p in dit.parameters()) / 1e6:.0f}M")
    return {"dit": dit, "vae": vae, "device": device, "dtype": dtype}


def _build_schedule(
    device: str,
    ddim_steps: int,
    alpha_bar_override: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (alphas_cumprod[T], noise_range[ddim_steps + 1]).

    Default schedule = sigmoid_beta_schedule from open-oasis/utils.py.
    Override path lets DIAMOND or other schedules be plugged in.
    """
    if alpha_bar_override is not None:
        alphas_cumprod = alpha_bar_override.to(device=device, dtype=torch.float32)
    else:
        from utils import sigmoid_beta_schedule  # type: ignore

        betas = sigmoid_beta_schedule(DDPM_T).float().to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

    # noise_range[0] = -1 (sentinel for "next step is t=0"), noise_range[ddim] = T-1
    noise_range = torch.linspace(-1, DDPM_T - 1, ddim_steps + 1, device=device)
    return alphas_cumprod, noise_range


@torch.inference_mode()
def generate(
    model: dict[str, Any],
    num_frames: int = 16,
    num_steps: int = DEFAULT_DDIM_STEPS,
    n_prompt: int = N_PROMPT,
    seed: int = 42,
    config: dict | None = None,
    optimizations: Any | None = None,
) -> dict[str, Any]:
    """
    Autoregressive Oasis generation.

    Parameters
    ----------
    model         : dict from load_model.
    num_frames    : NEW frames to generate (excludes n_prompt prior frames).
    num_steps     : DDIM denoising steps per frame. Oasis canonical = 10.
    n_prompt      : Number of conditioning prior frames (canonical = 1).
    seed          : RNG seed.
    config        : Full WorldServe config (optional, forwarded to optimizations).
    optimizations : OptimizationStack (optional). Hooks called per-step,
                    not per-frame, so step_caching / speculative work the same
                    as in our working benchmarks.
    """
    dit = model["dit"]
    vae = model["vae"]
    device = model["device"]
    dtype = model["dtype"]

    if optimizations is not None:
        kv_mgr = getattr(optimizations, "modules", {}).get("kv_cache")
        if kv_mgr is not None and hasattr(kv_mgr, "disarm_context_reuse"):
            kv_mgr.disarm_context_reuse()

    gen = torch.Generator(device=device).manual_seed(seed)
    B = 1
    total_frames = n_prompt + num_frames

    alphas_cumprod, noise_range = _build_schedule(
        device=device,
        ddim_steps=num_steps,
        alpha_bar_override=model.get("alpha_bar_override"),
    )

    # Prompt frames: random latent for benchmarking. In production this is
    # encoded from a real RGB frame via vae.encode().
    x = torch.randn(
        B, n_prompt, LATENT_C, LATENT_H, LATENT_W,
        device=device, dtype=dtype, generator=gen,
    )

    # External conditioning: per-frame Minecraft keyboard vector.
    # All zeros = "idle". DiT.external_cond is Linear(25, hidden) so dtype
    # must match model weights.
    actions = torch.zeros(B, total_frames, ACTION_DIM, device=device, dtype=dtype)

    # Step counter exposed to optimizations.pre/post_step (per-frame × per-noise step).
    global_step = 0
    total_dit_passes = num_frames * num_steps

    for frame_i in range(n_prompt, total_frames):
        # New frame starts from pure noise.
        chunk = torch.randn(
            B, 1, LATENT_C, LATENT_H, LATENT_W,
            device=device, dtype=dtype, generator=gen,
        )

        for noise_idx in reversed(range(1, num_steps + 1)):
            t_val = noise_range[noise_idx].long().clamp(0, DDPM_T - 1)

            # Past frames are "clean" (t=0); new frame is at current noise level.
            t_ctx = torch.zeros(B, frame_i, device=device, dtype=torch.long)
            t_cur = t_val.to(device).view(1, 1).expand(B, 1)
            t_in = torch.cat([t_ctx, t_cur], dim=1)
            x_in = torch.cat([x, chunk], dim=1)  # (B, frame_i + 1, C, H, W)

            if optimizations is not None and hasattr(optimizations, "pre_step"):
                if optimizations.pre_step(global_step, total_dit_passes, chunk):
                    global_step += 1
                    continue

            if hasattr(torch, "compiler") and hasattr(
                torch.compiler, "cudagraph_mark_step_begin"
            ):
                torch.compiler.cudagraph_mark_step_begin()

            spec = (
                getattr(optimizations, "modules", {}).get("speculative")
                if optimizations is not None else None
            )
            if spec is not None and hasattr(spec, "speculative_forward"):
                kv_mgr = getattr(optimizations, "modules", {}).get("kv_cache")
                v = spec.speculative_forward(
                    model=dit,
                    x=x_in,
                    kv_cache_manager=kv_mgr,
                    timestep=t_in,
                    step_idx=global_step,
                    total_steps=total_dit_passes,
                    external_cond=actions[:, : frame_i + 1],
                )
            else:
                v = dit(x_in, t_in, actions[:, : frame_i + 1])

            # v-prediction DDIM step on the current frame chunk only.
            # v[:, -1:] selects the predicted v for the new frame.
            t_idx = int(t_val.item())
            t_prev_idx = max(int(noise_range[noise_idx - 1].item()), 0)
            ab_t = alphas_cumprod[t_idx].to(dtype)
            ab_prev = alphas_cumprod[t_prev_idx].to(dtype)

            v_cur = v[:, -1:]
            x0_pred = ab_t.sqrt() * chunk - (1.0 - ab_t).sqrt() * v_cur
            eps_pred = (1.0 - ab_t).sqrt() * chunk + ab_t.sqrt() * v_cur
            chunk = (ab_prev.sqrt() * x0_pred + (1.0 - ab_prev).sqrt() * eps_pred)

            if optimizations is not None and hasattr(optimizations, "post_step"):
                optimizations.post_step(global_step, total_dit_passes, chunk)

            global_step += 1

        x = torch.cat([x, chunk], dim=1)

    # Decode latents → pixel frames via ViT-L VAE.
    # Latent layout (B, T, C=16, H=18, W=32) → VAE expects (B*T, H*W, C).
    B_, T_, C_, H_, W_ = x.shape
    z = x / VAE_SCALE
    z = z.reshape(B_ * T_, C_, H_ * W_).permute(0, 2, 1)  # (B*T, 576, 16)
    decoded = vae.decode(z)
    if decoded.shape[1] != 3:
        decoded = decoded.permute(0, 3, 1, 2)  # BHWC → BCHW
    frames = ((decoded + 1.0) / 2.0).reshape(B_, T_, *decoded.shape[1:])

    return {
        "frames": frames,
        "num_frames": num_frames,
        "num_steps": num_steps,
        "n_prompt": n_prompt,
        "total_dit_fwd_passes": total_dit_passes,
    }
