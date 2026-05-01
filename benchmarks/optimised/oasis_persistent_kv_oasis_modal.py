"""
Modal H100 prototype: persistent past-frame KV reuse (theory test) on Oasis.

Design constraints:
- New file only; no edits to existing Oasis or benchmark code.
- Hook only TemporalAxialAttention modules.
- Cache and reuse *past* K/V inside each frame's denoising loop.
- Keep current-frame K/V live each step.

Important note:
This prototype validates numerical behavior and the reuse mechanism. It does
NOT skip past-frame projection compute inside TemporalAxialAttention.forward,
so wall-clock speedup may be limited without deeper forward rewrites.
"""

from __future__ import annotations

import sys
sys.path.insert(0, "/root")
sys.path.insert(0, "/root/benchmarks")

from dataclasses import dataclass
from typing import Any, Dict

import modal
from modal_common import app, image, hf_secret, model_volume, MODEL_CACHE

SCALING_FACTOR = 0.07843137255


@dataclass
class _RuntimeState:
    past_tokens: int = 0
    step_idx: int = 0


class TemporalPastKVReuseHook:
    """
    Hook manager for TemporalAxialAttention.

    For each temporal attention module, cache the K/V prefix that corresponds to
    past frames, and reuse it on subsequent denoising steps in the same frame.
    """

    def __init__(self, mode: str = "reuse") -> None:
        # mode: "reuse" | "capture_only" | "noop"
        self._mode = mode
        self._state = _RuntimeState()
        self._cache: Dict[int, Dict[str, Any]] = {}
        self._wired: list[tuple[Any, Any]] = []

    def reset_frame(self) -> None:
        self._cache.clear()

    def set_runtime(self, past_tokens: int, step_idx: int) -> None:
        self._state.past_tokens = max(0, int(past_tokens))
        self._state.step_idx = int(step_idx)

    def wire(self, model: Any) -> int:
        import torch
        import torch.nn.functional as F

        temporal_mods = [
            (idx, mod)
            for idx, (_, mod) in enumerate(model.named_modules())
            if type(mod).__name__ == "TemporalAxialAttention"
        ]
        if not temporal_mods:
            return 0

        def _make_forward(orig_fwd: Any, layer_idx: int):
            @torch.compiler.disable
            def _wrapped(*args: Any, **kwargs: Any):
                orig_sdpa = F.scaled_dot_product_attention

                def _capture_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **extra):
                    # Only handle standard (B, H, S, D) layout
                    if key.dim() == 4 and value.dim() == 4:
                        S = key.shape[2]
                        if S > 1 and self._state.past_tokens > 0:
                            past = min(self._state.past_tokens, S - 1)
                            if past > 0 and self._mode != "noop":
                                if self._mode == "reuse" and layer_idx in self._cache and self._state.step_idx > 0:
                                    cached = self._cache[layer_idx]
                                    # Reuse cached prefix; keep live current suffix
                                    key = torch.cat([cached["k"], key[:, :, past:, :]], dim=2)
                                    value = torch.cat([cached["v"], value[:, :, past:, :]], dim=2)
                                elif layer_idx not in self._cache:
                                    # First step for this frame/layer: capture past prefix
                                    self._cache[layer_idx] = {
                                        "k": key[:, :, :past, :].detach(),
                                        "v": value[:, :, :past, :].detach(),
                                    }

                    return orig_sdpa(
                        query,
                        key,
                        value,
                        attn_mask=attn_mask,
                        dropout_p=dropout_p,
                        is_causal=is_causal,
                        scale=scale,
                        **extra,
                    )

                F.scaled_dot_product_attention = _capture_sdpa
                try:
                    return orig_fwd(*args, **kwargs)
                finally:
                    F.scaled_dot_product_attention = orig_sdpa

            return _wrapped

        for idx, mod in temporal_mods:
            orig = mod.forward
            wrapped = _make_forward(orig, idx)
            mod.forward = wrapped
            self._wired.append((mod, orig))

        return len(temporal_mods)

    def unwire(self) -> None:
        for mod, orig in self._wired:
            mod.forward = orig
        self._wired.clear()


@app.function(
    gpu="H100",
    image=image,
    secrets=[hf_secret],
    volumes={MODEL_CACHE: model_volume},
    timeout=3600,
    memory=32768,
)
def run_oasis_persistent_kv_theory(
    num_frames: int = 32,
    n_prompt: int = 1,
    ddim_steps: int = 10,
    max_noise_level: int = 1000,
    stabilization_level: int = 15,
    noise_abs_max: float = 20.0,
    seed: int = 42,
    prompt_file: str = "sample_image_0.png",
    actions_file: str = "sample_actions_0.one_hot_actions.pt",
) -> dict:
    import os
    import statistics
    import torch
    from einops import rearrange

    sys.path.insert(0, "/root/open-oasis")
    from huggingface_hub import hf_hub_download
    from dit import DiT_models
    from vae import VAE_models
    from safetensors.torch import load_file as safetensors_load
    from utils import sigmoid_beta_schedule, load_prompt, load_actions

    hf_token = os.environ.get("HF_TOKEN", "").strip() or None

    # Load model
    dit_ckpt = hf_hub_download("Etched/oasis-500m", "oasis500m.safetensors", cache_dir=MODEL_CACHE, token=hf_token)
    vae_ckpt = hf_hub_download("Etched/oasis-500m", "vit-l-20.safetensors", cache_dir=MODEL_CACHE, token=hf_token)
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
    prompt_latent = rearrange(z, "(b t) (h w) c -> b t c h w", t=n_prompt, h=H_lat, w=W_lat)

    betas = sigmoid_beta_schedule(max_noise_level).float().to("cuda")
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    alphas_cumprod = rearrange(alphas_cumprod, "T -> T 1 1 1")
    noise_range = torch.linspace(-1, max_noise_level - 1, ddim_steps + 1)
    model_max_frames = getattr(model, "max_frames", total_frames + 1)

    def _run_once(mode: str) -> tuple[torch.Tensor, float, int]:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        B = 1
        x = prompt_latent.clone()
        hook = TemporalPastKVReuseHook(mode=mode)
        n_temporal_wired = 0
        if mode != "off":
            n_temporal_wired = hook.wire(model)

        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        try:
            for i in range(n_prompt, total_frames):
                if mode != "off":
                    hook.reset_frame()

                chunk = torch.randn((B, 1, *x.shape[-3:]), device="cuda")
                chunk = torch.clamp(chunk, -noise_abs_max, +noise_abs_max)
                x = torch.cat([x, chunk], dim=1)
                start_frame = max(0, i + 1 - model_max_frames)

                for step_ctr, noise_idx in enumerate(reversed(range(1, ddim_steps + 1))):
                    t_ctx = torch.full((B, i), stabilization_level - 1, dtype=torch.long, device="cuda")
                    t = torch.full((B, 1), int(noise_range[noise_idx].item()), dtype=torch.long, device="cuda")
                    t_next = torch.full((B, 1), int(noise_range[noise_idx - 1].item()), dtype=torch.long, device="cuda")
                    t_next = torch.where(t_next < 0, t, t_next)
                    t = torch.cat([t_ctx, t], dim=1)
                    t_next = torch.cat([t_ctx, t_next], dim=1)
                    x_curr = x.clone()[:, start_frame:]
                    t = t[:, start_frame:]
                    t_next = t_next[:, start_frame:]

                    if mode != "off":
                        # In temporal attention, sequence tokens scale with past frame count.
                        # We estimate past-token prefix as a fraction of sequence length.
                        total_frames_in_window = max(1, x_curr.shape[1])
                        past_frames_in_window = max(0, total_frames_in_window - 1)
                        seq_len = int(x_curr.shape[-2] * x_curr.shape[-1])  # fallback proxy
                        est_past_tokens = int((past_frames_in_window / total_frames_in_window) * seq_len)
                        hook.set_runtime(past_tokens=est_past_tokens, step_idx=step_ctr)

                    with torch.no_grad():
                        with torch.cuda.amp.autocast(dtype=torch.float16):
                            v = model(x_curr, t, actions[:, start_frame : i + 1])

                    ab_t = alphas_cumprod[t]
                    ab_next = alphas_cumprod[t_next].clone()
                    ab_next[:, :-1] = torch.ones_like(ab_next[:, :-1])
                    if noise_idx == 1:
                        ab_next[:, -1:] = torch.ones_like(ab_next[:, -1:])
                    x_start = ab_t.sqrt() * x_curr - (1 - ab_t).sqrt() * v
                    eps_t = (1 - ab_t).sqrt() * x_curr + ab_t.sqrt() * v
                    x_pred = ab_next.sqrt() * x_start + (1 - ab_next).sqrt() * eps_t
                    x[:, -1:] = x_pred[:, -1:]
        finally:
            if mode != "off":
                hook.unwire()

        e.record()
        torch.cuda.synchronize()
        return x, s.elapsed_time(e), n_temporal_wired

    # Ablation modes:
    # off          : no hooks
    # noop         : hook + patched sdpa passthrough (overhead floor)
    # capture_only : hook + capture prefix, no reuse
    # reuse        : hook + capture + reuse prefix
    modes = ["off", "noop", "capture_only", "reuse"]
    lat_by_mode: Dict[str, list[float]] = {m: [] for m in modes}
    out_by_mode: Dict[str, torch.Tensor] = {}
    n_wired = 0

    for _ in range(2):
        for m in modes:
            x_out, ms, n_w = _run_once(mode=m)
            lat_by_mode[m].append(ms)
            out_by_mode[m] = x_out
            if m == "reuse":
                n_wired = n_w

    mean_by_mode = {m: statistics.mean(v) for m, v in lat_by_mode.items()}
    base = mean_by_mode["off"]
    diffs = {
        "noop_vs_off": float((out_by_mode["noop"] - out_by_mode["off"]).abs().max().item()),
        "capture_only_vs_off": float((out_by_mode["capture_only"] - out_by_mode["off"]).abs().max().item()),
        "reuse_vs_off": float((out_by_mode["reuse"] - out_by_mode["off"]).abs().max().item()),
    }

    # This is the projection-side theoretical saving if past K/V projections
    # were skipped after first step (not achieved by this hook-only prototype).
    past_steps = max(ddim_steps - 1, 0)
    theoretical_projection_reduction = 0.0 if ddim_steps <= 0 else (past_steps / ddim_steps)

    return {
        "model": "Etched/oasis-500m",
        "kernel": "persistent_past_kv_theory_temporal_hook",
        "num_frames": num_frames,
        "ddim_steps": ddim_steps,
        "latency_ms_mean_by_mode": {k: round(v, 2) for k, v in mean_by_mode.items()},
        "relative_speed_vs_off": {
            k: round(base / v, 3) if v > 0 else 0.0 for k, v in mean_by_mode.items()
        },
        "max_abs_diff_vs_off": diffs,
        "n_temporal_modules_wired": n_wired,
        "theoretical_projection_reduction_pct": round(100.0 * theoretical_projection_reduction, 2),
        "notes": (
            "Hook-only prototype reuses past K/V tensors in attention input but "
            "does not skip qkv projection FLOPs; use as correctness/theory harness."
        ),
        "gpu": "H100",
        "raw_latencies_ms_by_mode": lat_by_mode,
    }


@app.local_entrypoint(name="oasis_persistent_kv_oasis_modal")
def main(
    num_frames: int = 32,
    ddim_steps: int = 10,
    seed: int = 42,
):
    result = run_oasis_persistent_kv_theory.remote(
        num_frames=num_frames,
        ddim_steps=ddim_steps,
        seed=seed,
    )
    print("\n=== Persistent past-KV Oasis theory run ===")
    for k in [
        "latency_ms_mean_by_mode",
        "relative_speed_vs_off",
        "max_abs_diff_vs_off",
        "n_temporal_modules_wired",
        "theoretical_projection_reduction_pct",
    ]:
        print(f"{k}: {result.get(k)}")

    sys.path.insert(0, "/root/benchmarks")
    from result_store import save_result
    save_result(result, "optimised_kernels", f"oasis_persistent_kv_theory_f{num_frames}")
