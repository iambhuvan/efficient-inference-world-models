"""
Modal H100 benchmark: Open-Oasis 500M with stacked optimisations.

Kernels applied:
  1. SageAttention2  — monkey-patches F.scaled_dot_product_attention inside
                       SpatialAxialAttention + TemporalAxialAttention
  2. INT4 weight quantisation — torchao int4_weight_only on the DiT linear layers

TeaCache / PrediT are step-level caches and require hooking the custom DDIM loop;
SageAttn2 + INT4 are injected at the kernel level without loop changes.

Run:
    modal run modal/optimised/oasis_all_modal.py
"""

import sys
sys.path.insert(0, "/root")
sys.path.insert(0, "/root/benchmarks")

import modal
from modal_common import app, image, hf_secret, model_volume, MODEL_CACHE

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
def run_oasis_all(
    num_frames: int = 16,
    n_prompt: int = 1,
    ddim_steps: int = 10,
    max_noise_level: int = 1000,
    num_warmup: int = 1,
    num_iters: int = 1,
    use_sageattention: bool = True,
    use_int4: bool = True,
) -> dict:
    """
    Run Open-Oasis 500M with SageAttention2 + INT4 optimisations.
    """
    import json
    import os
    import statistics

    import torch
    import torch.nn.functional as F

    sys.path.insert(0, "/root/open-oasis")

    hf_token = os.environ.get("HF_TOKEN", "").strip() or None

    # -----------------------------------------------------------------------
    # Load model weights
    # -----------------------------------------------------------------------
    from huggingface_hub import hf_hub_download
    from dit import DiT_models
    from safetensors.torch import load_file as safetensors_load

    print("Downloading Etched/oasis-500m checkpoints ...")
    dit_ckpt = hf_hub_download(
        "Etched/oasis-500m", "oasis500m.safetensors",
        cache_dir=MODEL_CACHE, token=hf_token,
    )

    print("Loading DiT-S/2 ...")
    model = DiT_models["DiT-S/2"]().to("cuda").eval()
    state = safetensors_load(dit_ckpt)
    model.load_state_dict(state, strict=False)

    kernels_applied = []

    # -----------------------------------------------------------------------
    # Optimisation 1: SageAttention2 via F.sdpa monkey-patch
    # Both SpatialAxialAttention and TemporalAxialAttention call
    # F.scaled_dot_product_attention directly, so one patch covers both.
    # -----------------------------------------------------------------------
    if use_sageattention:
        try:
            import sageattention
            _orig_sdpa = F.scaled_dot_product_attention

            def _sage_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
                           is_causal=False, scale=None, **kwargs):
                # SageAttention requires fp16 or bf16; fall back for fp32
                if query.dtype in (torch.float16, torch.bfloat16):
                    try:
                        sage_kwargs = {"is_causal": is_causal}
                        if scale is not None:
                            sage_kwargs["sm_scale"] = scale
                        return sageattention.sageattn(query, key, value, **sage_kwargs)
                    except Exception:
                        pass
                return _orig_sdpa(query, key, value, attn_mask=attn_mask,
                                  dropout_p=dropout_p, is_causal=is_causal, scale=scale)

            F.scaled_dot_product_attention = _sage_sdpa
            torch.nn.functional.scaled_dot_product_attention = _sage_sdpa
            print("SageAttention2 active via F.sdpa monkey-patch")
            kernels_applied.append("sageattention2")
        except ImportError as exc:
            print(f"sageattention not importable: {exc} — skipping")

    # -----------------------------------------------------------------------
    # Optimisation 2: INT4 weight-only quantisation (torchao)
    # -----------------------------------------------------------------------
    if use_int4:
        try:
            from torchao.quantization import quantize_, int4_weight_only
            from dit import TimestepEmbedder

            # TimestepEmbedder hardcodes float32 for sinusoidal freqs but weights
            # become bfloat16 after quantization — patch forward to cast before MLP.
            _orig_te_forward = TimestepEmbedder.forward

            def _dtype_aware_te_forward(self, t):
                t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
                target_dtype = self.mlp[0].weight.dtype
                return self.mlp(t_freq.to(target_dtype))

            TimestepEmbedder.forward = _dtype_aware_te_forward

            model = model.to(torch.bfloat16)
            quantize_(model, int4_weight_only())
            print("INT4 weight-only quantisation applied via torchao (bfloat16)")
            kernels_applied.append("int4_weight_only")
        except Exception as exc:
            print(f"INT4 quantisation failed: {exc}")
            model = model.to(torch.float32)  # revert if quantization failed

    # -----------------------------------------------------------------------
    # Noise schedule
    # -----------------------------------------------------------------------
    from utils import sigmoid_beta_schedule

    betas = sigmoid_beta_schedule(max_noise_level).float().to("cuda")
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    noise_range = torch.linspace(-1, max_noise_level - 1, ddim_steps + 1)

    # -----------------------------------------------------------------------
    # Timed generation (same loop as baseline)
    # -----------------------------------------------------------------------
    # INT4 quantized params report a subclass dtype; track explicitly
    _model_dtype = torch.bfloat16 if "int4_weight_only" in kernels_applied else torch.float32

    def _run_once() -> float:
        B = 1
        total_frames = n_prompt + num_frames
        x = torch.randn(B, n_prompt, LATENT_C, LATENT_H, LATENT_W, device="cuda", dtype=_model_dtype)
        # external_cond is nn.Linear(25, hidden_size) — needs float (B, T, 25)
        actions = torch.zeros(B, total_frames, 25, device="cuda", dtype=_model_dtype)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()

        for frame_i in range(n_prompt, total_frames):
            chunk = torch.randn(B, 1, LATENT_C, LATENT_H, LATENT_W, device="cuda", dtype=_model_dtype)

            for noise_idx in reversed(range(1, ddim_steps + 1)):
                t_val = noise_range[noise_idx].long().clamp(0, max_noise_level - 1)

                t_ctx = torch.zeros(B, frame_i, device="cuda", dtype=torch.long)
                t_cur = t_val.to("cuda").view(1, 1).expand(B, 1)
                t_in = torch.cat([t_ctx, t_cur], dim=1)
                x_in = torch.cat([x, chunk], dim=1)

                with torch.no_grad():
                    v = model(x_in, t_in, actions[:, :frame_i + 1])

                t_idx = t_val.item()
                alpha_t = alphas_cumprod[t_idx].to(_model_dtype)
                t_prev_idx = max(int(noise_range[noise_idx - 1].item()), 0)
                alpha_t_prev = alphas_cumprod[t_prev_idx].to(_model_dtype)

                v_cur = v[:, -1:]
                x0_pred = alpha_t.sqrt() * chunk - (1 - alpha_t).sqrt() * v_cur
                eps_pred = (1 - alpha_t).sqrt() * chunk + alpha_t.sqrt() * v_cur
                chunk = (alpha_t_prev.sqrt() * x0_pred + (1 - alpha_t_prev).sqrt() * eps_pred).to(_model_dtype)

            x = torch.cat([x, chunk], dim=1)

        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end)

    print(f"Warmup ({num_warmup} iters) ...")
    for i in range(num_warmup):
        try:
            _run_once()
            print(f"  warmup {i + 1}/{num_warmup} done")
        except torch.cuda.OutOfMemoryError as oom:
            print(f"  warmup {i + 1}: OOM — {oom}; skipping")
            torch.cuda.empty_cache()

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
        return {
            "error": "All iterations OOM",
            "model": "Etched/oasis-500m",
            "kernels_applied": kernels_applied,
        }

    mean_ms = statistics.mean(latencies)
    results = {
        "model": "Etched/oasis-500m",
        "kernel": "+".join(kernels_applied) if kernels_applied else "none",
        "kernels_applied": kernels_applied,
        "num_frames": num_frames,
        "n_prompt": n_prompt,
        "ddim_steps": ddim_steps,
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


@app.local_entrypoint(name="oasis_all_modal")
def main(
    use_sageattention: bool = True,
    use_int4: bool = True,
    num_iters: int = 1,
    num_warmup: int = 1,
):
    result = run_oasis_all.remote(
        use_sageattention=use_sageattention,
        use_int4=use_int4,
        num_iters=num_iters,
        num_warmup=num_warmup,
    )
    print("\nOpen-Oasis 500M (all kernels) result:")
    import json
    print(json.dumps(result, indent=2))
    sys.path.insert(0, "/root/benchmarks")
    from result_store import save_result
    kernels = result.get("kernel", "unknown").replace("+", "_")
    save_result(result, "optimised_kernels", f"oasis_{kernels}")
