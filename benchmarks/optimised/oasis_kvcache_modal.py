"""
Modal H100 benchmark: Open-Oasis 500M with KVCacheManager-driven KV cache
compression (Phase 2 of the proposal).

Optimizations exposed (each toggleable via CLI flags):
  --use-temporal-merge      TemporalTokenMerger (cosine-sim merge across frames)
  --use-progressive-quant   ProgressiveKVQuantizer (FP16 -> FP8 @age 30 -> INT4 @age 90)
  --use-qvg                 QVGSmoothingQuantizer (k-means + 2-bit residual)
  --use-spatial-eviction    SpatialBlockEvictor (Oasis: 144-token grid is small,
                                                  expect minimal benefit)

Pipeline:
  1. Load Etched/oasis-500m DiT-S/2 (608M params).
  2. Instantiate KVCacheManager with the chosen techniques.
  3. KVCacheManager.wire_to_model(dit) installs F.sdpa capture hooks on every
     SpatialAxialAttention / TemporalAxialAttention module.
  4. Run the standard autoregressive generation loop (16 new frames, 10 DDIM
     steps each, v-prediction). Before each new frame, call
     manager.set_current_frame_idx(frame_i) so age-based quantization sees
     real frame indices, not (frame * ddim_step) inflated counters.
  5. Report measured latency + KVCacheManager.get_memory_report() with
     per-technique breakdown.

Important honesty caveat (documented in README):
  In the current plumbing, the KV compression operates as an *accountant* —
  it captures K/V tensors AFTER the model has already projected them in FP16
  and reports what the cache *would* look like if we substituted the
  quantized/merged copies back into attention. The model's actual numerical
  output is identical to baseline. This benchmark therefore measures:
    (a) Wall-clock OVERHEAD of running the manager hooks, and
    (b) THEORETICAL memory savings the optimizations would achieve.
  Substituting compressed K/V into the live attention forward is the next
  (more invasive) wiring step — it would change numerics and require
  end-to-end FVD verification.

Run:
    PYTHONPATH=benchmarks modal run benchmarks/optimised/oasis_kvcache_modal.py
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
def run_oasis_kvcache(
    num_frames: int = 16,
    n_prompt: int = 1,
    ddim_steps: int = 10,
    max_noise_level: int = 1000,
    num_warmup: int = 1,
    num_iters: int = 1,
    use_temporal_merge: bool = True,
    use_progressive_quant: bool = True,
    use_qvg: bool = False,
    use_spatial_eviction: bool = False,
    enable_quant_substitution: bool = True,
    # Oasis-tuned thresholds: 16-frame generation never reaches the canonical
    # 30/90-frame thresholds, so we default to 3/10 here. This means frames
    # 0-2 stay FP16, frames 3-9 attend through an FP8 roundtrip, frames 10+
    # through an INT4 roundtrip — every generation actually exercises both
    # quantization tiers.
    fp8_age_threshold: int = 3,
    int4_age_threshold: int = 10,
    similarity_threshold: float = 0.95,
    qvg_n_clusters: int = 16,
) -> dict:
    """Run Oasis-500M with KVCacheManager-driven KV cache compression."""
    import json
    import os
    import statistics

    import torch

    sys.path.insert(0, "/root/open-oasis")

    hf_token = os.environ.get("HF_TOKEN", "").strip() or None

    # -------------------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------------------
    from huggingface_hub import hf_hub_download

    print("Downloading Etched/oasis-500m checkpoints ...")
    dit_ckpt = hf_hub_download(
        "Etched/oasis-500m", "oasis500m.safetensors",
        cache_dir=MODEL_CACHE, token=hf_token,
    )

    from dit import DiT_models
    from safetensors.torch import load_file as safetensors_load

    print("Loading DiT-S/2 ...")
    model = DiT_models["DiT-S/2"]().to("cuda").eval()
    state = safetensors_load(dit_ckpt)
    model.load_state_dict(state, strict=False)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"DiT loaded: {n_params:.0f}M params")

    # -------------------------------------------------------------------------
    # Build KVCacheManager
    # -------------------------------------------------------------------------
    config: dict = {
        "num_layers": 16,        # Oasis DiT-S/2 has 16 SpatioTemporalDiTBlocks
        "max_frames": num_frames + n_prompt + 1,
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
    if use_qvg:
        config["qvg_smoothing"] = {
            "n_clusters": qvg_n_clusters,
            "residual_bits": 2,
        }
    if use_spatial_eviction:
        # Oasis spatial grid is 9x16 patches (144 tokens). Block_size=2 → 4x8 = 32 blocks.
        config["spatial_eviction"] = {
            "block_size": 2,
            "eviction_ratio": 0.3,
            "min_age_frames": 5,
            "H": 9,
            "W": 16,
        }

    print(f"KVCacheManager config: {json.dumps(config, indent=2)}")
    from worldserve.optimizations.system_level.kv_cache.manager import KVCacheManager

    manager = KVCacheManager(config=config)
    n_wired = manager.wire_to_model(model)
    print(f"KVCacheManager wired into {n_wired} attention modules")
    if n_wired == 0:
        print("WARNING: no attention modules were wired — KV capture inactive")

    techniques_active = [
        name for name, on in [
            ("temporal_merge", use_temporal_merge),
            ("progressive_quant", use_progressive_quant),
            ("qvg_smoothing", use_qvg),
            ("spatial_eviction", use_spatial_eviction),
        ] if on
    ]

    # -------------------------------------------------------------------------
    # DDPM sigmoid schedule (matches open-oasis/utils.sigmoid_beta_schedule)
    # -------------------------------------------------------------------------
    from utils import sigmoid_beta_schedule

    betas = sigmoid_beta_schedule(max_noise_level).float().to("cuda")
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    noise_range = torch.linspace(-1, max_noise_level - 1, ddim_steps + 1)

    # -------------------------------------------------------------------------
    # Timed run
    # -------------------------------------------------------------------------
    def _run_once() -> float:
        manager.create_cache(num_layers=16, max_frames=num_frames + n_prompt + 1)
        B = 1
        total_frames = n_prompt + num_frames

        x = torch.randn(B, n_prompt, LATENT_C, LATENT_H, LATENT_W, device="cuda")
        actions = torch.zeros(B, total_frames, 25, device="cuda", dtype=torch.float32)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()

        for frame_i in range(n_prompt, total_frames):
            # Critical: tell the manager which video frame we're working on
            # (not the denoising step), so age-based thresholds operate on
            # frame ages, not on (frame × ddim_step) over-inflated counters.
            manager.set_current_frame_idx(frame_i)

            chunk = torch.randn(B, 1, LATENT_C, LATENT_H, LATENT_W, device="cuda")

            for noise_idx in reversed(range(1, ddim_steps + 1)):
                t_val = noise_range[noise_idx].long().clamp(0, max_noise_level - 1)

                t_ctx = torch.zeros(B, frame_i, device="cuda", dtype=torch.long)
                t_cur = t_val.to("cuda").view(1, 1).expand(B, 1)
                t_in = torch.cat([t_ctx, t_cur], dim=1)
                x_in = torch.cat([x, chunk], dim=1)

                with torch.no_grad():
                    v = model(x_in, t_in, actions[:, : frame_i + 1])

                t_idx = int(t_val.item())
                alpha_t = alphas_cumprod[t_idx]
                t_prev_idx = max(int(noise_range[noise_idx - 1].item()), 0)
                alpha_t_prev = alphas_cumprod[t_prev_idx]

                v_cur = v[:, -1:]
                x0_pred = alpha_t.sqrt() * chunk - (1 - alpha_t).sqrt() * v_cur
                eps_pred = (1 - alpha_t).sqrt() * chunk + alpha_t.sqrt() * v_cur
                chunk = (
                    alpha_t_prev.sqrt() * x0_pred + (1 - alpha_t_prev).sqrt() * eps_pred
                )

            x = torch.cat([x, chunk], dim=1)

        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end)

    print(f"Warmup ({num_warmup} iters) ...")
    for i in range(num_warmup):
        _run_once()
        print(f"  warmup {i + 1}/{num_warmup} done")

    torch.cuda.reset_peak_memory_stats()
    latencies: list = []
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

    # -------------------------------------------------------------------------
    # Memory report from the manager
    # -------------------------------------------------------------------------
    mem_report = manager.get_memory_report()

    results = {
        "model": "Etched/oasis-500m",
        "kernel": "kvcache_" + ("+".join(techniques_active) if techniques_active else "none"),
        "techniques_active": techniques_active,
        "kvcache_config": config,
        "num_frames": num_frames,
        "n_prompt": n_prompt,
        "ddim_steps": ddim_steps,
        "total_dit_fwd_passes": num_frames * ddim_steps,
        "n_attention_modules_wired": n_wired,
        "latency_ms_mean": round(mean_ms, 2),
        "latency_ms_std": round(
            statistics.stdev(latencies) if len(latencies) > 1 else 0.0, 2
        ),
        "latency_per_frame_ms": round(mean_ms / num_frames, 2),
        "frames_per_sec": round(num_frames / (mean_ms / 1000.0), 3),
        "vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "kvcache_memory_report": {
            "total_fp16_bytes": mem_report.get("total_fp16_bytes", 0),
            "total_actual_bytes": mem_report.get("total_actual_bytes", 0),
            "savings_ratio": round(mem_report.get("savings_ratio", 0.0), 4),
            "per_technique": mem_report.get("per_technique", {}),
        },
        "gpu": "H100",
        "raw_latencies_ms": latencies,
    }
    print(json.dumps(results, indent=2, default=str))
    return results


@app.local_entrypoint(name="oasis_kvcache_modal")
def main(
    num_frames: int = 16,
    use_temporal_merge: bool = True,
    use_progressive_quant: bool = True,
    use_qvg: bool = False,
    use_spatial_eviction: bool = False,
    enable_quant_substitution: bool = True,
    fp8_age_threshold: int = 3,
    int4_age_threshold: int = 10,
    num_iters: int = 1,
    num_warmup: int = 1,
):
    result = run_oasis_kvcache.remote(
        num_frames=num_frames,
        use_temporal_merge=use_temporal_merge,
        use_progressive_quant=use_progressive_quant,
        use_qvg=use_qvg,
        use_spatial_eviction=use_spatial_eviction,
        enable_quant_substitution=enable_quant_substitution,
        fp8_age_threshold=fp8_age_threshold,
        int4_age_threshold=int4_age_threshold,
        num_iters=num_iters,
        num_warmup=num_warmup,
    )
    print("\nOpen-Oasis 500M KVCache result:")
    import json
    print(json.dumps(result, indent=2, default=str))

    sys.path.insert(0, "/root/benchmarks")
    from result_store import save_result
    techniques = result.get("kernel", "unknown")
    save_result(result, "optimised_kernels", f"oasis_{techniques}_f{num_frames}")
