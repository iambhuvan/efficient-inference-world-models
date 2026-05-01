"""
Helios-Distilled + Tiered KV cache quantization + H2O eviction — Layer 4.

Combined KV management:
  - Tiered quant: FP16 (recent ≤30 frames) → FP8 (30–90) → INT4 (>90 frames)
  - H2O token eviction: drops low-attention KV entries beyond a budget
  - Temporal merge: cosine-sim merge tokens across frames (τ=0.95)
  - K>V asymmetric quant (DiffKV finding — V tolerates more aggressive quant)

Wires worldserve.optimizations.system_level.kv_cache.manager.KVCacheManager
into the transformer's 40 attention modules.

Expected speedup: 1.2×–1.5× wall-clock + 3×–5× KV memory reduction.
DMD-distilled few-step regime limits absolute speedup since KV doesn't
grow over many denoising steps the way it does on undistilled diffusion.

Run:
    PYTHONPATH=benchmarks modal run benchmarks/optimised/helios_distilled_kvcache_modal.py
"""

import sys
sys.path.insert(0, "/root")
sys.path.insert(0, "/root/benchmarks")

import modal
from modal_common import app, image_cuda_devel, hf_secret, model_volume, MODEL_CACHE
image = image_cuda_devel

HELIOS_REPO = "BestWishYsh/Helios-Distilled"
PAPER_FPS = 19.5
BASELINE_FPS = 0.94


@app.function(
    gpu="H100",
    image=image,
    secrets=[hf_secret],
    volumes={MODEL_CACHE: model_volume},
    timeout=3600,
    memory=65536,
)
def run_helios_distilled_kvcache(
    enable_temporal_merge: bool = True,
    enable_progressive_quant: bool = True,
    enable_h2o_eviction: bool = True,
    fp8_age_threshold: int = 30,
    int4_age_threshold: int = 90,
    similarity_threshold: float = 0.95,
    h2o_keep_ratio: float = 0.5,
    num_warmup: int = 1,
    num_iters: int = 3,
    seed: int = 42,
    prompt: str = (
        "A camera slowly pans through a sunlit forest clearing, golden light "
        "filtering through tall trees, soft wind moving the leaves."
    ),
) -> dict:
    """Wire WorldServe's KVCacheManager into the transformer."""
    import os
    import statistics

    import torch
    from huggingface_hub import snapshot_download
    from diffusers import DiffusionPipeline

    hf_token = os.environ.get("HF_TOKEN", "").strip() or None

    print(f"Downloading {HELIOS_REPO} ...")
    local_dir = snapshot_download(HELIOS_REPO, cache_dir=MODEL_CACHE, token=hf_token)

    print("Loading HeliosPyramidPipeline (bf16, trust_remote_code=True) ...")
    pipe = DiffusionPipeline.from_pretrained(
        local_dir, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    # ── OPTIMIZATION: KVCacheManager wiring ─────────────────────────────
    cfg: dict = {
        "num_layers": 40,           # Wan2.1-T2V-14B blocks
        "max_frames": 256,          # generous upper bound
        "enable_quant_substitution": True,
    }
    if enable_temporal_merge:
        cfg["temporal_merge"] = {
            "similarity_threshold": similarity_threshold,
            "merge_strategy": "weighted_average",
        }
    if enable_progressive_quant:
        cfg["progressive_quant"] = {
            "fp8_age_threshold": fp8_age_threshold,
            "int4_age_threshold": int4_age_threshold,
            "key_value_asymmetry": True,        # K higher precision than V
        }
    if enable_h2o_eviction:
        cfg["h2o"] = {
            "keep_ratio": h2o_keep_ratio,
            "score_window_size": 16,
        }

    n_wired = 0
    try:
        from worldserve.optimizations.system_level.kv_cache.manager import KVCacheManager
        manager = KVCacheManager(config=cfg)
        n_wired = manager.wire_to_model(pipe.transformer)
        print(f"KVCacheManager wired into {n_wired} attention modules.")
    except Exception as exc:
        print(f"KVCacheManager wire failed: {exc}. Running un-optimized for diagnostic.")
        manager = None
    # ────────────────────────────────────────────────────────────────────

    n_params = sum(
        p.numel() for m in pipe.components.values()
        if isinstance(m, torch.nn.Module) for p in m.parameters()
    )

    @torch.inference_mode()
    def _run_once(seed_val: int) -> tuple[float, int]:
        gen = torch.Generator(device="cuda").manual_seed(seed_val)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        out = pipe(prompt=prompt, generator=gen, return_dict=True)
        e.record()
        torch.cuda.synchronize()

        frames = getattr(out, "frames", None)
        if frames is None:
            frames = getattr(out, "videos", None)
        return s.elapsed_time(e), _get_frame_count(frames)

    print(f"Warmup ({num_warmup} iters) ...")
    n_observed = 1
    for i in range(num_warmup):
        ms, n_observed = _run_once(seed_val=seed + i + 1000)
        print(f"  warmup {i + 1}: {ms:.0f} ms ({n_observed} frames)")

    torch.cuda.reset_peak_memory_stats()
    latencies: list[float] = []
    print(f"Timed runs ({num_iters} iters) ...")
    for i in range(num_iters):
        ms, n_observed = _run_once(seed_val=seed + i)
        latencies.append(ms)
        fps = n_observed / (ms / 1000.0)
        print(f"  iter {i + 1}: {ms:.0f} ms ({fps:.2f} fps)")

    mean_ms = statistics.mean(latencies)
    fps = n_observed / (mean_ms / 1000.0)

    mem_report: dict = {}
    if manager is not None:
        try:
            mem_report = manager.get_memory_report()
        except Exception:
            pass

    return {
        "model": HELIOS_REPO,
        "kernel": "tiered_kv_quant_plus_h2o",
        "config": cfg,
        "n_attention_modules_wired": n_wired,
        "num_frames_observed": n_observed,
        "n_params_B": round(n_params / 1e9, 3),
        "latency_ms_mean": round(mean_ms, 2),
        "latency_ms_std": round(statistics.stdev(latencies) if len(latencies) > 1 else 0.0, 2),
        "latency_per_frame_ms": round(mean_ms / n_observed, 2),
        "frames_per_sec": round(fps, 3),
        "speedup_vs_baseline": round(fps / BASELINE_FPS, 3),
        "baseline_fps": BASELINE_FPS,
        "paper_fps_target": PAPER_FPS,
        "vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "kv_memory_savings_ratio": mem_report.get("savings_ratio", None),
        "gpu": "H100",
        "raw_latencies_ms": latencies,
    }


def _get_frame_count(frames) -> int:
    if frames is None:
        return 1
    shape = getattr(frames, "shape", None)
    if shape is not None and len(shape) >= 4:
        return shape[0] if len(shape) == 4 else shape[1]
    if isinstance(frames, list):
        first = frames[0] if frames else None
        return len(first) if isinstance(first, list) else len(frames)
    return 1


@app.local_entrypoint(name="helios_distilled_kvcache_modal")
def main(num_iters: int = 3):
    result = run_helios_distilled_kvcache.remote(num_iters=num_iters)
    import json
    print("\n" + json.dumps(result, indent=2))
    print(f"\nSpeedup vs baseline: {result.get('speedup_vs_baseline', 'N/A')}×")

    sys.path.insert(0, "/root/benchmarks")
    from result_store import save_result
    save_result(result, "optimised_kernels", "helios_distilled_kvcache")
