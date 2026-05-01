"""
MAGI-1 4.5B-base ablation: cross-chunk K/V reuse DISABLED.

Why this script exists
----------------------
MAGI-1 doesn't have action conditioning; it's text-driven. So the
"action-aware K/V reuse" idea (the world-model variant) doesn't apply
directly. The natural analog is **cross-chunk** K/V reuse, which MAGI-1
performs natively via the runtime-config knobs:

    "noise2clean_kvrange": [5, 4, 3, 2],   # how many past clean chunks each
                                            #   noise level can attend to
    "clean_chunk_kvrange": 1,               # how many past clean chunks the
                                            #   clean chunks themselves see

Setting both to 0 disables cross-chunk K/V reuse — each chunk attends only
to itself.  This ablation isolates how much the temporal-K/V-reuse path
contributes to wall time, which is exactly the cross-frame K/V reuse
question we wanted to answer.

Important caveat
----------------
The model was trained with non-zero kvrange, so quality WILL degrade
when the temporal context is severed. We report FPS / VRAM here; quality
needs a separate FVD-style eval (out of scope for this run).

Run:
    PYTHONPATH=benchmarks modal run benchmarks/optimised/magi1_45b_no_temporal_kv_modal.py
"""

import sys
sys.path.insert(0, "/root")
sys.path.insert(0, "/root/benchmarks")

import os

# We reuse the baseline image (with all the MAGI-1 + flashinfer + cu124
# pin work). To keep the diff minimal, we import the baseline's image,
# entrypoint helpers, and constants.
from baseline.magi1_45b_baseline_modal import (    # type: ignore
    MAGI_REPO,
    MAGI_LOAD_REL,
    T5_PRETRAINED_REL,
    VAE_PRETRAINED_REL,
    MAGI_DOWNLOAD_REL,
    T5_DOWNLOAD_REL,
    VAE_DOWNLOAD_REL,
    MAGI_CONFIG_REL,
    DEFAULT_PROMPT,
    image,                                          # the configured Modal image
)
from modal_common import app, hf_secret, model_volume, MODEL_CACHE


@app.function(
    gpu="H100",
    image=image,
    secrets=[hf_secret],
    volumes={MODEL_CACHE: model_volume},
    timeout=10800,
    memory=131072,
)
def run_magi1_no_temporal_kv(
    num_frames: int = 96,
    num_steps: int = 64,
    width: int = 720,
    height: int = 720,
    seed: int = 42,
    num_warmup: int = 1,
    num_iters: int = 1,
    prompt: str = DEFAULT_PROMPT,
) -> dict:
    """Same baseline pipeline, but with cross-chunk K/V reuse disabled."""
    import json
    import statistics

    import torch
    from huggingface_hub import snapshot_download

    hf_token = os.environ.get("HF_TOKEN", "").strip() or None
    print(f"Downloading {MAGI_REPO} (4.5B + T5 + VAE) ...")
    local_dir = snapshot_download(
        MAGI_REPO,
        cache_dir=MODEL_CACHE,
        token=hf_token,
        allow_patterns=[
            f"{MAGI_DOWNLOAD_REL}/**",
            f"{T5_DOWNLOAD_REL}/**",
            f"{VAE_DOWNLOAD_REL}/**",
            "README.md",
        ],
    )
    print(f"  weights at: {local_dir}")

    load_dir = os.path.join(local_dir, MAGI_LOAD_REL)
    t5_dir = os.path.join(local_dir, T5_PRETRAINED_REL)
    vae_dir = os.path.join(local_dir, VAE_PRETRAINED_REL)

    src_cfg = os.path.join("/root/MAGI-1", MAGI_CONFIG_REL)
    with open(src_cfg) as fh:
        cfg = json.load(fh)

    cfg["runtime_config"]["load"] = load_dir
    cfg["runtime_config"]["t5_pretrained"] = t5_dir
    cfg["runtime_config"]["vae_pretrained"] = vae_dir
    cfg["runtime_config"]["num_frames"] = num_frames
    cfg["runtime_config"]["num_steps"] = num_steps
    cfg["runtime_config"]["video_size_h"] = height
    cfg["runtime_config"]["video_size_w"] = width
    cfg["runtime_config"]["seed"] = seed

    # ── ABLATION ──────────────────────────────────────────────────────────
    # Disable cross-chunk K/V reuse: each noise level sees 0 past clean
    # chunks, and clean chunks themselves see 0 prior clean chunks. Length
    # of noise2clean_kvrange must stay 4 (number of noise buckets), so we
    # use [0,0,0,0] rather than [].
    cfg["runtime_config"]["noise2clean_kvrange"] = [0, 0, 0, 0]
    cfg["runtime_config"]["clean_chunk_kvrange"] = 0
    print("  noise2clean_kvrange = [0,0,0,0]  (cross-chunk K/V reuse OFF)")
    print("  clean_chunk_kvrange = 0")
    # ──────────────────────────────────────────────────────────────────────

    patched_cfg = "/tmp/magi1_45b_no_temporal_kv.json"
    with open(patched_cfg, "w") as fh:
        json.dump(cfg, fh, indent=2)

    os.chdir("/root/MAGI-1")
    sys.path.insert(0, "/root/MAGI-1")
    from inference.pipeline import MagiPipeline  # type: ignore

    print("Constructing MagiPipeline (cross-chunk K/V disabled) ...")
    pipeline = MagiPipeline(patched_cfg)

    chunk_width = cfg["runtime_config"].get("chunk_width", 6)
    expected_chunks = max(1, num_frames // chunk_width)
    print(f"  num_frames={num_frames}, chunk_width={chunk_width} → "
          f"{expected_chunks} chunks × {num_steps} steps")

    @torch.inference_mode()
    def _run_once(seed_val: int, output_path: str) -> tuple[float, int]:
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        pipeline.run_text_to_video(prompt=prompt, output_path=output_path)
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e), num_frames

    print(f"Warmup ({num_warmup} iters) ...")
    n_obs = num_frames
    for i in range(num_warmup):
        ms, n_obs = _run_once(seed + i + 1000, f"/tmp/magi_no_kv_warmup_{i}.mp4")
        print(f"  warmup {i + 1}: {ms:.0f} ms ({n_obs} frames)")

    torch.cuda.reset_peak_memory_stats()
    latencies: list[float] = []
    print(f"Timed runs ({num_iters} iters) ...")
    for i in range(num_iters):
        try:
            ms, n_obs = _run_once(seed + i, f"/tmp/magi_no_kv_timed_{i}.mp4")
            latencies.append(ms)
            fps = n_obs / (ms / 1000.0)
            print(f"  iter {i + 1}: {ms:.0f} ms ({fps:.3f} fps)")
        except torch.cuda.OutOfMemoryError as oom:
            print(f"  iter {i + 1}: OOM — {oom}")
            torch.cuda.empty_cache()

    if not latencies:
        return {"error": "all OOM", "model": MAGI_REPO}

    mean_ms = statistics.mean(latencies)
    fps = n_obs / (mean_ms / 1000.0)

    return {
        "model": MAGI_REPO,
        "variant": "4.5B_base",
        "kernel": "magi1_no_cross_chunk_kv",
        "num_frames": num_frames,
        "num_steps": num_steps,
        "chunk_width": chunk_width,
        "expected_chunks": expected_chunks,
        "width": width,
        "height": height,
        "n_params_B": 4.5,
        "noise2clean_kvrange": [0, 0, 0, 0],
        "clean_chunk_kvrange": 0,
        "latency_ms_mean": round(mean_ms, 2),
        "latency_ms_std": round(
            statistics.stdev(latencies) if len(latencies) > 1 else 0.0, 2
        ),
        "latency_per_frame_ms": round(mean_ms / n_obs, 2),
        "frames_per_sec": round(fps, 3),
        "vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "gpu": "H100",
        "raw_latencies_ms": latencies,
        "note": (
            "Cross-chunk K/V reuse disabled. Quality will degrade vs "
            "baseline (model trained with non-zero kvrange); FPS/VRAM "
            "delta isolates the speed contribution of cross-chunk K/V."
        ),
    }


@app.local_entrypoint(name="magi1_45b_no_temporal_kv_modal")
def main(
    num_frames: int = 96,
    num_steps: int = 64,
    num_warmup: int = 1,
    num_iters: int = 1,
):
    result = run_magi1_no_temporal_kv.remote(
        num_frames=num_frames,
        num_steps=num_steps,
        num_warmup=num_warmup,
        num_iters=num_iters,
    )
    import json
    print("\nMAGI-1 4.5B no-cross-chunk-KV result:")
    print(json.dumps(result, indent=2))

    sys.path.insert(0, "/root/benchmarks")
    from result_store import save_result
    save_result(
        result, "optimised_kernels",
        f"magi1_45b_no_temporal_kv_f{num_frames}_s{num_steps}",
    )
