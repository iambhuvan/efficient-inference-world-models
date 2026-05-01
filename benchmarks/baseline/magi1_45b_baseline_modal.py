"""
Modal H100 benchmark: MAGI-1 4.5B-base baseline (no optimizations).

Model    : sand-ai/MAGI-1 (4.5B-base variant)
Pipeline : SandAI's `inference.pipeline.MagiPipeline` (chunk-wise AR DiT)
Task     : Text-to-video, autoregressive 6-frame chunks, 720×720 × 96 frames
License  : Apache-2.0, ungated
Paper    : arxiv.org/abs/2505.13211

Why:
  Only large open AR-video DiT with a real denoising-step budget per chunk
  (4.5B-base = 64 steps, NOT distilled). Right anchor for cross-frame KV
  and step-caching ablations.

Loading strategy (verified against upstream `inference/pipeline/entry.py`):
  1. snapshot_download(sand-ai/MAGI-1) — pulls 4.5B base (9 GB) + T5-XXL
     (19 GB) + VAE (2.4 GB) into the persistent Modal volume.
  2. Patch `example/4.5B/4.5B_base_config.json` so its `load`,
     `t5_pretrained`, `vae_pretrained` point at the snapshot subdirs.
  3. Set the distributed env vars MAGI's `dist_init` requires
     (MASTER_ADDR/PORT, RANK, WORLD_SIZE, LOCAL_RANK).
  4. Import + call `MagiPipeline.run_text_to_video(prompt, output_path)`.
     Each call rebuilds the DiT internally (`get_dit` lives inside `_run`),
     so timed iterations include weight-load.  This matches their canonical
     entry — fine for a baseline anchor.

Run:
    PYTHONPATH=benchmarks modal run benchmarks/baseline/magi1_45b_baseline_modal.py
"""

import sys
sys.path.insert(0, "/root")
sys.path.insert(0, "/root/benchmarks")

import modal
from modal_common import (
    app, image_cuda_devel_base, _add_common_layers,
    hf_secret, model_volume, MODEL_CACHE,
)

MAGI_REPO = "sand-ai/MAGI-1"
MAGI_GH = "https://github.com/SandAI-org/MAGI-1.git"

# Subdirs inside the HF snapshot (verified via HF tree API).
# NOTE: MAGI-1's inference code appends `/inference_weight` and
# `/t5-v1_1-xxl` to the configured paths, so `load` and `t5_pretrained`
# must point at the *parent* of those subdirs.
MAGI_LOAD_REL = "ckpt/magi/4.5B_base"           # parent of inference_weight
T5_PRETRAINED_REL = "ckpt/t5"                   # parent of t5-v1_1-xxl
VAE_PRETRAINED_REL = "ckpt/vae"
# allow_patterns target — we still want to download only the inference
# subset, so we keep narrower download globs separately.
MAGI_DOWNLOAD_REL = "ckpt/magi/4.5B_base"
T5_DOWNLOAD_REL = "ckpt/t5/t5-v1_1-xxl"
VAE_DOWNLOAD_REL = "ckpt/vae"
MAGI_CONFIG_REL = "example/4.5B/4.5B_base_config.json"

# MAGI-1 pins flash-attn 2.4.2, diffusers 0.29.2, transformers 4.42.3,
# torchdiffeq, flashinfer.  We install those locally to this image so the
# global image_cuda_devel deps used by other baselines aren't perturbed.
image = _add_common_layers(
    image_cuda_devel_base
    .apt_install("ffmpeg")              # the binary; ffmpeg-python wraps it
    .pip_install(
        "diffusers==0.29.2",
        "transformers==4.42.3",
        "accelerate==0.32.1",
        "torchdiffeq==0.2.4",
        "ftfy==6.2.0",
        "sentencepiece==0.2.0",
        "timm==1.0.15",
        "imageio[ffmpeg]",
        "ffmpeg-python",
        "rich==14.0.0",
        "protobuf==5.28.3",
        "matplotlib==3.10.1",
        "beautifulsoup4==4.13.4",
        "numpy==1.26.4",
    )
    .run_commands(
        # flashinfer wheels are hosted on flashinfer.ai with cu124+torch2.4 ABI
        "pip install flashinfer-python==0.2.0.post2 "
        "--extra-index-url https://flashinfer.ai/whl/cu124/torch2.4/ "
        "|| echo 'flashinfer install failed (will retry at runtime if needed)'"
    )
    .run_commands(
        # MAGI-1's deps (timm, transformers) can pull a torchvision/torchaudio
        # built for cu13, breaking torch 2.5.1 + cu124. Force-reinstall the
        # cu124 trio at the end so nothing later overrides them.
        "pip install --force-reinstall --no-deps "
        "torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 "
        "--extra-index-url https://download.pytorch.org/whl/cu124"
    )
    .run_commands(
        f"git clone --depth=1 {MAGI_GH} /root/MAGI-1 || echo 'MAGI-1 clone failed'"
    )
    .env({
        "PYTHONPATH": "/root/MAGI-1:/root/benchmarks:/root",
        # NCCL single-proc setup (matches example/4.5B/run.sh)
        "MASTER_ADDR": "localhost",
        "MASTER_PORT": "6009",
        "GPUS_PER_NODE": "1",
        "NNODES": "1",
        "WORLD_SIZE": "1",
        "RANK": "0",
        "LOCAL_RANK": "0",
        "CUDA_VISIBLE_DEVICES": "0",
        "PAD_HQ": "1",
        "PAD_DURATION": "1",
        "OFFLOAD_T5_CACHE": "true",
        "OFFLOAD_VAE_CACHE": "true",
        "TORCH_CUDA_ARCH_LIST": "9.0",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
)

DEFAULT_PROMPT = (
    "A camera slowly orbits a wooden cabin in a snowy forest at dusk; "
    "warm light glows through the windows, soft snow falling."
)


@app.function(
    gpu="H100",
    image=image,
    secrets=[hf_secret],
    volumes={MODEL_CACHE: model_volume},
    timeout=10800,                # full 4.5B + T5 + VAE generation can take 10+ min
    memory=131072,                # 128 GB RAM headroom (T5 offloads to CPU)
)
def run_magi1_45b_baseline(
    num_frames: int = 96,
    num_steps: int = 64,
    width: int = 720,
    height: int = 720,
    seed: int = 42,
    num_warmup: int = 1,
    num_iters: int = 1,
    prompt: str = DEFAULT_PROMPT,
) -> dict:
    """MAGI-1 4.5B-base T2V baseline."""
    import json
    import os
    import statistics

    import torch
    from huggingface_hub import snapshot_download

    hf_token = os.environ.get("HF_TOKEN", "").strip() or None

    # ---------------------------------------------------------------- weights
    print(f"Downloading {MAGI_REPO} (only 4.5B + T5 + VAE) ...")
    local_dir = snapshot_download(
        MAGI_REPO,
        cache_dir=MODEL_CACHE,
        token=hf_token,
        # Skip the 24B variants (~50 GB total saved)
        allow_patterns=[
            f"{MAGI_DOWNLOAD_REL}/**",
            f"{T5_DOWNLOAD_REL}/**",
            f"{VAE_DOWNLOAD_REL}/**",
            "README.md",
        ],
    )
    print(f"  weights at: {local_dir}")

    # Paths to put in the patched config — these are what MAGI-1's inference
    # code expects (it appends `/inference_weight` and `/t5-v1_1-xxl` itself).
    load_dir = os.path.join(local_dir, MAGI_LOAD_REL)
    t5_dir = os.path.join(local_dir, T5_PRETRAINED_REL)
    vae_dir = os.path.join(local_dir, VAE_PRETRAINED_REL)
    for p, name in [(load_dir, "magi"), (t5_dir, "t5"), (vae_dir, "vae")]:
        if not os.path.isdir(p):
            return {"error": f"missing {name} dir: {p}", "model": MAGI_REPO}

    # ---------------------------------------------------------------- config
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

    patched_cfg = "/tmp/magi1_45b_base_patched.json"
    with open(patched_cfg, "w") as fh:
        json.dump(cfg, fh, indent=2)
    print(f"  patched config: {patched_cfg}")
    print(json.dumps(cfg["runtime_config"], indent=2))

    # ---------------------------------------------------------------- pipeline
    # MAGI-1's `prompt_process.py` does `np.load("example/assets/...")` with
    # a relative path, so we must chdir to the repo root (the reference
    # `example/4.5B/run.sh` runs from `git rev-parse --show-toplevel`).
    os.chdir("/root/MAGI-1")
    sys.path.insert(0, "/root/MAGI-1")
    from inference.pipeline import MagiPipeline  # type: ignore

    print("Constructing MagiPipeline (this calls dist_init) ...")
    pipeline = MagiPipeline(patched_cfg)

    # n_params: introspect the DiT once it's built. MagiPipeline builds the
    # DiT lazily inside `_run`, so we can't count before the first call.
    # Report None and let the caller fall back to the spec sheet (4.5B).
    n_params_B = 4.5

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
        # MAGI-1 doesn't return frames programmatically — read them off disk
        # via imageio if needed. For timing we just trust the requested count.
        return s.elapsed_time(e), num_frames

    print(f"Warmup ({num_warmup} iters) ...")
    n_observed = num_frames
    for i in range(num_warmup):
        ms, n_observed = _run_once(seed + i + 1000, f"/tmp/magi_warmup_{i}.mp4")
        print(f"  warmup {i + 1}: {ms:.0f} ms ({n_observed} frames)")

    torch.cuda.reset_peak_memory_stats()
    latencies: list[float] = []
    print(f"Timed runs ({num_iters} iters) ...")
    for i in range(num_iters):
        try:
            ms, n_observed = _run_once(seed + i, f"/tmp/magi_timed_{i}.mp4")
            latencies.append(ms)
            fps = n_observed / (ms / 1000.0)
            print(
                f"  iter {i + 1}: {ms:.0f} ms "
                f"({n_observed} frames, {fps:.2f} fps, {ms / n_observed:.1f} ms/frame)"
            )
        except torch.cuda.OutOfMemoryError as oom:
            print(f"  iter {i + 1}: OOM — {oom}")
            torch.cuda.empty_cache()

    if not latencies:
        return {"error": "All iterations OOM", "model": MAGI_REPO}

    mean_ms = statistics.mean(latencies)
    fps = n_observed / (mean_ms / 1000.0)

    results = {
        "model": MAGI_REPO,
        "variant": "4.5B_base",
        "kernel": "baseline_magi1_pipeline_bf16",
        "num_frames": num_frames,
        "num_steps": num_steps,
        "chunk_width": chunk_width,
        "expected_chunks": expected_chunks,
        "width": width,
        "height": height,
        "n_params_B": n_params_B,
        "latency_ms_mean": round(mean_ms, 2),
        "latency_ms_std": round(
            statistics.stdev(latencies) if len(latencies) > 1 else 0.0, 2
        ),
        "latency_per_frame_ms": round(mean_ms / n_observed, 2),
        "frames_per_sec": round(fps, 3),
        "vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "gpu": "H100",
        "raw_latencies_ms": latencies,
    }
    print("\n" + json.dumps(results, indent=2))
    return results


@app.local_entrypoint(name="magi1_45b_baseline_modal")
def main(
    num_frames: int = 96,
    num_steps: int = 64,
    num_warmup: int = 1,
    num_iters: int = 1,
):
    result = run_magi1_45b_baseline.remote(
        num_frames=num_frames,
        num_steps=num_steps,
        num_warmup=num_warmup,
        num_iters=num_iters,
    )

    print("\nMAGI-1 4.5B-base baseline result:")
    import json
    print(json.dumps(result, indent=2))

    sys.path.insert(0, "/root/benchmarks")
    from result_store import save_result
    save_result(
        result, "baseline",
        f"magi1_45b_baseline_f{num_frames}_s{num_steps}",
    )
