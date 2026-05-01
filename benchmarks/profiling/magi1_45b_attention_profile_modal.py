"""
MAGI-1 4.5B-base attention profiler.

Goal: collect per-layer × per-step statistics on the *self/temporal* K
projection so we can choose the right pruning schedule (exponential vs
quadratic vs piecewise vs layer-wise) from data — instead of porting the
SAPS curve from a text model.

What we record per hook call (one record per layer per attention call):
  - layer_id            (0..33)
  - call_idx            global call counter (chronological)
  - shape               (n_tokens, d_model_kv)
  - k_norm_mean / p50 / p95 / max
  - k_top10pct_mass     fraction of total ‖K‖ in top-10% tokens
  - k_top1pct_mass      fraction of total ‖K‖ in top-1% tokens
  - cosine_sim_prev     cos(K_t, K_{t-1}) for same layer, same shape

Why K-norm and not attention probs:
  MAGI-1 uses flashinfer / flash-attn for its attention; neither exposes
  attention scores. K-norm is a defensible proxy: under standard scaled
  dot-product attention, large ‖K‖ tokens dominate the attention mass for
  any reasonable Q. So if K is heavy-tailed (top-1% mass ~> 0.5), pruning
  is safe; if K is uniform (top-1% mass ~ 0.01), pruning hurts.

Coverage:
  - Hooks `linear_qkv.k` per transformer layer (self + temporal K).
  - Skips V, T5 cross-attn K, qx — narrow and cheap first pass.
  - One full T2V generation (96 frames, 64 steps, 16 chunks → ~34*64*16
    ≈ 35k records). All records dumped to JSON for offline analysis.

Run:
    PYTHONPATH=benchmarks modal run benchmarks/profiling/magi1_45b_attention_profile_modal.py
"""

import sys
sys.path.insert(0, "/root")
sys.path.insert(0, "/root/benchmarks")

import os

# Reuse baseline image + constants — keeps deps consistent.
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
    image,
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
def run_magi1_attention_profile(
    num_frames: int = 96,
    num_steps: int = 64,
    width: int = 720,
    height: int = 720,
    seed: int = 42,
    prompt: str = DEFAULT_PROMPT,
) -> dict:
    """One T2V generation with K-projection hooks. Returns the records."""
    import json
    import time

    import torch
    from huggingface_hub import snapshot_download

    hf_token = os.environ.get("HF_TOKEN", "").strip() or None
    print(f"Downloading {MAGI_REPO} ...")
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
    patched_cfg = "/tmp/magi1_attn_profile.json"
    with open(patched_cfg, "w") as fh:
        json.dump(cfg, fh, indent=2)

    os.chdir("/root/MAGI-1")
    sys.path.insert(0, "/root/MAGI-1")
    from inference.pipeline import MagiPipeline  # type: ignore

    print("Constructing MagiPipeline ...")
    pipeline = MagiPipeline(patched_cfg)

    # ---------------------------------------------------------------- hooks
    # We need the DiT model to attach hooks. MagiPipeline builds it lazily
    # inside `_run`, so we monkey-patch `get_dit` to install hooks right
    # after construction.
    from inference.model.dit import get_dit as _orig_get_dit  # type: ignore
    import torch.nn as nn

    records: list[dict] = []
    last_k_per_layer: dict[int, torch.Tensor] = {}
    state = {"call_idx": 0}

    def _stats_hook(layer_id: int):
        def hook(module: nn.Module, inputs, output: torch.Tensor):
            with torch.no_grad():
                t = output.detach()
                # Collapse leading dims so we have (N, D) for per-token norms.
                # K projection output: (B, T_tokens, d_kv) typically.
                orig_shape = tuple(t.shape)
                if t.dim() >= 2:
                    flat = t.reshape(-1, t.shape[-1])
                else:
                    flat = t.reshape(1, -1)
                norms = flat.float().norm(dim=-1)            # (N,)
                n = norms.numel()
                if n == 0:
                    return

                total = norms.sum().clamp_min(1e-9)
                sorted_n = torch.sort(norms, descending=True).values
                k1 = max(1, int(0.01 * n))
                k10 = max(1, int(0.10 * n))
                rec = {
                    "layer_id": layer_id,
                    "call_idx": state["call_idx"],
                    "shape": orig_shape,
                    "n_tokens": n,
                    "k_norm_mean": float(norms.mean().item()),
                    "k_norm_p50": float(norms.median().item()),
                    "k_norm_p95": float(torch.quantile(norms, 0.95).item()),
                    "k_norm_max": float(norms.max().item()),
                    "k_top1pct_mass_frac": float(sorted_n[:k1].sum().item() / total.item()),
                    "k_top10pct_mass_frac": float(sorted_n[:k10].sum().item() / total.item()),
                }
                # Cross-step cosine similarity (only when shape matches the
                # previous call for this layer).
                prev = last_k_per_layer.get(layer_id)
                if prev is not None and prev.shape == flat.shape:
                    rec["cosine_sim_prev"] = float(
                        torch.nn.functional.cosine_similarity(
                            flat.float().flatten(), prev.flatten(), dim=0,
                        ).item()
                    )
                # Store a copy for the next call. Detach + cast to fp16 to
                # keep memory bounded across 35k calls × ~2 MB each.
                last_k_per_layer[layer_id] = flat.float().clone().contiguous()

                records.append(rec)
                state["call_idx"] += 1
        return hook

    def _patched_get_dit(*args, **kwargs):
        dit = _orig_get_dit(*args, **kwargs)
        # Walk modules; find the K projections. MAGI's CustomLayerNormLinear
        # exposes `.q`, `.qx`, `.k`, `.v` submodules.
        layer_id = 0
        n_hooked = 0
        seen_modules = set()
        for name, mod in dit.named_modules():
            if name.endswith(".linear_qkv.k") and id(mod) not in seen_modules:
                mod.register_forward_hook(_stats_hook(layer_id))
                seen_modules.add(id(mod))
                layer_id += 1
                n_hooked += 1
        print(f"  Hooked {n_hooked} K projections.")
        return dit

    # Install the patched factory before the pipeline calls it.
    import inference.pipeline.pipeline as _pl_mod   # type: ignore
    _pl_mod.get_dit = _patched_get_dit

    # ---------------------------------------------------------------- run
    @torch.inference_mode()
    def _generate():
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        pipeline.run_text_to_video(prompt=prompt, output_path="/tmp/magi_profile.mp4")
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e)

    print("Running one T2V generation with K-projection hooks ...")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    ms = _generate()
    wall_s = time.time() - t0

    print(f"  done — {ms:.0f} ms wall, {len(records)} hook records, "
          f"peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    # ---------------------------------------------------------------- summary
    # Aggregate per-layer summary stats across all calls so the output isn't
    # 35k records × 8 fields = unreadable. We keep the raw records too but
    # add a per-layer summary block for fast inspection.
    by_layer: dict[int, list[dict]] = {}
    for r in records:
        by_layer.setdefault(r["layer_id"], []).append(r)

    layer_summary = {}
    for lid, recs in by_layer.items():
        if not recs:
            continue
        n = len(recs)
        layer_summary[lid] = {
            "n_calls": n,
            "k_top1pct_mass_frac_mean":  sum(r["k_top1pct_mass_frac"]  for r in recs) / n,
            "k_top10pct_mass_frac_mean": sum(r["k_top10pct_mass_frac"] for r in recs) / n,
            "k_norm_mean_mean":          sum(r["k_norm_mean"]          for r in recs) / n,
            "k_norm_p95_mean":           sum(r["k_norm_p95"]           for r in recs) / n,
            "cosine_sim_prev_mean": (
                sum(r["cosine_sim_prev"] for r in recs if "cosine_sim_prev" in r)
                / max(1, sum(1 for r in recs if "cosine_sim_prev" in r))
            ),
            "step_count": sum(1 for r in recs if "cosine_sim_prev" in r) + 1,
        }

    return {
        "model": MAGI_REPO,
        "variant": "4.5B_base",
        "kernel": "magi1_attention_profile",
        "num_frames": num_frames,
        "num_steps": num_steps,
        "width": width,
        "height": height,
        "wall_ms": round(ms, 2),
        "wall_s": round(wall_s, 2),
        "n_records": len(records),
        "n_layers_hooked": len(by_layer),
        "vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "gpu": "H100",
        "layer_summary": layer_summary,
        "records": records,
    }


@app.local_entrypoint(name="magi1_45b_attention_profile_modal")
def main(
    num_frames: int = 96,
    num_steps: int = 64,
):
    result = run_magi1_attention_profile.remote(
        num_frames=num_frames, num_steps=num_steps,
    )

    # Strip the bulky records before the on-screen print; full records
    # land in the JSON file.
    print("\nMAGI-1 attention profile — layer summary:")
    import json
    summary_only = {k: v for k, v in result.items() if k != "records"}
    print(json.dumps(summary_only, indent=2)[:4000])

    sys.path.insert(0, "/root/benchmarks")
    from result_store import save_result
    save_result(
        result, "profiling",
        f"magi1_45b_attn_profile_f{num_frames}_s{num_steps}",
    )
