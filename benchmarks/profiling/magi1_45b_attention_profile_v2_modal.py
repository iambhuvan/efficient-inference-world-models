"""
MAGI-1 attention profiler — v2: adds T5 cross-attn vs self/temporal split.

Beyond v1 (which only hooked `linear_qkv.k`), this run hooks both:
  - `linear_qkv.k` → self/temporal K projection (per layer)
  - `linear_kv_xattn` → T5 cross-attention concatenated K|V (per layer)

This lets us answer: of the per-layer attention compute, how much is
self/temporal (large, can be broadcast) vs cross-T5 (smaller, fixed
prompt — usually cheaper to recompute)?  The answer drives whether we
should broadcast both paths uniformly or only self/temporal.

Same statistics as v1: norm distribution, top-1% / top-10% mass,
cosine_sim_prev. New per-record field: `path` ∈ {"self_temporal", "t5_xattn"}.

Run:
    PYTHONPATH=benchmarks modal run benchmarks/profiling/magi1_45b_attention_profile_v2_modal.py
"""

import sys
sys.path.insert(0, "/root")
sys.path.insert(0, "/root/benchmarks")

import os

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
def run_magi1_attention_profile_v2(
    num_frames: int = 96,
    num_steps: int = 64,
    width: int = 720,
    height: int = 720,
    seed: int = 42,
    prompt: str = DEFAULT_PROMPT,
) -> dict:
    """One T2V generation; hooks linear_qkv.k AND linear_kv_xattn per layer."""
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
    patched = "/tmp/magi1_attn_profile_v2.json"
    with open(patched, "w") as fh:
        json.dump(cfg, fh, indent=2)

    os.chdir("/root/MAGI-1")
    sys.path.insert(0, "/root/MAGI-1")
    from inference.pipeline import MagiPipeline  # type: ignore

    print("Constructing MagiPipeline ...")
    pipeline = MagiPipeline(patched)

    records: list[dict] = []
    last_per_layer: dict[tuple[int, str], torch.Tensor] = {}
    state = {"call_idx": 0}

    def _stats_hook(layer_id: int, path: str):
        def hook(module, inputs, output):
            with torch.no_grad():
                t = output.detach()
                if t.dim() >= 2:
                    flat = t.reshape(-1, t.shape[-1])
                else:
                    flat = t.reshape(1, -1)
                norms = flat.float().norm(dim=-1)
                n = norms.numel()
                if n == 0:
                    return
                total = norms.sum().clamp_min(1e-9)
                sorted_n = torch.sort(norms, descending=True).values
                k1, k10 = max(1, n // 100), max(1, n // 10)
                rec = {
                    "layer_id": layer_id,
                    "path": path,
                    "call_idx": state["call_idx"],
                    "shape": tuple(t.shape),
                    "n_tokens": n,
                    "norm_mean": float(norms.mean().item()),
                    "norm_p50": float(norms.median().item()),
                    "norm_p95": float(torch.quantile(norms, 0.95).item()),
                    "norm_max": float(norms.max().item()),
                    "top1pct_mass_frac":  float(sorted_n[:k1].sum().item() / total.item()),
                    "top10pct_mass_frac": float(sorted_n[:k10].sum().item() / total.item()),
                }
                key = (layer_id, path)
                prev = last_per_layer.get(key)
                if prev is not None and prev.shape == flat.shape:
                    rec["cosine_sim_prev"] = float(
                        torch.nn.functional.cosine_similarity(
                            flat.float().flatten(), prev.flatten(), dim=0,
                        ).item()
                    )
                last_per_layer[key] = flat.float().clone().contiguous()
                records.append(rec)
                state["call_idx"] += 1
        return hook

    from inference.model.dit import get_dit as _orig_get_dit  # type: ignore

    def _patched_get_dit(*args, **kwargs):
        dit = _orig_get_dit(*args, **kwargs)
        layer_id_qkv = 0
        layer_id_xattn = 0
        n_qkv = n_xattn = 0
        seen = set()
        # Two passes so qkv and xattn each get a clean 0..33 layer_id range.
        for name, mod in dit.named_modules():
            if name.endswith(".linear_qkv.k") and id(mod) not in seen:
                mod.register_forward_hook(_stats_hook(layer_id_qkv, "self_temporal"))
                seen.add(id(mod))
                layer_id_qkv += 1
                n_qkv += 1
        seen.clear()
        for name, mod in dit.named_modules():
            if name.endswith(".linear_kv_xattn") and id(mod) not in seen:
                mod.register_forward_hook(_stats_hook(layer_id_xattn, "t5_xattn"))
                seen.add(id(mod))
                layer_id_xattn += 1
                n_xattn += 1
        print(f"  Hooked {n_qkv} self/temporal K + {n_xattn} T5 cross-attn KV.")
        return dit

    import inference.pipeline.pipeline as _pl_mod   # type: ignore
    _pl_mod.get_dit = _patched_get_dit

    @torch.inference_mode()
    def _generate():
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        pipeline.run_text_to_video(prompt=prompt, output_path="/tmp/magi_profile_v2.mp4")
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e)

    print("Generating with hooks ...")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    ms = _generate()
    wall_s = time.time() - t0
    print(f"  done — {ms:.0f} ms wall, {len(records)} records")

    # Per-layer × per-path summary
    by = {}
    for r in records:
        by.setdefault((r["layer_id"], r["path"]), []).append(r)

    summary = {}
    for (lid, path), recs in by.items():
        n = len(recs)
        cs = [r["cosine_sim_prev"] for r in recs if "cosine_sim_prev" in r]
        summary[f"L{lid}_{path}"] = {
            "n_calls": n,
            "n_tokens_sample": recs[0]["n_tokens"],
            "top1pct_mass_frac_mean":  sum(r["top1pct_mass_frac"]  for r in recs) / n,
            "top10pct_mass_frac_mean": sum(r["top10pct_mass_frac"] for r in recs) / n,
            "norm_mean_mean":          sum(r["norm_mean"]          for r in recs) / n,
            "cosine_sim_prev_mean":    (sum(cs) / max(1, len(cs))) if cs else None,
        }

    return {
        "model": MAGI_REPO,
        "variant": "4.5B_base",
        "kernel": "magi1_attention_profile_v2",
        "num_frames": num_frames,
        "num_steps": num_steps,
        "wall_ms": round(ms, 2),
        "wall_s": round(wall_s, 2),
        "n_records": len(records),
        "vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "gpu": "H100",
        "summary": summary,
        "records": records,
    }


@app.local_entrypoint(name="magi1_45b_attention_profile_v2_modal")
def main(num_frames: int = 96, num_steps: int = 64):
    result = run_magi1_attention_profile_v2.remote(
        num_frames=num_frames, num_steps=num_steps,
    )
    sys.path.insert(0, "/root/benchmarks")
    from result_store import save_result
    save_result(
        result, "profiling",
        f"magi1_45b_attn_profile_v2_f{num_frames}_s{num_steps}",
    )
    import json
    summary_only = {k: v for k, v in result.items() if k != "records"}
    print("\n" + json.dumps(summary_only, indent=2)[:6000])
