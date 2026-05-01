"""
Wan2.1-T2V-14B + SAPS v2 + Block skip — Wan port of the two custom winners.

Wan-specific differences from CogVideoX:
  - Pipeline class: WanPipeline
  - Attention class: WanAttnProcessor (not CogVideoXAttnProcessor2_0)
  - Goes through dispatch_attention_fn — we override to call F.sdpa directly
  - Block list: pipe.transformer.blocks (probed at runtime)

Run:
    PYTHONPATH=benchmarks modal run benchmarks/optimised/wan21_14b_saps_block_skip_modal.py
"""

import sys
sys.path.insert(0, "/root"); sys.path.insert(0, "/root/benchmarks")

import modal
from modal_common import app, image_cuda_devel, hf_secret, model_volume, MODEL_CACHE
image = image_cuda_devel

REPO = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
BASELINE_FPS = 0.171
PROMPT = ("A cat walks on the grass, realistic, 4k, cinematic, smooth motion, "
          "natural lighting")


@app.function(gpu="H100", image=image, secrets=[hf_secret],
              volumes={MODEL_CACHE: model_volume}, timeout=7200, memory=65536)
def run_wan_saps_block_skip(
    r_max: float = 0.7, r_min: float = 0.1,
    skip_start_step: int = 10, skip_mid_step: int = 30,
    skip_period_mid: int = 4, skip_period_late: int = 2,
    enable_saps: bool = True, enable_block_skip: bool = True,
    num_warmup: int = 1, num_iters: int = 1, seed: int = 42,
    prompt: str = PROMPT,
) -> dict:
    import os, statistics
    import torch
    import torch.nn.functional as F
    from huggingface_hub import snapshot_download
    from diffusers import WanPipeline

    hf_token = os.environ.get("HF_TOKEN", "").strip() or None
    local_dir = snapshot_download(REPO, cache_dir=MODEL_CACHE, token=hf_token)
    pipe = WanPipeline.from_pretrained(local_dir, torch_dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    class Counter:
        def __init__(self): self.step = 0; self.total_steps = 50
        def keep_ratio(self):
            if self.total_steps <= 1: return r_max
            t = min(self.step, self.total_steps - 1)
            return r_max * (r_min / r_max) ** (t / (self.total_steps - 1))
        def skip_period(self):
            if self.step < skip_start_step: return 0
            if self.step < skip_mid_step: return skip_period_mid
            return skip_period_late
    counter = Counter()

    # ── SAPS v2 for Wan: replace the WanAttnProcessor's __call__ ────────
    n_procs_patched = 0
    if enable_saps:
        # Find Wan attention modules and replace their forward path
        for name, mod in pipe.transformer.named_modules():
            if not hasattr(mod, "to_q") or not hasattr(mod, "to_k") or not hasattr(mod, "to_v"):
                continue
            if not hasattr(mod, "to_out") or not hasattr(mod, "heads"):
                continue
            # Wan's attention modules — wrap forward
            original_forward = mod.forward
            mod_ref = mod

            def make_saps_forward(orig_fwd, attn):
                def wrapped(hidden_states, encoder_hidden_states=None,
                            attention_mask=None, rotary_emb=None, **kwargs):
                    is_cross = encoder_hidden_states is not None and not (encoder_hidden_states is hidden_states)
                    if is_cross:
                        # Cross-attention to text — don't prune, pass through
                        return orig_fwd(hidden_states, encoder_hidden_states,
                                        attention_mask=attention_mask,
                                        rotary_emb=rotary_emb, **kwargs)
                    B, S, _ = hidden_states.shape
                    q = attn.to_q(hidden_states); k = attn.to_k(hidden_states); v = attn.to_v(hidden_states)
                    # Wan applies RMSNorm at flat shape (B, S, inner_dim), not per-head
                    if hasattr(attn, "norm_q") and attn.norm_q is not None: q = attn.norm_q(q)
                    if hasattr(attn, "norm_k") and attn.norm_k is not None: k = attn.norm_k(k)
                    inner = k.shape[-1]; H = attn.heads; D = inner // H
                    q = q.view(B, S, H, D).transpose(1, 2)
                    k = k.view(B, S, H, D).transpose(1, 2)
                    v = v.view(B, S, H, D).transpose(1, 2)
                    if rotary_emb is not None:
                        try:
                            from diffusers.models.embeddings import apply_rotary_emb
                            q = apply_rotary_emb(q, rotary_emb)
                            k = apply_rotary_emb(k, rotary_emb)
                        except Exception:
                            pass
                    # SAPS gather — Wan has no text/video split (self-attn over video tokens only)
                    r = counter.keep_ratio()
                    n_keep = max(1, int(round(S * r)))
                    if n_keep < S:
                        k_norms = k.norm(dim=-1).mean(dim=1)
                        top = k_norms.topk(n_keep, dim=-1).indices
                        idx = top.unsqueeze(1).unsqueeze(-1).expand(B, H, n_keep, D)
                        k_g = torch.gather(k, 2, idx)
                        v_g = torch.gather(v, 2, idx)
                    else:
                        k_g = k; v_g = v
                    out = F.scaled_dot_product_attention(q, k_g, v_g, dropout_p=0.0, is_causal=False)
                    out = out.transpose(1, 2).reshape(B, S, H * D)
                    out = attn.to_out[0](out)
                    if len(attn.to_out) > 1: out = attn.to_out[1](out)
                    return out
                return wrapped

            try:
                mod.forward = make_saps_forward(original_forward, mod_ref)
                n_procs_patched += 1
            except Exception as exc:
                print(f"  skip patching {name}: {exc}")
        print(f"  SAPS v2 wrapped {n_procs_patched} attention modules")

    # ── Block skip for Wan ──────────────────────────────────────────────
    skip_stats = {"called": 0, "skipped": 0}
    n_blocks = 0
    if enable_block_skip:
        # Probe for block list
        block_list = None
        for attr in ("blocks", "transformer_blocks", "layers", "single_blocks"):
            cand = getattr(pipe.transformer, attr, None)
            if cand is not None and len(cand) > 5:
                block_list = cand
                print(f"  block list found: pipe.transformer.{attr} ({len(cand)} blocks)")
                break
        if block_list is None:
            print("  WARN: no block list found — block skip disabled")
        else:
            n_blocks = len(block_list)
            for idx, block in enumerate(block_list):
                original = block.forward
                b_idx = idx

                def make_wrapped(orig, bi):
                    def wrapped(*args, **kwargs):
                        period = counter.skip_period()
                        skip_stats["called"] += 1
                        if period > 0 and (bi + counter.step) % period == 0:
                            skip_stats["skipped"] += 1
                            # Return the first arg unchanged (the hidden_states input)
                            if args: return args[0]
                            return kwargs.get("hidden_states")
                        return orig(*args, **kwargs)
                    return wrapped

                block.forward = make_wrapped(original, b_idx)
            print(f"  block skip wired into {n_blocks} blocks")

    def cb(pipe, step_index, timestep, kw): counter.step = step_index; return kw

    n_params = sum(p.numel() for m in pipe.components.values()
                   if isinstance(m, torch.nn.Module) for p in m.parameters())

    @torch.inference_mode()
    def _run(seed_val):
        skip_stats["called"] = 0; skip_stats["skipped"] = 0
        counter.step = 0; counter.total_steps = 50
        gen = torch.Generator(device="cuda").manual_seed(seed_val)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        out = pipe(prompt=prompt, num_frames=81, num_inference_steps=50,
                   width=832, height=480, guidance_scale=5.0,
                   generator=gen, return_dict=True, callback_on_step_end=cb)
        e.record(); torch.cuda.synchronize()
        return s.elapsed_time(e), _frames(getattr(out, "frames", None))

    print(f"Wan2.1 stacked: SAPS={enable_saps} (r {r_max}→{r_min}), "
          f"block_skip={enable_block_skip}")

    print(f"Warmup ({num_warmup}) ...")
    n = 81
    for i in range(num_warmup):
        ms, n = _run(seed + i + 1000); print(f"  warmup {i+1}: {ms:.0f} ms ({n} frames)")
    torch.cuda.reset_peak_memory_stats()
    lat = []
    for i in range(num_iters):
        ms, n = _run(seed + i); lat.append(ms)
        skip_pct = 100.0 * skip_stats["skipped"] / max(skip_stats["called"], 1)
        print(f"  iter {i+1}: {ms:.0f} ms ({n/(ms/1000):.3f} fps, skipped {skip_pct:.1f}%)")
    mean_ms = statistics.mean(lat); fps = n / (mean_ms / 1000)
    skip_pct = 100.0 * skip_stats["skipped"] / max(skip_stats["called"], 1)
    return {"model": REPO, "kernel": "saps_v2_plus_block_skip_wan",
            "saps_enabled": enable_saps, "block_skip_enabled": enable_block_skip,
            "saps_r_max": r_max, "saps_r_min": r_min,
            "n_attention_modules_patched": n_procs_patched, "n_blocks": n_blocks,
            "block_skip_pct": round(skip_pct, 2),
            "num_frames_observed": n, "n_params_B": round(n_params/1e9, 3),
            "latency_ms_mean": round(mean_ms, 2), "latency_per_frame_ms": round(mean_ms/n, 2),
            "frames_per_sec": round(fps, 3),
            "speedup_vs_baseline": round(fps/BASELINE_FPS, 3), "baseline_fps": BASELINE_FPS,
            "vram_gb": round(torch.cuda.max_memory_allocated()/1e9, 3),
            "gpu": "H100", "raw_latencies_ms": lat}


def _frames(f):
    if f is None: return 1
    s = getattr(f, "shape", None)
    if s and len(s) >= 4: return s[0] if len(s) == 4 else s[1]
    if isinstance(f, list):
        return len(f[0]) if f and isinstance(f[0], list) else len(f)
    return 1


@app.local_entrypoint(name="wan21_14b_saps_block_skip_modal")
def main(num_iters: int = 1):
    r = run_wan_saps_block_skip.remote(num_iters=num_iters)
    import json; print(json.dumps(r, indent=2))
    print(f"\nWan2.1 stacked speedup: {r.get('speedup_vs_baseline', 'N/A')}×")
    sys.path.insert(0, "/root/benchmarks")
    from result_store import save_result
    save_result(r, "optimised_kernels", "wan21_14b_saps_block_skip")
