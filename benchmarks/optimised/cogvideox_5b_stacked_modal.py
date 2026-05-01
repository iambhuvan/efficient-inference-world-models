"""
CogVideoX-5B + STACKED: torch.compile + SAPS v2 (token gather) + Step-aware block skip.

Predicted compose:
  compile        : 1.259× (measured)
  SAPS v2        : 1.408× (measured)
  block skip     : 1.384× (measured)

  SAPS + block skip don't multiply cleanly (both reduce attention compute).
  Realistic: 1.7-2.1× from SAPS+block skip. Then compile adds 1.15-1.25×.
  Ceiling: ~2.2-2.6× over baseline = 0.97-1.15 FPS.

Ordering: install SAPS v2 processor + wrap block forwards FIRST, then
torch.compile so compile traces through the modifications.

Run:
    PYTHONPATH=benchmarks modal run benchmarks/optimised/cogvideox_5b_stacked_modal.py
"""

import sys
sys.path.insert(0, "/root"); sys.path.insert(0, "/root/benchmarks")

import modal
from modal_common import app, image_cuda_devel, hf_secret, model_volume, MODEL_CACHE
image = image_cuda_devel

REPO = "zai-org/CogVideoX-5b"
BASELINE_FPS = 0.442
PROMPT = ("A panda, dressed in a small, red jacket and a tiny hat, sits on a "
          "wooden stool in a serene bamboo forest. The panda's fluffy paws "
          "strum a miniature acoustic guitar.")


@app.function(gpu="H100", image=image, secrets=[hf_secret],
              volumes={MODEL_CACHE: model_volume}, timeout=3600, memory=65536)
def run_cogvideox_stacked(
    r_max: float = 0.7, r_min: float = 0.1,
    skip_start_step: int = 10, skip_mid_step: int = 30,
    skip_period_mid: int = 4, skip_period_late: int = 2,
    enable_compile: bool = True, compile_mode: str = "reduce-overhead",
    num_warmup: int = 2,        # extra warmup for compile JIT
    num_iters: int = 1, seed: int = 42,
    prompt: str = PROMPT,
) -> dict:
    import os, statistics
    import torch
    import torch.nn.functional as F
    from huggingface_hub import snapshot_download
    from diffusers import CogVideoXPipeline
    from diffusers.models.attention_processor import CogVideoXAttnProcessor2_0

    hf_token = os.environ.get("HF_TOKEN", "").strip() or None
    local_dir = snapshot_download(REPO, cache_dir=MODEL_CACHE, token=hf_token)
    pipe = CogVideoXPipeline.from_pretrained(local_dir, torch_dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    # ── Shared step counter ─────────────────────────────────────────────
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

    # ── Layer 1: SAPS v2 token-gather processor ─────────────────────────
    class SAPSv2Processor(CogVideoXAttnProcessor2_0):
        def __init__(self, c): super().__init__(); self.c = c

        def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                     attention_mask=None, image_rotary_emb=None):
            text_len = encoder_hidden_states.shape[1] if encoder_hidden_states is not None else 0
            hidden = torch.cat([encoder_hidden_states, hidden_states], dim=1) \
                if encoder_hidden_states is not None else hidden_states
            B, S, _ = hidden.shape
            q = attn.to_q(hidden); k = attn.to_k(hidden); v = attn.to_v(hidden)
            inner = k.shape[-1]; D = inner // attn.heads
            q = q.view(B, -1, attn.heads, D).transpose(1, 2)
            k = k.view(B, -1, attn.heads, D).transpose(1, 2)
            v = v.view(B, -1, attn.heads, D).transpose(1, 2)
            if attn.norm_q is not None: q = attn.norm_q(q)
            if attn.norm_k is not None: k = attn.norm_k(k)
            if image_rotary_emb is not None:
                from diffusers.models.embeddings import apply_rotary_emb
                q[:, :, text_len:] = apply_rotary_emb(q[:, :, text_len:], image_rotary_emb)
                if not attn.is_cross_attention:
                    k[:, :, text_len:] = apply_rotary_emb(k[:, :, text_len:], image_rotary_emb)
            r = self.c.keep_ratio()
            n_video = S - text_len
            n_keep_v = max(1, int(round(n_video * r)))
            if n_keep_v < n_video:
                video_k = k[:, :, text_len:, :]
                k_norms = video_k.norm(dim=-1).mean(dim=1)
                top = k_norms.topk(n_keep_v, dim=-1).indices
                idx_v = top.unsqueeze(1).unsqueeze(-1).expand(B, attn.heads, n_keep_v, D)
                video_k_kept = torch.gather(k[:, :, text_len:, :], 2, idx_v)
                video_v_kept = torch.gather(v[:, :, text_len:, :], 2, idx_v)
                k_full = torch.cat([k[:, :, :text_len, :], video_k_kept], dim=2)
                v_full = torch.cat([v[:, :, :text_len, :], video_v_kept], dim=2)
            else:
                k_full = k; v_full = v
            out = F.scaled_dot_product_attention(q, k_full, v_full, dropout_p=0.0, is_causal=False)
            out = out.transpose(1, 2).reshape(B, -1, attn.heads * D)
            out = attn.to_out[0](out); out = attn.to_out[1](out)
            enc, hid = out.split([text_len, out.size(1) - text_len], dim=1)
            return hid, enc

    # Install SAPS v2 on every attention module
    procs = []
    for name, mod in pipe.transformer.named_modules():
        if hasattr(mod, "set_processor") and hasattr(mod, "processor"):
            if isinstance(mod.processor, CogVideoXAttnProcessor2_0):
                p = SAPSv2Processor(counter); mod.set_processor(p); procs.append(p)
    print(f"  Layer 1: SAPS v2 installed on {len(procs)} attn modules")

    # ── Layer 2: Step-aware block skip ──────────────────────────────────
    blocks = pipe.transformer.transformer_blocks
    skip_stats = {"called": 0, "skipped": 0}
    for idx, block in enumerate(blocks):
        original_forward = block.forward
        b_idx = idx
        def make_wrapped(orig_fwd, bi):
            def wrapped(hidden_states, encoder_hidden_states, temb,
                        image_rotary_emb=None, attention_kwargs=None, **kwargs):
                period = counter.skip_period()
                skip_stats["called"] += 1
                if period > 0 and (bi + counter.step) % period == 0:
                    skip_stats["skipped"] += 1
                    return hidden_states, encoder_hidden_states
                return orig_fwd(hidden_states, encoder_hidden_states, temb,
                                image_rotary_emb=image_rotary_emb,
                                attention_kwargs=attention_kwargs, **kwargs)
            return wrapped
        block.forward = make_wrapped(original_forward, b_idx)
    print(f"  Layer 2: block skip wired into {len(blocks)} blocks")

    # ── Layer 3: torch.compile (after SAPS v2 + block skip in place) ────
    if enable_compile:
        print(f"  Layer 3: torch.compile(mode='{compile_mode}')")
        pipe.transformer = torch.compile(pipe.transformer, mode=compile_mode, dynamic=True)

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
        out = pipe(prompt=prompt, num_frames=49, num_inference_steps=50,
                   width=720, height=480, guidance_scale=6.0,
                   generator=gen, return_dict=True, callback_on_step_end=cb)
        e.record(); torch.cuda.synchronize()
        return s.elapsed_time(e), _frames(getattr(out, "frames", None))

    print(f"Schedule: SAPS r_max={r_max} → r_min={r_min}, "
          f"block_skip starts at step {skip_start_step}, "
          f"compile={enable_compile}({compile_mode if enable_compile else 'off'})")

    print(f"Warmup ({num_warmup}; first triggers compile JIT) ...")
    n = 49
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
    return {"model": REPO, "kernel": "stacked_compile_saps_v2_block_skip",
            "saps_r_max": r_max, "saps_r_min": r_min,
            "skip_start_step": skip_start_step, "skip_mid_step": skip_mid_step,
            "skip_period_mid": skip_period_mid, "skip_period_late": skip_period_late,
            "enable_compile": enable_compile, "compile_mode": compile_mode,
            "n_attention_modules_patched": len(procs), "n_blocks": len(blocks),
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


@app.local_entrypoint(name="cogvideox_5b_stacked_modal")
def main(num_iters: int = 1, enable_compile: bool = True):
    r = run_cogvideox_stacked.remote(num_iters=num_iters, enable_compile=enable_compile)
    import json; print(json.dumps(r, indent=2))
    print(f"\nStacked speedup: {r.get('speedup_vs_baseline', 'N/A')}×")
    sys.path.insert(0, "/root/benchmarks")
    from result_store import save_result
    save_result(r, "optimised_kernels", "cogvideox_5b_stacked")
