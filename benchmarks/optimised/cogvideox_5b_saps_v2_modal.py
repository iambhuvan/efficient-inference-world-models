"""
CogVideoX-5B + SAPS v2 (TOKEN GATHERING — actually reduces SDPA seq_len).

CRITICAL DIFFERENCE FROM v1:
  v1 used `attn_mask` argument with -inf for pruned positions.
  PROBLEM: SDPA still computes full Q×K^T at original seq_len. Masking only
  saturates softmax, doesn't save compute.

  v2 GATHERS kept K/V tokens, reducing K/V seq_len to S*r. Q stays full
  (all queries still active) but each query attends to S*r keys instead of S.
  Real Q × K^T compute scales as Q × (S*r) = r × baseline.

EXPECTED SPEEDUP:
  Avg keep ratio across 50 steps with r_max=0.7, r_min=0.1: ~0.31
  → Attention compute: 0.31× → 30% of original
  → Attention is ~50% of total → wall-clock: ~0.65× = 1.5× speedup

  Plus memory: kept K/V tensors smaller → ~30% KV memory reduction.

Run:
    PYTHONPATH=benchmarks modal run benchmarks/optimised/cogvideox_5b_saps_v2_modal.py
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
def run_cogvideox_saps_v2(
    r_max: float = 0.7, r_min: float = 0.1,
    num_warmup: int = 1, num_iters: int = 1, seed: int = 42,
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

    class StepCounter:
        def __init__(self): self.step = 0; self.total_steps = 50
        def keep_ratio(self):
            if self.total_steps <= 1: return r_max
            t = min(self.step, self.total_steps - 1)
            return r_max * (r_min / r_max) ** (t / (self.total_steps - 1))
    counter = StepCounter()

    class SAPSv2Processor(CogVideoXAttnProcessor2_0):
        def __init__(self, c): super().__init__(); self.c = c; self.total_kept = 0; self.total_seq = 0

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

            # ── SAPS v2: GATHER kept K/V tokens (actually shrinks seq_len) ──
            r = self.c.keep_ratio()
            n_video = S - text_len
            n_keep_v = max(1, int(round(n_video * r)))

            if n_keep_v < n_video:
                # K-norm importance per video token (head-averaged)
                video_k = k[:, :, text_len:, :]   # (B, H, n_video, D)
                k_norms = video_k.norm(dim=-1).mean(dim=1)  # (B, n_video)
                top = k_norms.topk(n_keep_v, dim=-1).indices  # (B, n_keep_v)

                # Gather kept K/V indices, expanded across heads. Note text
                # tokens always kept — they're the conditioning.
                idx_v = top.unsqueeze(1).unsqueeze(-1).expand(B, attn.heads, n_keep_v, D)
                # We gather from k[:, :, text_len:, :] then concat with text K/V
                video_k_kept = torch.gather(k[:, :, text_len:, :], 2, idx_v)
                video_v_kept = torch.gather(v[:, :, text_len:, :], 2, idx_v)
                k_full = torch.cat([k[:, :, :text_len, :], video_k_kept], dim=2)
                v_full = torch.cat([v[:, :, :text_len, :], video_v_kept], dim=2)
            else:
                k_full = k; v_full = v

            self.total_kept += k_full.shape[2]
            self.total_seq += S

            # SDPA with shrunk K/V — Q × K^T is now Q×(text_len + n_keep_v)
            out = F.scaled_dot_product_attention(q, k_full, v_full, dropout_p=0.0, is_causal=False)
            out = out.transpose(1, 2).reshape(B, -1, attn.heads * D)
            out = attn.to_out[0](out); out = attn.to_out[1](out)
            enc, hid = out.split([text_len, out.size(1) - text_len], dim=1)
            return hid, enc

    procs = []
    for name, mod in pipe.transformer.named_modules():
        if hasattr(mod, "set_processor") and hasattr(mod, "processor"):
            if isinstance(mod.processor, CogVideoXAttnProcessor2_0):
                p = SAPSv2Processor(counter); mod.set_processor(p); procs.append(p)
    print(f"  SAPS v2 (gather) installed on {len(procs)} attention modules")

    def cb(pipe, step_index, timestep, kw): counter.step = step_index; return kw

    n_params = sum(p.numel() for m in pipe.components.values()
                   if isinstance(m, torch.nn.Module) for p in m.parameters())

    @torch.inference_mode()
    def _run(seed_val):
        for p in procs: p.total_kept = 0; p.total_seq = 0
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

    print(f"Schedule: r_max={r_max}, r_min={r_min}, T=50 (TOKEN GATHERING — real compute reduction)")
    print(f"  step  0: r={r_max:.3f}")
    print(f"  step 25: r={r_max*(r_min/r_max)**(25/49):.3f}")
    print(f"  step 49: r={r_min:.3f}")

    print(f"Warmup ({num_warmup}) ...")
    n = 49
    for i in range(num_warmup):
        ms, n = _run(seed + i + 1000); print(f"  warmup {i+1}: {ms:.0f} ms ({n} frames)")
    torch.cuda.reset_peak_memory_stats()
    lat = []
    for i in range(num_iters):
        ms, n = _run(seed + i); lat.append(ms)
        keep = procs[0].total_kept / procs[0].total_seq if procs else 0
        print(f"  iter {i+1}: {ms:.0f} ms ({n/(ms/1000):.3f} fps, avg_keep={keep:.3f})")
    mean_ms = statistics.mean(lat); fps = n / (mean_ms / 1000)
    keep = procs[0].total_kept / procs[0].total_seq if procs else 0
    return {"model": REPO, "kernel": "saps_v2_token_gather",
            "r_max": r_max, "r_min": r_min, "schedule": "exponential_decay_gather",
            "n_attention_modules_patched": len(procs),
            "avg_keep_ratio_observed": round(keep, 4),
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


@app.local_entrypoint(name="cogvideox_5b_saps_v2_modal")
def main(r_max: float = 0.7, r_min: float = 0.1, num_iters: int = 1):
    r = run_cogvideox_saps_v2.remote(r_max=r_max, r_min=r_min, num_iters=num_iters)
    import json; print(json.dumps(r, indent=2))
    print(f"\nSAPS v2 speedup: {r.get('speedup_vs_baseline', 'N/A')}×, "
          f"avg keep: {r.get('avg_keep_ratio_observed', 'N/A')*100:.1f}%")
    sys.path.insert(0, "/root/benchmarks")
    from result_store import save_result
    save_result(r, "optimised_kernels", f"cogvideox_5b_saps_v2_r{r_max}_r{r_min}")
