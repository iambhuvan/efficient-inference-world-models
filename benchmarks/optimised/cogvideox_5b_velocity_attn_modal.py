"""
CogVideoX-5B + Velocity-Aware Attention Truncation — per-token early exit.

Insight: tokens whose latent velocity (||K_curr - K_prev||) is below threshold
drop their attention compute entirely; their attention output is reused from
the previous step. Per-token early exit on the denoising-step axis.

Different from SAPS v2:
  - SAPS v2 prunes K/V (key axis): all queries attend to fewer keys
  - Velocity prunes Q (query axis): some queries skip attention entirely

Compose: SAPS v2 × Velocity = double-axis pruning.

Mechanism: at each attention call, compare current K with cached prev-step K
per token. Tokens with low ΔK reuse cached attention output; high-velocity
tokens compute fresh.
"""
import sys; sys.path.insert(0, "/root"); sys.path.insert(0, "/root/benchmarks")
import modal
from modal_common import app, image_cuda_devel, hf_secret, model_volume, MODEL_CACHE
image = image_cuda_devel
REPO = "zai-org/CogVideoX-5b"; BASELINE_FPS = 0.442
PROMPT = ("A panda, dressed in a small, red jacket and a tiny hat, sits on a "
          "wooden stool in a serene bamboo forest. The panda's fluffy paws "
          "strum a miniature acoustic guitar.")


@app.function(gpu="H100", image=image, secrets=[hf_secret],
              volumes={MODEL_CACHE: model_volume}, timeout=3600, memory=65536)
def run_velocity_attn(
    velocity_threshold: float = 0.05, warmup_steps: int = 5,
    num_warmup: int = 1, num_iters: int = 1, seed: int = 42, prompt: str = PROMPT,
) -> dict:
    import os, statistics, torch
    import torch.nn.functional as F
    from huggingface_hub import snapshot_download
    from diffusers import CogVideoXPipeline
    from diffusers.models.attention_processor import CogVideoXAttnProcessor2_0

    hf_token = os.environ.get("HF_TOKEN", "").strip() or None
    local_dir = snapshot_download(REPO, cache_dir=MODEL_CACHE, token=hf_token)
    pipe = CogVideoXPipeline.from_pretrained(local_dir, torch_dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    class State:
        def __init__(self):
            self.step = 0
            self.q_skipped = 0; self.q_total = 0
        def reset(self):
            self.step = 0; self.q_skipped = 0; self.q_total = 0
    st = State()

    class VelocityProcessor(CogVideoXAttnProcessor2_0):
        def __init__(self):
            super().__init__()
            self.prev_k = None       # cached K from previous step (B, H, S, D)
            self.prev_out = None     # cached attention output (B, S_full, hidden)

        def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                     attention_mask=None, image_rotary_emb=None):
            text_len = encoder_hidden_states.shape[1] if encoder_hidden_states is not None else 0
            hidden = torch.cat([encoder_hidden_states, hidden_states], dim=1) \
                if encoder_hidden_states is not None else hidden_states
            B, S, _ = hidden.shape
            q = attn.to_q(hidden); k = attn.to_k(hidden); v = attn.to_v(hidden)
            inner = k.shape[-1]; H = attn.heads; D = inner // H
            q = q.view(B, S, H, D).transpose(1, 2)
            k = k.view(B, S, H, D).transpose(1, 2)
            v = v.view(B, S, H, D).transpose(1, 2)
            if attn.norm_q is not None: q = attn.norm_q(q)
            if attn.norm_k is not None: k = attn.norm_k(k)
            if image_rotary_emb is not None:
                from diffusers.models.embeddings import apply_rotary_emb
                q[:, :, text_len:] = apply_rotary_emb(q[:, :, text_len:], image_rotary_emb)
                if not attn.is_cross_attention:
                    k[:, :, text_len:] = apply_rotary_emb(k[:, :, text_len:], image_rotary_emb)

            if (st.step >= warmup_steps and self.prev_k is not None and
                self.prev_out is not None and self.prev_k.shape == k.shape):
                # Per-token velocity = ||k_curr - k_prev|| / ||k_prev||
                k_delta = (k - self.prev_k).norm(dim=-1)        # (B, H, S)
                k_prev_norm = self.prev_k.norm(dim=-1).clamp(min=1e-6)
                vel = (k_delta / k_prev_norm).mean(dim=1)        # (B, S) — head-avg
                # Active mask: tokens with high velocity
                active = vel > velocity_threshold                # (B, S) bool
                # Always keep text tokens
                if text_len > 0:
                    active[:, :text_len] = True
                n_active_per_batch = active.sum(dim=-1)          # (B,)
                n_active = int(n_active_per_batch[0].item())
                st.q_total += S; st.q_skipped += (S - n_active)
                if n_active < S and n_active > 0:
                    # Compute attention only on active queries
                    # Index gather active queries (treat batch=1 for simplicity)
                    act_idx = active[0].nonzero(as_tuple=True)[0]   # (n_active,)
                    q_active = q[:, :, act_idx, :]                  # (B, H, n_active, D)
                    out_active = F.scaled_dot_product_attention(
                        q_active, k, v, dropout_p=0.0, is_causal=False)
                    out_active = out_active.transpose(1, 2).reshape(B, n_active, H * D)
                    out_active = attn.to_out[0](out_active); out_active = attn.to_out[1](out_active)
                    # Scatter active outputs into cached prev_out
                    out_full = self.prev_out.clone()
                    out_full[:, act_idx, :] = out_active
                    # Update cache for next call
                    self.prev_k = k.detach(); self.prev_out = out_full.detach()
                    enc, hid = out_full.split([text_len, S - text_len], dim=1)
                    return hid, enc
            # Full attention path (warmup or cache miss)
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
            out = out.transpose(1, 2).reshape(B, S, H * D)
            out = attn.to_out[0](out); out = attn.to_out[1](out)
            self.prev_k = k.detach(); self.prev_out = out.detach()
            st.q_total += S
            enc, hid = out.split([text_len, S - text_len], dim=1)
            return hid, enc

    procs = []
    for name, mod in pipe.transformer.named_modules():
        if hasattr(mod, "set_processor") and hasattr(mod, "processor"):
            if isinstance(mod.processor, CogVideoXAttnProcessor2_0):
                p = VelocityProcessor(); mod.set_processor(p); procs.append(p)
    print(f"  velocity-attn processor installed on {len(procs)} attn modules (τ={velocity_threshold})")

    def cb(pipe, step_index, timestep, kw): st.step = step_index; return kw
    n_params = sum(p.numel() for m in pipe.components.values()
                   if isinstance(m, torch.nn.Module) for p in m.parameters())

    @torch.inference_mode()
    def _run(seed_val):
        st.reset()
        for p in procs: p.prev_k = None; p.prev_out = None
        gen = torch.Generator(device="cuda").manual_seed(seed_val)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        out = pipe(prompt=prompt, num_frames=49, num_inference_steps=50,
                   width=720, height=480, guidance_scale=6.0,
                   generator=gen, return_dict=True, callback_on_step_end=cb)
        e.record(); torch.cuda.synchronize()
        return s.elapsed_time(e), _frames(getattr(out, "frames", None))

    print(f"Warmup ({num_warmup}) ...")
    n = 49
    for i in range(num_warmup):
        ms, n = _run(seed + i + 1000); print(f"  warmup {i+1}: {ms:.0f} ms ({n} frames)")
    torch.cuda.reset_peak_memory_stats()
    lat = []
    for i in range(num_iters):
        ms, n = _run(seed + i); lat.append(ms)
        skip_pct = 100.0 * st.q_skipped / max(st.q_total, 1)
        print(f"  iter {i+1}: {ms:.0f} ms ({n/(ms/1000):.3f} fps, q_skip={skip_pct:.1f}%)")
    mean_ms = statistics.mean(lat); fps = n / (mean_ms / 1000)
    skip_pct = 100.0 * st.q_skipped / max(st.q_total, 1)
    return {"model": REPO, "kernel": "velocity_aware_attn_truncation",
            "velocity_threshold": velocity_threshold, "warmup_steps": warmup_steps,
            "n_attention_modules_patched": len(procs),
            "query_skip_pct": round(skip_pct, 2),
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


@app.local_entrypoint(name="cogvideox_5b_velocity_attn_modal")
def main(velocity_threshold: float = 0.05, num_iters: int = 1):
    r = run_velocity_attn.remote(velocity_threshold=velocity_threshold, num_iters=num_iters)
    import json; print(json.dumps(r, indent=2))
    print(f"\nVelocity-attn speedup: {r.get('speedup_vs_baseline', 'N/A')}×")
    sys.path.insert(0, "/root/benchmarks"); from result_store import save_result
    save_result(r, "optimised_kernels", "cogvideox_5b_velocity_attn")
