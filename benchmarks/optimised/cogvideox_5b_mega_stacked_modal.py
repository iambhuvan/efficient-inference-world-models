"""
CogVideoX-5B + MEGA STACK: per-layer Taylor (constant h) + SAPS v2 + velocity-attn.

Composes the three top single-layer custom contributions on orthogonal axes:
  - Per-Layer Taylor h=6 (block-output cache axis): 3.975×
  - SAPS v2 (attention-input K/V token-gather axis): 1.408×
  - Velocity-Aware Attention @ 0.10 (query-axis prune): 1.606×

When block-cache hits (frequent at h=6), block forward is skipped entirely —
SAPS and velocity don't fire. When block forward runs, the wrapped attn
processor does both SAPS K/V gather and velocity Q-skip.

Predicted ceiling: ~5-7× if mostly orthogonal. Sub-multiplicative if attention
savings overlap.
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
def run_mega_stacked(
    horizon: int = 6, warmup_cache: int = 3,
    saps_r_max: float = 0.7, saps_r_min: float = 0.1,
    velocity_threshold: float = 0.10, warmup_velocity: int = 5,
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

    class Counter:
        def __init__(self):
            self.step = 0; self.total_steps = 50
            # Per-block cache state
            self.cache_hits = 0; self.cache_misses = 0
            self.prev_out: dict[int, tuple] = {}
            self.prev2_out: dict[int, tuple] = {}
            self.last_refresh: dict[int, int] = {}
            # Velocity stats
            self.q_skipped = 0; self.q_total = 0
        def saps_keep_ratio(self):
            t = min(self.step, self.total_steps - 1)
            return saps_r_max * (saps_r_min / saps_r_max) ** (t / max(self.total_steps - 1, 1))
    counter = Counter()

    # ── Layer 1+2 fused: SAPS v2 token gather + Velocity-Aware Q-skip ───
    class FusedSAPSVelocityProcessor(CogVideoXAttnProcessor2_0):
        def __init__(self):
            super().__init__()
            self.prev_k = None; self.prev_out_full = None
        def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                     attention_mask=None, image_rotary_emb=None):
            text_len = encoder_hidden_states.shape[1] if encoder_hidden_states is not None else 0
            hidden = torch.cat([encoder_hidden_states, hidden_states], dim=1) \
                if encoder_hidden_states is not None else hidden_states
            B, S, _ = hidden.shape
            q = attn.to_q(hidden); k = attn.to_k(hidden); v = attn.to_v(hidden)
            inner = k.shape[-1]; H = attn.heads; D = inner // H
            q = q.view(B, -1, H, D).transpose(1, 2)
            k = k.view(B, -1, H, D).transpose(1, 2)
            v = v.view(B, -1, H, D).transpose(1, 2)
            if attn.norm_q is not None: q = attn.norm_q(q)
            if attn.norm_k is not None: k = attn.norm_k(k)
            if image_rotary_emb is not None:
                from diffusers.models.embeddings import apply_rotary_emb
                q[:, :, text_len:] = apply_rotary_emb(q[:, :, text_len:], image_rotary_emb)
                if not attn.is_cross_attention:
                    k[:, :, text_len:] = apply_rotary_emb(k[:, :, text_len:], image_rotary_emb)

            # Velocity-aware Q-skip path (when prev cached + warmup done)
            if (counter.step >= warmup_velocity and self.prev_k is not None and
                self.prev_out_full is not None and self.prev_k.shape == k.shape):
                k_delta = (k - self.prev_k).norm(dim=-1)
                k_prev_norm = self.prev_k.norm(dim=-1).clamp(min=1e-6)
                vel = (k_delta / k_prev_norm).mean(dim=1)
                active = vel > velocity_threshold
                if text_len > 0: active[:, :text_len] = True
                n_active = int(active[0].sum().item())
                counter.q_total += S; counter.q_skipped += (S - n_active)
                if 0 < n_active < S:
                    act_idx = active[0].nonzero(as_tuple=True)[0]
                    q_act = q[:, :, act_idx, :]
                    # SAPS K/V gather on full sequence
                    r = counter.saps_keep_ratio()
                    n_video = S - text_len
                    n_keep_v = max(1, int(round(n_video * r)))
                    if n_keep_v < n_video:
                        video_k = k[:, :, text_len:, :]
                        k_norms = video_k.norm(dim=-1).mean(dim=1)
                        top = k_norms.topk(n_keep_v, dim=-1).indices
                        idx_v = top.unsqueeze(1).unsqueeze(-1).expand(B, H, n_keep_v, D)
                        vk = torch.gather(k[:, :, text_len:, :], 2, idx_v)
                        vv = torch.gather(v[:, :, text_len:, :], 2, idx_v)
                        k_use = torch.cat([k[:, :, :text_len, :], vk], dim=2)
                        v_use = torch.cat([v[:, :, :text_len, :], vv], dim=2)
                    else:
                        k_use = k; v_use = v
                    out_act = F.scaled_dot_product_attention(q_act, k_use, v_use,
                                                             dropout_p=0.0, is_causal=False)
                    out_act = out_act.transpose(1, 2).reshape(B, n_active, H * D)
                    out_act = attn.to_out[0](out_act); out_act = attn.to_out[1](out_act)
                    out_full = self.prev_out_full.clone()
                    out_full[:, act_idx, :] = out_act
                    self.prev_k = k.detach(); self.prev_out_full = out_full.detach()
                    enc, hid = out_full.split([text_len, S - text_len], dim=1)
                    return hid, enc

            # Full path with SAPS gather only (no velocity skip)
            r = counter.saps_keep_ratio()
            n_video = S - text_len
            n_keep_v = max(1, int(round(n_video * r)))
            if n_keep_v < n_video:
                video_k = k[:, :, text_len:, :]
                k_norms = video_k.norm(dim=-1).mean(dim=1)
                top = k_norms.topk(n_keep_v, dim=-1).indices
                idx_v = top.unsqueeze(1).unsqueeze(-1).expand(B, H, n_keep_v, D)
                vk = torch.gather(k[:, :, text_len:, :], 2, idx_v)
                vv = torch.gather(v[:, :, text_len:, :], 2, idx_v)
                k_use = torch.cat([k[:, :, :text_len, :], vk], dim=2)
                v_use = torch.cat([v[:, :, :text_len, :], vv], dim=2)
            else:
                k_use = k; v_use = v
            out = F.scaled_dot_product_attention(q, k_use, v_use, dropout_p=0.0, is_causal=False)
            out = out.transpose(1, 2).reshape(B, S, H * D)
            out = attn.to_out[0](out); out = attn.to_out[1](out)
            self.prev_k = k.detach(); self.prev_out_full = out.detach()
            counter.q_total += S
            enc, hid = out.split([text_len, S - text_len], dim=1)
            return hid, enc

    procs = []
    for name, mod in pipe.transformer.named_modules():
        if hasattr(mod, "set_processor") and hasattr(mod, "processor"):
            if isinstance(mod.processor, CogVideoXAttnProcessor2_0):
                p = FusedSAPSVelocityProcessor(); mod.set_processor(p); procs.append(p)
    print(f"  fused SAPS+velocity processor on {len(procs)} attn modules")

    # ── Layer 3: Per-layer Taylor cache (constant horizon) wrapping blocks ─
    blocks = pipe.transformer.transformer_blocks
    print(f"  block-cache (constant h={horizon}) wrapped on {len(blocks)} blocks")

    for idx, block in enumerate(blocks):
        original_forward = block.forward; bi = idx
        def make_fwd(orig_fwd, bii, h):
            def wrapped(hidden_states, encoder_hidden_states, temb, image_rotary_emb=None,
                        attention_kwargs=None, **kwargs):
                last = counter.last_refresh.get(bii, -1)
                steps_since = counter.step - last
                if (counter.step >= warmup_cache and steps_since < h and last >= 0 and
                    bii in counter.prev_out and bii in counter.prev2_out):
                    ph, pe = counter.prev_out[bii]
                    p2h, p2e = counter.prev2_out[bii]
                    delta_h = ph - p2h; delta_e = pe - p2e
                    pred_h = ph + delta_h * (steps_since / max(1, h - 1))
                    pred_e = pe + delta_e * (steps_since / max(1, h - 1))
                    counter.cache_hits += 1
                    return pred_h, pred_e
                counter.cache_misses += 1
                out_h, out_e = orig_fwd(hidden_states, encoder_hidden_states, temb,
                                        image_rotary_emb=image_rotary_emb,
                                        attention_kwargs=attention_kwargs, **kwargs)
                if bii in counter.prev_out: counter.prev2_out[bii] = counter.prev_out[bii]
                counter.prev_out[bii] = (out_h.detach(), out_e.detach())
                counter.last_refresh[bii] = counter.step
                return out_h, out_e
            return wrapped
        block.forward = make_fwd(original_forward, bi, horizon)

    def cb(pipe, step_index, timestep, kw): counter.step = step_index; return kw
    n_params = sum(p.numel() for m in pipe.components.values()
                   if isinstance(m, torch.nn.Module) for p in m.parameters())

    @torch.inference_mode()
    def _run(seed_val):
        for d in (counter.prev_out, counter.prev2_out, counter.last_refresh): d.clear()
        for p in procs: p.prev_k = None; p.prev_out_full = None
        counter.step = 0; counter.cache_hits = 0; counter.cache_misses = 0
        counter.q_skipped = 0; counter.q_total = 0
        gen = torch.Generator(device="cuda").manual_seed(seed_val)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        out = pipe(prompt=prompt, num_frames=49, num_inference_steps=50,
                   width=720, height=480, guidance_scale=6.0,
                   generator=gen, return_dict=True, callback_on_step_end=cb)
        e.record(); torch.cuda.synchronize()
        return s.elapsed_time(e), _frames(getattr(out, "frames", None))

    print(f"Mega-stack: cache h={horizon}, SAPS r={saps_r_max}→{saps_r_min}, vel τ={velocity_threshold}")
    print(f"Warmup ({num_warmup}) ...")
    n = 49
    for i in range(num_warmup):
        ms, n = _run(seed + i + 1000); print(f"  warmup {i+1}: {ms:.0f} ms ({n} frames)")
    torch.cuda.reset_peak_memory_stats()
    lat = []
    for i in range(num_iters):
        ms, n = _run(seed + i); lat.append(ms)
        c_total = counter.cache_hits + counter.cache_misses
        c_rate = 100.0 * counter.cache_hits / max(c_total, 1)
        q_rate = 100.0 * counter.q_skipped / max(counter.q_total, 1)
        print(f"  iter {i+1}: {ms:.0f} ms ({n/(ms/1000):.3f} fps, "
              f"cache={c_rate:.1f}%, q_skip={q_rate:.1f}%)")
    mean_ms = statistics.mean(lat); fps = n / (mean_ms / 1000)
    c_total = counter.cache_hits + counter.cache_misses
    c_rate = 100.0 * counter.cache_hits / max(c_total, 1)
    q_rate = 100.0 * counter.q_skipped / max(counter.q_total, 1)
    return {"model": REPO, "kernel": "mega_stack_cache_saps_velocity",
            "horizon": horizon, "saps_r_max": saps_r_max, "saps_r_min": saps_r_min,
            "velocity_threshold": velocity_threshold,
            "n_blocks": len(blocks), "n_attention_modules_patched": len(procs),
            "cache_hit_rate_pct": round(c_rate, 2),
            "query_skip_pct": round(q_rate, 2),
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


@app.local_entrypoint(name="cogvideox_5b_mega_stacked_modal")
def main(horizon: int = 6, velocity_threshold: float = 0.10, num_iters: int = 1):
    r = run_mega_stacked.remote(horizon=horizon, velocity_threshold=velocity_threshold, num_iters=num_iters)
    import json; print(json.dumps(r, indent=2))
    print(f"\nMega-stack speedup: {r.get('speedup_vs_baseline', 'N/A')}×")
    sys.path.insert(0, "/root/benchmarks"); from result_store import save_result
    save_result(r, "optimised_kernels", "cogvideox_5b_mega_stacked")
