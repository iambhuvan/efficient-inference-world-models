"""
CogVideoX-5B + Cache Drift Detector — adaptive replacement for fixed-threshold block cache.

Block-output cache with adaptive drift detection on a held-out token sample.
Refreshes cache when measured drift exceeds an EMA-based threshold rather than
a fixed L1 threshold.

Algorithm:
  1. Cache block output + cached input.
  2. Each step, sample N=128 tokens from current input; compute L2 distance
     to same tokens in cached input.
  3. Maintain running mean+std of drift across recent steps (EMA).
  4. If drift > mean + tau_sigmas * std OR drift > absolute_max → refresh.
  5. Otherwise reuse cached output (linear extrapolation as in TeaCache).

Beats fixed-threshold cache by adapting per-prompt and per-step variance.
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
def run_drift_cache(
    sample_n: int = 128, tau_sigmas: float = 1.5, abs_max_drift: float = 0.30,
    warmup_steps: int = 3, ema_alpha: float = 0.7,
    num_warmup: int = 1, num_iters: int = 1, seed: int = 42, prompt: str = PROMPT,
) -> dict:
    import os, statistics, torch
    from huggingface_hub import snapshot_download
    from diffusers import CogVideoXPipeline

    hf_token = os.environ.get("HF_TOKEN", "").strip() or None
    local_dir = snapshot_download(REPO, cache_dir=MODEL_CACHE, token=hf_token)
    pipe = CogVideoXPipeline.from_pretrained(local_dir, torch_dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    class DriftState:
        def __init__(self):
            self.step = 0; self.hits = 0; self.misses = 0
            self.prev_in: dict[int, torch.Tensor] = {}
            self.prev_out: dict[int, tuple] = {}
            self.prev2_out: dict[int, tuple] = {}
            # per-block EMA stats of drift
            self.drift_mean: dict[int, float] = {}
            self.drift_var: dict[int, float] = {}
            self.sample_idx: dict[int, torch.Tensor] = {}
        def reset(self):
            for d in (self.prev_in, self.prev_out, self.prev2_out,
                      self.drift_mean, self.drift_var, self.sample_idx): d.clear()
            self.step = 0; self.hits = 0; self.misses = 0
    st = DriftState()

    blocks = pipe.transformer.transformer_blocks
    print(f"  drift-cache wired into {len(blocks)} blocks (sample_n={sample_n}, τ_σ={tau_sigmas})")

    for idx, block in enumerate(blocks):
        original_forward = block.forward; bi = idx
        def make_fwd(orig_fwd, bii):
            def wrapped(hidden_states, encoder_hidden_states, temb, image_rotary_emb=None,
                        attention_kwargs=None, **kwargs):
                # Decide cache hit via drift on sample
                if (st.step >= warmup_steps and bii in st.prev_in and
                    bii in st.prev_out and bii in st.prev2_out):
                    prev_in = st.prev_in[bii]
                    if prev_in.shape == hidden_states.shape:
                        # Use deterministic sample indices per block (cached)
                        if bii not in st.sample_idx:
                            S = hidden_states.shape[1]
                            n = min(sample_n, S)
                            st.sample_idx[bii] = torch.randperm(S, device=hidden_states.device)[:n]
                        idx_t = st.sample_idx[bii]
                        cur_s = hidden_states[:, idx_t]
                        prev_s = prev_in[:, idx_t]
                        drift = (cur_s - prev_s).norm(p=2) / prev_s.norm(p=2).clamp(min=1e-6)
                        drift_v = drift.item()
                        # EMA update + threshold check
                        m = st.drift_mean.get(bii, drift_v)
                        v = st.drift_var.get(bii, drift_v * drift_v)
                        std = max((v - m * m) ** 0.5, 1e-6)
                        threshold = max(m + tau_sigmas * std, 0.001)
                        # Update EMA
                        st.drift_mean[bii] = ema_alpha * m + (1 - ema_alpha) * drift_v
                        st.drift_var[bii] = ema_alpha * v + (1 - ema_alpha) * (drift_v ** 2)
                        if drift_v < threshold and drift_v < abs_max_drift:
                            # Cache hit — Taylor extrapolation: 2*prev - prev2
                            ph, pe = st.prev_out[bii]
                            p2h, p2e = st.prev2_out[bii]
                            pred_h = 2 * ph - p2h; pred_e = 2 * pe - p2e
                            st.hits += 1
                            return pred_h, pred_e
                # Cache miss — full forward
                st.misses += 1
                out_h, out_e = orig_fwd(hidden_states, encoder_hidden_states, temb,
                                        image_rotary_emb=image_rotary_emb,
                                        attention_kwargs=attention_kwargs, **kwargs)
                if bii in st.prev_out: st.prev2_out[bii] = st.prev_out[bii]
                st.prev_out[bii] = (out_h.detach(), out_e.detach())
                st.prev_in[bii] = hidden_states.detach()
                return out_h, out_e
            return wrapped
        block.forward = make_fwd(original_forward, bi)

    def cb(pipe, step_index, timestep, kw): st.step = step_index; return kw
    n_params = sum(p.numel() for m in pipe.components.values()
                   if isinstance(m, torch.nn.Module) for p in m.parameters())

    @torch.inference_mode()
    def _run(seed_val):
        st.reset()
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
        total = st.hits + st.misses
        rate = 100.0 * st.hits / max(total, 1)
        print(f"  iter {i+1}: {ms:.0f} ms ({n/(ms/1000):.3f} fps, "
              f"hit_rate={rate:.1f}% [{st.hits}/{total}])")
    mean_ms = statistics.mean(lat); fps = n / (mean_ms / 1000)
    total = st.hits + st.misses; rate = 100.0 * st.hits / max(total, 1)
    return {"model": REPO, "kernel": "cache_drift_detector",
            "sample_n": sample_n, "tau_sigmas": tau_sigmas,
            "abs_max_drift": abs_max_drift, "ema_alpha": ema_alpha,
            "n_blocks": len(blocks), "cache_hit_rate_pct": round(rate, 2),
            "cache_hits": st.hits, "cache_misses": st.misses,
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


@app.local_entrypoint(name="cogvideox_5b_drift_cache_modal")
def main(tau_sigmas: float = 1.5, abs_max_drift: float = 0.30, num_iters: int = 1):
    r = run_drift_cache.remote(tau_sigmas=tau_sigmas, abs_max_drift=abs_max_drift, num_iters=num_iters)
    import json; print(json.dumps(r, indent=2))
    print(f"\nDrift-cache speedup: {r.get('speedup_vs_baseline', 'N/A')}× hit_rate: {r.get('cache_hit_rate_pct', 'N/A')}%")
    sys.path.insert(0, "/root/benchmarks"); from result_store import save_result
    save_result(r, "optimised_kernels", "cogvideox_5b_drift_cache")
