"""
CogVideoX-5B + Per-Layer Taylor Cache Schedule — different cache horizons per block.

Insight (from ∆-DiT extension): blocks have different convergence rates.
  - Early blocks (composition): refresh every step
  - Middle blocks (refinement): cache 2-3 steps
  - Late blocks (detail): cache aggressively (3-5 steps)

Per-block cache horizon h(b) = round(1 + (b/B) * (h_max - 1))
Cache predicts via 2nd-order Taylor: pred = 2*prev - prev2.
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
def run_per_layer_taylor(
    h_min: int = 1, h_max: int = 4, warmup_steps: int = 3,
    schedule: str = "linear",       # "linear" | "u_shape" | "constant"
    num_warmup: int = 1, num_iters: int = 1, seed: int = 42, prompt: str = PROMPT,
) -> dict:
    import os, statistics, torch
    from huggingface_hub import snapshot_download
    from diffusers import CogVideoXPipeline

    hf_token = os.environ.get("HF_TOKEN", "").strip() or None
    local_dir = snapshot_download(REPO, cache_dir=MODEL_CACHE, token=hf_token)
    pipe = CogVideoXPipeline.from_pretrained(local_dir, torch_dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    blocks = pipe.transformer.transformer_blocks
    B = len(blocks)
    # Per-block cache horizon h(b), b in [0, B-1]. Normalize t = b/(B-1).
    def t_norm(b: int) -> float:
        return b / max(B - 1, 1)
    if schedule == "linear":
        horizons = [round(h_min + t_norm(b) * (h_max - h_min)) for b in range(B)]
    elif schedule == "quadratic":
        # Slow rise early, faster late: h(b) = h_min + t² × (h_max - h_min)
        horizons = [round(h_min + (t_norm(b) ** 2) * (h_max - h_min)) for b in range(B)]
    elif schedule == "exponential":
        # Smooth multiplicative growth: h(b) = h_min × (h_max/h_min)^t
        import math
        ratio = max(h_max / max(h_min, 1), 1.0)
        horizons = [round(h_min * (ratio ** t_norm(b))) for b in range(B)]
    elif schedule == "inverse_linear":
        # Early caches more, late refreshes more — opposite of linear
        horizons = [round(h_max - t_norm(b) * (h_max - h_min)) for b in range(B)]
    elif schedule == "u_shape":
        mid = B // 2
        horizons = [round(h_min + (1 - abs(b - mid) / mid) * (h_max - h_min)) for b in range(B)]
    elif schedule == "constant":
        horizons = [h_max] * B
    else:
        horizons = [h_max] * B
    horizons = [max(1, h) for h in horizons]  # clamp to positive horizons
    print(f"  per-layer Taylor: schedule={schedule}, h_min={h_min}, h_max={h_max}")
    print(f"  block horizons: {horizons[:5]}...{horizons[-5:]}")

    class State:
        def __init__(self):
            self.step = 0; self.hits = 0; self.misses = 0
            self.prev_out: dict[int, tuple] = {}
            self.prev2_out: dict[int, tuple] = {}
            self.last_refresh: dict[int, int] = {}
        def reset(self):
            for d in (self.prev_out, self.prev2_out, self.last_refresh): d.clear()
            self.step = 0; self.hits = 0; self.misses = 0
    st = State()

    for idx, block in enumerate(blocks):
        original_forward = block.forward; bi = idx; horizon = horizons[idx]
        def make_fwd(orig_fwd, bii, h):
            def wrapped(hidden_states, encoder_hidden_states, temb, image_rotary_emb=None,
                        attention_kwargs=None, **kwargs):
                last = st.last_refresh.get(bii, -1)
                # Cache hit if (step - last_refresh) < horizon AND we have prev2
                steps_since = st.step - last
                if (st.step >= warmup_steps and steps_since < h and last >= 0 and
                    bii in st.prev_out and bii in st.prev2_out):
                    ph, pe = st.prev_out[bii]; p2h, p2e = st.prev2_out[bii]
                    if ph.shape == hidden_states.shape[:1] + ph.shape[1:]:  # batch match
                        # 2nd-order Taylor extrapolation
                        delta_h = ph - p2h; delta_e = pe - p2e
                        pred_h = ph + delta_h * (steps_since / max(1, h - 1))
                        pred_e = pe + delta_e * (steps_since / max(1, h - 1))
                        st.hits += 1
                        return pred_h, pred_e
                # Cache miss / refresh
                st.misses += 1
                out_h, out_e = orig_fwd(hidden_states, encoder_hidden_states, temb,
                                        image_rotary_emb=image_rotary_emb,
                                        attention_kwargs=attention_kwargs, **kwargs)
                if bii in st.prev_out: st.prev2_out[bii] = st.prev_out[bii]
                st.prev_out[bii] = (out_h.detach(), out_e.detach())
                st.last_refresh[bii] = st.step
                return out_h, out_e
            return wrapped
        block.forward = make_fwd(original_forward, bi, horizon)

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
        total = st.hits + st.misses; rate = 100.0 * st.hits / max(total, 1)
        print(f"  iter {i+1}: {ms:.0f} ms ({n/(ms/1000):.3f} fps, hit_rate={rate:.1f}%)")
    mean_ms = statistics.mean(lat); fps = n / (mean_ms / 1000)
    total = st.hits + st.misses; rate = 100.0 * st.hits / max(total, 1)
    return {"model": REPO, "kernel": "per_layer_taylor_schedule",
            "schedule": schedule, "h_min": h_min, "h_max": h_max,
            "horizons_first5": horizons[:5], "horizons_last5": horizons[-5:],
            "n_blocks": B, "cache_hit_rate_pct": round(rate, 2),
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


@app.local_entrypoint(name="cogvideox_5b_per_layer_taylor_modal")
def main(h_min: int = 1, h_max: int = 4, schedule: str = "linear", num_iters: int = 1):
    r = run_per_layer_taylor.remote(h_min=h_min, h_max=h_max, schedule=schedule, num_iters=num_iters)
    import json; print(json.dumps(r, indent=2))
    print(f"\nPer-layer Taylor speedup: {r.get('speedup_vs_baseline', 'N/A')}×")
    sys.path.insert(0, "/root/benchmarks"); from result_store import save_result
    save_result(r, "optimised_kernels", f"cogvideox_5b_per_layer_taylor_{schedule}")
