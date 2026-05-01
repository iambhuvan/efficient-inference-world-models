"""
Wan2.1-T2V-14B + Per-Layer Taylor Cache — port of CogVideoX winner.

Tests whether the "constant horizon wins" finding generalizes to 14B.

Wan transformer block forward returns single tensor (not tuple), so cache
storage is simpler than CogVideoX. Block list found via attribute probing.
"""
import sys; sys.path.insert(0, "/root"); sys.path.insert(0, "/root/benchmarks")
import modal
from modal_common import app, image_cuda_devel, hf_secret, model_volume, MODEL_CACHE
image = image_cuda_devel
REPO = "Wan-AI/Wan2.1-T2V-14B-Diffusers"; BASELINE_FPS = 0.171
PROMPT = ("A cat walks on the grass, realistic, 4k, cinematic, smooth motion, "
          "natural lighting")


@app.function(gpu="H100", image=image, secrets=[hf_secret],
              volumes={MODEL_CACHE: model_volume}, timeout=7200, memory=65536)
def run_wan_per_layer_taylor(
    schedule: str = "constant", h_min: int = 6, h_max: int = 6,
    warmup_steps: int = 3, num_warmup: int = 1, num_iters: int = 1,
    seed: int = 42, prompt: str = PROMPT,
) -> dict:
    import os, statistics, torch
    from huggingface_hub import snapshot_download
    from diffusers import WanPipeline

    hf_token = os.environ.get("HF_TOKEN", "").strip() or None
    local_dir = snapshot_download(REPO, cache_dir=MODEL_CACHE, token=hf_token)
    pipe = WanPipeline.from_pretrained(local_dir, torch_dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    # Probe for block list
    block_list = None; attr_name = None
    for attr in ("blocks", "transformer_blocks", "layers", "single_blocks"):
        cand = getattr(pipe.transformer, attr, None)
        if cand is not None and len(cand) > 5:
            block_list = cand; attr_name = attr; break
    if block_list is None:
        return {"error": "no Wan block list found", "model": REPO}
    print(f"  Wan block list: pipe.transformer.{attr_name} ({len(block_list)} blocks)")
    B = len(block_list)

    def t_norm(b): return b / max(B - 1, 1)
    if schedule == "constant":
        horizons = [h_max] * B
    elif schedule == "linear":
        horizons = [round(h_min + t_norm(b) * (h_max - h_min)) for b in range(B)]
    elif schedule == "quadratic":
        horizons = [round(h_min + (t_norm(b) ** 2) * (h_max - h_min)) for b in range(B)]
    elif schedule == "exponential":
        ratio = max(h_max / max(h_min, 1), 1.0)
        horizons = [round(h_min * (ratio ** t_norm(b))) for b in range(B)]
    elif schedule == "u_shape":
        mid = B // 2
        horizons = [round(h_min + (1 - abs(b - mid) / mid) * (h_max - h_min)) for b in range(B)]
    else:
        horizons = [h_max] * B
    horizons = [max(1, h) for h in horizons]
    print(f"  horizons[:5]={horizons[:5]} horizons[-5:]={horizons[-5:]}")

    class State:
        def __init__(self):
            self.step = 0; self.hits = 0; self.misses = 0
            self.prev_out: dict[int, torch.Tensor] = {}
            self.prev2_out: dict[int, torch.Tensor] = {}
            self.last_refresh: dict[int, int] = {}
        def reset(self):
            for d in (self.prev_out, self.prev2_out, self.last_refresh): d.clear()
            self.step = 0; self.hits = 0; self.misses = 0
    st = State()

    for idx, block in enumerate(block_list):
        original = block.forward; bi = idx; h = horizons[idx]
        def make_fwd(orig, bii, hh):
            def wrapped(*args, **kwargs):
                last = st.last_refresh.get(bii, -1)
                steps_since = st.step - last
                if (st.step >= warmup_steps and steps_since < hh and last >= 0 and
                    bii in st.prev_out and bii in st.prev2_out):
                    ph = st.prev_out[bii]; p2h = st.prev2_out[bii]
                    if ph.shape == p2h.shape:
                        delta = ph - p2h
                        pred = ph + delta * (steps_since / max(1, hh - 1))
                        st.hits += 1
                        return pred
                st.misses += 1
                out = orig(*args, **kwargs)
                # Wan blocks return single tensor (not tuple)
                if bii in st.prev_out: st.prev2_out[bii] = st.prev_out[bii]
                if isinstance(out, torch.Tensor):
                    st.prev_out[bii] = out.detach()
                elif isinstance(out, tuple) and len(out) > 0 and isinstance(out[0], torch.Tensor):
                    st.prev_out[bii] = out[0].detach()
                st.last_refresh[bii] = st.step
                return out
            return wrapped
        block.forward = make_fwd(original, bi, h)

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
        out = pipe(prompt=prompt, num_frames=81, num_inference_steps=50,
                   width=832, height=480, guidance_scale=5.0,
                   generator=gen, return_dict=True, callback_on_step_end=cb)
        e.record(); torch.cuda.synchronize()
        return s.elapsed_time(e), _frames(getattr(out, "frames", None))

    print(f"Warmup ({num_warmup}) ...")
    n = 81
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
    return {"model": REPO, "kernel": f"wan_per_layer_taylor_{schedule}",
            "schedule": schedule, "h_min": h_min, "h_max": h_max,
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


@app.local_entrypoint(name="wan21_14b_per_layer_taylor_modal")
def main(schedule: str = "constant", h_min: int = 6, h_max: int = 6, num_iters: int = 1):
    r = run_wan_per_layer_taylor.remote(schedule=schedule, h_min=h_min, h_max=h_max, num_iters=num_iters)
    import json; print(json.dumps(r, indent=2))
    print(f"\nWan per-layer Taylor speedup: {r.get('speedup_vs_baseline', 'N/A')}×")
    sys.path.insert(0, "/root/benchmarks"); from result_store import save_result
    save_result(r, "optimised_kernels", f"wan21_14b_per_layer_taylor_{schedule}_h{h_max}")
