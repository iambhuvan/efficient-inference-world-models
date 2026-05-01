"""
CogVideoX-5B + STEP-AXIS Per-Layer Taylor — SAPS-style equation form.

DIFFERENT AXIS from existing per_layer_taylor (which scheduled by BLOCK):
  This version schedules cache horizon h(t) across DENOISING STEPS t.
  Same h applies to all blocks at step t; h grows over 50 steps.

THE SAPS-FORM EQUATION:
  exponential:  h(t) = h_min × (h_max/h_min)^(t / (T-1))
  quadratic:    h(t) = h_min + (t/(T-1))² × (h_max - h_min)
  linear:       h(t) = h_min + (t/(T-1)) × (h_max - h_min)
  constant:     h(t) = h_max

Direct port of SAPS poster's equation form to our cache scheduling problem.
SAPS scheduled token-prune ratio across denoising steps; we schedule cache
horizon across denoising steps.

Run examples:
    modal run cogvideox_5b_step_axis_taylor_modal.py --schedule exponential --h-min 2 --h-max 8
    modal run cogvideox_5b_step_axis_taylor_modal.py --schedule quadratic --h-min 2 --h-max 8
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
def run_step_axis_taylor(
    schedule: str = "exponential",       # "constant" | "linear" | "quadratic" | "exponential"
    h_min: int = 2, h_max: int = 8,
    warmup_steps: int = 3, total_steps: int = 50,
    num_warmup: int = 1, num_iters: int = 1, seed: int = 42, prompt: str = PROMPT,
) -> dict:
    import os, statistics, torch
    from huggingface_hub import snapshot_download
    from diffusers import CogVideoXPipeline

    hf_token = os.environ.get("HF_TOKEN", "").strip() or None
    local_dir = snapshot_download(REPO, cache_dir=MODEL_CACHE, token=hf_token)
    pipe = CogVideoXPipeline.from_pretrained(local_dir, torch_dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    # ── SAPS-form schedule equation ─────────────────────────────────────
    def h_at_step(t: int) -> int:
        T = total_steps
        if T <= 1: return h_max
        u = t / (T - 1)  # u ∈ [0, 1]
        if schedule == "constant":
            return h_max
        if schedule == "linear":
            return max(1, round(h_min + u * (h_max - h_min)))
        if schedule == "quadratic":
            return max(1, round(h_min + (u ** 2) * (h_max - h_min)))
        if schedule == "cubic":
            return max(1, round(h_min + (u ** 3) * (h_max - h_min)))
        if schedule == "quartic":
            return max(1, round(h_min + (u ** 4) * (h_max - h_min)))
        if schedule == "exponential":
            ratio = max(h_max / max(h_min, 1), 1.0)
            return max(1, round(h_min * (ratio ** u)))
        return h_max

    # Print schedule preview
    sample_steps = [0, 5, 10, 20, 25, 30, 40, 49]
    print(f"  schedule={schedule}, h_min={h_min}, h_max={h_max}, T={total_steps}")
    print(f"  h(t) preview: " + ", ".join(f"t={t}:h={h_at_step(t)}" for t in sample_steps))

    blocks = pipe.transformer.transformer_blocks
    print(f"  step-axis schedule applied across {len(blocks)} blocks")

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
        original_forward = block.forward; bi = idx
        def make_fwd(orig_fwd, bii):
            def wrapped(hidden_states, encoder_hidden_states, temb, image_rotary_emb=None,
                        attention_kwargs=None, **kwargs):
                # h depends on CURRENT step, not block — same h for all blocks at step t
                h = h_at_step(st.step)
                last = st.last_refresh.get(bii, -1)
                steps_since = st.step - last
                if (st.step >= warmup_steps and steps_since < h and last >= 0 and
                    bii in st.prev_out and bii in st.prev2_out):
                    ph, pe = st.prev_out[bii]; p2h, p2e = st.prev2_out[bii]
                    delta_h = ph - p2h; delta_e = pe - p2e
                    pred_h = ph + delta_h * (steps_since / max(1, h - 1))
                    pred_e = pe + delta_e * (steps_since / max(1, h - 1))
                    st.hits += 1
                    return pred_h, pred_e
                st.misses += 1
                out_h, out_e = orig_fwd(hidden_states, encoder_hidden_states, temb,
                                        image_rotary_emb=image_rotary_emb,
                                        attention_kwargs=attention_kwargs, **kwargs)
                if bii in st.prev_out: st.prev2_out[bii] = st.prev_out[bii]
                st.prev_out[bii] = (out_h.detach(), out_e.detach())
                st.last_refresh[bii] = st.step
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
        total = st.hits + st.misses; rate = 100.0 * st.hits / max(total, 1)
        print(f"  iter {i+1}: {ms:.0f} ms ({n/(ms/1000):.3f} fps, hit_rate={rate:.1f}%)")
    mean_ms = statistics.mean(lat); fps = n / (mean_ms / 1000)
    total = st.hits + st.misses; rate = 100.0 * st.hits / max(total, 1)
    horizons_at_steps = {t: h_at_step(t) for t in sample_steps}
    return {"model": REPO, "kernel": f"step_axis_taylor_{schedule}",
            "schedule": schedule, "h_min": h_min, "h_max": h_max, "total_steps": total_steps,
            "h_at_steps": horizons_at_steps,
            "n_blocks": len(blocks), "cache_hit_rate_pct": round(rate, 2),
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


@app.local_entrypoint(name="cogvideox_5b_step_axis_taylor_modal")
def main(schedule: str = "exponential", h_min: int = 2, h_max: int = 8, num_iters: int = 1):
    r = run_step_axis_taylor.remote(schedule=schedule, h_min=h_min, h_max=h_max, num_iters=num_iters)
    import json; print(json.dumps(r, indent=2))
    print(f"\n[{schedule} h={h_min}→{h_max}] speedup: {r.get('speedup_vs_baseline', 'N/A')}×")
    sys.path.insert(0, "/root/benchmarks"); from result_store import save_result
    save_result(r, "optimised_kernels", f"cogvideox_5b_step_axis_taylor_{schedule}")
