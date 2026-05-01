"""
Wan2.1-T2V-14B + Quality Evaluation — baseline vs each top optimization.

Cross-model port of `cogvideox_5b_quality_eval_modal.py`. Validates whether
Pareto winners on CogVideoX-5B (notably step_axis_exponential_h1_4 at
1.64×/PSNR 34.2 dB) generalize to the larger 14B Wan model.

Wan transformer block forward returns a single tensor (not a tuple), so the
cache hook is simpler than CogVideoX. Block list found via attribute probing.

Run:
    cd benchmarks && modal run optimised/wan21_14b_quality_eval_modal.py \
        --optimization step_axis_exponential_h1_4
"""
import sys; sys.path.insert(0, "/root"); sys.path.insert(0, "/root/benchmarks")
import modal
from modal_common import app, image_cuda_devel, hf_secret, model_volume, MODEL_CACHE
image = image_cuda_devel
REPO = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
PROMPT = ("A cat walks on the grass, realistic, 4k, cinematic, smooth motion, "
          "natural lighting")


@app.function(gpu="H100", image=image, secrets=[hf_secret],
              volumes={MODEL_CACHE: model_volume}, timeout=10800, memory=65536)
def run_quality_eval(
    optimization: str = "step_axis_exponential_h1_4",
    seed: int = 42, prompt: str = PROMPT,
) -> dict:
    """Run baseline + optimization at same seed; compute quality metrics."""
    import os, math, json
    import torch
    import numpy as np
    from huggingface_hub import snapshot_download
    from diffusers import WanPipeline

    hf_token = os.environ.get("HF_TOKEN", "").strip() or None
    local_dir = snapshot_download(REPO, cache_dir=MODEL_CACHE, token=hf_token)

    def _load_pipe():
        p = WanPipeline.from_pretrained(local_dir, torch_dtype=torch.bfloat16).to("cuda")
        p.set_progress_bar_config(disable=True)
        return p

    @torch.inference_mode()
    def _generate(p, gen, callback=None):
        kwargs = dict(prompt=prompt, num_frames=81, num_inference_steps=50,
                      width=832, height=480, guidance_scale=5.0,
                      generator=gen, return_dict=True, output_type="np")
        if callback is not None:
            kwargs["callback_on_step_end"] = callback
        out = p(**kwargs)
        frames = getattr(out, "frames", None)
        # output_type="np" returns numpy float [0,1] either as ndarray or list
        if isinstance(frames, np.ndarray):
            arr_f = frames
        elif isinstance(frames, list):
            if frames and isinstance(frames[0], list): frames = frames[0]
            arr_f = np.stack([np.asarray(im) for im in frames], axis=0)
        elif isinstance(frames, torch.Tensor):
            t = frames
            if t.ndim == 5: t = t[0]
            if t.shape[-1] == 3: arr_f = t.float().cpu().numpy()
            else: arr_f = t.permute(0, 2, 3, 1).float().cpu().numpy()
        else:
            print(f"  unexpected frames type: {type(frames)}; out keys: {dir(out)}")
            return None
        # Normalize to (T, H, W, 3) uint8
        if arr_f.ndim == 5: arr_f = arr_f[0]   # drop batch
        if arr_f.dtype != np.uint8:
            arr_f = (np.clip(arr_f, 0, 1) * 255).astype(np.uint8)
        return arr_f

    # ── Pass 1: BASELINE ────────────────────────────────────────────────
    print("\n=== Pass 1: BASELINE ===")
    pipe1 = _load_pipe()
    g1 = torch.Generator(device="cuda").manual_seed(seed)
    torch.cuda.synchronize()
    s1, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s1.record(); base_frames = _generate(pipe1, g1); e1.record(); torch.cuda.synchronize()
    base_ms = s1.elapsed_time(e1)
    print(f"  baseline: {base_ms:.0f} ms, frames shape={base_frames.shape}")
    del pipe1; torch.cuda.empty_cache()

    # ── Pass 2: OPTIMIZATION ────────────────────────────────────────────
    print(f"\n=== Pass 2: {optimization} ===")
    pipe2 = _load_pipe()
    state = _apply_optimization(pipe2, optimization)
    g2 = torch.Generator(device="cuda").manual_seed(seed)
    torch.cuda.synchronize()
    s2, e2 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s2.record()
    cb = state.get("callback") if state else None
    opt_frames = _generate(pipe2, g2, callback=cb)
    e2.record(); torch.cuda.synchronize()
    opt_ms = s2.elapsed_time(e2)
    print(f"  {optimization}: {opt_ms:.0f} ms, frames shape={opt_frames.shape}")

    # ── Compute per-frame metrics ───────────────────────────────────────
    print("\n=== Per-frame quality metrics ===")
    n_frames = min(base_frames.shape[0], opt_frames.shape[0])
    per_frame = []
    psnrs = []; ssims = []; l1s = []
    for f in range(n_frames):
        b = base_frames[f].astype(np.float32) / 255.0
        o = opt_frames[f].astype(np.float32) / 255.0
        mse = float(((b - o) ** 2).mean())
        psnr = 10.0 * math.log10(1.0 / max(mse, 1e-12))
        l1 = float(np.abs(b - o).mean())
        ssim = None
        try:
            from skimage.metrics import structural_similarity
            ssim = float(structural_similarity(b, o, channel_axis=2, data_range=1.0))
            ssims.append(ssim)
        except ImportError: pass
        psnrs.append(psnr); l1s.append(l1)
        per_frame.append({"frame": f, "psnr_db": round(psnr, 3),
                          "l1": round(l1, 5), "ssim": round(ssim, 4) if ssim else None})

    aggregate = {
        "psnr_mean_db": round(float(np.mean(psnrs)), 3),
        "psnr_min_db": round(float(np.min(psnrs)), 3),
        "psnr_max_db": round(float(np.max(psnrs)), 3),
        "l1_mean": round(float(np.mean(l1s)), 5),
        "ssim_mean": round(float(np.mean(ssims)), 4) if ssims else None,
    }
    speedup = base_ms / opt_ms

    print(f"\n=== AGGREGATE ===")
    print(json.dumps(aggregate, indent=2))
    print(f"  base_ms={base_ms:.0f}, opt_ms={opt_ms:.0f}, speedup={speedup:.3f}×")
    print(f"  Pareto check: speedup={speedup:.2f}× at PSNR={aggregate['psnr_mean_db']:.1f} dB")
    return {
        "model": REPO, "optimization": optimization, "seed": seed,
        "baseline_ms": round(base_ms, 2), "optimization_ms": round(opt_ms, 2),
        "speedup": round(speedup, 3),
        "aggregate_quality": aggregate,
        "per_frame": per_frame[:10] + ([{"...": "truncated"}] if n_frames > 10 else []),
        "n_frames_compared": n_frames,
    }


def _wan_blocks(pipe):
    """Probe Wan transformer for block list (varies across diffusers versions)."""
    for attr in ("blocks", "transformer_blocks", "layers", "single_blocks"):
        cand = getattr(pipe.transformer, attr, None)
        if cand is not None and len(cand) > 5:
            return cand, attr
    raise RuntimeError("Wan block list not found")


def _apply_optimization(pipe, optimization: str) -> dict:
    """Apply named optimization to pipe in-place. Returns state dict with callback."""
    import re
    if optimization == "noop":
        return {"step": 0, "callback": None}
    m = re.match(r"^constant_h(\d+)$", optimization)
    if m:
        return _apply_const_horizon(pipe, h=int(m.group(1)), warmup_steps=3)
    m = re.match(r"^step_axis_(exponential|quadratic|cubic|linear)_h(\d+)_(\d+)(_delta)?$", optimization)
    if m:
        sched, h_min, h_max, delta_suffix = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
        mode = "delta" if delta_suffix else "output"
        return _apply_step_axis_horizon(pipe, schedule=sched, h_min=h_min, h_max=h_max,
                                        warmup_steps=3, total_steps=50, mode=mode)
    raise ValueError(f"Unknown optimization: {optimization}")


def _wrap_wan_block(block, state, h_lookup, mode: str = "output"):
    """Wrap a Wan transformer block. Wan blocks return a single tensor.

    mode='output': cache full block output (= block(x)). 1st-order extrapolation
                   `pred = ph + (ph - p2h) * scale`. Verified PSNR ~20 dB on Wan.
    mode='delta':  cache residual `block(x) - x` and apply to fresh input. Goal:
                   preserve x's per-step variation while caching the slow-changing
                   block residual.
    """
    import torch
    original = block.forward
    bi = state["_next_idx"]; state["_next_idx"] += 1

    def wrapped(*args, **kwargs):
        h = h_lookup(state["step"])
        last = state["last_refresh"].get(bi, -1)
        steps_since = state["step"] - last
        x_in = args[0] if args else kwargs.get("hidden_states")  # first positional
        if (state["step"] >= state["warmup_steps"] and steps_since < h and last >= 0 and
            bi in state["prev_out"] and bi in state["prev2_out"]):
            ph = state["prev_out"][bi]; p2h = state["prev2_out"][bi]
            if ph.shape == p2h.shape:
                delta_step = ph - p2h
                pred = ph + delta_step * (steps_since / max(1, h - 1))
                if mode == "delta" and isinstance(x_in, torch.Tensor) and x_in.shape == pred.shape:
                    # pred is the cached residual; combine with fresh input
                    return x_in + pred
                return pred
        out = original(*args, **kwargs)
        if bi in state["prev_out"]: state["prev2_out"][bi] = state["prev_out"][bi]
        if isinstance(out, torch.Tensor):
            stored = (out - x_in).detach() if (mode == "delta" and isinstance(x_in, torch.Tensor)
                                               and x_in.shape == out.shape) else out.detach()
            state["prev_out"][bi] = stored
        elif isinstance(out, tuple) and len(out) > 0 and isinstance(out[0], torch.Tensor):
            t = out[0]
            stored = (t - x_in).detach() if (mode == "delta" and isinstance(x_in, torch.Tensor)
                                             and x_in.shape == t.shape) else t.detach()
            state["prev_out"][bi] = stored
        state["last_refresh"][bi] = state["step"]
        return out
    block.forward = wrapped


def _apply_const_horizon(pipe, h: int, warmup_steps: int) -> dict:
    blocks, attr = _wan_blocks(pipe)
    print(f"  const horizon h={h} on pipe.transformer.{attr} ({len(blocks)} blocks)")
    state = {"step": 0, "warmup_steps": warmup_steps,
             "prev_out": {}, "prev2_out": {}, "last_refresh": {}, "_next_idx": 0}
    for block in blocks:
        _wrap_wan_block(block, state, h_lookup=lambda t, hh=h: hh)
    def cb(p, step_idx, t, kw): state["step"] = step_idx; return kw
    state["callback"] = cb
    return state


def _apply_step_axis_horizon(pipe, schedule: str, h_min: int, h_max: int,
                              warmup_steps: int, total_steps: int,
                              mode: str = "output") -> dict:
    blocks, attr = _wan_blocks(pipe)
    print(f"  step-axis {schedule} h={h_min}->{h_max} mode={mode} on pipe.transformer.{attr} ({len(blocks)} blocks)")

    def h_at(t: int) -> int:
        T = total_steps
        if T <= 1: return h_max
        u = t / (T - 1)
        if schedule == "linear":
            return max(1, round(h_min + u * (h_max - h_min)))
        if schedule == "quadratic":
            return max(1, round(h_min + (u ** 2) * (h_max - h_min)))
        if schedule == "cubic":
            return max(1, round(h_min + (u ** 3) * (h_max - h_min)))
        if schedule == "exponential":
            ratio = max(h_max / max(h_min, 1), 1.0)
            return max(1, round(h_min * (ratio ** u)))
        return h_max

    state = {"step": 0, "warmup_steps": warmup_steps,
             "prev_out": {}, "prev2_out": {}, "last_refresh": {}, "_next_idx": 0}
    for block in blocks:
        _wrap_wan_block(block, state, h_lookup=h_at, mode=mode)
    def cb(p, step_idx, t, kw): state["step"] = step_idx; return kw
    state["callback"] = cb
    return state


@app.local_entrypoint(name="wan21_14b_quality_eval_modal")
def main(optimization: str = "step_axis_exponential_h1_4"):
    r = run_quality_eval.remote(optimization=optimization)
    import json; print(json.dumps(r, indent=2))
    sys.path.insert(0, "/root/benchmarks"); from result_store import save_result
    save_result(r, "optimised_kernels", f"wan21_14b_quality_eval_{optimization}")
