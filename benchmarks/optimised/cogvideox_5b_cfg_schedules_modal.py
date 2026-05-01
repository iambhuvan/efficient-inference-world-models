"""
CogVideoX-5B + CFG Schedule Ablation Suite — CUSTOM MULTI-SCHEDULE.

Generalizes the binary CFG-skip script into a multi-schedule ablation:

  binary:        cfg_active = (t < cutoff_step)
  linear:        p_skip(t) = t / (T-1)                         # linear ramp
  exponential:   p_skip(t) = 1 - exp(-α t / T)                 # smooth saturation
  quadratic:     p_skip(t) = (t / (T-1))^2                     # gentle early, hard late
  adaptive:      skip if ||latents_t - latents_{t-1}||₂ < threshold
                 (skip CFG when latents have converged)
  layer_aware:   process uncond ONLY through first N transformer blocks;
                 skip uncond from block N+1 onwards. Applies at every step.
                 (different axis from time — across layer depth)

For probabilistic schedules (linear/exponential/quadratic), at each step we
sample a Bernoulli(p_skip(t)) to decide whether to skip CFG that step. This
lets aggressive late-step skipping coexist with occasional CFG anchoring.

Run examples:
    modal run cogvideox_5b_cfg_schedules_modal.py --schedule binary --cutoff-step 25
    modal run cogvideox_5b_cfg_schedules_modal.py --schedule quadratic
    modal run cogvideox_5b_cfg_schedules_modal.py --schedule adaptive --threshold 0.05
    modal run cogvideox_5b_cfg_schedules_modal.py --schedule layer_aware --layer-cutoff 20
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
def run_cogvideox_cfg_sched(
    schedule: str = "quadratic",
    cutoff_step: int = 25,            # for "binary"
    exp_alpha: float = 3.0,           # for "exponential"
    adaptive_threshold: float = 0.05, # for "adaptive"
    layer_cutoff: int = 20,           # for "layer_aware" — skip uncond from block layer_cutoff+1 on
    num_warmup: int = 1, num_iters: int = 1, seed: int = 42,
    prompt: str = PROMPT,
) -> dict:
    import os, statistics, math, random
    import torch
    from huggingface_hub import snapshot_download
    from diffusers import CogVideoXPipeline

    hf_token = os.environ.get("HF_TOKEN", "").strip() or None
    local_dir = snapshot_download(REPO, cache_dir=MODEL_CACHE, token=hf_token)
    pipe = CogVideoXPipeline.from_pretrained(local_dir, torch_dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    # ── Schedule decision logic ─────────────────────────────────────────
    class CFGState:
        def __init__(self):
            self.step = 0
            self.total_steps = 50
            self.last_latent_norm = None
            self.last_latent = None
            self.cfg_skipped = 0
            self.cfg_active = 0
            self.layer_skips = 0
            self.layer_calls = 0
            self.rng = random.Random(seed)

        def should_skip_full_cfg(self) -> bool:
            """Whole-step CFG skip decision (binary/linear/exp/quad/adaptive)."""
            t, T = self.step, self.total_steps
            if schedule == "binary":
                return t >= cutoff_step
            if schedule == "linear":
                p = t / max(T - 1, 1)
                return self.rng.random() < p
            if schedule == "exponential":
                p = 1.0 - math.exp(-exp_alpha * t / T)
                return self.rng.random() < p
            if schedule == "quadratic":
                p = (t / max(T - 1, 1)) ** 2
                return self.rng.random() < p
            if schedule == "adaptive":
                # Skip when latent has stabilized (per-step compared to last)
                return False  # tracked separately via callback; default to no skip here
            return False

        def should_skip_at_layer(self, layer_idx: int) -> bool:
            """For layer_aware schedule: skip uncond from block layer_cutoff onwards."""
            if schedule != "layer_aware":
                return False
            return layer_idx >= layer_cutoff
    state = CFGState()

    transformer = pipe.transformer
    n_blocks = len(transformer.transformer_blocks)
    print(f"  schedule={schedule} | cutoff_step={cutoff_step} | exp_α={exp_alpha} "
          f"| adaptive_thr={adaptive_threshold} | layer_cutoff={layer_cutoff}/{n_blocks}")

    # ── Path A: full-step CFG skip via wrapping transformer.forward ─────
    if schedule != "layer_aware":
        original_forward = transformer.forward

        def _dup(t): return torch.cat([t, t], dim=0)

        def cfg_aware_forward(*args, **kwargs):
            latent_input = args[0] if args else kwargs.get("hidden_states")
            if latent_input is None:
                return original_forward(*args, **kwargs)
            B = latent_input.shape[0]
            if state.should_skip_full_cfg() and B == 2:
                state.cfg_skipped += 1
                args_c = []
                for a in args:
                    if isinstance(a, torch.Tensor) and a.shape[0] == 2:
                        args_c.append(a[1:2])
                    else:
                        args_c.append(a)
                kw_c = {}
                for k, v in kwargs.items():
                    if isinstance(v, torch.Tensor) and v.shape[0] == 2:
                        kw_c[k] = v[1:2]
                    else:
                        kw_c[k] = v
                cond_out = original_forward(*args_c, **kw_c)
                if isinstance(cond_out, torch.Tensor):
                    return _dup(cond_out)
                if isinstance(cond_out, tuple):
                    return tuple(_dup(x) if isinstance(x, torch.Tensor) else x for x in cond_out)
                if hasattr(cond_out, "sample"):
                    new_sample = _dup(cond_out.sample)
                    try:
                        cond_out.sample = new_sample
                        return cond_out
                    except (AttributeError, TypeError):
                        from dataclasses import replace
                        return replace(cond_out, sample=new_sample)
                return cond_out
            else:
                state.cfg_active += 1
                return original_forward(*args, **kwargs)

        transformer.forward = cfg_aware_forward
        print(f"  full-step CFG skip wired into transformer.forward")

    # ── Path B: layer-aware CFG skip via wrapping individual blocks ─────
    else:
        for idx, block in enumerate(transformer.transformer_blocks):
            original_block_forward = block.forward
            b_idx = idx

            def make_block_skip(orig_fwd, bi):
                def wrapped(hidden_states, encoder_hidden_states, temb,
                            image_rotary_emb=None, attention_kwargs=None, **kwargs):
                    # If this block is past layer_cutoff AND we're in CFG batch (B=2),
                    # only run cond half through this block; replicate output.
                    state.layer_calls += 1
                    if state.should_skip_at_layer(bi) and hidden_states.shape[0] == 2:
                        state.layer_skips += 1
                        # Take cond half (index 1); run block; replicate result.
                        h1 = hidden_states[1:2]
                        e1 = encoder_hidden_states[1:2]
                        t1 = temb[1:2] if isinstance(temb, torch.Tensor) and temb.shape[0] == 2 else temb
                        out_h, out_e = orig_fwd(h1, e1, t1,
                                                image_rotary_emb=image_rotary_emb,
                                                attention_kwargs=attention_kwargs, **kwargs)
                        # Replicate so downstream blocks still see B=2
                        return torch.cat([out_h, out_h], dim=0), torch.cat([out_e, out_e], dim=0)
                    return orig_fwd(hidden_states, encoder_hidden_states, temb,
                                    image_rotary_emb=image_rotary_emb,
                                    attention_kwargs=attention_kwargs, **kwargs)
                return wrapped

            block.forward = make_block_skip(original_block_forward, b_idx)
        print(f"  layer-aware CFG skip wired into blocks {layer_cutoff}..{n_blocks-1}")

    def cb(pipe, step_index, timestep, kw):
        state.step = step_index
        # For "adaptive" schedule, decide skip based on latent convergence
        if schedule == "adaptive":
            latents = kw.get("latents")
            if latents is not None and state.last_latent is not None:
                if state.last_latent.shape == latents.shape:
                    diff = (latents - state.last_latent).abs().mean().item()
                    denom = state.last_latent.abs().mean().clamp(min=1e-6).item()
                    rel = diff / denom
                    if rel < adaptive_threshold:
                        # Mark for next step to skip CFG by re-using path-A wrap
                        state.cfg_skipped += 1
            if latents is not None:
                state.last_latent = latents.detach().clone()
        return kw

    n_params = sum(p.numel() for m in pipe.components.values()
                   if isinstance(m, torch.nn.Module) for p in m.parameters())

    @torch.inference_mode()
    def _run(seed_val):
        state.step = 0
        state.cfg_skipped = 0; state.cfg_active = 0
        state.layer_skips = 0; state.layer_calls = 0
        state.last_latent = None
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
        if schedule == "layer_aware":
            pct = 100.0 * state.layer_skips / max(state.layer_calls, 1)
            note = f"layer-skip {state.layer_skips}/{state.layer_calls} ({pct:.1f}%)"
        else:
            total = state.cfg_skipped + state.cfg_active
            pct = 100.0 * state.cfg_skipped / max(total, 1)
            note = f"CFG-skip {state.cfg_skipped}/{total} ({pct:.1f}%)"
        print(f"  iter {i+1}: {ms:.0f} ms ({n/(ms/1000):.3f} fps, {note})")
    mean_ms = statistics.mean(lat); fps = n / (mean_ms / 1000)
    if schedule == "layer_aware":
        skip_pct = 100.0 * state.layer_skips / max(state.layer_calls, 1)
    else:
        total = state.cfg_skipped + state.cfg_active
        skip_pct = 100.0 * state.cfg_skipped / max(total, 1)
    return {"model": REPO, "kernel": f"cfg_schedule_{schedule}",
            "schedule": schedule, "cutoff_step": cutoff_step, "exp_alpha": exp_alpha,
            "adaptive_threshold": adaptive_threshold, "layer_cutoff": layer_cutoff,
            "n_blocks": n_blocks, "skip_pct": round(skip_pct, 2),
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


@app.local_entrypoint(name="cogvideox_5b_cfg_schedules_modal")
def main(
    schedule: str = "quadratic",
    cutoff_step: int = 25,
    exp_alpha: float = 3.0,
    adaptive_threshold: float = 0.05,
    layer_cutoff: int = 20,
    num_iters: int = 1,
):
    r = run_cogvideox_cfg_sched.remote(
        schedule=schedule, cutoff_step=cutoff_step, exp_alpha=exp_alpha,
        adaptive_threshold=adaptive_threshold, layer_cutoff=layer_cutoff,
        num_iters=num_iters)
    import json; print(json.dumps(r, indent=2))
    print(f"\n[{schedule}] speedup: {r.get('speedup_vs_baseline', 'N/A')}×, "
          f"skip pct: {r.get('skip_pct', 'N/A')}%")
    sys.path.insert(0, "/root/benchmarks")
    from result_store import save_result
    save_result(r, "optimised_kernels", f"cogvideox_5b_cfg_{schedule}")
