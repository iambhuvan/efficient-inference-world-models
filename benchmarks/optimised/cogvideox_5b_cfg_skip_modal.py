"""
CogVideoX-5B + Step-Aware CFG Skip — CUSTOM IMPLEMENTATION.

THE PROBLEM (analogous to SAPS):
  Classifier-Free Guidance (CFG) fires at *every* denoising step. It works
  by running the transformer TWICE per step:
    - Once on the unconditional embedding (negative prompt / empty)
    - Once on the conditional embedding (the actual prompt)
  Then combines: `noise_pred = uncond + scale * (cond - uncond)`
  → CFG roughly DOUBLES per-step compute (factor of 2 batch size).

THE INSIGHT:
  - Early steps: CFG matters most — semantic alignment forms here
  - Late steps: refinement — uncond contribution is mostly noise
  - Late-step CFG is largely wasted compute

THE SCHEDULE:
  cfg_active(t) = (t < cfg_cutoff_step)    # binary cutoff, simplest variant
  Default: skip CFG after step 25 (half of 50)

CUSTOM CODE:
  Wraps pipe.transformer.forward to detect CFG-doubled batches at skip steps
  and just process the conditional half (skip uncond compute entirely).
  Returns cond_pred duplicated so the pipeline's CFG-combine math works out
  to no-op (uncond = cond → combined = cond).

Expected: 1.4-1.7× wall-clock at cutoff_step=25
          1.6-1.9× at cutoff_step=15 (more aggressive)

Run:
    PYTHONPATH=benchmarks modal run benchmarks/optimised/cogvideox_5b_cfg_skip_modal.py
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
def run_cogvideox_cfg_skip(
    cfg_cutoff_step: int = 25,        # binary cutoff at step 25 (= half of 50)
    num_warmup: int = 1, num_iters: int = 1, seed: int = 42,
    prompt: str = PROMPT,
) -> dict:
    import os, statistics
    import torch
    from huggingface_hub import snapshot_download
    from diffusers import CogVideoXPipeline

    hf_token = os.environ.get("HF_TOKEN", "").strip() or None
    local_dir = snapshot_download(REPO, cache_dir=MODEL_CACHE, token=hf_token)
    pipe = CogVideoXPipeline.from_pretrained(local_dir, torch_dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    # ── Shared step counter ─────────────────────────────────────────────
    class CFGState:
        def __init__(self):
            self.step = 0
            self.cfg_skipped = 0
            self.cfg_active = 0
        def should_skip_cfg(self) -> bool:
            return self.step >= cfg_cutoff_step
    state = CFGState()

    # ── Wrap transformer.forward to skip uncond half on skip steps ──────
    transformer = pipe.transformer
    original_forward = transformer.forward

    def cfg_aware_forward(*args, **kwargs):
        # First positional arg is hidden_states / latents (B, T, C, H, W) or similar
        latent_input = args[0] if args else kwargs.get("hidden_states")
        if latent_input is None:
            return original_forward(*args, **kwargs)

        B = latent_input.shape[0]
        # CFG-doubled batch comes in as B=2 (uncond, cond stacked).
        # On skip steps: process only the cond half, then duplicate output.
        if state.should_skip_cfg() and B == 2:
            state.cfg_skipped += 1
            # Take cond half (index 1)
            args_cond = []
            for a in args:
                if isinstance(a, torch.Tensor) and a.shape[0] == 2:
                    args_cond.append(a[1:2])
                else:
                    args_cond.append(a)
            kwargs_cond = {}
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor) and v.shape[0] == 2:
                    kwargs_cond[k] = v[1:2]
                else:
                    kwargs_cond[k] = v
            cond_out = original_forward(*args_cond, **kwargs_cond)
            # Duplicate output along batch dim. CogVideoX returns either a
            # Transformer2DModelOutput (with .sample), a tuple (output,), or
            # a bare tensor depending on return_dict and version.
            def _dup(t: torch.Tensor) -> torch.Tensor:
                return torch.cat([t, t], dim=0)

            if isinstance(cond_out, torch.Tensor):
                return _dup(cond_out)
            if isinstance(cond_out, tuple):
                return tuple(_dup(x) if isinstance(x, torch.Tensor) else x for x in cond_out)
            if hasattr(cond_out, "sample") and isinstance(cond_out.sample, torch.Tensor):
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
    print(f"  CFG-skip wired into transformer.forward")
    print(f"  Cutoff: step {cfg_cutoff_step} of 50 → skip uncond after that")

    def cb(pipe, step_index, timestep, kw): state.step = step_index; return kw

    n_params = sum(p.numel() for m in pipe.components.values()
                   if isinstance(m, torch.nn.Module) for p in m.parameters())

    @torch.inference_mode()
    def _run(seed_val):
        state.step = 0; state.cfg_skipped = 0; state.cfg_active = 0
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
        total = state.cfg_skipped + state.cfg_active
        skip_pct = 100.0 * state.cfg_skipped / max(total, 1)
        print(f"  iter {i+1}: {ms:.0f} ms ({n/(ms/1000):.3f} fps, "
              f"CFG skipped at {skip_pct:.0f}% of forwards = {state.cfg_skipped}/{total})")
    mean_ms = statistics.mean(lat); fps = n / (mean_ms / 1000)
    total = state.cfg_skipped + state.cfg_active
    skip_pct = 100.0 * state.cfg_skipped / max(total, 1)
    return {"model": REPO, "kernel": "step_aware_cfg_skip",
            "cfg_cutoff_step": cfg_cutoff_step,
            "cfg_skip_pct_of_forwards": round(skip_pct, 2),
            "cfg_skipped": state.cfg_skipped, "cfg_active": state.cfg_active,
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


@app.local_entrypoint(name="cogvideox_5b_cfg_skip_modal")
def main(cfg_cutoff_step: int = 25, num_iters: int = 1):
    r = run_cogvideox_cfg_skip.remote(cfg_cutoff_step=cfg_cutoff_step, num_iters=num_iters)
    import json; print(json.dumps(r, indent=2))
    print(f"\nCFG-skip speedup: {r.get('speedup_vs_baseline', 'N/A')}×, "
          f"skipped {r.get('cfg_skip_pct_of_forwards', 'N/A')}% of CFG forwards")
    sys.path.insert(0, "/root/benchmarks")
    from result_store import save_result
    save_result(r, "optimised_kernels", f"cogvideox_5b_cfg_skip_cut{cfg_cutoff_step}")
