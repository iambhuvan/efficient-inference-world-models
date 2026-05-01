"""
CogVideoX-5B + Step-Aware Block Skipping — CUSTOM IMPLEMENTATION.

Schedule a per-step block-skip ratio. At step t, skip every K-th transformer
block where K = skip_period(t):
  - Steps 0-9   (high noise, structure formation): skip NOTHING
  - Steps 10-29 (mid):                              skip every 4th block
  - Steps 30-49 (refinement):                       skip every 2nd block

When a block is skipped, its forward = identity. The hidden_states pass
through unchanged. Residual connections in the surrounding architecture
preserve the gradient path.

Custom code: replaces transformer block forward with conditional dispatch
based on shared step counter.

Expected: 1.3-1.5× wall-clock; ~5-10% FVD regression.

Run:
    PYTHONPATH=benchmarks modal run benchmarks/optimised/cogvideox_5b_block_skip_modal.py
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
def run_cogvideox_block_skip(
    skip_start_step: int = 10,
    skip_mid_step: int = 30,
    skip_period_mid: int = 4,
    skip_period_late: int = 2,
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

    # ── Shared step counter + skip schedule ─────────────────────────────
    class Counter:
        def __init__(self):
            self.step = 0
            self.total_skipped = 0
            self.total_called = 0
        def skip_period(self) -> int:
            if self.step < skip_start_step: return 0
            if self.step < skip_mid_step: return skip_period_mid
            return skip_period_late
    counter = Counter()

    # ── Wrap each transformer block with skip logic ─────────────────────
    blocks = pipe.transformer.transformer_blocks
    block_indices = list(range(len(blocks)))
    print(f"  found {len(blocks)} transformer_blocks")

    for idx, block in enumerate(blocks):
        original_forward = block.forward
        block_idx = idx

        def make_wrapped_forward(orig_fwd, b_idx):
            def wrapped(hidden_states, encoder_hidden_states, temb, image_rotary_emb=None,
                        attention_kwargs=None, **kwargs):
                period = counter.skip_period()
                counter.total_called += 1
                # Skip when (block_idx + step) % period == 0; period 0 = no skip
                if period > 0 and (b_idx + counter.step) % period == 0:
                    counter.total_skipped += 1
                    # Identity skip: return inputs unchanged. CogVideoX returns
                    # (hidden_states, encoder_hidden_states) tuple.
                    return hidden_states, encoder_hidden_states
                return orig_fwd(hidden_states, encoder_hidden_states, temb,
                                image_rotary_emb=image_rotary_emb,
                                attention_kwargs=attention_kwargs, **kwargs)
            return wrapped

        block.forward = make_wrapped_forward(original_forward, block_idx)

    print(f"  block-skip wired into all {len(blocks)} blocks")

    def cb(pipe, step_index, timestep, kw):
        counter.step = step_index
        return kw

    n_params = sum(p.numel() for m in pipe.components.values()
                   if isinstance(m, torch.nn.Module) for p in m.parameters())

    @torch.inference_mode()
    def _run(seed_val):
        counter.step = 0; counter.total_skipped = 0; counter.total_called = 0
        gen = torch.Generator(device="cuda").manual_seed(seed_val)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        out = pipe(prompt=prompt, num_frames=49, num_inference_steps=50,
                   width=720, height=480, guidance_scale=6.0,
                   generator=gen, return_dict=True, callback_on_step_end=cb)
        e.record(); torch.cuda.synchronize()
        return s.elapsed_time(e), _frames(getattr(out, "frames", None))

    print(f"Schedule:")
    print(f"  steps 0-{skip_start_step-1}:           NO skip")
    print(f"  steps {skip_start_step}-{skip_mid_step-1}:           skip 1/{skip_period_mid}")
    print(f"  steps {skip_mid_step}-49:           skip 1/{skip_period_late}")

    print(f"Warmup ({num_warmup}) ...")
    n = 49
    for i in range(num_warmup):
        ms, n = _run(seed + i + 1000); print(f"  warmup {i+1}: {ms:.0f} ms ({n} frames)")
    torch.cuda.reset_peak_memory_stats()
    lat = []
    for i in range(num_iters):
        ms, n = _run(seed + i); lat.append(ms)
        skip_pct = 100.0 * counter.total_skipped / max(counter.total_called, 1)
        print(f"  iter {i+1}: {ms:.0f} ms ({n/(ms/1000):.3f} fps, skipped {skip_pct:.1f}%)")
    mean_ms = statistics.mean(lat); fps = n / (mean_ms / 1000)
    skip_pct = 100.0 * counter.total_skipped / max(counter.total_called, 1)
    return {"model": REPO, "kernel": "step_aware_block_skip",
            "skip_start_step": skip_start_step, "skip_mid_step": skip_mid_step,
            "skip_period_mid": skip_period_mid, "skip_period_late": skip_period_late,
            "n_blocks": len(blocks),
            "block_skip_pct": round(skip_pct, 2),
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


@app.local_entrypoint(name="cogvideox_5b_block_skip_modal")
def main(skip_start_step: int = 10, skip_mid_step: int = 30,
         skip_period_mid: int = 4, skip_period_late: int = 2, num_iters: int = 1):
    r = run_cogvideox_block_skip.remote(
        skip_start_step=skip_start_step, skip_mid_step=skip_mid_step,
        skip_period_mid=skip_period_mid, skip_period_late=skip_period_late,
        num_iters=num_iters,
    )
    import json; print(json.dumps(r, indent=2))
    print(f"\nBlock-skip speedup: {r.get('speedup_vs_baseline', 'N/A')}×, "
          f"skipped {r.get('block_skip_pct', 'N/A')}%")
    sys.path.insert(0, "/root/benchmarks")
    from result_store import save_result
    save_result(r, "optimised_kernels", "cogvideox_5b_block_skip")
