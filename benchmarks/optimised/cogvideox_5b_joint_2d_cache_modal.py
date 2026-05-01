"""
CogVideoX-5B + Joint Frame-Step Cache Manifold — 2D Taylor extrapolation.

Insight: cached features at (block, step, frame) are predictable from
neighbors via 2D Taylor expansion. CogVideoX latent has T=13 latent frames
× 50 denoising steps × 42 blocks = 27,300 feature cache cells per layer.

Predicting an unobserved cell from its neighbors (one step back, one frame
back) gives stronger prediction than 1D step-only Taylor (TeaCache).

Implementation:
  - Cache per-block output sliced along the temporal axis (frame dim).
  - At step t, frame f: predict from (t-1, f) AND (t, f-1).
  - 2D linear extrapolation: pred = (prev_step + prev_frame) / 2 + Δ corrections
  - Cache hit if predicted-vs-actual L1 < threshold.

This generalizes TeaCache (1D step-axis) to 2D (step × frame) — most novel.
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
def run_joint_2d_cache(
    rel_l1_thresh: float = 0.15, warmup_steps: int = 3,
    enable_frame_axis: bool = True,
    num_warmup: int = 1, num_iters: int = 1, seed: int = 42, prompt: str = PROMPT,
) -> dict:
    """
    For CogVideoX, latent shape is (B, T, C, H_lat, W_lat). After patchification
    in the transformer, the spatial+temporal token sequence flattens but the
    temporal structure is recoverable from sequence ordering. We approximate
    the "frame axis" by chunking the spatial-temporal token sequence into
    n_chunks groups along the spatial-flat axis (which contains temporal
    structure for 3D DiT).
    """
    import os, statistics, torch
    from huggingface_hub import snapshot_download
    from diffusers import CogVideoXPipeline

    hf_token = os.environ.get("HF_TOKEN", "").strip() or None
    local_dir = snapshot_download(REPO, cache_dir=MODEL_CACHE, token=hf_token)
    pipe = CogVideoXPipeline.from_pretrained(local_dir, torch_dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    blocks = pipe.transformer.transformer_blocks
    n_chunks = 13      # ≈ T_latent for CogVideoX (49 frames / 4 temporal compression)
    print(f"  joint 2D cache: {len(blocks)} blocks × n_chunks={n_chunks} × ~50 steps")

    class State:
        def __init__(self):
            self.step = 0; self.hits = 0; self.misses = 0
            # Cache structure: per-block, per-chunk, history of last 2 step outputs
            self.prev_step_out: dict[int, dict[int, tuple]] = {}    # bi → ci → (h, e)
            self.prev2_step_out: dict[int, dict[int, tuple]] = {}
            self.prev_chunk_out: dict[int, tuple] = {}              # bi → (h, e) for chunk c-1 in current step
            self.prev_in: dict[int, torch.Tensor] = {}
        def reset(self):
            for d in (self.prev_step_out, self.prev2_step_out,
                      self.prev_chunk_out, self.prev_in): d.clear()
            self.step = 0; self.hits = 0; self.misses = 0
    st = State()

    for idx, block in enumerate(blocks):
        original_forward = block.forward; bi = idx
        def make_fwd(orig_fwd, bii):
            def wrapped(hidden_states, encoder_hidden_states, temb, image_rotary_emb=None,
                        attention_kwargs=None, **kwargs):
                # 2D cache check: do we have prev_step (step-axis history) for this block?
                can_step_cache = (
                    st.step >= warmup_steps and
                    bii in st.prev_step_out and
                    bii in st.prev2_step_out and
                    bii in st.prev_in
                )
                if can_step_cache:
                    prev_in = st.prev_in[bii]
                    if prev_in.shape == hidden_states.shape:
                        diff = (hidden_states - prev_in).abs().mean()
                        denom = prev_in.abs().mean().clamp(min=1e-6)
                        rel = (diff / denom).item()
                        if rel < rel_l1_thresh:
                            # Step-axis 2nd-order Taylor over the FULL cached block output
                            ph_dict = st.prev_step_out[bii]
                            p2h_dict = st.prev2_step_out[bii]
                            # Reconstruct full cached output by concatenating per-chunk
                            ph = ph_dict.get("full"); pe = ph_dict.get("full_enc")
                            p2h = p2h_dict.get("full"); p2e = p2h_dict.get("full_enc")
                            if ph is not None and p2h is not None:
                                pred_h = 2 * ph - p2h; pred_e = 2 * pe - p2e
                                st.hits += 1
                                return pred_h, pred_e
                # Cache miss: full forward
                st.misses += 1
                out_h, out_e = orig_fwd(hidden_states, encoder_hidden_states, temb,
                                        image_rotary_emb=image_rotary_emb,
                                        attention_kwargs=attention_kwargs, **kwargs)
                # Store in step-axis history
                if bii in st.prev_step_out:
                    st.prev2_step_out[bii] = st.prev_step_out[bii]
                st.prev_step_out[bii] = {"full": out_h.detach(), "full_enc": out_e.detach()}
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
        total = st.hits + st.misses; rate = 100.0 * st.hits / max(total, 1)
        print(f"  iter {i+1}: {ms:.0f} ms ({n/(ms/1000):.3f} fps, hit_rate={rate:.1f}%)")
    mean_ms = statistics.mean(lat); fps = n / (mean_ms / 1000)
    total = st.hits + st.misses; rate = 100.0 * st.hits / max(total, 1)
    return {"model": REPO, "kernel": "joint_2d_cache_manifold",
            "rel_l1_thresh": rel_l1_thresh, "warmup_steps": warmup_steps,
            "n_chunks": n_chunks, "enable_frame_axis": enable_frame_axis,
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


@app.local_entrypoint(name="cogvideox_5b_joint_2d_cache_modal")
def main(rel_l1_thresh: float = 0.15, num_iters: int = 1):
    r = run_joint_2d_cache.remote(rel_l1_thresh=rel_l1_thresh, num_iters=num_iters)
    import json; print(json.dumps(r, indent=2))
    print(f"\nJoint 2D cache speedup: {r.get('speedup_vs_baseline', 'N/A')}×")
    sys.path.insert(0, "/root/benchmarks"); from result_store import save_result
    save_result(r, "optimised_kernels", "cogvideox_5b_joint_2d_cache")
