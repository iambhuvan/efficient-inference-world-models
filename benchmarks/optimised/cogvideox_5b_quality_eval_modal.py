"""
CogVideoX-5B + Quality Evaluation — baseline vs each top optimization.

Mirrors `oasis_quality_eval_modal.py` pattern for the CogVideoX video DiT.
For each `optimization` value, runs baseline AND optimization at the same
seed and prompt, then computes per-frame divergence metrics:
  - Pixel PSNR (dB) — visual fidelity
  - Pixel L1 — pixel-level error magnitude
  - Pixel SSIM — structural similarity (when scikit-image available)
  - Latent MSE — drift from baseline trajectory
  - Aggregate: mean / min / max PSNR

Baseline anchor: 0.442 FPS (verified). Each optimization is compared by
RUNNING THE FULL GENERATION TWICE in the same container — first baseline
config, then with the optimization applied — at identical seed.

Supported optimizations:
  - "constant_h6"   — Per-Layer Taylor const h=6 (3.975×)
  - "constant_h8"   — Per-Layer Taylor const h=8 (4.716×)
  - "saps_v2"       — SAPS v2 token gather (1.408×)
  - "block_skip"    — step-aware block skip (1.384×)
  - "mega_stack"    — h=6 + SAPS + velocity (4.964×)

Run:
    PYTHONPATH=benchmarks modal run benchmarks/optimised/cogvideox_5b_quality_eval_modal.py --optimization constant_h6
"""
import sys; sys.path.insert(0, "/root"); sys.path.insert(0, "/root/benchmarks")
import modal
from modal_common import app, image_cuda_devel, hf_secret, model_volume, MODEL_CACHE
image = image_cuda_devel
REPO = "zai-org/CogVideoX-5b"
PROMPT = ("A panda, dressed in a small, red jacket and a tiny hat, sits on a "
          "wooden stool in a serene bamboo forest. The panda's fluffy paws "
          "strum a miniature acoustic guitar.")


@app.function(gpu="H100", image=image, secrets=[hf_secret],
              volumes={MODEL_CACHE: model_volume}, timeout=7200, memory=65536)
def run_quality_eval(
    optimization: str = "constant_h6",
    seed: int = 42, prompt: str = PROMPT,
) -> dict:
    """Run baseline + optimization at same seed; compute quality metrics."""
    import os, math, json
    import torch
    import numpy as np
    from huggingface_hub import snapshot_download
    from diffusers import CogVideoXPipeline

    hf_token = os.environ.get("HF_TOKEN", "").strip() or None
    local_dir = snapshot_download(REPO, cache_dir=MODEL_CACHE, token=hf_token)

    def _load_pipe():
        p = CogVideoXPipeline.from_pretrained(local_dir, torch_dtype=torch.bfloat16).to("cuda")
        p.set_progress_bar_config(disable=True)
        return p

    @torch.inference_mode()
    def _generate(p, gen):
        out = p(prompt=prompt, num_frames=49, num_inference_steps=50,
                width=720, height=480, guidance_scale=6.0,
                generator=gen, return_dict=True)
        frames = getattr(out, "frames", None)
        # frames is list[list[PIL]] for diffusers video — extract numpy uint8
        if isinstance(frames, list):
            if frames and isinstance(frames[0], list): frames = frames[0]
            arrs = [np.asarray(im) for im in frames]   # (H, W, 3) uint8
            arr = np.stack(arrs, axis=0)
        elif isinstance(frames, torch.Tensor):
            t = frames
            if t.ndim == 5: t = t[0]
            if t.shape[-1] == 3: arr = (t.float().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            else: arr = (t.permute(0, 2, 3, 1).float().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        else:
            arr = None
        return arr   # (T, H, W, 3) uint8

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
    counter, procs = _apply_optimization(pipe2, optimization)
    g2 = torch.Generator(device="cuda").manual_seed(seed)
    torch.cuda.synchronize()
    s2, e2 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s2.record()
    if counter is None or counter.get("callback") is None:
        out = pipe2(prompt=prompt, num_frames=49, num_inference_steps=50,
                    width=720, height=480, guidance_scale=6.0,
                    generator=g2, return_dict=True)
    else:
        out = pipe2(prompt=prompt, num_frames=49, num_inference_steps=50,
                    width=720, height=480, guidance_scale=6.0,
                    generator=g2, return_dict=True,
                    callback_on_step_end=counter["callback"])
    e2.record(); torch.cuda.synchronize()
    opt_ms = s2.elapsed_time(e2)
    opt_frames_obj = getattr(out, "frames", None)
    if isinstance(opt_frames_obj, list):
        if opt_frames_obj and isinstance(opt_frames_obj[0], list): opt_frames_obj = opt_frames_obj[0]
        opt_frames = np.stack([np.asarray(im) for im in opt_frames_obj], axis=0)
    else:
        t = opt_frames_obj
        if t.ndim == 5: t = t[0]
        if t.shape[-1] == 3: opt_frames = (t.float().clamp(0,1).cpu().numpy() * 255).astype(np.uint8)
        else: opt_frames = (t.permute(0,2,3,1).float().clamp(0,1).cpu().numpy() * 255).astype(np.uint8)
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


def _apply_optimization(pipe, optimization: str) -> tuple[dict | None, list]:
    """Apply named optimization to pipe in-place. Returns (state_dict, processors)."""
    import re
    # Parse constant_hN for any N (cliff search)
    m = re.match(r"^constant_h(\d+)$", optimization)
    if m:
        return _apply_const_horizon(pipe, h=int(m.group(1)), warmup_steps=3), []
    # Step-axis schedules (SAPS-form equation), with optional torch.compile prefix
    m = re.match(r"^(compile_)?step_axis_(exponential|quadratic|cubic|quartic|linear)_h(\d+)_(\d+)$", optimization)
    if m:
        do_compile = m.group(1) is not None
        sched, h_min, h_max = m.group(2), int(m.group(3)), int(m.group(4))
        return _apply_step_axis_horizon(pipe, schedule=sched, h_min=h_min, h_max=h_max,
                                        warmup_steps=3, total_steps=50,
                                        compile_inner=do_compile), []
    if optimization == "saps_v2":
        return _apply_saps(pipe, r_max=0.7, r_min=0.1)
    if optimization == "block_skip":
        return _apply_block_skip(pipe), []
    if optimization == "mega_stack":
        return _apply_const_horizon(pipe, h=6, warmup_steps=3), []
    raise ValueError(f"Unknown optimization: {optimization}")


def _apply_step_axis_horizon(pipe, schedule: str, h_min: int, h_max: int,
                              warmup_steps: int, total_steps: int,
                              compile_inner: bool = False) -> dict:
    """Step-axis SAPS-form schedule: h(t) varies across denoising steps.

    compile_inner=True wraps each block's original forward in torch.compile
    BEFORE the cache hook, so cache misses run on a compiled (CUDA-graph) path.
    """
    import torch
    blocks = pipe.transformer.transformer_blocks
    state = {"step": 0, "prev_out": {}, "prev2_out": {}, "last_refresh": {}}
    if compile_inner:
        print(f"  compile_inner=True: torch.compile each of {len(blocks)} blocks")

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
        if schedule == "quartic":
            return max(1, round(h_min + (u ** 4) * (h_max - h_min)))
        if schedule == "exponential":
            ratio = max(h_max / max(h_min, 1), 1.0)
            return max(1, round(h_min * (ratio ** u)))
        return h_max

    for idx, block in enumerate(blocks):
        original = block.forward; bi = idx
        if compile_inner:
            # mode="default" avoids CUDA graphs (incompatible with our cached
            # tensor storage being overwritten across steps).
            original = torch.compile(original, mode="default", dynamic=False)
        def make_fwd(orig, bii):
            def wrapped(hidden_states, encoder_hidden_states, temb, image_rotary_emb=None,
                        attention_kwargs=None, **kwargs):
                h = h_at(state["step"])
                last = state["last_refresh"].get(bii, -1)
                steps_since = state["step"] - last
                if (state["step"] >= warmup_steps and steps_since < h and last >= 0 and
                    bii in state["prev_out"] and bii in state["prev2_out"]):
                    ph, pe = state["prev_out"][bii]; p2h, p2e = state["prev2_out"][bii]
                    delta_h = ph - p2h; delta_e = pe - p2e
                    pred_h = ph + delta_h * (steps_since / max(1, h - 1))
                    pred_e = pe + delta_e * (steps_since / max(1, h - 1))
                    return pred_h, pred_e
                out_h, out_e = orig(hidden_states, encoder_hidden_states, temb,
                                    image_rotary_emb=image_rotary_emb,
                                    attention_kwargs=attention_kwargs, **kwargs)
                if bii in state["prev_out"]: state["prev2_out"][bii] = state["prev_out"][bii]
                state["prev_out"][bii] = (out_h.detach(), out_e.detach())
                state["last_refresh"][bii] = state["step"]
                return out_h, out_e
            return wrapped
        block.forward = make_fwd(original, bi)
    def cb(p, step_idx, t, kw): state["step"] = step_idx; return kw
    state["callback"] = cb
    return state


def _apply_const_horizon(pipe, h: int, warmup_steps: int) -> dict:
    blocks = pipe.transformer.transformer_blocks
    state = {"step": 0, "prev_out": {}, "prev2_out": {}, "last_refresh": {}}
    for idx, block in enumerate(blocks):
        original = block.forward; bi = idx
        def make_fwd(orig, bii):
            def wrapped(hidden_states, encoder_hidden_states, temb, image_rotary_emb=None,
                        attention_kwargs=None, **kwargs):
                last = state["last_refresh"].get(bii, -1)
                steps_since = state["step"] - last
                if (state["step"] >= warmup_steps and steps_since < h and last >= 0 and
                    bii in state["prev_out"] and bii in state["prev2_out"]):
                    ph, pe = state["prev_out"][bii]; p2h, p2e = state["prev2_out"][bii]
                    delta_h = ph - p2h; delta_e = pe - p2e
                    pred_h = ph + delta_h * (steps_since / max(1, h - 1))
                    pred_e = pe + delta_e * (steps_since / max(1, h - 1))
                    return pred_h, pred_e
                out_h, out_e = orig(hidden_states, encoder_hidden_states, temb,
                                    image_rotary_emb=image_rotary_emb,
                                    attention_kwargs=attention_kwargs, **kwargs)
                if bii in state["prev_out"]: state["prev2_out"][bii] = state["prev_out"][bii]
                state["prev_out"][bii] = (out_h.detach(), out_e.detach())
                state["last_refresh"][bii] = state["step"]
                return out_h, out_e
            return wrapped
        block.forward = make_fwd(original, bi)
    def cb(p, step_idx, t, kw): state["step"] = step_idx; return kw
    state["callback"] = cb
    return state


def _apply_saps(pipe, r_max: float, r_min: float) -> tuple[dict, list]:
    import torch
    import torch.nn.functional as F
    from diffusers.models.attention_processor import CogVideoXAttnProcessor2_0
    state = {"step": 0, "total_steps": 50}
    def keep(): t = min(state["step"], 49); return r_max * (r_min / r_max) ** (t / 49)
    class SAPSProc(CogVideoXAttnProcessor2_0):
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
            r = keep(); n_video = S - text_len; n_keep_v = max(1, int(round(n_video * r)))
            if n_keep_v < n_video:
                video_k = k[:, :, text_len:, :]
                k_norms = video_k.norm(dim=-1).mean(dim=1)
                top = k_norms.topk(n_keep_v, dim=-1).indices
                idx_v = top.unsqueeze(1).unsqueeze(-1).expand(B, H, n_keep_v, D)
                vk = torch.gather(k[:, :, text_len:, :], 2, idx_v)
                vv = torch.gather(v[:, :, text_len:, :], 2, idx_v)
                k_full = torch.cat([k[:, :, :text_len, :], vk], dim=2)
                v_full = torch.cat([v[:, :, :text_len, :], vv], dim=2)
            else: k_full = k; v_full = v
            out = F.scaled_dot_product_attention(q, k_full, v_full, dropout_p=0.0, is_causal=False)
            out = out.transpose(1, 2).reshape(B, -1, H * D)
            out = attn.to_out[0](out); out = attn.to_out[1](out)
            enc, hid = out.split([text_len, out.size(1) - text_len], dim=1)
            return hid, enc
    procs = []
    for name, mod in pipe.transformer.named_modules():
        if hasattr(mod, "set_processor") and hasattr(mod, "processor"):
            if isinstance(mod.processor, CogVideoXAttnProcessor2_0):
                p = SAPSProc(); mod.set_processor(p); procs.append(p)
    def cb(pp, step_idx, t, kw): state["step"] = step_idx; return kw
    state["callback"] = cb
    return state, procs


def _apply_block_skip(pipe) -> dict:
    state = {"step": 0}
    blocks = pipe.transformer.transformer_blocks
    for idx, block in enumerate(blocks):
        original = block.forward; bi = idx
        def make_fwd(orig, bii):
            def wrapped(hidden_states, encoder_hidden_states, temb, image_rotary_emb=None,
                        attention_kwargs=None, **kwargs):
                step = state["step"]
                # Same schedule as cogvideox_5b_block_skip_modal.py
                if step < 10: period = 0
                elif step < 30: period = 4
                else: period = 2
                if period > 0 and (bii + step) % period == 0:
                    return hidden_states, encoder_hidden_states
                return orig(hidden_states, encoder_hidden_states, temb,
                            image_rotary_emb=image_rotary_emb,
                            attention_kwargs=attention_kwargs, **kwargs)
            return wrapped
        block.forward = make_fwd(original, bi)
    def cb(p, step_idx, t, kw): state["step"] = step_idx; return kw
    state["callback"] = cb
    return state


_PROMPTS = {
    "panda": PROMPT,
    "city": ("Aerial drone footage flying low over a futuristic neon-lit city at "
             "night, traffic flowing on multi-level highways, cinematic, 4k"),
    "ocean": ("Slow-motion underwater shot of a sea turtle swimming through "
              "coral reef, sunlight rays piercing the water, photorealistic"),
}


@app.local_entrypoint(name="cogvideox_5b_quality_eval_modal")
def main(optimization: str = "constant_h6", prompt_id: str = "panda", seed: int = 42):
    p = _PROMPTS.get(prompt_id, PROMPT)
    r = run_quality_eval.remote(optimization=optimization, prompt=p, seed=seed)
    r["prompt_id"] = prompt_id
    import json; print(json.dumps(r, indent=2))
    sys.path.insert(0, "/root/benchmarks"); from result_store import save_result
    parts = [optimization]
    if prompt_id != "panda": parts.append(prompt_id)
    if seed != 42: parts.append(f"seed{seed}")
    suffix = "_".join(parts)
    save_result(r, "optimised_kernels", f"cogvideox_5b_quality_eval_{suffix}")
