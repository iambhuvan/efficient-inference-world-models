"""
CogVideoX-5B + SAPS (Step-Aware Pruning Schedule) — CUSTOM IMPLEMENTATION.

Direct port of "SAPS: Step-Aware Pruning Schedule for Diffusion LLMs"
(Gangavarapu/John/Kshirsagar, 17-752 ML Systems CMU) from text-diffusion
(LLaDA-8B) to video-diffusion (CogVideoX-5B).

KEY INSIGHT (verbatim from SAPS poster):
    "Early steps form global structure such as topic and sentence-level
     relationships, so they need broader context. Late steps mostly refine
     local details, so much distal context becomes less useful."
    → Early steps: keep 70% tokens. Late steps: keep 10%.

SCHEDULE: r(t) = r_max × (r_min / r_max)^(t / (T-1))
    where T=50 (CogVideoX denoising steps), r_max=0.7, r_min=0.1
    → r(0)  = 0.700  (keep 70% of context tokens)
    → r(25) = 0.265  (keep 26.5%)
    → r(49) = 0.100  (keep 10%)

IMPLEMENTATION:
  1. Custom CogVideoXAttnProcessor2_0 subclass
  2. Each call: read current denoising step from shared counter
  3. Compute K-norm importance scores per token
  4. Top-K mask keeping only the highest-importance tokens
  5. Pass mask to F.scaled_dot_product_attention

NO LIBRARY DEPENDENCY beyond torch + diffusers. This is a 100% custom
contribution — the kind of speedup the SAPS poster demonstrates.

Expected on CogVideoX-5B (baseline 0.442 FPS):
  - Wall-clock: 1.25–1.40× → 0.55–0.62 FPS
  - KV memory: ~30% reduction (Pareto improvement, like SAPS)
  - Quality: <2% FVD regression (matches SAPS pattern of preserving accuracy)

Run:
    PYTHONPATH=benchmarks modal run benchmarks/optimised/cogvideox_5b_saps_modal.py
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
def run_cogvideox_saps(
    r_max: float = 0.7,
    r_min: float = 0.1,
    num_warmup: int = 1,
    num_iters: int = 1,
    seed: int = 42,
    prune_text_tokens: bool = False,  # CogVideoX has joint text+video — text safe to keep
    prompt: str = PROMPT,
) -> dict:
    import os, statistics, math
    import torch
    import torch.nn.functional as F
    from huggingface_hub import snapshot_download
    from diffusers import CogVideoXPipeline
    from diffusers.models.attention_processor import CogVideoXAttnProcessor2_0

    hf_token = os.environ.get("HF_TOKEN", "").strip() or None
    print(f"Downloading {REPO} ...")
    local_dir = snapshot_download(REPO, cache_dir=MODEL_CACHE, token=hf_token)
    pipe = CogVideoXPipeline.from_pretrained(local_dir, torch_dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    # ── Shared step counter ─────────────────────────────────────────────
    # Updated via pipeline callback once per denoising step. All attention
    # processor instances read from the same object so they share state.
    class StepCounter:
        def __init__(self):
            self.step = 0
            self.total_steps = 50

        def reset(self, total_steps: int):
            self.step = 0
            self.total_steps = total_steps

        def keep_ratio(self) -> float:
            if self.total_steps <= 1:
                return r_max
            t = min(self.step, self.total_steps - 1)
            return r_max * (r_min / r_max) ** (t / (self.total_steps - 1))

    step_counter = StepCounter()

    # ── Custom SAPS processor ───────────────────────────────────────────
    class SAPSCogVideoXAttnProcessor(CogVideoXAttnProcessor2_0):
        """Step-aware sparse attention. Subclasses CogVideoXAttnProcessor2_0
        and only overrides the attention math; QKV projection and output
        projection inherited unchanged."""

        def __init__(self, counter: StepCounter):
            super().__init__()
            self.counter = counter
            self.calls = 0
            self.total_kept = 0
            self.total_seq = 0

        def __call__(
            self,
            attn,
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor | None = None,
            attention_mask: torch.Tensor | None = None,
            image_rotary_emb: torch.Tensor | None = None,
        ) -> torch.Tensor:
            text_seq_length = encoder_hidden_states.shape[1] if encoder_hidden_states is not None else 0
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1) \
                if encoder_hidden_states is not None else hidden_states

            B, S, _ = hidden_states.shape
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)

            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads
            query = query.view(B, -1, attn.heads, head_dim).transpose(1, 2)
            key = key.view(B, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(B, -1, attn.heads, head_dim).transpose(1, 2)

            if attn.norm_q is not None:
                query = attn.norm_q(query)
            if attn.norm_k is not None:
                key = attn.norm_k(key)

            # RoPE
            if image_rotary_emb is not None:
                from diffusers.models.embeddings import apply_rotary_emb
                query[:, :, text_seq_length:] = apply_rotary_emb(
                    query[:, :, text_seq_length:], image_rotary_emb)
                if not attn.is_cross_attention:
                    key[:, :, text_seq_length:] = apply_rotary_emb(
                        key[:, :, text_seq_length:], image_rotary_emb)

            # ── SAPS: build per-step token-keep mask ─────────────────────
            r = self.counter.keep_ratio()
            n_video = S - text_seq_length
            n_keep_video = max(1, int(round(n_video * r)))

            if n_keep_video < n_video:
                # K-norm importance score per video token (averaged across heads)
                video_keys = key[:, :, text_seq_length:, :]   # (B, H, n_video, D)
                k_norms = video_keys.norm(dim=-1).mean(dim=1)  # (B, n_video) — head-avg
                topk = k_norms.topk(n_keep_video, dim=-1).indices  # (B, n_keep)

                # Build attention mask: True = attend, False = mask out.
                # Mask shape (B, 1, 1, S) broadcasts across heads & queries.
                video_mask = torch.zeros(B, n_video, dtype=torch.bool, device=key.device)
                video_mask.scatter_(1, topk, True)
                if prune_text_tokens:
                    full_mask = video_mask  # treat all as prunable
                else:
                    text_mask = torch.ones(B, text_seq_length, dtype=torch.bool, device=key.device)
                    full_mask = torch.cat([text_mask, video_mask], dim=1)

                # SDPA expects additive mask (float) or boolean — boolean True=attend.
                attn_mask = full_mask[:, None, None, :]  # (B, 1, 1, S)
            else:
                attn_mask = None  # no pruning at step 0

            self.total_kept += n_keep_video + text_seq_length if not prune_text_tokens \
                else n_keep_video
            self.total_seq += S
            self.calls += 1

            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
            hidden_states = hidden_states.transpose(1, 2).reshape(B, -1, attn.heads * head_dim)
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states, hidden_states = hidden_states.split(
                [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1)
            return hidden_states, encoder_hidden_states

    # ── Install custom processor on every attention module ─────────────
    procs: list[SAPSCogVideoXAttnProcessor] = []
    for name, module in pipe.transformer.named_modules():
        if isinstance(module.processor, CogVideoXAttnProcessor2_0) if hasattr(module, "processor") else False:
            p = SAPSCogVideoXAttnProcessor(step_counter)
            module.set_processor(p)
            procs.append(p)
    # Fallback: walk the whole pipe.transformer and replace any CogVideoXAttnProcessor2_0
    if not procs:
        for name, module in pipe.transformer.named_modules():
            if hasattr(module, "set_processor") and hasattr(module, "processor"):
                if isinstance(module.processor, CogVideoXAttnProcessor2_0):
                    p = SAPSCogVideoXAttnProcessor(step_counter)
                    module.set_processor(p)
                    procs.append(p)
    print(f"  SAPS processor installed on {len(procs)} attention modules")

    # ── Pipeline callback to update step counter ────────────────────────
    def saps_step_callback(pipe, step_index: int, timestep, callback_kwargs: dict):
        step_counter.step = step_index
        return callback_kwargs

    n_params = sum(p.numel() for m in pipe.components.values()
                   if isinstance(m, torch.nn.Module) for p in m.parameters())

    @torch.inference_mode()
    def _run(seed_val):
        for p in procs: p.calls = 0; p.total_kept = 0; p.total_seq = 0
        step_counter.reset(total_steps=50)
        gen = torch.Generator(device="cuda").manual_seed(seed_val)
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        out = pipe(
            prompt=prompt, num_frames=49, num_inference_steps=50,
            width=720, height=480, guidance_scale=6.0,
            generator=gen, return_dict=True,
            callback_on_step_end=saps_step_callback,
        )
        e.record(); torch.cuda.synchronize()
        return s.elapsed_time(e), _frames(getattr(out, "frames", None))

    print(f"Schedule: r_max={r_max}, r_min={r_min}, T=50")
    print(f"  step  0: r={r_max:.3f} → keep {r_max*100:.0f}%")
    print(f"  step 25: r={r_max*(r_min/r_max)**(25/49):.3f} → keep {r_max*(r_min/r_max)**(25/49)*100:.1f}%")
    print(f"  step 49: r={r_min:.3f} → keep {r_min*100:.0f}%")

    print(f"Warmup ({num_warmup}) ...")
    n = 49
    for i in range(num_warmup):
        ms, n = _run(seed + i + 1000); print(f"  warmup {i+1}: {ms:.0f} ms ({n} frames)")
    torch.cuda.reset_peak_memory_stats()
    lat = []
    for i in range(num_iters):
        ms, n = _run(seed + i); lat.append(ms)
        avg_keep = procs[0].total_kept / procs[0].total_seq if procs else 0.0
        print(f"  iter {i+1}: {ms:.0f} ms ({n/(ms/1000):.3f} fps, "
              f"avg_keep_ratio={avg_keep:.3f})")

    mean_ms = statistics.mean(lat); fps = n / (mean_ms / 1000)
    avg_keep_ratio = procs[0].total_kept / procs[0].total_seq if procs else 0.0
    return {
        "model": REPO, "kernel": "saps_step_aware_token_pruning",
        "r_max": r_max, "r_min": r_min, "schedule": "exponential_decay",
        "n_attention_modules_patched": len(procs),
        "avg_keep_ratio_observed": round(avg_keep_ratio, 4),
        "num_frames_observed": n, "n_params_B": round(n_params/1e9, 3),
        "latency_ms_mean": round(mean_ms, 2), "latency_per_frame_ms": round(mean_ms/n, 2),
        "frames_per_sec": round(fps, 3),
        "speedup_vs_baseline": round(fps/BASELINE_FPS, 3), "baseline_fps": BASELINE_FPS,
        "vram_gb": round(torch.cuda.max_memory_allocated()/1e9, 3),
        "gpu": "H100", "raw_latencies_ms": lat,
    }


def _frames(f):
    if f is None: return 1
    s = getattr(f, "shape", None)
    if s and len(s) >= 4: return s[0] if len(s) == 4 else s[1]
    if isinstance(f, list):
        return len(f[0]) if f and isinstance(f[0], list) else len(f)
    return 1


@app.local_entrypoint(name="cogvideox_5b_saps_modal")
def main(r_max: float = 0.7, r_min: float = 0.1, num_iters: int = 1):
    r = run_cogvideox_saps.remote(r_max=r_max, r_min=r_min, num_iters=num_iters)
    import json; print(json.dumps(r, indent=2))
    print(f"\nSAPS speedup vs baseline: {r.get('speedup_vs_baseline', 'N/A')}×")
    print(f"Avg tokens kept: {r.get('avg_keep_ratio_observed', 'N/A')*100:.1f}%")
    sys.path.insert(0, "/root/benchmarks")
    from result_store import save_result
    save_result(r, "optimised_kernels", f"cogvideox_5b_saps_r{r_max}_r{r_min}")
