"""
CogVideoX-5B + Step-Aware K/V Precision Schedule — CUSTOM IMPLEMENTATION.

THE PROBLEM (orthogonal to SAPS):
  K and V tensors are bf16 at every denoising step. Quantization papers
  use a single fixed precision throughout inference.

THE INSIGHT:
  - Early steps (high noise): tolerate aggressive K/V quantization
  - Late steps (low noise): need precision (visible artifacts)
  - INVERSE direction from SAPS: precision RISES across steps

THE SCHEDULE:
  step  0-19  (high noise, structure):    K/V → INT4-equivalent (per-token quant)
  step 20-39  (mid):                      K/V → FP8-equivalent
  step 40-49  (refinement):               K/V → BF16 (no quant)

CUSTOM CODE:
  Wraps CogVideoXAttnProcessor2_0 to apply per-token symmetric quantization
  to K and V before SDPA. Quant simulates lower precision via fakequant
  (round-clamp-dequantize cycle) — gives the numerical error of low-bit
  computation without needing fused INT4/FP8 GEMM kernels.

Expected:
  - With fakequant only: 1.0-1.1× (overhead ~ savings — fakequant adds compute)
  - Real win comes from + flash-attn FP8 path or + INT4 fused kernel
  - At minimum, demonstrates the SCHEDULE works without quality regression

Run:
    PYTHONPATH=benchmarks modal run benchmarks/optimised/cogvideox_5b_kv_precision_modal.py
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
def run_cogvideox_kv_precision(
    int4_until_step: int = 20,        # steps 0..19: INT4-equivalent
    fp8_until_step: int = 40,         # steps 20..39: FP8-equivalent; rest = BF16
    num_warmup: int = 1, num_iters: int = 1, seed: int = 42,
    prompt: str = PROMPT,
) -> dict:
    import os, statistics
    import torch
    import torch.nn.functional as F
    from huggingface_hub import snapshot_download
    from diffusers import CogVideoXPipeline
    from diffusers.models.attention_processor import CogVideoXAttnProcessor2_0

    hf_token = os.environ.get("HF_TOKEN", "").strip() or None
    local_dir = snapshot_download(REPO, cache_dir=MODEL_CACHE, token=hf_token)
    pipe = CogVideoXPipeline.from_pretrained(local_dir, torch_dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    class StepCounter:
        def __init__(self): self.step = 0
        def precision_bits(self) -> int:
            if self.step < int4_until_step: return 4
            if self.step < fp8_until_step:  return 8
            return 16
    counter = StepCounter()

    def fake_quantize(x: torch.Tensor, bits: int) -> torch.Tensor:
        """Symmetric per-(batch, head) quant + dequant. bits=16 → no-op."""
        if bits >= 16:
            return x
        # Per (B, head, group_dim=last) absmax — group along head_dim for stability
        absmax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        levels = (1 << (bits - 1)) - 1   # signed range: 7 for INT4, 127 for INT8
        scale = absmax / levels
        q = (x / scale).round().clamp(-(levels + 1), levels)
        return (q * scale).to(x.dtype)

    class KVPrecisionProcessor(CogVideoXAttnProcessor2_0):
        def __init__(self, c): super().__init__(); self.c = c

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

            # ── KV Precision: fake-quantize K and V at scheduled bits ────
            bits = self.c.precision_bits()
            k = fake_quantize(k, bits)
            v = fake_quantize(v, bits)

            out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
            out = out.transpose(1, 2).reshape(B, -1, H * D)
            out = attn.to_out[0](out); out = attn.to_out[1](out)
            enc, hid = out.split([text_len, out.size(1) - text_len], dim=1)
            return hid, enc

    procs = []
    for name, mod in pipe.transformer.named_modules():
        if hasattr(mod, "set_processor") and hasattr(mod, "processor"):
            if isinstance(mod.processor, CogVideoXAttnProcessor2_0):
                p = KVPrecisionProcessor(counter); mod.set_processor(p); procs.append(p)
    print(f"  KV-precision processor installed on {len(procs)} attn modules")
    print(f"  Schedule: INT4 [0,{int4_until_step}), FP8 [{int4_until_step},{fp8_until_step}), BF16 [{fp8_until_step}, 50)")

    def cb(pipe, step_index, timestep, kw): counter.step = step_index; return kw

    n_params = sum(p.numel() for m in pipe.components.values()
                   if isinstance(m, torch.nn.Module) for p in m.parameters())

    @torch.inference_mode()
    def _run(seed_val):
        counter.step = 0
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
        print(f"  iter {i+1}: {ms:.0f} ms ({n/(ms/1000):.3f} fps)")
    mean_ms = statistics.mean(lat); fps = n / (mean_ms / 1000)
    return {"model": REPO, "kernel": "step_aware_kv_precision",
            "int4_until_step": int4_until_step, "fp8_until_step": fp8_until_step,
            "n_attention_modules_patched": len(procs),
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


@app.local_entrypoint(name="cogvideox_5b_kv_precision_modal")
def main(int4_until_step: int = 20, fp8_until_step: int = 40, num_iters: int = 1):
    r = run_cogvideox_kv_precision.remote(
        int4_until_step=int4_until_step, fp8_until_step=fp8_until_step, num_iters=num_iters)
    import json; print(json.dumps(r, indent=2))
    print(f"\nKV-precision speedup: {r.get('speedup_vs_baseline', 'N/A')}×")
    sys.path.insert(0, "/root/benchmarks")
    from result_store import save_result
    save_result(r, "optimised_kernels", "cogvideox_5b_kv_precision")
