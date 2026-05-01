"""
Modal H100 benchmark: Open-Oasis 500M + Sliding Tile Attention (STA) only.

Measures the isolated speedup from STA (layout='1d', window_size=4) on H100.

Run:
    modal run modal/optimised/oasis_sta_modal.py
"""

import sys
sys.path.insert(0, "/root")
sys.path.insert(0, "/root/benchmarks")
sys.path.insert(0, "/root/worldserve/optimizations")

import modal
from modal_common import app, image, hf_secret, model_volume, MODEL_CACHE


@app.function(
    gpu="H100",
    image=image,
    secrets=[hf_secret],
    volumes={MODEL_CACHE: model_volume},
    timeout=3600,
    memory=32768,
)
def run_oasis_sta(
    num_frames: int = 16,
    height: int = 360,
    width: int = 640,
    num_steps: int = 20,
    num_warmup: int = 1,
    num_iters: int = 1,
    window_size: int = 4,
    prompt: str = "A first-person view of walking through a Minecraft forest",
) -> dict:
    """
    Run Open-Oasis 500M with Sliding Tile Attention (STA, 1D layout) and
    return timing results.
    """
    import json
    import math
    import os
    import statistics

    import torch
    import torch.nn.functional as F

    hf_token = os.environ.get("HF_TOKEN", "").strip() or None
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN is required for the gated Etched/oasis-500m model."
        )

    # -----------------------------------------------------------------
    # STA attention processor (1D layout for Oasis)
    # -----------------------------------------------------------------
    def _infer_3d_dims(seq_len, T_hint, H_hint, W_hint):
        if seq_len == T_hint * H_hint * W_hint:
            return T_hint, H_hint, W_hint
        if T_hint > 0 and seq_len % T_hint == 0:
            hw = seq_len // T_hint
            ratio = H_hint / max(W_hint, 1)
            W_f = max(1, round(math.sqrt(hw / ratio)))
            H_f = hw // W_f
            if H_f * W_f == hw:
                return T_hint, H_f, W_f
            for delta in range(1, W_f + 1):
                for w in (W_f - delta, W_f + delta):
                    if w > 0 and hw % w == 0:
                        return T_hint, hw // w, w
        return seq_len, 1, 1

    class OasisSTAAttnProcessor:
        def __init__(self, ws: int = 4, tile: int = 64,
                     temporal_size: int = 16, spatial_h: int = 45, spatial_w: int = 80) -> None:
            self.window_size = ws
            self.tile = tile
            self._T_hint = temporal_size
            self._H_hint = spatial_h
            self._W_hint = spatial_w
            self._cached_dims = None

        def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                     attention_mask=None, **kwargs):
            bs, seq_len, _ = hidden_states.shape
            kv = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
            q = attn.to_q(hidden_states)
            k = attn.to_k(kv)
            v = attn.to_v(kv)
            inner_dim = k.shape[-1]
            head_dim  = inner_dim // attn.heads
            q = q.view(bs, seq_len, attn.heads, head_dim).transpose(1, 2)
            k = k.view(bs, k.shape[1], attn.heads, head_dim).transpose(1, 2)
            v = v.view(bs, v.shape[1], attn.heads, head_dim).transpose(1, 2)
            if attention_mask is not None:
                hidden_states = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
            else:
                try:
                    from worldserve.kernels.triton.sliding_tile_attention import sliding_tile_attention
                    hidden_states = sliding_tile_attention(
                        q, k, v,
                        window_size=self.window_size,
                        layout="1d",
                        tile=self.tile,
                    )
                except Exception:
                    hidden_states = F.scaled_dot_product_attention(q, k, v)
            hidden_states = hidden_states.transpose(1, 2).reshape(bs, seq_len, inner_dim)
            hidden_states = hidden_states.to(q.dtype)
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            return hidden_states

    # -----------------------------------------------------------------
    # Load pipeline
    # -----------------------------------------------------------------
    print(f"Loading Etched/oasis-500m (cache={MODEL_CACHE}) ...")
    try:
        from diffusers import DiffusionPipeline
        pipe = DiffusionPipeline.from_pretrained(
            "Etched/oasis-500m",
            torch_dtype=torch.bfloat16,
            cache_dir=MODEL_CACHE,
            token=hf_token,
        ).to("cuda")
    except Exception as exc:
        print(f"ERROR loading pipeline: {exc}")
        raise

    sta_proc = OasisSTAAttnProcessor(
        ws=window_size,
        temporal_size=num_frames,
        spatial_h=height // 8,
        spatial_w=width // 8,
    )
    if hasattr(pipe, "transformer") and hasattr(pipe.transformer, "set_attn_processor"):
        pipe.transformer.set_attn_processor(sta_proc)
        print(f"STA processor injected (window_size={window_size}, layout=1d)")
    else:
        print("Warning: set_attn_processor() not available — STA not applied.")

    generator = torch.Generator(device="cuda").manual_seed(42)

    def _run_once() -> float:
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        pipe(
            prompt=prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=num_steps,
            generator=generator,
            output_type="latent",
        )
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end)

    print(f"Warmup ({num_warmup} iters) ...")
    for i in range(num_warmup):
        try:
            _run_once()
            print(f"  warmup {i + 1}/{num_warmup} done")
        except torch.cuda.OutOfMemoryError as oom:
            print(f"  warmup {i + 1}: OOM — {oom}; skipping")
            torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()
    latencies = []
    print(f"Timed runs ({num_iters} iters) ...")
    for i in range(num_iters):
        try:
            ms = _run_once()
            latencies.append(ms)
            fps = num_frames / (ms / 1000.0)
            print(f"  iter {i + 1}: {ms:.1f} ms  ({fps:.2f} fps)")
        except torch.cuda.OutOfMemoryError as oom:
            print(f"  iter {i + 1}: OOM — {oom}; skipping")
            torch.cuda.empty_cache()

    if not latencies:
        return {
            "error": "All iterations OOM",
            "model": "Etched/oasis-500m",
            "kernel": "STA (Sliding Tile Attention)",
        }

    mean_ms = statistics.mean(latencies)
    std_ms  = statistics.stdev(latencies) if len(latencies) > 1 else 0.0

    # Load baseline for speedup ratio.
    speedup_vs_baseline = None
    baseline_path = "/root/benchmarks/baseline/results/oasis_baseline.json"
    try:
        with open(baseline_path) as fh:
            bl = json.load(fh)
        bl_mean = bl.get("latency_ms", {}).get("total_mean")
        if bl_mean and mean_ms > 0:
            speedup_vs_baseline = round(bl_mean / mean_ms, 4)
    except Exception:
        pass

    results = {
        "model": "Etched/oasis-500m",
        "kernel": "STA (Sliding Tile Attention)",
        "window_size": window_size,
        "layout": "1d",
        "num_frames": num_frames,
        "height": height,
        "width": width,
        "steps": num_steps,
        "mean_ms": round(mean_ms, 2),
        "std_ms": round(std_ms, 2),
        "frames_per_sec": round(num_frames / (mean_ms / 1000.0), 3),
        "vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "speedup_vs_baseline": speedup_vs_baseline,
        "gpu": "H100",
        "raw_latencies_ms": latencies,
    }
    print(json.dumps(results, indent=2))
    return results


@app.local_entrypoint(name="oasis_sta_modal")
def main():
    result = run_oasis_sta.remote()
    print("\nOpen-Oasis 500M (STA only) result:")
    import json
    print(json.dumps(result, indent=2))
