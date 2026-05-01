"""
Modal H100 benchmark: Open-Oasis 500M baseline + torch.compile(mode="reduce-overhead").

Hypothesis: the bottleneck on the canonical generate.py sampler is per-SDPA
Python dispatch + kernel launch overhead, not GPU compute. torch.compile with
reduce-overhead mode installs CUDA graphs that fuse launches and skip Python
dispatch, which is the standard 1.5-2x win on PyTorch DiT inference.

Caveats specific to Oasis:
  * Sliding window in generate.py grows the temporal-attention K/V from
    T=1 to T=model.max_frames before plateauing. Each new T triggers a
    recompile with reduce-overhead mode → warmup pays N recompiles.
  * After frame `model.max_frames` the shape is static and graphs are reused.
  * dynamic=True lets one compiled artifact cover multiple shapes (slower
    per-call but no recompiles); we use dynamic=False so each shape gets
    its own CUDA graph (faster per-call once warmed).

Run:
    PYTHONPATH=benchmarks modal run benchmarks/optimised/oasis_compile_modal.py \
        --num-frames 75 --num-iters 2 --num-warmup 1
"""

import sys
sys.path.insert(0, "/root")
sys.path.insert(0, "/root/benchmarks")

import modal
from modal_common import app, image, hf_secret, model_volume, MODEL_CACHE

SCALING_FACTOR = 0.07843137255


@app.function(
    gpu="H100",
    image=image,
    secrets=[hf_secret],
    volumes={MODEL_CACHE: model_volume},
    timeout=3600,
    memory=32768,
)
def run_oasis_compile(
    num_frames: int = 75,
    n_prompt: int = 1,
    ddim_steps: int = 10,
    max_noise_level: int = 1000,
    stabilization_level: int = 15,
    noise_abs_max: float = 20.0,
    seed: int = 42,
    num_warmup: int = 1,
    num_iters: int = 2,
    compile_mode: str = "reduce-overhead",
    prompt_file: str = "sample_image_0.png",
    actions_file: str = "sample_actions_0.one_hot_actions.pt",
) -> dict:
    """Baseline canonical-sampler Oasis + torch.compile on the DiT."""
    import json
    import os
    import statistics

    import torch

    sys.path.insert(0, "/root/open-oasis")

    from einops import rearrange
    from huggingface_hub import hf_hub_download
    from dit import DiT_models
    from vae import VAE_models
    from safetensors.torch import load_file as safetensors_load
    from utils import sigmoid_beta_schedule, load_prompt, load_actions

    hf_token = os.environ.get("HF_TOKEN", "").strip() or None

    # ─────────────────────────────────────────────────────────────────────
    # Load model + VAE
    # ─────────────────────────────────────────────────────────────────────
    print("Downloading Etched/oasis-500m checkpoints ...")
    dit_ckpt = hf_hub_download("Etched/oasis-500m", "oasis500m.safetensors",
                               cache_dir=MODEL_CACHE, token=hf_token)
    vae_ckpt = hf_hub_download("Etched/oasis-500m", "vit-l-20.safetensors",
                               cache_dir=MODEL_CACHE, token=hf_token)

    print("Loading DiT-S/2 ...")
    model = DiT_models["DiT-S/2"]()
    model.load_state_dict(safetensors_load(dit_ckpt), strict=False)
    model = model.to("cuda").eval()

    print("Loading ViT-L VAE ...")
    vae = VAE_models["vit-l-20-shallow-encoder"]()
    vae.load_state_dict(safetensors_load(vae_ckpt), strict=False)
    vae = vae.to("cuda").eval()

    # ─────────────────────────────────────────────────────────────────────
    # torch.compile the DiT
    #
    # Oasis's rotary_embedding_torch library has buffer-mutation patterns
    # that torch.compile can't trace cleanly (cached_freqs_seq_len.copy_(int)).
    # We suppress those errors so dynamo falls back to eager for that subgraph
    # and still compiles everything else (the heavy attention + MLP blocks).
    # ─────────────────────────────────────────────────────────────────────
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.cache_size_limit = 256

    # Save eager reference for baseline timing
    eager_model = model

    # Fix: rotary_embedding_torch mutates an int into a tensor buffer
    # (cached_freqs_seq_len.copy_(int)) which AOT autograd refuses to trace.
    # Don't delete the buffer (the forward needs it) — instead wrap each
    # RotaryEmbedding.forward in torch._dynamo.disable so dynamo treats it
    # as a black box and stops trying to trace the mutation.
    rotary_count = 0
    for module in model.modules():
        if module.__class__.__name__ in ("RotaryEmbedding", "AxialRotaryEmbedding"):
            module.forward = torch._dynamo.disable(module.forward)
            rotary_count += 1
    print(f"Wrapped {rotary_count} RotaryEmbedding forwards with torch._dynamo.disable")

    compiled = False
    compiled_model = eager_model
    try:
        print(f"\nCompiling DiT with torch.compile(mode={compile_mode!r}) ...")
        compiled_model = torch.compile(
            eager_model,
            mode=compile_mode,
            fullgraph=False,
            dynamic=True,
        )
        compiled = True
        print("Compile dispatch wrapper installed (compile-on-first-call)")
    except Exception as exc:
        print(f"Compile failed, running eager: {exc}")

    # ─────────────────────────────────────────────────────────────────────
    # Real prompt + actions
    # ─────────────────────────────────────────────────────────────────────
    prompt_path = f"/root/open-oasis/sample_data/{prompt_file}"
    actions_path = f"/root/open-oasis/sample_data/{actions_file}"
    total_frames = n_prompt + num_frames

    x_pixels = load_prompt(prompt_path, n_prompt_frames=n_prompt).to("cuda")
    actions = load_actions(actions_path).to("cuda")[:, :total_frames]

    H_pix, W_pix = x_pixels.shape[-2:]
    flat = rearrange(x_pixels, "b t c h w -> (b t) c h w")
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            z = vae.encode(flat * 2 - 1).mean * SCALING_FACTOR
    H_lat = H_pix // vae.patch_size
    W_lat = W_pix // vae.patch_size
    prompt_latent = rearrange(z, "(b t) (h w) c -> b t c h w",
                              t=n_prompt, h=H_lat, w=W_lat)

    # ─────────────────────────────────────────────────────────────────────
    # Schedule
    # ─────────────────────────────────────────────────────────────────────
    betas = sigmoid_beta_schedule(max_noise_level).float().to("cuda")
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    alphas_cumprod = rearrange(alphas_cumprod, "T -> T 1 1 1")
    noise_range = torch.linspace(-1, max_noise_level - 1, ddim_steps + 1)

    model_max_frames = getattr(eager_model, "max_frames", total_frames + 1)

    # ─────────────────────────────────────────────────────────────────────
    # Canonical generate.py loop, parameterized on the network handle
    # ─────────────────────────────────────────────────────────────────────
    def _run_once(net, seed_val: int = 42) -> float:
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        x = prompt_latent.clone()
        B = x.shape[0]

        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()

        for i in range(n_prompt, total_frames):
            chunk = torch.randn((B, 1, *x.shape[-3:]), device="cuda")
            chunk = torch.clamp(chunk, -noise_abs_max, +noise_abs_max)
            x = torch.cat([x, chunk], dim=1)
            start_frame = max(0, i + 1 - model_max_frames)

            for noise_idx in reversed(range(1, ddim_steps + 1)):
                t_ctx = torch.full((B, i), stabilization_level - 1,
                                   dtype=torch.long, device="cuda")
                t = torch.full((B, 1), int(noise_range[noise_idx].item()),
                               dtype=torch.long, device="cuda")
                t_next = torch.full((B, 1), int(noise_range[noise_idx - 1].item()),
                                    dtype=torch.long, device="cuda")
                t_next = torch.where(t_next < 0, t, t_next)
                t = torch.cat([t_ctx, t], dim=1)
                t_next = torch.cat([t_ctx, t_next], dim=1)

                x_curr = x.clone()[:, start_frame:]
                t = t[:, start_frame:]
                t_next = t_next[:, start_frame:]

                with torch.no_grad():
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        v = net(x_curr, t, actions[:, start_frame: i + 1])

                x_start = (alphas_cumprod[t].sqrt() * x_curr
                           - (1 - alphas_cumprod[t]).sqrt() * v)
                x_noise = (((1 / alphas_cumprod[t]).sqrt() * x_curr - x_start)
                           / (1 / alphas_cumprod[t] - 1).sqrt())
                alpha_next = alphas_cumprod[t_next].clone()
                alpha_next[:, :-1] = torch.ones_like(alpha_next[:, :-1])
                if noise_idx == 1:
                    alpha_next[:, -1:] = torch.ones_like(alpha_next[:, -1:])
                x_pred = alpha_next.sqrt() * x_start + x_noise * (1 - alpha_next).sqrt()
                x[:, -1:] = x_pred[:, -1:]

        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e)

    # ─────────────────────────────────────────────────────────────────────
    # Same-session BASELINE (eager) for fair speedup comparison
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n=== BASELINE eager ({num_warmup} warmup + {num_iters} timed) ===")
    for i in range(num_warmup):
        ms = _run_once(eager_model, seed_val=seed)
        print(f"  warmup {i + 1}/{num_warmup}: {ms:.1f} ms ({num_frames / (ms / 1000.0):.2f} fps)")
    baseline_latencies: list = []
    for i in range(num_iters):
        ms = _run_once(eager_model, seed_val=seed)
        baseline_latencies.append(ms)
        print(f"  iter {i + 1}: {ms:.1f} ms ({num_frames / (ms / 1000.0):.2f} fps)")
    baseline_mean = statistics.mean(baseline_latencies)
    baseline_fps = num_frames / (baseline_mean / 1000.0)

    # ─────────────────────────────────────────────────────────────────────
    # Compiled warmup (triggers JIT) + timed
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n=== COMPILED ({compile_mode}) warmup + timed ===")
    print(f"Warmup ({num_warmup} iters — first run includes JIT compilation) ...")
    for i in range(num_warmup):
        ms = _run_once(compiled_model, seed_val=seed)
        print(f"  warmup {i + 1}/{num_warmup}: {ms:.1f} ms ({num_frames / (ms / 1000.0):.2f} fps)")

    torch.cuda.reset_peak_memory_stats()
    latencies: list = []
    print(f"Timed runs ({num_iters} iters) ...")
    for i in range(num_iters):
        try:
            ms = _run_once(compiled_model, seed_val=seed)
            latencies.append(ms)
            print(f"  iter {i + 1}: {ms:.1f} ms ({num_frames / (ms / 1000.0):.2f} fps)")
        except torch.cuda.OutOfMemoryError as oom:
            print(f"  iter {i + 1}: OOM — {oom}; skipping")
            torch.cuda.empty_cache()

    if not latencies:
        return {"error": "All iterations OOM", "model": "Etched/oasis-500m"}

    mean_ms = statistics.mean(latencies)
    speedup = baseline_mean / mean_ms

    results = {
        "model": "Etched/oasis-500m",
        "kernel": f"torch_compile_{compile_mode.replace('-', '_')}",
        "compile_mode": compile_mode,
        "compiled_ok": compiled,
        "num_frames": num_frames,
        "n_prompt": n_prompt,
        "ddim_steps": ddim_steps,
        "stabilization_level": stabilization_level,
        "total_dit_fwd_passes": num_frames * ddim_steps,
        "baseline_latency_ms_mean": round(baseline_mean, 2),
        "baseline_fps": round(baseline_fps, 3),
        "latency_ms_mean": round(mean_ms, 2),
        "latency_ms_std": round(
            statistics.stdev(latencies) if len(latencies) > 1 else 0.0, 2
        ),
        "latency_per_frame_ms": round(mean_ms / num_frames, 2),
        "frames_per_sec": round(num_frames / (mean_ms / 1000.0), 3),
        "speedup": round(speedup, 3),
        "vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "raw_latencies_ms": latencies,
        "raw_baseline_latencies_ms": baseline_latencies,
        "gpu": "H100",
    }
    print(f"\n=== SUMMARY ===")
    print(f"Baseline:  {baseline_mean:.1f} ms ({baseline_fps:.2f} fps)")
    print(f"Compiled:  {mean_ms:.1f} ms ({num_frames / (mean_ms / 1000.0):.2f} fps)")
    print(f"Speedup:   {speedup:.3f}x")
    print("\n" + json.dumps(results, indent=2, default=str))
    return results


@app.local_entrypoint(name="oasis_compile_modal")
def main(
    num_frames: int = 75,
    num_iters: int = 2,
    num_warmup: int = 1,
    compile_mode: str = "reduce-overhead",
):
    result = run_oasis_compile.remote(
        num_frames=num_frames,
        num_iters=num_iters,
        num_warmup=num_warmup,
        compile_mode=compile_mode,
    )
    print("\n=== torch.compile result ===")
    print(f"compile_mode: {result.get('compile_mode')}")
    print(f"latency_ms_mean: {result.get('latency_ms_mean')}")
    print(f"frames_per_sec:  {result.get('frames_per_sec')}")

    sys.path.insert(0, "/root/benchmarks")
    from result_store import save_result
    save_result(result, "optimised_kernels", f"oasis_compile_{compile_mode}_f{num_frames}")
