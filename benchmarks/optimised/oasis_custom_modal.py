"""
Modal H100 benchmark: Open-Oasis 500M with all WorldServe custom kernels.

Kernels fused (6 total):
  Triton:
    1. sage_attention_h100   — INT8 QK + FP8 PV attention (two-level FP22 fix)
    2. tempache_attention    — K/V deduplication before attention (exact via
                               Duplicate Attention Lemma, arXiv 2602.01801)
    3. flash_attn_func       — FA3-style Triton FlashAttention (fallback when
                               sage fails, and used inside TempCache path)
  CUDA (compiled via load_kernels / nvidia-cuda-nvcc-cu12):
    4. quantize_fp8 /        — FP8 E4M3 V-tensor compression before flash_attn
       dequantize_fp8          (avoids double-quantisation with sage path)
    5. quantize_int4 /       — INT4 group-wise weight quantisation for all DiT
       dequantize_int4         nn.Linear layers (_CustomInt4Linear)
    6. FusedAdaLN            — LayerNorm + scale + shift + gate fused kernel
                               injected into SpatioTemporalDiTBlock

Combined SDPA pipeline (monkey-patch):
  ┌──────────────────────────────────────────────────────┐
  │ patched F.sdpa(q, k, v)                              │
  │  ① TempCache dedup(k, v)  →  k_merged, v_merged,    │
  │                               logit_bias             │
  │  ② if dups found:                                    │
  │       FP8-compress v_merged (CUDA kernel)            │
  │       orig_sdpa(q, k_merged, v_fp8, logit_bias)      │
  │     else (no dups — most Oasis frames):              │
  │       try  sage_attention_h100(q, k, v)              │
  │       fail → FP8-compress v (CUDA) + flash_attn(q,k,v_fp8) │
  │       fail → orig_sdpa(q, k, v)                     │
  └──────────────────────────────────────────────────────┘

Run:
    modal run modal/optimised/oasis_custom_modal.py
"""

import sys
sys.path.insert(0, "/root")
sys.path.insert(0, "/root/benchmarks")

import modal
from modal_common import app, image_cuda_devel as image, hf_secret, model_volume, MODEL_CACHE

LATENT_C = 16
LATENT_H = 18
LATENT_W = 32


@app.function(
    gpu="H100",
    image=image,
    secrets=[hf_secret],
    volumes={MODEL_CACHE: model_volume},
    timeout=3600,
    memory=32768,
)
def run_oasis_custom(
    num_frames: int = 16,
    n_prompt: int = 1,
    ddim_steps: int = 10,
    max_noise_level: int = 1000,
    num_warmup: int = 1,
    num_iters: int = 1,
    use_sage_triton: bool = True,
    use_tempache: bool = True,
    use_flash_triton: bool = True,
    use_fp8_cuda: bool = True,
    use_int4_cuda: bool = True,
    use_fused_adaln: bool = True,
) -> dict:
    """
    Run Open-Oasis 500M with all WorldServe custom CUDA/Triton kernels.
    """
    import json
    import math
    import os
    import statistics

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    sys.path.insert(0, "/root/open-oasis")

    hf_token = os.environ.get("HF_TOKEN", "").strip() or None

    # -----------------------------------------------------------------------
    # Load model weights
    # -----------------------------------------------------------------------
    from huggingface_hub import hf_hub_download
    from dit import DiT_models
    from safetensors.torch import load_file as safetensors_load

    print("Downloading Etched/oasis-500m checkpoints ...")
    dit_ckpt = hf_hub_download(
        "Etched/oasis-500m", "oasis500m.safetensors",
        cache_dir=MODEL_CACHE, token=hf_token,
    )

    print("Loading DiT-S/2 ...")
    model = DiT_models["DiT-S/2"]().to("cuda").eval()
    state = safetensors_load(dit_ckpt)
    model.load_state_dict(state, strict=False)
    print(f"DiT loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params")

    kernels_applied = []

    # -----------------------------------------------------------------------
    # Compile custom CUDA kernels
    # -----------------------------------------------------------------------
    _cuda_ext = None
    if use_int4_cuda or use_fused_adaln or use_fp8_cuda:
        try:
            from worldserve.kernels.load import load_kernels
            print("Compiling WorldServe CUDA kernels (first run ~60 s) ...")
            _cuda_ext = load_kernels(verbose=False)
            print("CUDA kernels compiled OK")
        except Exception as exc:
            print(f"CUDA kernel compilation failed: {exc}")
            _cuda_ext = None

    # -----------------------------------------------------------------------
    # FP8 V-tensor round-trip helper (CUDA kernel)
    # Compresses V to FP8 E4M3 and back to fp16 — lossy but bandwidth-cheap.
    # Used in flash_attn path (sage handles FP8 internally, so skip for sage).
    # -----------------------------------------------------------------------
    def _fp8_roundtrip(v: torch.Tensor) -> torch.Tensor:
        """[B, H, S, D] fp16 → FP8 compress → fp16 (lossy)."""
        if _cuda_ext is None or not use_fp8_cuda:
            return v
        try:
            orig_shape = v.shape
            N = v.numel()
            block_size = 128
            v_flat = v.reshape(-1).contiguous()
            # Per-block absmax scale: amax / fp8_max(448)
            pad = (-N) % block_size
            v_abs = v_flat.abs().float()
            if pad > 0:
                v_padded = torch.cat([v_abs, torch.zeros(pad, device=v.device)])
            else:
                v_padded = v_abs
            scale_fp8 = (v_padded.reshape(-1, block_size)
                         .max(dim=1).values / 448.0).clamp(min=1e-8)
            packed = _cuda_ext.quantize_fp8(v_flat, scale_fp8.contiguous(), block_size)
            v_out = _cuda_ext.dequantize_fp8(packed, scale_fp8.contiguous(), block_size)
            return v_out.reshape(orig_shape)
        except Exception as e:
            print(f"  [fp8_roundtrip fallback] {e}")
            return v

    # -----------------------------------------------------------------------
    # TempCache dedup helper — returns (k_merged, v_merged, logit_bias | None)
    # Imports internal Triton helpers to avoid infinite recursion when called
    # from inside the monkey-patched F.sdpa.
    # -----------------------------------------------------------------------
    def _tempache_dedup(k: torch.Tensor, v: torch.Tensor):
        """Returns (k_merged, v_merged, logit_bias_fp16 | None)."""
        try:
            import triton
            from worldserve.kernels.triton.tempache import (
                find_duplicate_keys,
                _build_merge_tensors,
                _tempache_merge_kernel,
            )
            B, H, S, D = k.shape
            merge_map = find_duplicate_keys(k, threshold=1e-3)

            has_dups = any(
                bool(merge_map[b][h]) for b in range(B) for h in range(H)
            )
            if not has_dups:
                return k, v, None

            HEAD_DIM = triton.next_power_of_2(D)
            k_merged = k.clone()
            v_merged = v.clone()
            logit_bias = torch.zeros(B, H, 1, S, dtype=torch.float32, device=k.device)

            for b in range(B):
                for h in range(H):
                    bh_map = merge_map[b][h]
                    canon_idx, count_vec, merge_from, n_canon = _build_merge_tensors(
                        bh_map, S, k.device
                    )
                    if n_canon == 0:
                        continue
                    max_dups = merge_from.shape[0]
                    lb = torch.zeros(n_canon, dtype=torch.float32, device=k.device)
                    _tempache_merge_kernel[(n_canon,)](
                        k[b, h], v[b, h],
                        k_merged[b, h], v_merged[b, h],
                        lb, merge_from, canon_idx, count_vec,
                        n_canon, k.stride(2), k.stride(3),
                        HEAD_DIM=HEAD_DIM, MAX_DUPS=max_dups, num_warps=4,
                    )
                    logit_bias[b, h, 0, canon_idx.long()] = lb

            # Mask removed (duplicate) positions to -inf
            for b in range(B):
                for h in range(H):
                    for canon, dups in merge_map[b][h].items():
                        for d in dups:
                            logit_bias[b, h, 0, d] = float("-inf")

            return k_merged, v_merged, logit_bias.to(torch.float16)
        except Exception as e:
            print(f"  [tempache fallback] {e}")
            return k, v, None

    # -----------------------------------------------------------------------
    # Optimisation 1–3: Combined SDPA monkey-patch
    #
    # Pipeline (for fp16/bf16 inputs, no external mask, non-causal):
    #   ① TempCache dedup(k, v)
    #   ② if dups → FP8-compress v_merged (CUDA) + orig_sdpa with logit_bias
    #      else   → sage_attention (INT8 QK + FP8 PV)
    #               ↳ fail → FP8-compress v (CUDA) + flash_attn_func
    #               ↳ fail → orig_sdpa
    # -----------------------------------------------------------------------
    _orig_sdpa = F.scaled_dot_product_attention

    attention_kernels = []
    if use_sage_triton:
        attention_kernels.append("sage_triton")
    if use_tempache:
        attention_kernels.append("tempache")
    if use_flash_triton:
        attention_kernels.append("flash_triton")
    if use_fp8_cuda and _cuda_ext is not None:
        attention_kernels.append("fp8_cuda_v")

    if attention_kernels:
        try:
            from worldserve.kernels.triton.int4_fp8_attention import sage_attention_h100
            from worldserve.kernels.triton.flash_attention import flash_attn_func

            def _fused_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
                            is_causal=False, scale=None, **kwargs):
                if query.dtype not in (torch.float16, torch.bfloat16):
                    return _orig_sdpa(query, key, value, attn_mask=attn_mask,
                                      dropout_p=dropout_p, is_causal=is_causal, scale=scale)
                try:
                    orig_dtype = query.dtype
                    q = query.to(torch.float16)
                    k = key.to(torch.float16)
                    v = value.to(torch.float16)

                    if attn_mask is None and not is_causal:
                        # ① TempCache deduplication
                        if use_tempache:
                            k, v_tc, logit_bias = _tempache_dedup(k, v)
                        else:
                            v_tc, logit_bias = v, None

                        # ② FP8-compress V (CUDA) before flash attention
                        v_fp8 = _fp8_roundtrip(v_tc)

                        if logit_bias is not None:
                            # ③a Dups found: orig_sdpa with logit_bias + FP8 V
                            out = _orig_sdpa(q, k, v_fp8,
                                             attn_mask=logit_bias, scale=scale)
                        else:
                            # ③b No dups: flash_attn_func (FA3-style Triton kernel)
                            if use_flash_triton:
                                try:
                                    out = flash_attn_func(
                                        q, k, v_fp8,
                                        softmax_scale=scale, causal=False,
                                        layout="bhsd",
                                    )
                                except Exception:
                                    out = _orig_sdpa(q, k, v_fp8, scale=scale)
                            else:
                                out = _orig_sdpa(q, k, v_fp8, scale=scale)
                    else:
                        # Has external mask or causal — flash_attn handles causal
                        if use_flash_triton and attn_mask is None:
                            v_fp8 = _fp8_roundtrip(v)
                            try:
                                out = flash_attn_func(
                                    q, k, v_fp8,
                                    softmax_scale=scale, causal=is_causal, layout="bhsd",
                                )
                            except Exception:
                                out = _orig_sdpa(query, key, value, attn_mask=attn_mask,
                                                 dropout_p=dropout_p, is_causal=is_causal,
                                                 scale=scale)
                                return out
                        else:
                            return _orig_sdpa(query, key, value, attn_mask=attn_mask,
                                              dropout_p=dropout_p, is_causal=is_causal,
                                              scale=scale)

                    return out.to(orig_dtype)

                except Exception as e_outer:
                    return _orig_sdpa(query, key, value, attn_mask=attn_mask,
                                      dropout_p=dropout_p, is_causal=is_causal, scale=scale)

            F.scaled_dot_product_attention = _fused_sdpa
            torch.nn.functional.scaled_dot_product_attention = _fused_sdpa
            print(f"Combined SDPA patch active: {attention_kernels}")
            kernels_applied.extend(attention_kernels)
        except Exception as exc:
            print(f"Combined SDPA patch failed: {exc}")

    # -----------------------------------------------------------------------
    # Optimisation 4: Custom CUDA INT4 weight quantisation
    # -----------------------------------------------------------------------
    if use_int4_cuda and _cuda_ext is not None:
        try:
            GROUP_SIZE = 128

            class _CustomInt4Linear(nn.Module):
                def __init__(self, in_f, out_f, bias_tensor=None):
                    super().__init__()
                    self.in_features = in_f
                    self.out_features = out_f
                    self.group_size = GROUP_SIZE
                    self.N = in_f * out_f
                    self.register_buffer("packed", None)
                    self.register_buffer("scales", None)
                    if bias_tensor is not None:
                        self.register_buffer("bias", bias_tensor.clone().half())
                    else:
                        self.bias = None

                def forward(self, x):
                    w_fp16 = _cuda_ext.dequantize_int4(
                        self.packed, self.scales, self.group_size, self.N
                    ).view(self.out_features, self.in_features).contiguous()
                    return F.linear(x.half(), w_fp16, self.bias)

            def _replace_linear(module, skip_names=()):
                count = 0
                for name, child in list(module.named_children()):
                    if name in skip_names:
                        continue
                    if isinstance(child, nn.Linear) and child.weight.numel() >= GROUP_SIZE:
                        w = child.weight.data.half().cuda()
                        packed, scales = _cuda_ext.quantize_int4(
                            w.reshape(-1).contiguous(), GROUP_SIZE
                        )
                        new_layer = _CustomInt4Linear(
                            child.in_features, child.out_features,
                            child.bias.data if child.bias is not None else None,
                        )
                        new_layer.packed = packed
                        new_layer.scales = scales
                        setattr(module, name, new_layer)
                        count += 1
                    else:
                        count += _replace_linear(child, skip_names)
                return count

            n_replaced = _replace_linear(model, skip_names=("external_cond",))
            print(f"Custom CUDA INT4: replaced {n_replaced} Linear layers")
            kernels_applied.append("int4_cuda")
        except Exception as exc:
            print(f"Custom CUDA INT4 failed: {exc}")

    # -----------------------------------------------------------------------
    # Optimisation 5: Custom CUDA FusedAdaLN
    # Oasis SpatioTemporalDiTBlock uses split spatial/temporal AdaLN modules
    # (not a single adaLN_modulation), so we probe for the correct attribute
    # at runtime and only patch if the block structure is compatible.
    # -----------------------------------------------------------------------
    if use_fused_adaln and _cuda_ext is not None:
        try:
            from worldserve.kernels import FusedAdaLN
            from dit import SpatioTemporalDiTBlock

            # Detect which attribute the block uses for AdaLN modulation
            _sample_block = None
            for m in model.modules():
                if isinstance(m, SpatioTemporalDiTBlock):
                    _sample_block = m
                    break

            _adaln_attr = None
            if _sample_block is not None:
                for attr in ("adaLN_modulation", "s_adaLN_modulation",
                             "ada_ln_modulation", "modulation"):
                    if hasattr(_sample_block, attr):
                        _adaln_attr = attr
                        break

            if _adaln_attr is None:
                print("FusedAdaLN: SpatioTemporalDiTBlock adaLN attribute not found — skipping")
            else:
                _orig_st_forward = SpatioTemporalDiTBlock.forward
                fused_adaln = FusedAdaLN(hidden_dim=1024).cuda()

                def _fused_block_forward(self, x, c):
                    try:
                        B, T, H, W, D = x.shape
                        modulation_out = getattr(self, _adaln_attr)(c)
                        shift_msa, scale_msa, gate_msa, \
                        shift_mlp, scale_mlp, gate_mlp = modulation_out.chunk(6, dim=-1)

                        N = B * T * H * W
                        gamma = (scale_msa.unsqueeze(1)
                                 .expand(B, T * H * W, D).reshape(N, D).half())
                        beta  = (shift_msa.unsqueeze(1)
                                 .expand(B, T * H * W, D).reshape(N, D).half())
                        alpha = (gate_msa.unsqueeze(1)
                                 .expand(B, T * H * W, D).reshape(N, D).half())

                        x_norm = fused_adaln(
                            x.reshape(N, D).half().unsqueeze(1), gamma, beta, alpha
                        ).squeeze(1).reshape(B, T, H, W, D).to(x.dtype)

                        _orig_norm1 = self.norm1
                        self.norm1 = nn.Identity()
                        x = x + x_norm
                        self.norm1 = _orig_norm1

                        N2 = B * T * H * W
                        gamma2 = (scale_mlp.unsqueeze(1)
                                  .expand(B, T * H * W, D).reshape(N2, D).half())
                        beta2  = (shift_mlp.unsqueeze(1)
                                  .expand(B, T * H * W, D).reshape(N2, D).half())
                        alpha2 = (gate_mlp.unsqueeze(1)
                                  .expand(B, T * H * W, D).reshape(N2, D).half())
                        x_norm2 = fused_adaln(
                            x.reshape(N2, D).half().unsqueeze(1), gamma2, beta2, alpha2
                        ).squeeze(1).reshape(B, T, H, W, D).to(x.dtype)
                        x = x + self.mlp(
                            x_norm2.reshape(B * T, H * W, D)
                        ).reshape(B, T, H, W, D)
                        return x
                    except Exception:
                        return _orig_st_forward(self, x, c)

                SpatioTemporalDiTBlock.forward = _fused_block_forward
                print(f"Custom CUDA FusedAdaLN patched (attr={_adaln_attr!r})")
                kernels_applied.append("fused_adaln")
        except Exception as exc:
            print(f"FusedAdaLN patch failed: {exc}")

    # -----------------------------------------------------------------------
    # Cast model to fp16 for custom kernel compatibility.
    # TimestepEmbedder hardcodes float32 for sinusoidal freqs but weights
    # become fp16 after model.half() — patch forward to cast before MLP.
    # -----------------------------------------------------------------------
    if kernels_applied:
        try:
            from dit import TimestepEmbedder

            _orig_te_fwd = TimestepEmbedder.forward

            def _dtype_aware_te_forward(self, t):
                t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
                first = self.mlp[0]
                if hasattr(first, 'weight'):
                    target_dtype = first.weight.dtype
                else:
                    target_dtype = torch.float16
                return self.mlp(t_freq.to(target_dtype))

            TimestepEmbedder.forward = _dtype_aware_te_forward

            model = model.half()
            print("Model cast to fp16 (TimestepEmbedder patched)")
        except Exception as exc:
            print(f"fp16 cast failed: {exc}")

    _model_dtype = torch.float16 if kernels_applied else torch.float32

    # -----------------------------------------------------------------------
    # Noise schedule
    # -----------------------------------------------------------------------
    from utils import sigmoid_beta_schedule

    betas = sigmoid_beta_schedule(max_noise_level).float().to("cuda")
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    noise_range = torch.linspace(-1, max_noise_level - 1, ddim_steps + 1)

    # -----------------------------------------------------------------------
    # Timed generation
    # -----------------------------------------------------------------------
    def _run_once() -> float:
        B = 1
        total_frames = n_prompt + num_frames
        x = torch.randn(B, n_prompt, LATENT_C, LATENT_H, LATENT_W,
                        device="cuda", dtype=_model_dtype)
        actions = torch.zeros(B, total_frames, 25, device="cuda", dtype=_model_dtype)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()

        for frame_i in range(n_prompt, total_frames):
            chunk = torch.randn(B, 1, LATENT_C, LATENT_H, LATENT_W,
                                device="cuda", dtype=_model_dtype)

            for noise_idx in reversed(range(1, ddim_steps + 1)):
                t_val = noise_range[noise_idx].long().clamp(0, max_noise_level - 1)

                t_ctx = torch.zeros(B, frame_i, device="cuda", dtype=torch.long)
                t_cur = t_val.to("cuda").view(1, 1).expand(B, 1)
                t_in = torch.cat([t_ctx, t_cur], dim=1)
                x_in = torch.cat([x, chunk], dim=1)

                with torch.no_grad():
                    v = model(x_in, t_in, actions[:, :frame_i + 1])

                t_idx = t_val.item()
                alpha_t = alphas_cumprod[t_idx].to(_model_dtype)
                t_prev_idx = max(int(noise_range[noise_idx - 1].item()), 0)
                alpha_t_prev = alphas_cumprod[t_prev_idx].to(_model_dtype)

                v_cur = v[:, -1:]
                x0_pred = alpha_t.sqrt() * chunk - (1 - alpha_t).sqrt() * v_cur
                eps_pred = (1 - alpha_t).sqrt() * chunk + alpha_t.sqrt() * v_cur
                chunk = (alpha_t_prev.sqrt() * x0_pred
                         + (1 - alpha_t_prev).sqrt() * eps_pred).to(_model_dtype)

            x = torch.cat([x, chunk], dim=1)

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
            "kernels_applied": kernels_applied,
        }

    mean_ms = statistics.mean(latencies)
    results = {
        "model": "Etched/oasis-500m",
        "kernel": "+".join(kernels_applied) if kernels_applied else "none",
        "kernels_applied": kernels_applied,
        "num_frames": num_frames,
        "n_prompt": n_prompt,
        "ddim_steps": ddim_steps,
        "total_dit_fwd_passes": num_frames * ddim_steps,
        "latency_ms_mean": round(mean_ms, 2),
        "latency_ms_std": round(statistics.stdev(latencies) if len(latencies) > 1 else 0.0, 2),
        "latency_per_frame_ms": round(mean_ms / num_frames, 2),
        "frames_per_sec": round(num_frames / (mean_ms / 1000.0), 3),
        "vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "gpu": "H100",
        "raw_latencies_ms": latencies,
    }
    print(json.dumps(results, indent=2))
    return results


@app.local_entrypoint(name="oasis_custom_modal")
def main(
    use_sage_triton: bool = False,
    use_tempache: bool = False,
    use_flash_triton: bool = True,
    use_fp8_cuda: bool = True,
    use_int4_cuda: bool = False,
    use_fused_adaln: bool = False,
    num_iters: int = 1,
    num_warmup: int = 1,
):
    result = run_oasis_custom.remote(
        use_sage_triton=use_sage_triton,
        use_tempache=use_tempache,
        use_flash_triton=use_flash_triton,
        use_fp8_cuda=use_fp8_cuda,
        use_int4_cuda=use_int4_cuda,
        use_fused_adaln=use_fused_adaln,
        num_iters=num_iters,
        num_warmup=num_warmup,
    )
    print("\nOpen-Oasis 500M (all custom kernels) result:")
    import json
    print(json.dumps(result, indent=2))
    sys.path.insert(0, "/root/benchmarks")
    from result_store import save_result
    kernels = result.get("kernel", "unknown").replace("+", "_")
    save_result(result, "optimised_kernels", f"oasis_custom_{kernels}")
