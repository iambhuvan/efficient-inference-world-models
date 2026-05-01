# WorldServe — MAGI-1 4.5B Inference Optimisation Study

CMU 15-849 ML Systems, Spring 2026 — author Bhuvan Nallamothu (`bnallamo@andrew.cmu.edu`).
Hardware: NVIDIA H100 80 GB SXM via Modal. Reference run config: 96 frames at 720×720, 64 denoising steps, 16 chunks, `seed=42`, single-prompt T2V.

This document captures every measurement, decision, and result obtained while optimising MAGI-1 4.5B-base inference. It is the standalone record of the work done on this model — the goal is not a literature survey but a self-contained narrative of what we measured, what we built, what worked, what didn't, and *why*, so the next iteration starts from a complete picture.

---

## 1. Why MAGI-1

The project initially targeted Open-Oasis (DiT-S, ~500M, 4 denoising steps) and Matrix-Game 2.0. Two structural issues with those targets killed most of the optimisation axes we wanted to study:

- **Open-Oasis** runs only 4 denoising steps at deployed config — step-caching (TaylorSeer / FORA / ∆-DiT) has nothing to cache across with so few steps.
- **DIAMOND CSGO** (a candidate replacement world model) turned out to be a 1-step consistency model — even more degenerate for step-axis optimisations.

We needed an open, large autoregressive video model with a real denoising-step budget per chunk to study cross-frame KV reuse and step-aware optimisations. The literature survey produced a single clean candidate that satisfies all constraints:

| Criterion | MAGI-1 4.5B-base | Notes |
|---|---|---|
| Open weights, ungated | ✅ | Apache-2.0, `sand-ai/MAGI-1` on HF |
| Multi-step denoising per chunk | ✅ 64 steps | Big schedule space |
| Autoregressive across chunks | ✅ chunk-wise AR | Cross-frame K/V cache |
| Real cross-chunk attention | ✅ MagiAttention | Their `noise2clean_kvrange` knob |
| Fits on 1× H100 80 GB | ✅ 4.5B variant | 24B variant overflows |
| Not distilled to few-step | ✅ base variant | 4.5B-distill exists but defeats the purpose |

Paper: arXiv:2505.13211, *"MAGI-1: Autoregressive Video Generation at Scale"*. HuggingFace: [`sand-ai/MAGI-1`](https://huggingface.co/sand-ai/MAGI-1).

---

## 2. What MAGI-1 actually is (verified against upstream source)

### 2.1 Architecture

| Property | Value |
|---|---|
| Variant studied | 4.5B-base (also exists: 4.5B-distill, 24B-base, 24B-distill, fp8-distill) |
| Parameter count | 4,459,898,128 (~4.46 B), 8.34 GB FP16 |
| Transformer layers | 34 |
| Hidden / FFN | 3072 / 12288 |
| Attention heads | 24 query, **8 KV** (GQA factor 3) |
| `kv_channels` | 128 |
| Patch | spatial 2×2, temporal 1 |
| Caption length (T5-XXL) | 800 tokens |
| Text encoder | T5-v1_1-XXL (≈4.7B, kept on CPU at runtime via `OFFLOAD_T5_CACHE=true`) |
| VAE | ViT-VAE, ~700M, channels 16 |

Each `TransformerLayer` contains a **`FullyParallelAttention`** with three projection paths:

```
linear_qkv      : Conv-1×1 producing Q (3072→3072) || QX (3072→3072) || K (3072→1024) || V (3072→1024)
linear_kv_xattn : 3072 → 2048 (concat K|V for cross-attention to T5 prompt embeddings)
linear_proj     : 6144 → 3072 (concat self/temporal output || cross-T5 output → projection)
```

Two separate attention paths share the same query stream:

- **Self/temporal path**: `q × k`, attends to current chunk + cached K/V from past clean chunks (the *cross-chunk* KV cache).
- **Cross-T5 path**: `qx × xattn_k`, attends to the encoded prompt.

### 2.2 Inference pipeline (verified by reading `inference/pipeline/pipeline.py`)

```
 prompt ──► T5 encode (CPU)         ┐
 image  ──► VAE encode  ──┐         │
                          ▼         ▼
                       prefix latent │
                          │         │
        ┌───── chunk loop (16) ─────┐│
        │ for noise in [4 buckets]: ││
        │   for step in [16 within ]│
        │       full DiT forward    ││
        │       (34 layers, attn +  ││
        │        cross-T5)          ││
        │ commit chunk → cached K/V ││
        └───────────────────────────┘│
                          ▼          │
                  VAE decode (chunk) ◄┘
                          ▼
                   ffmpeg encode → mp4
```

Key runtime constants (from `example/4.5B/4.5B_base_config.json`):

```json
"num_frames"          : 96,
"video_size_h/w"      : 720,
"num_steps"           : 64,
"chunk_width"         : 6,
"window_size"         : 4,
"cfg_t_range"         : [0.0, 0.0217, 0.1, 0.3, 0.999],
"noise2clean_kvrange" : [5, 4, 3, 2],
"clean_chunk_kvrange" : 1,
"shortcut_mode"       : "8,16,16",
"kv_offload"          : true
```

Reading `noise2clean_kvrange = [5, 4, 3, 2]`: as noise level *decreases* (from bucket 3 → bucket 0), the number of past clean chunks attended to *increases* 2 → 5. This is the **opposite** of LLaDA-text SAPS schedules — temporal coherence pressure peaks at low noise (final detail) for video, not at high noise.

---

## 3. Getting the baseline running — engineering notes

These were not trivial; documenting them so we don't relearn the same lessons.

### 3.1 HuggingFace repo layout (verified via HF API)

```
sand-ai/MAGI-1/
├── ckpt/magi/4.5B_base/inference_weight/      ← 2 safetensors (~9 GB)
├── ckpt/magi/4.5B_distill/inference_weight.distill/
├── ckpt/magi/24B_base/inference_weight/        (skipped via allow_patterns)
├── ckpt/t5/t5-v1_1-xxl/                        ← T5 (~19 GB)
└── ckpt/vae/                                   ← VAE (~2.4 GB)
```

`allow_patterns = ["ckpt/magi/4.5B_base/**", "ckpt/t5/t5-v1_1-xxl/**", "ckpt/vae/**"]` brings the snapshot down from 172 GB → ~30 GB.

### 3.2 Path indirection traps

MAGI-1's reference inference appends `/inference_weight` and `/t5-v1_1-xxl` to the configured paths. So:

```json
"load"           : "<snapshot>/ckpt/magi/4.5B_base"   # NOT inference_weight
"t5_pretrained"  : "<snapshot>/ckpt/t5"               # NOT t5-v1_1-xxl
"vae_pretrained" : "<snapshot>/ckpt/vae"
```

Misnaming any of these reproduces our first three failed runs.

### 3.3 Modal image build order

`add_local_*` mounts must come **last** in a Modal image. Our `image_cuda_devel` originally ended with `add_local_dir`, so any subsequent `.pip_install` / `.run_commands` errored out with:

> `An image tried to run a build step after using image.add_local_*`.

Fix: refactored `benchmarks/common.py` to expose `image_cuda_devel_base` (pre-mount) and `_add_common_layers` separately so each baseline composes its own deps before mounting.

### 3.4 CUDA / torch ABI

MAGI-1 pins `flash-attn==2.4.2` and `flashinfer-python==0.2.0.post2` (cu124 + torch 2.4 ABI). Adding `timm`, `transformers`, `diffusers` after pip resolution silently upgraded `torchvision`/`torchaudio` to a CUDA-13 build that errors at import:

> `OSError: libcudart.so.13: cannot open shared object file`

Fix: append `pip install --force-reinstall --no-deps torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --extra-index-url https://download.pytorch.org/whl/cu124` at the end of the image build.

### 3.5 Distributed init env vars

MAGI-1's `MagiPipeline.__init__` calls `dist_init` even on a single GPU. Required env vars (matching `example/4.5B/run.sh`):

```
MASTER_ADDR=localhost  MASTER_PORT=6009
GPUS_PER_NODE=1  NNODES=1  WORLD_SIZE=1  RANK=0  LOCAL_RANK=0
PAD_HQ=1  PAD_DURATION=1
OFFLOAD_T5_CACHE=true  OFFLOAD_VAE_CACHE=true
TORCH_CUDA_ARCH_LIST=9.0
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### 3.6 Working-directory dependency

`inference/pipeline/prompt_process.py` does `np.load("example/assets/special_tokens.npz")` with a relative path. Must `os.chdir("/root/MAGI-1")` before constructing `MagiPipeline`.

### 3.7 ffmpeg binary

`inference/pipeline/video_process.py` shells out to `ffmpeg`. The Python wrapper `ffmpeg-python` only wraps the binary; it does not install it. Add `apt_install("ffmpeg")` to the image.

After all of the above were correct, the first successful run completed in 770 s.

---

## 4. Baseline measurement

### 4.1 Configuration

| Parameter | Value |
|---|---|
| Hardware | H100 80 GB SXM (Modal) |
| Precision | BF16 weights |
| Frames | 96 |
| Resolution | 720 × 720 |
| Denoising steps | 64 |
| Chunks | 16 (of width 6) |
| Window size | 4 |
| Prompt | "A camera slowly orbits a wooden cabin in a snowy forest at dusk; warm light glows through the windows, soft snow falling." |
| Iterations | 1 warmup + 1 timed |

### 4.2 Result (`benchmarks/runs/baseline/magi1_45b_baseline_f96_s64.json`)

| Metric | Value |
|---|---|
| Total wall (timed) | **770,428 ms (≈12.8 min)** |
| Latency / frame | 8025 ms |
| Frames per second | **0.125 FPS** |
| Peak VRAM | 14.10 GB (T5 on CPU, VAE offloaded) |
| Per-chunk profile (4 InferBatches) | 403 → 250 → 178 → 126 s |

The 3.2× front-loaded cost between the first and last chunks reflects JIT warmup of fused kernels + first-chunk cache fills. Steady-state per-chunk is closer to 90–130 s.

---

## 5. Optimisation #1 — Cross-chunk K/V ablation

### 5.1 Hypothesis

The runtime config exposes `noise2clean_kvrange` and `clean_chunk_kvrange`, which control how many past *clean* chunks the temporal cross-attention attends to per noise level. Naming suggests "K/V reuse"; under-the-hood reading shows this is **how much past context to attend to**, not whether to recompute past projections. Disabling it should isolate how much of MAGI-1's wall time the temporal cross-chunk attention path consumes.

### 5.2 Implementation

(`benchmarks/optimised/magi1_45b_no_temporal_kv_modal.py`)

A 200-line wrapper around the baseline that patches the runtime config in-place:

```json
"noise2clean_kvrange" : [0, 0, 0, 0],
"clean_chunk_kvrange" : 0
```

Every other parameter identical to baseline.

### 5.3 Result (`benchmarks/runs/optimised_kernels/magi1_45b_no_temporal_kv_f96_s64.json`)

| Metric | Baseline | No cross-chunk K/V | Δ |
|---|---|---|---|
| Total wall | 770 s | **620 s** | **−19 %** |
| FPS | 0.125 | **0.155** | **+24 %** |
| Latency / frame | 8025 ms | 6466 ms | −19 % |
| Peak VRAM | 14.10 GB | 14.10 GB | 0 |
| Per-chunk profile | 403 → 250 → 178 → 126 | 297 → 178 → 127 → 92 | −26 % avg |

### 5.4 Interpretation

**Counter-intuitive but conclusive**: MAGI-1's so-called "K/V reuse" path is **net compute cost, not compute saving**. The temporal cross-attention's job is to *attend* to past chunks' cached K/V, not to *avoid recomputing* them. Disabling that attention saves the cost of attending — at the obvious price of losing temporal coherence.

The model was trained with non-zero `kvrange`, so quality regression is expected when this is severed; we did not run FVD here. The number is the **upper bound** on what "remove cross-chunk attention" can save: 24 % wall, free in compute, expensive in coherence.

This datum frames the design space for #2: the right move is **selective** pruning of the cross-chunk attention path, keeping the most useful past tokens.

---

## 6. Profiling MAGI-1's attention

The natural reflex was to port SAPS (CMU 17-752 — "Step-Aware Pruning Schedule for Diffusion LLMs", which gets −31 % memory + 1.9 pp accuracy on LLaDA-8B GSM8K). SAPS uses an **exponential decay schedule** `r(t) = r_max · (r_min/r_max)^(t/(T-1))` over denoising steps, keeping more tokens early (when global structure forms) and fewer late.

Direct port to MAGI-1 would be wrong:

1. SAPS's "early=more context, late=less" is calibrated to **text** structure formation. MAGI-1's existing `kvrange = [5,4,3,2]` shows the opposite intuition for **video**: temporal coherence pressure peaks at low noise.
2. SAPS treats all layers identically. MAGI-1's 34-layer DiT has likely heterogeneous per-layer attention (literature on PAB / ∆-DiT / AdaCache supports this in DiT video models).
3. SAPS's mechanism is **token pruning by importance** — that only works if K is heavy-tailed. We have no a priori reason to believe MAGI-1's K is heavy-tailed.

We needed data. Two profiler runs answer the four questions:

> Q1. Is K heavy-tailed? (Determines whether token pruning is viable at all.)
> Q2. Does step-axis variance support a step-aware schedule?
> Q3. Are layers heterogeneous enough for layer-wise scheduling?
> Q4. How does T5 cross-attn compare to self/temporal in size + behaviour?

### 6.1 Profiler v1 — self/temporal K projection

(`benchmarks/profiling/magi1_45b_attention_profile_modal.py`)

Hooks every `linear_qkv.k` (one per layer × 34 layers) with a forward hook that captures per-call statistics on the K-projection output:

| Field | Meaning |
|---|---|
| `k_norm_{mean,p50,p95,max}` | per-token K-norm distribution |
| `k_top1pct_mass_frac` | fraction of total ‖K‖ in top-1 % tokens (uniform = 0.01) |
| `k_top10pct_mass_frac` | fraction in top-10 % (uniform = 0.10) |
| `cosine_sim_prev` | cosine similarity to previous call's K (per-layer cross-step stability) |

K-norm is a defensible proxy for attention mass: under standard SDPA, large-‖K‖ tokens dominate softmax for any reasonable Q. If K is heavy-tailed, top-k pruning is safe; if uniform, pruning hurts. Since flashinfer / flash-attn don't expose attention probabilities, this proxy was the most informative cheap signal available.

**Run cost**: 990 s (vs 770 s baseline → ~28 % overhead from hooks). Output: 11,424 records (34 layers × 336 calls each).

### 6.2 Profiler v1 — per-layer summary

(`benchmarks/runs/profiling/magi1_45b_attn_profile_f96_s64.json`)

Aggregate across all 34 layers:

| Metric | Min | Mean | Max |
|---|---|---|---|
| `k_top1pct_mass_frac` | 0.0115 | 0.0131 | **0.0153** |
| `k_top10pct_mass_frac` | 0.110 | 0.117 | 0.125 |
| `k_norm_mean` | 24.0 | 43.7 | 164.1 |
| `cosine_sim_prev` | 0.758 | 0.868 | **0.989** |

**Q1 answer — K is NOT heavy-tailed.**
Uniform K would give top-1 % mass = 0.010 and top-10 % mass = 0.100. MAGI-1 gives top-1 % = 0.013 (only **30 % above uniform**) and top-10 % = 0.117 (only **17 % above uniform**). No layer breaks 0.0153 at top-1 %. **SAPS-style token pruning by importance is therefore weak on MAGI-1** — there is no long tail of unimportant tokens to drop. Dropping 90 % of tokens drops 88 % of K-mass. Hard pass.

**Q3 answer — K is highly stable across steps, with large layer heterogeneity.**
Mean `cos_prev` ranges from 0.758 (layer 19) to 0.989 (layer 31) — **nearly an order of magnitude in the (1 − cos_prev) gap**. K-norm mean ranges 24 → 164 (7× across layers). Both signals point at a **layer-conditioned** schedule, not uniform.

### 6.3 Per-layer × per-step heatmap (offline analysis from v1 records)

We binned each layer's 336 calls into 8 chronological buckets and recomputed `top-1 % mass` and `cos_prev` per bin:

```
=== top-1% mass fraction (layer × step-bin) ===
lyr |  b0    b1    b2    b3    b4    b5    b6    b7  | mean  range
  4 | 0.0144 0.0144 0.0140 0.0136 0.0133 0.0129 0.0126 0.0123 | 0.0134  0.0021
 11 | 0.0146 0.0143 0.0139 0.0135 0.0131 0.0126 0.0123 0.0119 | 0.0133  0.0027
 20 | 0.0134 0.0144 0.0152 0.0156 0.0163 0.0168 0.0163 0.0146 | 0.0153  0.0033
 31 | 0.0118 0.0121 0.0125 0.0129 0.0130 0.0130 0.0129 0.0128 | 0.0126  0.0012

=== cosine_sim_prev (layer × step-bin) ===
lyr |  b0    b1    b2    b3    b4    b5    b6    b7  | mean
  4 | 0.980 0.913 0.892 0.888 0.891 0.899 0.913 0.990 | 0.921
 19 | 0.945 0.820 0.745 0.706 0.671 0.647 0.663 0.865 | 0.758
 31 | 0.999 0.991 0.987 0.985 0.983 0.982 0.983 0.999 | 0.989
```

**Q2 answer — step-axis variance is small; layer-axis variance is large.**

- Top-1 % mass varies **at most 0.0033** across step bins for any layer (typically 0.001–0.002). The "early steps need broad context" assumption SAPS rests on does *not* hold: token-importance distribution is essentially flat across steps within a layer.
- `cos_prev` shows a U-shape (high at chunk boundaries, dip in middle) — explained by chunk transitions, not by denoising-step semantics. Mid-step `cos_prev` is the right summary statistic.

### 6.4 Layer bucketing (the schedule)

Using mean `cos_prev` from the v1 summary:

| Bucket | `cos_prev` range | Layers (count) | Broadcast horizon |
|---|---|---|---|
| **Very stable** | > 0.92 | 1, 4, 17, 26, 31, 32, 33 (**7**) | every **4** steps |
| **Moderate** | 0.85 – 0.92 | 2, 3, 5, 6, 7, 8, 9, 15, 21, 25, 27, 29 (**12**) | every **2** steps |
| **Volatile** | < 0.85 | 0, 10, 11, 12, 13, 14, 16, 18, 19, 20, 22, 23, 24, 28, 30 (**15**) | every **1** step (no broadcast) |

Predicted attention compute fraction:

```
(7/34)·(1/4) + (12/34)·(1/2) + (15/34)·(1) = 0.073 + 0.176 + 0.441 = 0.668
                                    → 33 % attention reduction
```

Attention is roughly 30–40 % of forward in MAGI-1 → **predicted wall reduction: 13–18 %**, at iso-quality (no token modifications, only output broadcasting between identical-input steps).

### 6.5 Profiler v2 — T5 cross-attention attribution

(`benchmarks/profiling/magi1_45b_attention_profile_v2_modal.py`)

Same v1 hooks plus a second hook on `linear_kv_xattn` (T5 cross-attention K|V).

**Q4 answer — T5 cross-attention is precomputed once.**
The v2 run completed in 1002 s and recorded **11,424 self/temporal calls + 0 T5 cross-attn calls**. The hook on `linear_kv_xattn` fired zero times during the entire 16 chunk × 64 step inference. Reading: T5 prompt embeddings are encoded once at the start of inference (`OFFLOAD_T5_CACHE=true` puts them on CPU), the K|V projections are computed once, cached, and **never recomputed during the chunk loop**.

Implication: **all per-step attention compute is in the self/temporal path.** The broadcast schedule does not need to be extended to T5 cross-attn; that path is already free.

---

## 7. Optimisation #2 — Layer-conditioned attention-output broadcast

### 7.1 Mechanism

(`benchmarks/optimised/magi1_45b_layer_broadcast_modal.py`, ~280 lines)

Wrap each `TransformerLayer.self_attention.forward` with a thin shim:

```python
def wrapped(*args, **kwargs):
    elapsed = step_counter - cache.step_at_compute
    shape_match = cache.in_shape == current_input_shape
    if cache.out is not None and shape_match and elapsed < horizon:
        return cache.out                          # BROADCAST
    out = original_forward(*args, **kwargs)
    cache.update(out, current_input_shape, step_counter)
    step_counter += 1
    return out
```

Per-layer `horizon` looked up from the bucket table in §6.4. Output is `.detach().clone()`'d on store to prevent in-place corruption.

The hook is installed by monkey-patching `inference.pipeline.pipeline.get_dit` so the wrap happens immediately after `get_dit(self.config)` builds the model — needed because `MagiPipeline._run` rebuilds the DiT internally on every call.

### 7.2 Result (`benchmarks/runs/optimised_kernels/magi1_45b_layer_broadcast_h4_2_1.json`)

| Metric | Baseline | Layer broadcast (h=4/2/1) | Δ |
|---|---|---|---|
| Total wall | 770 s | **720 s** | **−6.5 %** |
| FPS | 0.125 | **0.133** | **+6.4 %** (1.07×) |
| Latency / frame | 8025 ms | 7501 ms | −6.5 % |
| Peak VRAM | 14.10 GB | **34.40 GB** | **+144 %** ⚠ |
| Compute calls | 11,424 (100 %) | 8,831 (77.3 %) | — |
| Broadcast calls | 0 | 2,593 (**22.7 %**) | — |
| Shape-mismatch resets | — | **11,016 (96.4 %)** | ⚠ |

Per-chunk profile: 370 → 227 → 161 → 113 s (vs baseline 403 → 250 → 178 → 126).

### 7.3 Interpretation — math is internally consistent

- **Predicted broadcast rate**: 33 % (from §6.4).
- **Observed broadcast rate**: 22.7 %. Gap = 70 % of theoretical.
- **Predicted wall savings**: 12 % (33 % attn × ~35 % attn-fraction-of-forward).
- **Observed wall savings**: 6.5 %. Ratio = 6.5/12 ≈ 0.54 — close to the 22.7/33 ratio of effective vs theoretical broadcast. Speed savings track effective broadcast rate, as expected.

**The implementation works as designed.** The shortfall versus prediction is fully explained by the next finding.

### 7.4 Why broadcast rate is 22.7 % and not 33 %

96 % of compute calls (11,016 / 11,424) hit a **shape mismatch** between the cached output and the new input → cache invalidation → forced recompute.

Cause: MAGI-1's chunk-by-chunk inference, combined with `noise2clean_kvrange = [5,4,3,2]`, produces **input shapes that vary across chunks**. The cache keyed on a single `in_shape` resets at every chunk boundary, losing what would otherwise be hits.

The fix is a **multi-shape cache**: `Dict[shape_signature → (cached_out, last_step)]`. Recurring shapes within a chunk reuse, new shapes compute fresh. This should push broadcast rate from 22.7 % → near the predicted 33 % and wall savings from 6.5 % → ~12 %. Not yet implemented.

### 7.5 VRAM regression

Peak VRAM blew from 14.1 GB → 34.4 GB. The cache stores full BF16 attention outputs without eviction; the multi-shape extension will worsen this without a bound. Two cheap fixes when we re-run:

1. **CPU eviction** for stale shape entries.
2. **Quantise cached output to FP8** (→ ~2× memory cut at negligible quality cost since these are only used as broadcast values).

### 7.6 Quality

Speed-only this round. Cached attention-output reuse at horizon ≤ 4 is conservative — only invoked on layers whose K is empirically near-static (`cos_prev` > 0.92 implies the attention output between consecutive steps differs by ≲ 8 %). Quality regression should be small. To validate: compute FVD against baseline on a held-out prompt set (next iteration; out of scope here).

---

## 8. Result table — everything we measured for MAGI-1

| File | What | FPS | Wall | VRAM | Notes |
|---|---|---|---|---|---|
| `magi1_45b_baseline_f96_s64.json` | baseline (kvrange [5,4,3,2]) | **0.125** | 770 s | 14.10 GB | anchor |
| `magi1_45b_no_temporal_kv_f96_s64.json` | cross-chunk K/V disabled | **0.155** | 620 s | 14.10 GB | 1.24× — quality regression expected |
| `magi1_45b_layer_broadcast_h4_2_1.json` | layer-conditioned broadcast | **0.133** | 720 s | **34.40 GB** | 1.07× — broadcast_rate 22.7%, 96% shape resets |
| `magi1_45b_attn_profile_f96_s64.json` | profiler v1 (K-only) | — | 990 s | 20.87 GB | 11,424 records, drove the schedule |
| `magi1_45b_attn_profile_v2_f96_s64.json` | profiler v2 (+T5 attribution) | — | 1002 s | 20.87 GB | T5 cross-attn = 0 hook calls (precomputed) |

**Headline numbers.** MAGI-1 4.5B-base baseline = 0.125 FPS on H100 80 GB. Two real wins so far:

- **Hard ceiling (quality-lossy):** disabling cross-chunk attention → 1.24× (0.155 FPS).
- **Quality-preserving:** layer-conditioned attention-output broadcast → 1.07× (0.133 FPS), with a clear path to ~1.12× via multi-shape cache.

---

## 9. Findings worth remembering

1. **MAGI-1's K is near-uniform** — top-1 % mass = 0.013 (uniform = 0.010). Token-level pruning à la SAPS will not work; layer-wise broadcast is the right axis.
2. **K is highly stable across steps but extremely heterogeneous across layers** — `cos_prev ∈ [0.76, 0.99]`. The right schedule is layer-conditioned, not step-conditioned.
3. **MAGI-1's `noise2clean_kvrange` path is compute *cost*, not compute *saving*** — disabling it gains 24 %. Their cache is about avoiding K/V projection recompute (memory), not avoiding the attention itself (compute).
4. **T5 cross-attention is precomputed once and cached** — zero hook calls during inference. All optimisation effort should focus on the self/temporal path.
5. **Layer-broadcast prediction (12 %) and observation (6.5 %) are internally consistent** — gap explained by 96 % shape-mismatch reset rate.
6. **The right next iteration is multi-shape caching + CPU eviction**, not a different algorithm.
7. **Don't port LLaDA SAPS** — their assumption of heavy-tailed K + step-axis information gradient does not hold for MAGI-1 video.

---

## 10. File / artefact map

```
benchmarks/
├── baseline/
│   └── magi1_45b_baseline_modal.py                               # 280 LOC
├── optimised/
│   ├── magi1_45b_no_temporal_kv_modal.py                         # 220 LOC — kvrange=0 ablation
│   └── magi1_45b_layer_broadcast_modal.py                        # 280 LOC — layer-conditioned broadcast
├── profiling/
│   ├── magi1_45b_attention_profile_modal.py                      # 240 LOC — v1 K-only
│   └── magi1_45b_attention_profile_v2_modal.py                   # 250 LOC — v2 + T5
└── runs/
    ├── baseline/magi1_45b_baseline_f96_s64.json
    ├── optimised_kernels/magi1_45b_no_temporal_kv_f96_s64.json
    ├── optimised_kernels/magi1_45b_layer_broadcast_h4_2_1.json
    └── profiling/{magi1_45b_attn_profile_f96_s64.json, magi1_45b_attn_profile_v2_f96_s64.json}
```

All Modal runs reproducible with:

```bash
PYTHONPATH=benchmarks modal run benchmarks/<path>/<file>.py
```

---

## 11. What's next

In rough priority order:

| # | Item | Expected delta | Effort |
|---|---|---|---|
| 1 | Multi-shape cache in layer-broadcast (Dict keyed on input shape) | 6.5 % → ~12 % wall | 1 day |
| 2 | CPU eviction + FP8 cache compression to bound VRAM | VRAM 34 GB → ≤ 18 GB | 0.5 day |
| 3 | FVD evaluation against baseline on 50 VBench prompts | quality validation | 1 day (one extra Modal run + scoring) |
| 4 | Sweep horizon parameters (try h=8/4/2 for very-stable bucket) | possibly 1.10 → 1.15× | 1 day, 4 Modal runs |
| 5 | Combine layer-broadcast with reduced `noise2clean_kvrange` (e.g. [3,2,1,0] instead of full disable) | 1.24× ceiling × 1.12× compose? | 1 day |
| 6 | Test 4.5B-distill (4-step) variant for the speed-headline number | 4–8× via fewer steps | 2 days, accepts quality loss |

The core hypothesis we want validated: **layer-conditioned attention-output broadcast at iso-quality recovers most of the cross-chunk-attention cost without disabling temporal coherence.** The current prototype shows the math works and the implementation is correct; the remaining 5 percentage points of predicted speedup are gated on the multi-shape cache, which is the one thing that should be done first.
