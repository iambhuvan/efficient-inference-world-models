# WorldServe — H100 Inference Engine for Open-Oasis 500M

CMU 15-849 ML Systems, Spring 2026 Final Project — `bnallamo@andrew.cmu.edu`

**Goal.** Build an optimized inference engine for an autoregressive interactive
world model (`Etched/oasis-500m`, DiT-S/2, 608M params) and measure end-to-end
speedup on H100 SXM. Two systems bottlenecks dominate: (1) the sequential
encode → transformer → decode pipeline, and (2) KV cache growth that limits
generation length. We attack the transformer step with custom CUDA + Triton
kernels and (planned) tackle KV growth with spatial-temporal compression.

---

## What Oasis Actually Does (verified against open-oasis source)

| Property | Value |
|---|---|
| Architecture | DiT-S/2 (Diffusion Transformer, patch_size=2) |
| Latent shape | `(B, T, C=16, H=18, W=32)` per VAE-encoded frame |
| Spatial tokens / frame | **144** (post-patchify, 9×16) |
| Temporal tokens | T (one per past frame) |
| Diffusion process | DDPM, **1000** timesteps, `sigmoid_beta_schedule` |
| Parameterization | **v-prediction** (Salimans & Ho 2022) |
| Inference sampler | **10-step DDIM** (canonical) |
| Generation order | **Autoregressive per frame**, conditioning on all prior frames |
| Action conditioning | 25-dim Minecraft keyboard vector per frame |

A 16-frame run does `16 frames × 10 DDIM steps = 160` DiT forward passes, each
on the growing prefix `[prior_frames + new_frame]`. The temporal axis lengthens
every frame.

---

## Repository Layout

```
MLSYS_FINAL_PROJECT/
├── worldserve/                      # The package
│   ├── kernels/                     # Custom CUDA + Triton kernels
│   │   ├── load.py                  # JIT compiles the 3 .cu files
│   │   ├── fused_adaln.cu/.cuh      # K1: fused LayerNorm + scale + shift + gate
│   │   ├── int4_quantize.cu/.cuh    # K2: INT4 per-group quant (oasis_int4 path)
│   │   ├── fp8_quantize.cu/.cuh     # K3: FP8 E4M3 V-tensor compression
│   │   └── triton/                  # 4 Triton kernels:
│   │       ├── flash_attention.py        # FA3-style FlashAttention
│   │       ├── int4_fp8_attention.py     # SageAttention2 (INT8 QK + FP8 PV)
│   │       ├── tempache.py               # K/V deduplication (Duplicate Attention Lemma)
│   │       └── sliding_tile_attention.py # STA (arXiv 2502.04507)
│   ├── models/                      # Model wrappers (Oasis-only)
│   │   ├── base.py                  # BaseWorldModel ABC + OasisWrapper
│   │   ├── oasis.py                 # Corrected v-prediction autoregressive sampler
│   │   └── attn_processors.py       # SageAttnProcessor (drop-in replacement)
│   ├── optimizations/               # Higher-level optimization library
│   │   ├── model_level/             # attention, distillation, dynamic_compute,
│   │   │                            # feature_caching, feature_prediction,
│   │   │                            # guidance, noise_schedule, samplers
│   │   └── system_level/            # kv_cache, sparse_attention, speculative,
│   │                                # step_caching
│   └── utils/                       # metrics, profiler, visualization
├── benchmarks/                      # Modal H100 benchmark scripts
│   ├── common.py                    # Modal app + CUDA-devel image
│   ├── modal_common.py              # re-export alias
│   ├── result_store.py              # JSON result writer
│   ├── run_all.py                   # parallel runner across all variants
│   ├── baseline/
│   │   └── oasis_modal.py           # Native SDPA baseline
│   └── optimised/
│       ├── oasis_custom_modal.py    # Custom kernels (FlashAttn + FP8 + opt-in stack)
│       ├── oasis_all_modal.py       # Third-party SageAttn2 + INT4
│       ├── oasis_int4_modal.py      # torchao INT4 weight-only
│       ├── oasis_sageattention_modal.py
│       ├── oasis_sta_modal.py       # Sliding tile attention
│       ├── oasis_teacache_modal.py  # TeaCache step-skip
│       └── oasis_prediT_modal.py    # PrediT AB-2 extrapolation
└── docs/                            # Markdown notes (kernels, models, world-models)
```

---

## Custom Kernel Inventory

### CUDA (compiled JIT via `worldserve/kernels/load.py`, sm_90a, CUDA 12.4)

#### K1 — Fused AdaLN-Zero (`fused_adaln.cu`)
Fuses LayerNorm + scale + shift + gate into one kernel — saves 3 HBM round-trips per AdaLN call.
- Two-pass Welford variance in SMEM, warp `__shfl_xor_sync` reduction
- `__half2` SIMD pairs, `float4` 128-bit loads (8 FP16 / thread)
- Templated `D ∈ {128, 256, 512, 1024}` for unrolled inner loops
- **Status:** Compiles correctly; injection into Oasis `SpatioTemporalDiTBlock` disabled because the block uses split `s_adaLN_modulation` / `t_adaLN_modulation` (2-vector output) rather than the 6-chunk pattern this kernel assumes.

#### K2 — Per-Group INT4 Quantization (`int4_quantize.cu`)
4× weight bandwidth reduction via 4-bit packed weights.
- Warp absmax (`__shfl_xor_sync` tree reduction); scale = `absmax / 7`
- Two nibbles per byte via inline PTX `lop3.b32`
- Dequantize: masked extract → sign-extend → multiply by group scale
- **Status:** Works; replaces 164 `nn.Linear` layers in Oasis. Net **slower** at batch=1 because dequantize overhead exceeds bandwidth savings. Same pattern observed for the third-party torchao INT4 path. Disabled by default in `oasis_custom_modal`.

#### K3 — FP8 E4M3 Quantization (`fp8_quantize.cu`)
Halves V-tensor memory; cheap V-roundtrip during attention.
- Each thread: 8 FP16 values via `float4` 128-bit load
- Pack via `__nv_fp8x4_e4m3(make_float4(...))` (4 FP8 / 32-bit word)
- `cuda_fp8.h`, requires CUDA 12.1+
- **Status:** Works; used in custom SDPA pipeline as the V-tensor compressor before FlashAttn.

### Triton

#### A1 — FlashAttention FA3-style (`flash_attention.py`)
2-stage tiled Flash with online softmax.
- `BLOCK_M × HEAD_DIM` Q tile loaded once, `BLOCK_N` K/V tiles streamed
- `num_stages=2` software-pipelined K/V loads
- `NUM_HEADS: tl.constexpr` parameter — fixes a critical bug where `tl.num_programs(2)` returned 1 on a 2D grid, causing OOB memory access on multi-head models
- **Status:** Working; primary attention path in `oasis_custom_modal`.

#### A2 — Sliding Tile Attention (`sliding_tile_attention.py`)
Skips entire K/V tiles outside the window (not masked, never loaded).
- Validity table indexed by `(q_tile, kv_tile)`
- 1D / 2D / 3D builders (Oasis uses 1D over the 144-token spatial axis)
- arXiv 2502.04507 (ICML 2025)
- **Status:** Available via `oasis_sta_modal`.

#### A3 — TempCache K-Dedup (`tempache.py`)
Merges duplicate K vectors — Duplicate Attention Lemma.
- `find_duplicate_keys`: block-wise cosine similarity (avoids O(S²D) memory spike)
- `_tempache_merge_kernel`: averages K/V within each equivalence class
- Logit bias `log(count)` injected into attention; duplicate positions masked to `-inf`
- **Status:** Available but disabled in custom pipeline — Python `for b,h` loop wraps the Triton kernel and is called per SDPA invocation (~3,840 times per generation), causing ~30 min host-side overhead. Needs a batched-over-(B,H) Triton merge to be useful.

#### A5 — SageAttention2 H100 INT8/FP8 (`int4_fp8_attention.py`)
INT8 QK + FP8 PV with two-level FP22 accumulation fix.
- Level 1: `tl.dot(q_int8, k_int8, out_dtype=tl.int32)` → INT8 tensor core
- Level 2: explicit FP32 rescale `acc = alpha[:,None] * acc` before cross-tile add (prevents FP22 drift)
- Outlier smoothing: `Q_smooth = Q − q_bar`
- arXiv 2411.10958 (ICML 2025)
- **Status:** Compiles, but caused recurring CUDA illegal-memory-access faults in the Oasis pipeline even after fixing the same `NUM_HEADS` bug as FlashAttn and adding sequence-length padding to BLOCK_M/BLOCK_N multiples. Removed from active pipeline; FlashAttn is primary instead.

---

## High-Level Optimization Library (`worldserve/optimizations/`)

Imported but not yet wired into the Modal benchmarks. Provides:

| Subsystem | Components | Purpose |
|---|---|---|
| `model_level/attention` | radial_attention | Sparse attention pattern fitting |
| `model_level/distillation` | PCM, DCM, ProReflow, RCM, ContextForcing | Reduce inference steps via distillation |
| `model_level/dynamic_compute` | DC-DiT, DyDiT++ | Token/layer compute skipping |
| `model_level/feature_caching` | TaylorSeer, ERTA, MAG | Per-block output caching |
| `model_level/guidance` | APG, NAG | Adaptive guidance |
| `model_level/noise_schedule` | DIAMOND | Game-video-tuned cosine schedule |
| `model_level/samplers` | DPM-Solver++ | Higher-order DDIM solvers |
| `system_level/kv_cache` | progressive_quant, spatial_eviction, temporal_merge, qvg_smoothing | Tiered FP16→FP8→INT4 KV |
| `system_level/sparse_attention` | NATTEN, SVG2, profiler | Local-window kernels |
| `system_level/speculative` | KVQuantSpec, LayerSkipSpec, ParaDiGMS | Draft + verify |
| `system_level/step_caching` | FlowCache, MixCache, SeaCache, SpecA | DDIM-step output caching |

The `oasis.py` sampler exposes `pre_step` / `post_step` hooks so any of these
modules can plug in without changing the per-frame autoregressive structure.

---

## Verified Bug Fixes During Integration

| Issue | Root cause | Fix |
|---|---|---|
| `CUBLAS_STATUS_INTERNAL_ERROR` after attention | `tl.num_programs(2)` returns 1 on a 2D grid → all heads mapped to `h_idx=0` → OOB pointer arithmetic in attention kernels → GPU MMU fault corrupts CUDA context for all subsequent ops | Added `NUM_HEADS: tl.constexpr` to both `flash_attention.py` and `int4_fp8_attention.py`; pass `H` explicitly at launch |
| `mat1 and mat2 must have the same dtype` in TimestepEmbedder after `model.half()` | Sinusoidal `t_freq` stays float32 while MLP weights become fp16 | Monkey-patch `TimestepEmbedder.forward` to cast `t_freq` to `self.mlp[0].weight.dtype` (with `hasattr` guard for INT4-replaced layers) |
| `CUDA_HOME not set` when JIT-compiling .cu files | `nvidia-cuda-nvcc-cu12` PyPI wheel didn't expose nvcc on PATH | Switched the custom-kernel image to `nvidia/cuda:12.4.1-devel-ubuntu22.04` (ships nvcc + headers at `/usr/local/cuda`) |
| `__nv_fp8x4_e4m3` constructor compile error | Constructor takes a single `float4`, not 4 separate floats | `__nv_fp8x4_e4m3(make_float4(a, b, c, d))` |
| Sage kernel CUDA illegal access at small `Sq` | Triton masked `tl.load` still computes addresses past tensor allocation when `Sq < BLOCK_M` | Pad q/k/v to BLOCK_M / BLOCK_N multiples before launch, slice output back |
| Worldserve's original `oasis.py` had wrong sampler | Used eps-prediction formula, linear β schedule, parallel multi-frame denoise, `num_steps=50` default | Rewrote: v-prediction DDIM, `sigmoid_beta_schedule`, autoregressive per-frame, `num_steps=10` |

---

## Measured Results (H100 SXM, 16 new frames + 1 prompt frame, 10 DDIM steps)

| Variant | Latency (mean) | FPS | vs Baseline | VRAM |
|---|---:|---:|---:|---:|
| Baseline (native SDPA) | **8,250 ms** | **1.94** | 1.00× | — |
| Third-party (SageAttn2 + INT4) | 11,155 ms | 1.43 | **0.74×** (slower) | — |
| **Custom (FlashAttn + FP8 V)** | **7,863 ms** | **2.04** | **1.05×** (faster) | 1.31 GB |

The custom Triton FlashAttn + CUDA FP8 V-roundtrip stack delivers a real
+5% over baseline, while the third-party SageAttn2+INT4 combination is slower
than baseline at this batch/sequence (the INT4 dequantize overhead and
SageAttn2's small-S overhead dominate the bandwidth savings). The same
INT4-is-net-slower pattern showed up in our custom INT4 path, confirming
this is a workload property, not an implementation issue.

---

## Running on Modal H100

```bash
# One-time
modal secret create huggingface-secret HF_TOKEN=hf_...

# Native SDPA baseline
PYTHONPATH=benchmarks modal run benchmarks/baseline/oasis_modal.py

# Custom kernel stack (FlashAttn + FP8, others opt-in)
PYTHONPATH=benchmarks modal run benchmarks/optimised/oasis_custom_modal.py \
    --num-iters 3 --num-warmup 1

# Toggle individual kernels
PYTHONPATH=benchmarks modal run benchmarks/optimised/oasis_custom_modal.py \
    --use-flash-triton true \
    --use-fp8-cuda    true \
    --use-tempache    false  # Python loop overhead
    --use-sage-triton false  # CUDA fault — see bug table
    --use-int4-cuda   false  # net slower at batch=1
    --use-fused-adaln false  # block structure mismatch

# Other variants (each saves JSON to benchmarks/runs/)
modal run benchmarks/optimised/oasis_int4_modal.py
modal run benchmarks/optimised/oasis_sageattention_modal.py
modal run benchmarks/optimised/oasis_sta_modal.py
modal run benchmarks/optimised/oasis_teacache_modal.py
modal run benchmarks/optimised/oasis_prediT_modal.py
modal run benchmarks/optimised/oasis_all_modal.py
```

Modal mounts (`benchmarks/common.py`):
- `/root/open-oasis` — upstream Oasis source (DiT-S/2 + ViT-L VAE)
- `/root/worldserve` — this package
- `/root/benchmarks` — entrypoints + result_store

Two images: `image` (debian_slim) for non-CUDA-compile benchmarks; `image_cuda_devel` (nvidia/cuda:12.4.1-devel) for `oasis_custom_modal` which JIT-builds the .cu files.

---

## Hardware Target

**NVIDIA H100 SXM** — sm_90a
- HBM3: 3.35 TB/s memory bandwidth
- FP8 tensor cores: 3,958 TFLOPS
- INT8 tensor cores: 1,979 TOPS
- SMEM: 228 KB/SM, L2: 50 MB

INT4 attention is Ada/sm89 only — not available on H100; the SageAttention2 H100 path uses INT8 QK + FP8 PV instead.

---

## Status & Roadmap

**Done**
- End-to-end Oasis baseline + custom-kernel benchmarks on Modal H100
- Working FlashAttn (Triton) + FP8 (CUDA) custom path with measured speedup
- Corrected `worldserve/models/oasis.py` (v-prediction, sigmoid β, autoregressive)
- Consolidated repo into 2 top-level dirs: `worldserve/` and `benchmarks/`
- Removed all non-Oasis benchmarks and unused kernels (RoPE, top-K, KV gather/scatter, radial mask)

**In progress / planned (per project proposal)**
- Wire up `worldserve/optimizations/system_level/kv_cache` (progressive quant + spatial eviction + temporal merge + QVG smoothing) for spatial-temporal KV compression — target 5–7× memory reduction, 30+ s generation
- Stage-overlap pipelining (CUDA streams: encode N+1 || transformer N || decode N-1)
- Speculative decoding paths (KVQuantSpec, LayerSkipSpec, ParaDiGMS)

**Out of scope** (per project memory): Matrix-Game 2.0 and other video models — Oasis-only.
