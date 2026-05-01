# Universal CUDA Kernels for Transformer Inference Optimization

> **Scope**: This document covers general-purpose CUDA kernels that are **architecture-agnostic** — they apply to any transformer, DiT (Diffusion Transformer), video model, or LLM. None of these kernels are restricted to a specific model family. They constitute the reusable kernel library layer of WorldServe and similar inference systems.
>
> **Target Hardware**: NVIDIA H100 SXM5 (Hopper architecture, sm_90a). Kernels exploit H100-specific features (TMA, WGMMA, FP8 tensor cores, NVLink, HBM3) but degrade gracefully on A100 (sm_80).
>
> **CMU 15-849 ML Systems — Final Project Reference, Spring 2026**

---

## Table of Contents

1. [fused_adaln](#1-fused_adaln)
2. [int4_fp8_attention](#2-int4_fp8_attention)
3. [fa3_warp_specialized_attention](#3-fa3_warp_specialized_attention)
4. [tiered_kv_quantization](#4-tiered_kv_quantization)
5. [h2o_token_eviction](#5-h2o_token_eviction)
6. [tome_token_merge](#6-tome_token_merge)
7. [per_group_int4_quantize](#7-per_group_int4_quantize)
8. [per_group_fp8_quantize](#8-per_group_fp8_quantize)
9. [fused_conv_norm_act](#9-fused_conv_norm_act)
10. [warp_topk_selection](#10-warp_topk_selection)
11. [indexed_gather_scatter](#11-indexed_gather_scatter)
12. [streaming_kv_ringbuffer](#12-streaming_kv_ringbuffer)
13. [pyramidkv_score_extraction](#13-pyramidkv_score_extraction)
14. [fused_rope_embedding](#14-fused_rope_embedding)
15. [snapkv_3d_pooling](#15-snapkv_3d_pooling)
16. [cuda_graph_pipeline](#16-cuda_graph_pipeline)
17. [int2_kv_pack_unpack](#17-int2_kv_pack_unpack)
18. [wgmma_int4_gemm](#18-wgmma_int4_gemm)

---

## 1. `fused_adaln`

### What It Does

Adaptive LayerNorm (adaLN) modulation is the conditioning mechanism used by virtually every modern Diffusion Transformer. It applies a learned scale (`γ`) and shift (`β`) — derived from a conditioning signal such as a timestep embedding or class embedding — to a layer-normalized activation. The fused kernel collapses the sequence: `LayerNorm → scale → shift → (optional gate)` into a single warp-cooperative pass over the input tensor, eliminating two intermediate HBM round-trips.

### Applicable Models / Architectures

Any model using `adaLN` or `adaLN-zero` conditioning blocks: Stable Diffusion 3, FLUX.1, CogVideoX, HunyuanVideo, Wan2.1, OpenSora, DiT (Peebles & Xie 2023), SiT, PixArt-α/Σ, Latte, Open-Sora-Plan, Movie Gen (Meta), any custom DiT backbone. Also applies to any LLM that uses conditional LayerNorm gating (e.g., T5 with FiLM conditioning).

### Key Algorithm / Technique

The kernel performs a **two-pass Welford online variance** computation within shared memory across the token dimension, then in a single second pass applies:

```
y = gamma_cond * ((x - mean) / sqrt(var + eps)) + beta_cond
```

The two passes are fused into one kernel launch. Optionally a gate vector `α` (for adaLN-zero) multiplies the output before it feeds into the residual. The conditioning vectors `gamma_cond`, `beta_cond`, `alpha` are loaded once into registers and reused across the token dimension — they are broadcast from a `[B, C]` tensor to a `[B, T, C]` activation.

```cuda
// Simplified adaLN-zero fused kernel (illustrative, not full production code)
__global__ void fused_adaln_zero_kernel(
    const __half* __restrict__ x,        // [B, T, C]
    const __half* __restrict__ gamma,    // [B, C]
    const __half* __restrict__ beta,     // [B, C]
    const __half* __restrict__ alpha,    // [B, C] — gate (adaLN-zero only)
          __half* __restrict__ y,        // [B, T, C]
    int T, int C, float eps)
{
    // One block per (batch, token) pair
    int b = blockIdx.x / T;
    int t = blockIdx.x % T;
    int tid = threadIdx.x;

    // Welford mean/var in shared memory
    extern __shared__ float smem[];  // [C] float accumulation
    float local_sum = 0.f, local_sq = 0.f;

    // Vectorized load: 8x __half per thread via float4
    for (int c = tid * 8; c < C; c += blockDim.x * 8) {
        float4 v = *reinterpret_cast<const float4*>(&x[(b*T+t)*C + c]);
        // unpack and accumulate ...
    }

    // Warp reduce then block reduce for mean/var
    // __shfl_xor_sync(0xffffffff, val, lane_offset) for warp reduction

    // Second pass: normalize + scale + shift + gate
    // ...
}
```

### CUDA Primitives Used

- `__shfl_xor_sync(0xffffffff, ...)` — warp-level Welford reduction for mean and variance
- `__half2` vectorized arithmetic via `__hadd2`, `__hmul2`, `__hsub2`
- `float4` / `uint4` vectorized global memory loads (128-bit transactions)
- `__ldg()` for read-only conditioning loads through texture cache
- `__syncthreads()` for two-pass block-level reduction

### H100 Hardware Features Exploited

- **L2 cache persistence**: conditioning vectors `gamma`/`beta`/`alpha` are small (`[B, C]`); mark with `cudaAccessPropertyStreaming` to pin them in L2 across token iterations
- **HBM3 bandwidth** (3.35 TB/s): two-pass design is bandwidth-bound; H100's higher BW vs A100 (2.0 TB/s) gives ~1.67x raw BW advantage
- **FP16 tensor cores not needed here** (element-wise op); the kernel is **compute-light and BW-heavy**, so it benefits primarily from H100's memory subsystem

### Expected Speedup

- **1.8x–2.5x** vs. unfused PyTorch implementation (`nn.LayerNorm` + pointwise multiply + add)
- **1.3x–1.6x** vs. Apex FusedLayerNorm (which doesn't include the conditioning modulation)
- Speedup scales with batch size; at batch=1 the kernel is memory-latency bound; at batch=32+ it becomes BW-bound and speedup is maximized
- Reference: DiT paper ablations, xDiT team benchmarks (2024)

### Benchmark Metrics

| Metric | Tool |
|--------|------|
| Kernel time (µs) | `nsys profile` + NVTX ranges |
| HBM bandwidth utilization | `ncu --metrics l1tex__t_bytes,lts__t_bytes` |
| L2 hit rate for conditioning tensors | `ncu --metrics lts__t_sectors_hit_rate` |
| Arithmetic intensity (FLOP/byte) | manual: `2*B*T*C / bytes_read` |
| End-to-end DiT block latency | wall-clock, averaged over 1000 iters |

### Implementation Complexity

**Medium** — The Welford reduction pattern is well-understood; the main complexity is handling vectorized `__half2` loads aligned to 128-bit boundaries and correctly fusing the optional gate for adaLN-zero vs. standard adaLN.

---

## 2. `int4_fp8_attention`

### What It Does

Implements SageAttention2-style quantized attention: Q and K matrices are quantized to **INT4** per-warp before the QK^T GEMM, and the softmax probability matrix P is accumulated in **FP8 E4M3** during the P·V multiply. This dramatically reduces the arithmetic intensity of the attention operation while keeping output quality close to full-precision attention. The dequantization and quantization steps are fused into the GEMM prologues/epilogues.

### Applicable Models / Architectures

Any transformer with multi-head or grouped-query attention (MHA, GQA, MQA): GPT-4 style LLMs, LLaMA 3, Mistral, Qwen2, Gemma 2, FLUX.1 (MM-DiT attention), CogVideoX (3D attention), HunyuanVideo, any ViT/DeiT, any cross-attention in encoder-decoder models, SDXL U-Net cross-attention.

### Key Algorithm / Technique

**INT4 QK quantization**: For each warp's tile of Q (or K), compute the per-warp absmax, then quantize to INT4 using:
```
q_int4 = clamp(round(q_fp16 / scale), -8, 7)
scale  = absmax(q_fp16) / 7.0
```
The INT4 values are packed two-per-byte and fed into the INT4 tensor core GEMM (`mma.m16n8k64.s4` on H100) to produce INT32 accumulators, which are immediately rescaled to FP16 via the stored scale factors.

**FP8 PV accumulation**: After softmax, the P matrix is quantized to FP8 E4M3 before the P·V GEMM. This uses H100's native FP8 tensor cores.

The two stages (QK and PV) tile the sequence dimension to keep intermediate matrices in SRAM, following FlashAttention's tiling strategy.

```cuda
// INT4 packing: two INT4 values -> one byte
__device__ __forceinline__ uint8_t pack_int4x2(int8_t hi, int8_t lo) {
    return ((uint8_t)(hi & 0x0F) << 4) | (uint8_t)(lo & 0x0F);
}

// Per-warp absmax using warp-level reduction
__device__ __forceinline__ float warp_absmax(float val) {
    val = fabsf(val);
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    return val;
}
```

### CUDA Primitives Used

- `mma.m16n8k64.row.col.s32.s4.s4.s32` PTX instruction for INT4 tensor core GEMM
- `mma.m16n8k32.row.col.f32.e4m3.e4m3.f32` for FP8 PV GEMM (H100 only)
- `__shfl_xor_sync(0xffffffff, ...)` for per-warp absmax computation
- `__ldmatrix` for efficient shared-memory → register matrix loads
- `cp.async.cg` for async global→shared copies of Q, K, V tiles
- `lop3.b32` for fast INT4 bit manipulation and packing

### H100 Hardware Features Exploited

- **FP8 tensor cores**: H100 is the first datacenter GPU with native FP8 E4M3/E5M2 GEMM support (989 TFLOPS FP8 vs 312 TFLOPS FP16)
- **INT4 tensor cores**: `mma.m16n8k64.s4` doubles the throughput of INT8 tensor cores
- **SRAM capacity (228 KB per SM)**: H100's larger L1/shared memory vs A100 (164 KB) allows larger attention tiles, reducing HBM traffic
- **Warp specialization**: producer warps handle async copies while consumer warps run tensor core GEMMs

### Expected Speedup

- **1.8x–3.1x** vs. FlashAttention-2 FP16 (SageAttention2 paper reports 2.23x on H100 for 4096 seqlen)
- **1.4x–2.0x** vs. FlashAttention-3 FP16
- Most pronounced at long sequences (>4096 tokens) where attention is FLOPs-bound
- Quality: <0.01% output MSE vs. FP16 attention in practice (per SageAttention2 evaluation)

### Benchmark Metrics

| Metric | Tool |
|--------|------|
| Attention kernel TFLOPS | `ncu --metrics sm__ops_path_tensor_src_fp8` |
| Peak memory bandwidth | `ncu --metrics l1tex__t_bytes_pipe_lsu_mem_global` |
| Output MSE vs FP16 reference | custom validation harness |
| Tokens/second end-to-end | throughput benchmark |
| Quantization scale calibration error | histogram of relative errors |

### Implementation Complexity

**High** — Correct INT4 packing, scale computation, and the coupling with FlashAttention-style tiling require careful register management. The FP8 epilogue requires H100-specific PTX. Not trivially portable to A100.

---

## 3. `fa3_warp_specialized_attention`

### What It Does

Implements the FlashAttention-3 attention kernel design featuring **warp specialization**: some warps act as producers (issuing TMA async copies of Q/K/V tiles from HBM into SMEM) while other warps act as consumers (executing WGMMA tensor core GEMMs and softmax). This overlaps data movement with computation at the warp level, hiding the full HBM latency and achieving near-peak utilization of both the memory subsystem and the tensor cores simultaneously.

### Applicable Models / Architectures

Any self-attention or cross-attention in any transformer: LLMs (GPT, LLaMA, Mistral, Falcon, Gemma), vision transformers (ViT, DeiT, SAM), diffusion transformers (DiT, FLUX, SD3, CogVideoX), multimodal models (LLaVA, Flamingo, Idefics), encoder-decoder models (T5, BART), any model using multi-head, grouped-query, or multi-query attention.

### Key Algorithm / Technique

**Producer-Consumer warp split**: The thread block is divided into producer warps and consumer warps via `cutlass::arch::warpgroup_reg_alloc`. Producer warps use TMA (`cp.async.bulk.tensor`) to issue bulk copies of the next K/V tile while consumer warps run WGMMA on the current tile. A barrier (`mbarrier`) synchronizes the pipeline stages.

**WGMMA (Warpgroup Matrix Multiply-Accumulate)**: Unlike `mma` (single-warp), WGMMA operates across an entire warpgroup (4 warps = 128 threads), enabling larger tile sizes and better register utilization.

```cuda
// Conceptual FA3 warp-specialized loop structure (pseudocode)
// Producer warp (warp 0):
for (int tile = 0; tile < num_kv_tiles; ++tile) {
    // Async bulk copy next K/V tile via TMA descriptor
    cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
        [smem_kv_ping_pong[(tile+1)%2]], [gmem_kv_desc], {tile+1, 0},
        mbar[fill_stage];
    // Wait on current tile's mbarrier
    mbarrier_wait(mbar[use_stage]);
}

// Consumer warpgroup (warps 1-4):
// WGMMA: Q_tile @ K_tile^T -> S_tile (accumulate in RF)
wgmma::mma_async(acc_s, smem_q, smem_k);
// Softmax online update (rescaling)
// WGMMA: P_tile @ V_tile -> O_tile
wgmma::mma_async(acc_o, smem_p, smem_v);
```

### CUDA Primitives Used

- `cp.async.bulk.tensor.2d.shared::cluster.global` — TMA (Tensor Memory Accelerator) bulk copy with descriptor
- `mbarrier.init` / `mbarrier.arrive` / `mbarrier.try_wait.parity` — pipeline stage synchronization
- `wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16` — WGMMA FP16 warpgroup GEMM
- `wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3` — WGMMA FP8 variant
- `__shfl_xor_sync` for per-row online softmax normalization across warp lanes
- `cutlass::arch::fence_view_async_shared()` for async SMEM visibility

### H100 Hardware Features Exploited

- **TMA (Tensor Memory Accelerator)**: H100's dedicated hardware unit offloads address calculation and memory copy from CUDA cores; a single TMA instruction initiates a 2D strided copy with no software loop
- **WGMMA**: 4x larger tiles than single-warp `mma`, achieves >90% of peak 989 TFLOPS FP8 / 494 TFLOPS FP16
- **Distributed shared memory / cluster**: CTA clusters share SMEM across SMs for Q broadcasting
- **HBM3 + NVLink bandwidth**: FA3's pipeline maximizes bandwidth utilization by fully hiding latency

### Expected Speedup

- **1.5x–2.0x** vs. FlashAttention-2 (FA3 paper: 1.5x–2.0x on H100 for FP16; up to 2.6x for FP8)
- Reaches **75% of H100 peak FP8 throughput** at seqlen ≥ 8192
- **1.35x** prefill speedup in LLaMA-3 70B benchmark (reported by Together AI)
- Prefill-dominant workloads benefit most; decode (batch=1, seqlen=1) is memory-bound and less affected

### Benchmark Metrics

| Metric | Tool |
|--------|------|
| TFLOPS achieved | `ncu --metrics sm__ops_path_tensor` |
| TMA efficiency | `ncu --metrics l1tex__data_pipe_lsu_wavefronts_mem_shared` |
| WGMMA utilization | `ncu --metrics sm__inst_executed_pipe_tensor_op_hmma` |
| Warp stall cycles | `ncu --metrics smsp__warp_issue_stalled_*` |
| MFU (model FLOP utilization) | `actual_FLOPS / peak_FLOPS` |

### Implementation Complexity

**High** — Requires deep knowledge of Hopper-specific PTX (TMA descriptors, mbarrier, WGMMA), warpgroup coordination, and register pressure management. The CUTLASS 3.x library provides reference implementations but custom kernels require non-trivial expertise.

---

## 4. `tiered_kv_quantization`

### What It Does

Implements a **temporal depth-aware KV cache compression** scheme: as tokens age (i.e., as more new tokens are generated), their KV entries are progressively re-quantized from FP16 → FP8 → INT4. The kernel handles the in-place re-quantization of aging cache slots during the decoding loop, maintaining a generation-count index per token slot to trigger tier transitions automatically.

### Applicable Models / Architectures

Any autoregressive (causal) language model with a growing KV cache: LLaMA, Mistral, Qwen, Falcon, GPT-NeoX, OPT, Phi, Gemma, Mamba (if using KV cache), and any DiT or video model that uses autoregressive decoding (e.g., LlamaGen, VAR). Also applicable to speculative decoding targets where the KV cache is shared between draft and target models.

### Key Algorithm / Technique

A **sliding tier structure** maintains three cache regions in a single preallocated buffer:

- **Tier 0 (FP16)**: most recent `W0` tokens — full precision
- **Tier 1 (FP8 E4M3)**: tokens at positions `[W0, W0+W1)` — FP8 quantized
- **Tier 2 (INT4)**: tokens older than `W0+W1` — INT4 quantized

On each decoding step, a lightweight `tiered_kv_quantization` kernel scans the cache metadata, identifies slots that have crossed a tier threshold, and re-quantizes them in-place. The kernel also maintains per-slot scale factors in a compact auxiliary buffer.

```cuda
__global__ void tiered_kv_requant_kernel(
    void*  __restrict__ kv_cache,       // unified buffer, mixed precision
    float* __restrict__ scale_buf,      // per-slot scale factors
    int*   __restrict__ tier_meta,      // tier assignment per slot
    int*   __restrict__ gen_count,      // generation count per slot
    int total_slots, int w0, int w1)
{
    int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= total_slots) return;

    int age = gen_count[slot];
    int current_tier = tier_meta[slot];

    if (age == w0 && current_tier == 0) {
        // FP16 -> FP8 in-place
        __half* src = (__half*)kv_cache + slot * HEAD_DIM;
        // compute absmax, store scale, pack to FP8
        float absmax = 0.f;
        // ... (warp_absmax over HEAD_DIM elements)
        float scale = absmax / 448.f;  // FP8 E4M3 max = 448
        scale_buf[slot] = scale;
        // rewrite slot in FP8
        __nv_fp8_e4m3* dst = (__nv_fp8_e4m3*)kv_cache + slot * HEAD_DIM;
        // ... pack loop
        tier_meta[slot] = 1;
    } else if (age == w0 + w1 && current_tier == 1) {
        // FP8 -> INT4: dequant FP8, requant INT4
        // ...
        tier_meta[slot] = 2;
    }
}
```

### CUDA Primitives Used

- `__shfl_xor_sync(0xffffffff, ...)` for warp-level absmax over `HEAD_DIM` elements
- `__nv_fp8_e4m3` intrinsics (`__nv_cvt_float_to_fp8`) for FP8 packing
- `lop3.b32` for INT4 nibble packing (two INT4 values per byte)
- `atomicCAS` for tier metadata updates (prevent double-requantization race)
- `cp.async` for prefetching the next batch of slots

### H100 Hardware Features Exploited

- **FP8 native storage**: H100 is the first GPU with hardware FP8 load/store in the memory pipeline; `__nv_fp8_e4m3` values can be stored and loaded without software emulation
- **Large L2 (50 MB)**: The scale factor auxiliary buffer fits entirely in L2 at typical context lengths (4096 tokens × 32 heads × 4 bytes = 512 KB)
- **Memory compression**: INT4 representation gives 4x compression over FP16, allowing 4x longer context at the same VRAM budget

### Expected Speedup

- **2.0x–4.0x** KV cache memory reduction (FP16 baseline vs. INT4 oldest tier)
- **1.3x–1.8x** end-to-end decode throughput improvement at long context (>16K tokens) due to reduced HBM bandwidth pressure during KV prefetch
- Reference: KVQuant (Hooper et al., 2024) reports 4x compression with <0.5% perplexity degradation; tiered approach trades some accuracy for better recent-token precision

### Benchmark Metrics

| Metric | Tool |
|--------|------|
| KV cache memory footprint (MB) | `torch.cuda.memory_allocated()` |
| Decode tokens/sec at varying context lengths | throughput benchmark |
| Perplexity vs. FP16 KV cache | WikiText-103 PPL |
| Requantization kernel overhead (µs/step) | `nsys profile` |
| HBM read bytes during attention | `ncu --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld` |

### Implementation Complexity

**Medium** — The quantization math is straightforward; complexity arises from managing the mixed-precision buffer layout (variable-width slots), maintaining consistent scale factor metadata, and ensuring correct behavior during speculative decoding rollbacks.

---

## 5. `h2o_token_eviction`

### What It Does

Implements the **Heavy-Hitter Oracle (H2O)** token eviction policy as a fused CUDA kernel. During the forward pass of each attention layer, the kernel accumulates per-token **attention score sums** across all heads and queries seen so far. At eviction time, it performs an in-kernel top-K selection to retain only the `K` most "attended-to" tokens plus a fixed recent-token window, discarding the rest from the KV cache.

### Applicable Models / Architectures

Any transformer using causal (autoregressive) attention with a bounded KV cache budget: LLaMA 2/3, Mistral, Falcon, GPT-NeoX, Qwen, Phi-3, CodeLlama, any instruction-tuned LLM used in chat/completion settings. Also applicable to diffusion models with iterative refinement steps where attention patterns are consistent across denoising steps.

### Key Algorithm / Technique

Two cooperating kernels implement H2O:

1. **Score accumulation kernel**: During each attention forward pass, an auxiliary kernel reads the attention weight matrix A (shape `[B, H, T_q, T_k]`) and atomically accumulates column sums into a score buffer `s[b, k]` representing total attention received by each past token.

2. **Eviction kernel**: At the eviction threshold, performs warp-parallel top-K selection on `s[b, :]` using a **register-resident max-heap** of size K, then compacts the KV cache by gathering surviving token slots to contiguous positions.

```cuda
// Score accumulation: called after each attention softmax
__global__ void h2o_score_accumulate(
    const float* __restrict__ attn_weights,  // [B, H, T_q, T_k]
    float* __restrict__ token_scores,         // [B, T_k] accumulated
    int B, int H, int T_q, int T_k)
{
    int b = blockIdx.z;
    int k = blockIdx.x * blockDim.x + threadIdx.x;  // key token index
    if (k >= T_k) return;

    float sum = 0.f;
    for (int h = 0; h < H; h++)
        for (int q = 0; q < T_q; q++)
            sum += attn_weights[((b*H + h)*T_q + q)*T_k + k];

    atomicAdd(&token_scores[b*T_k + k], sum);
}
```

### CUDA Primitives Used

- `__shfl_xor_sync(0xffffffff, ...)` for warp-level max reduction during top-K selection
- `atomicAdd` for concurrent score accumulation across multiple attention layers
- `float4` vectorized loads for attention weight matrix reads
- `__ldg()` for read-only cache-line-optimized loads of score buffer
- Warp-level bitonic sort (using `__shfl_xor_sync`) for small-K top-K (K ≤ 32)

### H100 Hardware Features Exploited

- **L2 cache (50 MB)**: Score buffer `[B, T_k]` fits in L2 for typical batch sizes, making atomic accumulation fast
- **Distributed atomics**: H100's improved atomic throughput reduces contention in the accumulation kernel
- **Stream multi-processor count (132 SMs)**: High parallelism allows score accumulation to overlap with the main attention compute

### Expected Speedup

- **1.5x–2.0x** decode throughput at context length >8192 with 50% KV budget (H2O paper, Zhang et al. 2023)
- Quality: <2% accuracy degradation on LongBench at 20% KV retention rate
- Memory: linear reduction in KV cache size (50% budget → 50% memory)

### Benchmark Metrics

| Metric | Tool |
|--------|------|
| KV cache hit rate | custom counter |
| Eviction kernel overhead (µs) | `nsys profile` |
| LongBench accuracy vs. full KV | benchmark suite |
| Peak memory at 32K context | `torch.cuda.max_memory_allocated()` |
| Top-K selection throughput (tokens/ms) | microbenchmark |

### Implementation Complexity

**Medium** — The accumulation pattern is straightforward; the main challenge is correctly integrating with FlashAttention (which doesn't materialize the full attention matrix), requiring a modified FA kernel that also writes out row-wise attention sums.

---

## 6. `tome_token_merge`

### What It Does

Implements **Token Merging (ToMe)** as a fused CUDA kernel. ToMe reduces token count in vision/diffusion transformers by finding the most similar token pairs across a bipartite partition of the token sequence and merging (averaging) them. This reduces the effective sequence length by up to 50% per layer, linearly reducing the cost of all subsequent operations.

### Applicable Models / Architectures

Any transformer operating on tokenized 2D or 3D spatial data: ViT (all variants), DeiT, MAE, DINO, SAM, CLIP visual encoder, DiT, FLUX, SD3 (image patches), CogVideoX, HunyuanVideo (video patches), ViViT, VideoMAE, any UNet with transformer bottleneck. Most effective when token similarity is high (smooth image regions, video frames with low motion).

### Key Algorithm / Technique

**Bipartite soft matching**: The token sequence is split into two sets A (odd positions) and B (even positions). For each token in A, the kernel finds its nearest neighbor in B using cosine similarity computed on a **reduced-dimension key projection** (typically the Q or K vectors from the preceding attention). Matched pairs are merged by averaging (with optional importance weighting).

```cuda
// Phase 1: compute cosine similarity between A and B partitions
__global__ void tome_similarity_kernel(
    const __half* __restrict__ keys,  // [B, T, D] — attention keys
          float* __restrict__ sim,    // [B, T/2] — max similarity per A token
          int*   __restrict__ match,  // [B, T/2] — matched B index per A token
    int T, int D)
{
    int b = blockIdx.y;
    int a_idx = blockIdx.x * blockDim.x + threadIdx.x;  // A partition index
    if (a_idx >= T/2) return;

    int best_b = -1;
    float best_sim = -1.f;

    // Dot product with each B token (vectorized with float4)
    for (int b_idx = T/2; b_idx < T; b_idx++) {
        float dot = 0.f;
        for (int d = 0; d < D; d += 8) {
            // vectorized half dot product
        }
        if (dot > best_sim) { best_sim = dot; best_b = b_idx; }
    }
    sim[b*T/2 + a_idx]   = best_sim;
    match[b*T/2 + a_idx] = best_b;
}

// Phase 2: merge matched pairs (scatter-add with normalization)
```

### CUDA Primitives Used

- `__shfl_xor_sync` for warp-level dot product reduction (D-dim inner product split across warp lanes)
- `float4` / `__half2` vectorized loads for key vectors
- `atomicAdd` for scatter-merge (accumulate matching B token into A slot)
- `__ldg()` for read-only key cache access
- Optional `cub::DeviceRadixSort` for sorting by similarity score to select top-r merges

### H100 Hardware Features Exploited

- **Large SRAM (228 KB)**: For small D (e.g., D=64 projected keys), the entire B partition's key matrix for a CTA can fit in shared memory, converting O(T²) global loads to O(T) global reads
- **FP16 tensor cores**: If D is large (≥ 64), the A×B^T similarity matrix can be computed as a small GEMM using tensor cores
- **Persistent kernel**: For video models with many frames, a persistent kernel can process multiple frames in a single launch

### Expected Speedup

- **1.4x–2.0x** end-to-end ViT inference at 50% token reduction (ToMe paper, Bolya et al. 2023)
- **1.2x–1.6x** for DiT inference (less speedup because DiT blocks are heavier per token)
- Quality: <0.5% top-1 accuracy drop on ImageNet at r=8 merges/layer
- Memory: up to 50% reduction in attention FLOPs and KV memory

### Benchmark Metrics

| Metric | Tool |
|--------|------|
| Tokens remaining after each block | custom counter |
| Similarity computation throughput | microbenchmark (TFLOPS) |
| Merge kernel overhead (µs) | `nsys profile` |
| ImageNet/FID quality vs. no-ToMe | evaluation harness |
| End-to-end image generation latency | wall-clock benchmark |

### Implementation Complexity

**Medium** — The bipartite matching is conceptually simple; challenges are efficient vectorized similarity computation, handling variable merge rates per layer, and correctly propagating merged token positions through downstream operations.

---

## 7. `per_group_int4_quantize`

### What It Does

A general-purpose CUDA kernel that quantizes a floating-point tensor (FP16 or BF16) to **INT4 with per-group scaling**. The input tensor is divided into groups of `G` elements (typically G=128); within each group, the absmax is computed and used as a scale factor, and each element is quantized to a signed 4-bit integer. The output packs two INT4 values per byte. This is the fundamental building block for weight-only INT4 quantization (GPTQ, AWQ, QMoE) and activation quantization.

### Applicable Models / Architectures

**Universal** — any weight matrix or activation tensor in any neural network: LLM linear layers (QKV projections, FFN, output projections), DiT MLP layers, VAE encoder/decoder convolution weights, any LoRA adapter weights, any tensor that benefits from INT4 compression.

### Key Algorithm / Technique

One warp (32 threads) processes one group of G elements. The warp performs:
1. Cooperative absmax computation via `__shfl_xor_sync` tree reduction
2. Scale computation: `scale = absmax / 7.0` (for signed INT4 range [-8, 7])
3. Quantization loop: each thread quantizes its elements: `q = clamp(round(x / scale), -8, 7)`
4. Packing: pairs of INT4 values are packed into bytes using bit manipulation

```cuda
__global__ void per_group_int4_quantize_kernel(
    const __half* __restrict__ input,   // [N] flattened
    uint8_t*      __restrict__ output,  // [N/2] packed INT4
    __half*       __restrict__ scales,  // [N/G] per-group scales
    int N, int G)
{
    int group_id = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int lane     = threadIdx.x % 32;
    int base     = group_id * G;
    if (base >= N) return;

    // Step 1: compute absmax over the group (each lane handles G/32 elements)
    float local_max = 0.f;
    for (int i = lane; i < G; i += 32) {
        float val = __half2float(input[base + i]);
        local_max = fmaxf(local_max, fabsf(val));
    }
    // Warp reduction
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, mask));

    float scale = local_max / 7.0f;
    if (lane == 0) scales[group_id] = __float2half(scale);
    float inv_scale = (scale > 0.f) ? (1.f / scale) : 0.f;

    // Step 2: quantize and pack
    for (int i = lane * 2; i < G; i += 64) {
        int8_t q0 = (int8_t)__float2int_rn(
            fminf(fmaxf(__half2float(input[base+i])   * inv_scale, -8.f), 7.f));
        int8_t q1 = (int8_t)__float2int_rn(
            fminf(fmaxf(__half2float(input[base+i+1]) * inv_scale, -8.f), 7.f));
        output[(base + i) / 2] = pack_int4x2(q0, q1);
    }
}
```

### CUDA Primitives Used

- `__shfl_xor_sync(0xffffffff, val, mask)` — warp-level tree reduction for absmax
- `__float2int_rn` — round-to-nearest float-to-int conversion
- `__half2float` / `__float2half` — FP16 conversion intrinsics
- Bitwise packing: `((q0 & 0xF) << 4) | (q1 & 0xF)` using 32-bit integer ops
- `__ldg()` for read-only input loads through texture cache

### H100 Hardware Features Exploited

- **132 SMs**: embarrassingly parallel over groups; H100's SM count allows quantizing large weight matrices in <100 µs
- **L2 cache (50 MB)**: scale factor output buffer (typically small) cached in L2 for subsequent dequantization access
- **Memory bandwidth (3.35 TB/s)**: kernel is bandwidth-bound (2:1 read:write ratio after packing); H100's BW advantage over A100 is fully exploited

### Expected Speedup

- **Quantization throughput**: ~500 GB/s effective (limited by input read bandwidth)
- **Memory savings**: 4x compression (INT4 vs FP16), e.g., LLaMA-3 70B weights shrink from ~140 GB to ~35 GB
- **Downstream GEMM speedup**: 2x–4x for INT4 weight-only GEMM vs FP16 (per ExLlama, AutoAWQ benchmarks)

### Benchmark Metrics

| Metric | Tool |
|--------|------|
| Quantization throughput (GB/s) | `nsys`, bandwidth = input_bytes / kernel_time |
| Quantization error (MSE, SQNR) | custom validation |
| Model perplexity (WIKITEXT-2 PPL) | lm-eval-harness |
| Downstream GEMM latency | microbenchmark |
| Scale buffer memory overhead | `sizeof(scales) / sizeof(weights)` |

### Implementation Complexity

**Low** — Well-understood algorithm with a straightforward warp-cooperative implementation. The main variation is supporting different group sizes (G=32, 64, 128) and output dtypes (asymmetric INT4 with zero-point vs. symmetric).

---

## 8. `per_group_fp8_quantize`

### What It Does

General-purpose kernel for quantizing any FP16/BF16 tensor to **FP8 E4M3** with per-group scaling, using H100's native `__nv_fp8_e4m3` type. Unlike INT4, FP8 preserves the floating-point distribution shape and is directly supported by H100 tensor cores for GEMM. This kernel is the prerequisite for all FP8 linear layer and attention implementations.

### Applicable Models / Architectures

Universal — any activation or weight tensor. Particularly valuable for: attention QKV projections (enabling FP8 attention GEMMs), FFN linear layers, LoRA adapter weights, embedding tables, any tensor where FP8's dynamic range advantage over INT8 matters (e.g., activations with outliers).

### Key Algorithm / Technique

Similar group structure to INT4 quantization but outputs to `__nv_fp8_e4m3`. Per-group scale is set to map the group's max absolute value to the FP8 maximum (448.0 for E4M3):

```cuda
__global__ void per_group_fp8_e4m3_quantize_kernel(
    const __half*        __restrict__ input,
    __nv_fp8_e4m3*       __restrict__ output,
    float*               __restrict__ scales,
    int N, int G)
{
    int group_id = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int lane     = threadIdx.x % 32;
    int base     = group_id * G;
    if (base >= N) return;

    // Warp absmax
    float local_max = 0.f;
    for (int i = lane; i < G; i += 32)
        local_max = fmaxf(local_max, fabsf(__half2float(input[base + i])));
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, mask));

    const float FP8_E4M3_MAX = 448.f;
    float scale = local_max / FP8_E4M3_MAX;
    if (lane == 0) scales[group_id] = scale;
    float inv_scale = (scale > 1e-8f) ? (1.f / scale) : 0.f;

    // Quantize to FP8 E4M3
    for (int i = lane; i < G; i += 32) {
        float val = __half2float(input[base + i]) * inv_scale;
        output[base + i] = __nv_cvt_float_to_fp8(val, __NV_SATFINITE, __NV_E4M3);
    }
}
```

### CUDA Primitives Used

- `__nv_cvt_float_to_fp8(val, __NV_SATFINITE, __NV_E4M3)` — H100 hardware FP8 conversion intrinsic
- `__shfl_xor_sync(0xffffffff, ...)` — warp absmax reduction
- `__half2float` / `__float2half` for FP16 I/O
- `__ldg()` for read-only input tensor access
- Optional: `__nv_fp8x4_e4m3` for vectorized 4-element FP8 stores

### H100 Hardware Features Exploited

- **Native FP8 type support**: H100 introduces hardware support for `__nv_fp8_e4m3` and `__nv_fp8_e5m2`; the conversion intrinsic maps to a single hardware instruction (vs. software emulation on A100)
- **FP8 tensor core throughput**: 989 TFLOPS FP8 (2x FP16) — quantized outputs are directly consumable by FP8 WGMMA/mma instructions
- **FP8 memory footprint**: 1 byte per value vs 2 bytes for FP16, reducing HBM traffic 2x in subsequent GEMMs

### Expected Speedup

- **Quantization kernel**: ~600 GB/s effective throughput (faster than INT4 due to simpler packing)
- **Downstream FP8 GEMM**: 1.5x–2.0x vs FP16 GEMM on H100 (NVIDIA transformer engine benchmarks)
- **Combined with FA3 FP8 attention**: up to 2.6x end-to-end attention speedup (FA3 paper)

### Benchmark Metrics

| Metric | Tool |
|--------|------|
| Conversion throughput (GB/s) | kernel time + bytes transferred |
| Saturation rate (values clipped) | histogram analysis |
| Downstream GEMM TFLOPS | `ncu --metrics sm__ops_path_tensor_src_fp8` |
| Output MSE vs FP16 reference | validation harness |
| Scale calibration overhead | profiler |

### Implementation Complexity

**Low** — Structurally identical to INT4 but simpler (no packing step). H100-specific due to `__nv_cvt_float_to_fp8` intrinsic; A100 requires software FP8 emulation.

---

## 9. `fused_conv_norm_act`

### What It Does

Fuses a **Conv2D → GroupNorm → SiLU (or GELU) activation** sequence into a single kernel pass. This is the ubiquitous building block of VAE encoders/decoders, U-Net-based diffusion models, and any CNN backbone used in a larger system. The fusion eliminates two intermediate tensor materializations (after Conv and after GroupNorm), saving 2× the activation memory and 2× the HBM round-trips for those intermediate buffers.

### Applicable Models / Architectures

Any model with a convolutional encoder or decoder: Stable Diffusion VAE (encoder and decoder), SDXL VAE, FLUX VAE, CogVideoX VAE (3D), HunyuanVideo VAE, any U-Net with ResBlock (ResNet + GroupNorm + SiLU), ControlNet, IP-Adapter encoder, image tokenizers (VQVAE, VQGAN, FSQ), video tokenizers.

### Key Algorithm / Technique

The kernel implements **implicit GEMM** for Conv2D (reshaping the convolution as a matrix multiply of the input im2col-transformed patches against the filter matrix) followed by **in-register GroupNorm + SiLU**. The key insight is that GroupNorm operates per-channel-group per spatial position, so its mean/variance can be computed immediately after the GEMM accumulation without writing to HBM.

For small spatial sizes (H×W ≤ 32×32, common in deep VAE layers), the entire output feature map for one group fits in shared memory:

1. Compute Conv2D output tile into SMEM via tensor core GEMM
2. Warp-cooperative Welford mean/var over the group channels in SMEM
3. Normalize + apply SiLU: `y = x_norm * sigmoid(x_norm)`, all in registers
4. Write final output to HBM

```cuda
// SiLU fused with GroupNorm normalization (in-register)
__device__ __forceinline__ float apply_groupnorm_silu(
    float val, float mean, float inv_std, float gamma, float beta)
{
    float x_norm = (val - mean) * inv_std * gamma + beta;
    return x_norm / (1.f + expf(-x_norm));  // SiLU = x * sigmoid(x)
}
```

### CUDA Primitives Used

- `mma.m16n8k16.row.col.f32.f16.f16.f32` — tensor core GEMM for Conv2D (implicit GEMM)
- `__shfl_xor_sync(0xffffffff, ...)` — warp Welford reduction for GroupNorm
- `cp.async.cg` — async copy of input image tiles into shared memory
- `__expf` — fast approximate exp for SiLU
- `float4` vectorized loads for input image patches

### H100 Hardware Features Exploited

- **TMA 2D**: H100's TMA can copy 2D image tiles (with stride) directly, making im2col-free convolution easier to implement efficiently
- **Large SRAM**: 228 KB allows larger feature map tiles per SM, improving reuse
- **WGMMA for implicit GEMM**: H100's WGMMA tiles are better suited to the irregular-shaped GEMMs that arise from convolution (K dimension = filter_h × filter_w × C_in)

### Expected Speedup

- **1.5x–2.2x** vs. unfused cuDNN Conv + PyTorch GroupNorm + pointwise SiLU
- **1.2x–1.5x** vs. cuDNN fused Conv+BN (cuDNN fusion does BN not GroupNorm)
- Memory savings: 2× for intermediate activations (significant for large-batch VAE encoding)

### Benchmark Metrics

| Metric | Tool |
|--------|------|
| VRAM peak during VAE encode/decode | `torch.cuda.max_memory_allocated()` |
| Kernel time vs. unfused baseline | `nsys profile` |
| Conv arithmetic intensity | `ncu --metrics sm__ops_path_tensor` |
| GroupNorm accuracy (mean/var error) | unit test vs. PyTorch reference |
| End-to-end VAE encode latency | wall-clock |

### Implementation Complexity

**High** — Combining implicit GEMM (non-trivial tiling) with in-register normalization requires careful handling of GroupNorm's cross-channel dependency and padding for non-power-of-2 channel counts.

---

## 10. `warp_topk_selection`

### What It Does

A general-purpose warp-parallel top-K selection kernel using shuffle-based sorting networks. Given an array of (score, index) pairs per row (or per batch element), the kernel selects the top K entries by score using a **warp-level bitonic sort** for K ≤ 32, or a **register-resident heap** for larger K. This is a fundamental utility primitive used by token eviction (H2O), expert routing (MoE), speculative decoding, and any other system component that needs fast top-K.

### Applicable Models / Architectures

**Universal utility kernel** — used as a subroutine by: H2O eviction, ToMe similarity ranking, MoE expert selection (any MoE LLM: Mixtral, DeepSeek-MoE, OLMoE), speculative decoding draft candidate ranking, SnapKV importance scoring, LoRA adapter selection, any beam search / sampling algorithm.

### Key Algorithm / Technique

**For K ≤ 32 (fits in one warp)**: Warp-level bitonic sort using only `__shfl_xor_sync` comparisons. Each thread holds one (score, index) pair; bitonic sort exchanges are implemented as:

```cuda
__device__ __forceinline__
void bitonic_step(float& score, int& idx, int partner_lane, bool ascending) {
    float other_score = __shfl_xor_sync(0xffffffff, score, partner_lane);
    int   other_idx   = __shfl_xor_sync(0xffffffff, idx,   partner_lane);
    bool swap = ascending ? (score < other_score) : (score > other_score);
    if (swap) { score = other_score; idx = other_idx; }
}

// Full bitonic sort for warp of 32 elements
__device__ void warp_bitonic_sort_32(float& score, int& idx) {
    #pragma unroll
    for (int k = 2; k <= 32; k <<= 1)
        for (int j = k >> 1; j > 0; j >>= 1) {
            int partner = threadIdx.x ^ j;
            bool ascending = !((threadIdx.x & k) != 0);
            bitonic_step(score, idx, j, ascending);
        }
}
```

**For K > 32**: A register-resident min-heap of size K is maintained per thread-group; each new element is compared against the heap minimum and inserted if larger, using `__shfl_xor_sync` for inter-thread communication.

### CUDA Primitives Used

- `__shfl_xor_sync(0xffffffff, val, mask)` — the sole primitive for all data exchange in warp sorts
- `__shfl_sync(0xffffffff, val, src_lane)` — for heap root broadcasts
- `__ballot_sync` — for checking convergence conditions
- Integer bit manipulation (`x ^ j`, `x & k`) for bitonic network index computation

### H100 Hardware Features Exploited

- **Warp-level primitives have zero overhead on H100**: warp shuffle executes in a single clock cycle with no barrier needed
- **Register file (256 KB per SM)**: For K=32, the top-K fits entirely in registers across one warp with no SMEM needed
- **High SM count (132)**: Enables processing many independent top-K problems in parallel (one per batch element per layer)

### Expected Speedup

- **3x–8x** vs. `cub::DeviceRadixSort` for small K (K ≤ 32) due to zero SMEM and zero global memory usage
- **2x–3x** vs. `cub::DeviceSelect` for K up to 128
- Latency: ~2–5 µs for top-K with K=32 over T=4096 tokens per batch element

### Benchmark Metrics

| Metric | Tool |
|--------|------|
| Sort throughput (elements/sec) | microbenchmark |
| Latency at various K and T values | `nsys profile` |
| Register occupancy | `ncu --metrics sm__maximum_warps_per_active_cycle_pct` |
| Correctness vs. `thrust::sort` | unit test |
| Warp efficiency | `ncu --metrics smsp__thread_inst_executed_per_inst_executed` |

### Implementation Complexity

**Low** — For K ≤ 32, pure warp primitives with no SMEM. For K > 32, the heap approach is more complex but still self-contained. Well-documented in CUB and custom CUDA literature.

---

## 11. `indexed_gather_scatter`

### What It Does

A general-purpose coalesced **indexed gather and scatter kernel** for KV cache compaction. Given an index array specifying which token slots to keep, the kernel gathers the corresponding KV entries from a sparse (fragmented) cache into a dense (contiguous) output buffer, using vectorized `float4` loads/stores to maximize memory throughput. The inverse operation (scatter) places computed KV values at indexed positions.

### Applicable Models / Architectures

**Universal utility kernel** — used by: KV cache compaction after token eviction (H2O, SnapKV, StreamingLLM), vLLM-style PagedAttention block relocation, MoE expert output scatter, embedding table lookup (any model with learned embeddings), token merge/unmerge in ToMe, batch repacking for dynamic sequence lengths.

### Key Algorithm / Technique

**Vectorized gather** with `float4` (128-bit loads): Each thread reads 8 consecutive FP16 values (= 128 bits = one cache line element) from an indexed source position. The index array is sorted to improve source access locality.

```cuda
// float4-vectorized indexed gather for KV compaction
// KV cache layout: [num_slots, num_heads, head_dim]
__global__ void indexed_gather_kv_kernel(
    const __half*  __restrict__ src_kv,     // [total_slots, H, D]
    const int32_t* __restrict__ keep_ids,   // [K] sorted indices to keep
          __half*  __restrict__ dst_kv,     // [K, H, D] compacted output
    int K, int H, int D)
{
    // One warp per (kept_slot, head) pair
    int slot_head = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int lane      = threadIdx.x % 32;
    if (slot_head >= K * H) return;

    int k = slot_head / H;
    int h = slot_head % H;
    int src_slot = keep_ids[k];

    // D elements per head, load 8 halfs (128 bits) per lane per iteration
    const float4* src_ptr = reinterpret_cast<const float4*>(
        src_kv + (src_slot * H + h) * D);
    float4* dst_ptr = reinterpret_cast<float4*>(
        dst_kv + (k * H + h) * D);

    // Each lane handles D/(32*8) float4 elements
    for (int i = lane; i < D / 8; i += 32)
        dst_ptr[i] = src_ptr[i];
}
```

### CUDA Primitives Used

- `float4` (128-bit) vectorized global memory loads/stores — the primary optimization
- `int4` for 4-index batch loading from the index array
- `__ldg()` for index array and read-only source loads (L1 texture cache)
- `__builtin_assume_aligned` hints to ensure 16-byte aligned addresses for float4
- Optional: `cp.async` for pipelining multiple gather batches

### H100 Hardware Features Exploited

- **128-byte cache line alignment**: float4 loads are exactly one cache line element; aligned access is critical for achieving peak BW
- **HBM3 bandwidth (3.35 TB/s)**: gather is purely BW-bound after coalescing; H100's BW advantage directly translates to speedup
- **L1 cache (192 KB)**: index arrays for typical eviction budgets (K ≤ 2048 tokens) fit in L1

### Expected Speedup

- **Near-peak HBM bandwidth** (>80% utilization) for aligned, coalesced access patterns
- **2x–3x** vs. naive non-vectorized gather (which achieves only 32-bit transactions)
- Compaction throughput: ~400 GB/s effective for typical KV head dimensions

### Benchmark Metrics

| Metric | Tool |
|--------|------|
| Effective bandwidth (GB/s) | `ncu --metrics l1tex__t_bytes` |
| L1 cache hit rate for index array | `ncu --metrics l1tex__t_sector_hit_rate` |
| Compaction latency vs. token count | profiler sweep |
| Alignment validation | `assert(ptr % 16 == 0)` |

### Implementation Complexity

**Low** — Straightforward vectorized memory copy with indirection. Main complexity is ensuring pointer alignment for float4 and handling non-multiples of 8 head dimensions.

---

## 12. `streaming_kv_ringbuffer`

### What It Does

Implements a **sliding-window KV cache ring buffer** with **attention sink tokens** (StreamingLLM-style). Rather than evicting all tokens outside a window, this kernel maintains the first `A` tokens (attention sinks, which receive disproportionate attention in LLMs) and the most recent `W` tokens in a circular buffer. The kernel handles the ring buffer pointer arithmetic and ensures attention is only computed over the valid window, enabling infinite-length generation without memory growth.

### Applicable Models / Architectures

Any causal autoregressive LLM: LLaMA 2/3, Mistral, Falcon, Qwen 2, GPT-NeoX, Phi-3, CodeLlama, WizardLM, any chat model used for long-context conversations, any model deployed in streaming inference (e.g., continuous batching servers like vLLM, TGI, TensorRT-LLM).

### Key Algorithm / Technique

The ring buffer maintains a circular array of `A + W` slots. On each new token:
1. The ring write pointer is incremented modulo `W` (in the sliding window portion)
2. The attention sink region (first `A` slots) is never overwritten
3. An attention mask is generated that marks valid positions for the current window

```cuda
// Ring buffer KV write kernel
__global__ void streaming_kv_write_kernel(
    const __half* __restrict__ new_k,    // [B, H, D] new key
    const __half* __restrict__ new_v,    // [B, H, D] new value
          __half* __restrict__ kv_ring,  // [B, A+W, H, D] ring buffer
          int*    __restrict__ ring_ptr, // [B] current write position in window
    int A, int W, int H, int D, int B)
{
    int b    = blockIdx.z;
    int head = blockIdx.y;
    int d    = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D || b >= B) return;

    // Write new key into window slot (ring buffer with offset A for sinks)
    int write_pos = A + (ring_ptr[b] % W);
    kv_ring[((b * (A + W) + write_pos) * H + head) * D + d] =
        new_k[(b * H + head) * D + d];
    // Similarly for new_v ...

    // Update ring pointer (only lane 0 of each (b, head) group)
    if (head == 0 && d == 0)
        atomicAdd(&ring_ptr[b], 1);
}
```

The complementary attention kernel reads valid positions using:
```cuda
// Map logical position to ring buffer physical position
__device__ int ring_pos(int logical_pos, int ring_ptr, int A, int W) {
    if (logical_pos < A) return logical_pos;  // sink region
    int window_logical = logical_pos - A;
    int window_physical = (ring_ptr - W + window_logical + W) % W;
    return A + window_physical;
}
```

### CUDA Primitives Used

- `atomicAdd` for ring pointer increment (single atomic per batch element per step)
- `float4` vectorized loads/stores for KV write (128-bit coalesced)
- `__shfl_sync` for broadcasting ring pointer to all threads in a warp
- `__ldg()` for read-only sink region access

### H100 Hardware Features Exploited

- **L2 cache persistence**: The sliding window portion is accessed in a predictable circular pattern; marking the ring buffer with persistent L2 policy keeps hot slots in L2
- **Atomic throughput**: H100's improved atomic unit handles the single `atomicAdd` per step with minimal contention
- **Memory coalescing**: Since D (head dimension) is typically 64–128 and float4-aligned, KV writes achieve near-peak BW

### Expected Speedup

- **Memory growth eliminated**: O(constant) vs. O(n) VRAM with generation length
- **1.5x–2.5x** decode throughput at generation lengths > 4096 tokens (vs. full KV cache that grows linearly and stresses bandwidth)
- StreamingLLM paper: maintains perplexity near full-cache baseline with A=4 sink tokens and W=1024 window

### Benchmark Metrics

| Metric | Tool |
|--------|------|
| VRAM at 10K, 50K, 100K generated tokens | `torch.cuda.memory_allocated()` |
| Perplexity at long context | PassageKey / StreamEval benchmarks |
| Ring buffer write latency (µs) | `nsys profile` |
| L2 hit rate for window region | `ncu --metrics lts__t_sectors_hit_rate` |
| Attention mask generation overhead | profiler |

### Implementation Complexity

**Low-Medium** — The ring buffer arithmetic is simple; the main complexity is correctly computing physical-to-logical position mappings in the attention kernel and handling the boundary between sink and window regions.

---

## 13. `pyramidkv_score_extraction`

### What It Does

A modified FlashAttention kernel that performs the standard attention computation AND **simultaneously accumulates per-token importance scores** into an auxiliary output buffer. During the tiled attention computation, each tile's attention weights are summed along the query dimension and accumulated into a `[B, H, T_k]` score tensor. This avoids a second pass over the attention weights to extract importance scores, enabling token eviction (SnapKV, H2O, PyramidKV) without any overhead beyond the attention computation itself.

### Applicable Models / Architectures

Any transformer where attention-based token importance scoring drives KV compression: GPT-family LLMs (for H2O, SnapKV, PyramidKV), diffusion transformers (for spatial token pruning), video transformers (for frame/patch eviction), multimodal models (for image token compression in VLMs like LLaVA, InternVL).

### Key Algorithm / Technique

The standard FlashAttention tiling accumulates output `O` via running rescaling. This kernel adds a **score accumulation step** inside the tile loop: after computing softmax(QK^T) for a tile, the column sums of the attention weight tile are atomically accumulated into the score buffer.

```cuda
// Inside the FlashAttention tile loop (simplified):
// After computing softmax tile P (shape [T_q_tile, T_k_tile]):

// Standard: accumulate O += P @ V_tile
// ADDED: accumulate scores[b, h, k] += sum_q(P[q, k]) for k in tile

// Score accumulation using warp-level reduction
__device__ void accumulate_scores_tile(
    const float* __restrict__ P_tile,  // [T_q_tile, T_k_tile] in registers
    float*       __restrict__ score_smem,  // [T_k_tile] in shared memory
    int T_q_tile, int T_k_tile, float lse_rescale)
{
    // Each thread owns P_tile[lane_row, lane_col] fragments
    // Sum along query dimension using warp transpose + reduce
    for (int k = 0; k < T_k_tile; k++) {
        float col_sum = 0.f;
        for (int q = 0; q < T_q_tile; q++)
            col_sum += P_tile[q * T_k_tile + k];
        // Warp reduce across query tiles
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1)
            col_sum += __shfl_xor_sync(0xffffffff, col_sum, mask);
        if (threadIdx.x % 32 == 0)
            atomicAdd(&score_smem[k], col_sum * lse_rescale);
    }
}
```

The scores must be rescaled by the log-sum-exp (LSE) normalization factor to account for FlashAttention's online softmax rescaling, ensuring they correspond to true attention probabilities.

### CUDA Primitives Used

- All FlashAttention-3 primitives (TMA, WGMMA, mbarrier) — this kernel extends FA3
- `atomicAdd` to shared memory score accumulation buffer
- `__shfl_xor_sync` for warp-level query-dimension reduction of score columns
- `__expf` for LSE rescaling factor computation
- `cp.async` for score buffer writeback from SMEM to HBM

### H100 Hardware Features Exploited

- **SRAM (228 KB)**: Score buffer for a KV tile (`T_k_tile × H` floats) fits alongside Q/K/V tiles
- All FA3 hardware features: TMA, WGMMA, warp specialization (score accumulation runs in consumer warps alongside O accumulation)
- **Atomic SMEM**: `atomicAdd` to shared memory is extremely fast (~1 cycle) on H100

### Expected Speedup

- **Score extraction for free**: <5% overhead over standard FA3 (score accumulation is negligible vs. GEMM cost)
- **Avoids second pass**: eliminates a separate score-extraction kernel that would add 20–40% overhead
- PyramidKV paper: 1.4x–2.0x speedup over SnapKV (which uses a separate scoring pass) at equivalent quality

### Benchmark Metrics

| Metric | Tool |
|--------|------|
| Score extraction overhead vs. standard FA3 | kernel time delta |
| Score accuracy vs. full softmax materialization | MSE on score tensors |
| End-to-end pipeline speedup | wall-clock |
| Memory overhead for score buffer | `sizeof(float) * B * H * T_k` |
| WGMMA utilization with score accumulation | `ncu --metrics sm__ops_path_tensor` |

### Implementation Complexity

**High** — Requires modifying FA3's inner loop to add score accumulation while preserving FA3's careful warp specialization and register management. The LSE rescaling is subtle and errors cause incorrect scores.

---

## 14. `fused_rope_embedding`

### What It Does

Fused **Rotary Positional Embedding (RoPE)** kernel supporting 1D (standard LLM), 2D axial (ViT/DiT image patches), and 3D spatio-temporal (video model patches). Rather than computing sin/cos tables separately and then applying them in a second kernel, this kernel precomputes the rotation matrices on-the-fly in registers and applies them to Q and K directly, avoiding all intermediate tensor materializations.

### Applicable Models / Architectures

Any transformer using RoPE: LLaMA 1/2/3, Mistral, Falcon, Qwen, GPT-NeoX, CodeLlama (1D); ViT with RoPE, DeiT-3, SAM2 (2D); CogVideoX, HunyuanVideo, Wan2.1 (3D T×H×W); multimodal models with mixed 1D text + 2D image RoPE (e.g., Qwen-VL, InternVL). Also supports ALiBi and other positional schemes via parameterization.

### Key Algorithm / Technique

RoPE rotates pairs of dimensions in the head vector by an angle `θ_i = pos / 10000^(2i/D)`. The rotation is:

```
[x_{2i}, x_{2i+1}] -> [x_{2i} cos(θ) - x_{2i+1} sin(θ),
                        x_{2i} sin(θ) + x_{2i+1} cos(θ)]
```

**Fast sin/cos via `__sincosf`**: CUDA's `__sincosf` computes both sin and cos simultaneously with a single hardware instruction (~4 cycles on H100 vs. ~20 cycles for separate `sinf`/`cosf`).

**3D RoPE for video**: The position index is decomposed as `pos = t * H * W + h * W + w`, and three separate frequency sets are used for temporal, height, and width dimensions.

```cuda
__global__ void fused_rope_3d_kernel(
          __half* __restrict__ q,    // [B, T_s, n_heads, D] — modified in-place
          __half* __restrict__ k,    // same layout
    const int*   __restrict__ pos_t, // [B, T_s] temporal positions
    const int*   __restrict__ pos_h, // [B, T_s] height positions
    const int*   __restrict__ pos_w, // [B, T_s] width positions
    int D, float theta_t, float theta_h, float theta_w)
{
    int b     = blockIdx.z;
    int token = blockIdx.y;
    int head  = blockIdx.x;
    int d2    = threadIdx.x;  // pairs: 0..D/2

    if (d2 >= D / 2) return;

    // Compute rotation angle for this dimension pair
    // D/3 dims each for T, H, W
    float angle = 0.f;
    if      (d2 < D/6)       angle = pos_t[b*T_s + token] / powf(theta_t, 6.f*d2/D);
    else if (d2 < D/3)       angle = pos_h[b*T_s + token] / powf(theta_h, 6.f*(d2-D/6)/D);
    else                     angle = pos_w[b*T_s + token] / powf(theta_w, 6.f*(d2-D/3)/D);

    float cos_a, sin_a;
    __sincosf(angle, &sin_a, &cos_a);

    // Apply rotation to Q
    int base = ((b * T_s + token) * n_heads + head) * D + d2 * 2;
    float q0 = __half2float(q[base]);
    float q1 = __half2float(q[base + 1]);
    q[base]     = __float2half(q0 * cos_a - q1 * sin_a);
    q[base + 1] = __float2half(q0 * sin_a + q1 * cos_a);
    // Apply same rotation to K ...
}
```

### CUDA Primitives Used

- `__sincosf(angle, &sin, &cos)` — single hardware instruction for simultaneous sin/cos (2x faster than separate calls)
- `__half2float` / `__float2half` — FP16 I/O with FP32 accumulation
- `__half2` arithmetic for paired rotation (can process 2 elements simultaneously)
- `__powf(base, exp)` — fast hardware power for frequency computation (or precomputed inv_freq table)
- `float4` loads for reading Q/K pairs when D ≥ 8

### H100 Hardware Features Exploited

- **Fast transcendental hardware**: H100 has a dedicated SFU (Special Function Unit) per SM for `sin`, `cos`, `exp`, `rcp` with 4-cycle throughput; `__sincosf` amortizes this to effectively 1 SFU call
- **FP32 accumulation in FP16 kernels**: mixed-precision math prevents rotation error accumulation, critical for long sequences (>32K tokens) with extended RoPE
- **Memory bandwidth**: kernel is bandwidth-bound; H100's 3.35 TB/s vs A100's 2.0 TB/s gives direct speedup

### Expected Speedup

- **2x–3x** vs. PyTorch RoPE (which uses sin/cos tables + two separate tensor operations)
- **1.3x–1.5x** vs. pre-built tables + vectorized apply (Flash-Decoding approach)
- 3D RoPE: ~1.8x vs. three sequential 1D RoPE applications

### Benchmark Metrics

| Metric | Tool |
|--------|------|
| Kernel time vs. PyTorch reference | `nsys profile` |
| SFU utilization | `ncu --metrics smsp__inst_executed_pipe_xu` |
| Numerical accuracy at 1M token positions | MSE vs. double-precision reference |
| Memory bandwidth utilization | `ncu --metrics l1tex__t_bytes` |
| 3D vs. 1D RoPE overhead | kernel time ratio |

### Implementation Complexity

**Low-Medium** — 1D RoPE is straightforward; 3D requires careful dimension assignment and position index decoding. The main pitfall is numerical accuracy at very large position values (requires careful frequency scaling for extended RoPE).

---

## 15. `snapkv_3d_pooling`

### What It Does

Extends SnapKV's 1D importance pooling (which computes a max-pool over consecutive token scores) to a **3D attention-weighted average pooling** across the T×H×W (temporal × height × width) structure of video token sequences. This identifies spatially and temporally clustered regions of high attention importance, enabling structured KV eviction that respects video's spatial locality rather than treating all token positions as independent.

### Applicable Models / Architectures

Any video transformer or 3D spatial transformer: CogVideoX (3D causal attention over T×H×W patches), HunyuanVideo, Wan2.1, ViViT, VideoMAE, TimeSFormer, Video-DINO, any model processing 3D feature maps with spatial attention, 3D medical image transformers.

### Key Algorithm / Technique

Given importance scores `s[t, h, w]` for each video patch position (T×H×W total), the kernel applies a **3D max-pool with configurable kernel sizes `(kt, kh, kw)`** to compute neighborhood-smoothed scores, then selects the top-K regions for retention. This is more effective than 1D pooling because video has strong spatial locality — if one patch is important, its neighbors likely are too.

```cuda
__global__ void snapkv_3d_pool_kernel(
    const float* __restrict__ scores_in,   // [B, H_attn, T*sH*sW]
    float*       __restrict__ scores_out,  // [B, H_attn, T*sH*sW] smoothed
    int T, int sH, int sW,                 // spatial dimensions
    int kt, int kh, int kw)                // pooling kernel sizes
{
    int b    = blockIdx.z;
    int head = blockIdx.y;
    int idx  = blockIdx.x * blockDim.x + threadIdx.x;  // T*sH*sW linear index

    if (idx >= T * sH * sW) return;

    int t = idx / (sH * sW);
    int h = (idx / sW) % sH;
    int w = idx % sW;

    // 3D max pooling over neighborhood
    float max_val = 0.f;
    int t0 = max(0, t - kt/2), t1 = min(T-1,  t + kt/2);
    int h0 = max(0, h - kh/2), h1 = min(sH-1, h + kh/2);
    int w0 = max(0, w - kw/2), w1 = min(sW-1, w + kw/2);

    for (int tt = t0; tt <= t1; tt++)
        for (int hh = h0; hh <= h1; hh++)
            for (int ww = w0; ww <= w1; ww++) {
                int src = (b * n_heads + head) * T*sH*sW + tt*sH*sW + hh*sW + ww;
                max_val = fmaxf(max_val, scores_in[src]);
            }

    scores_out[(b * n_heads + head) * T*sH*sW + idx] = max_val;
}
```

For the subsequent top-K selection, the warp-level `warp_topk_selection` kernel (Section 10) is used directly.

### CUDA Primitives Used

- `__shfl_xor_sync` for warp-level max reduction over the 3D neighborhood (if pooling window fits in a warp)
- `float4` for vectorized loading of score rows
- `__ldg()` for read-only score input access
- `fmaxf` intrinsic for fast floating-point max
- Shared memory tiling for neighborhood reuse when `kt*kh*kw > 8`

### H100 Hardware Features Exploited

- **L1 spatial locality**: 3D pooling accesses memory in a structured neighborhood pattern; with `sH×sW` mapped to row-major, spatial neighbors have good cache locality
- **High SM count**: T×H×W can be tens of thousands of tokens (e.g., 4s video at 8fps, 16×16 patches = 4×32×32 = 4096 tokens); H100's 132 SMs process all positions in parallel
- **SRAM tiling**: For typical video sizes (T=16, sH=32, sW=32), pooling tiles fit in SRAM

### Expected Speedup

- 3D pooling kernel itself: <1% overhead on total attention time
- **KV compression**: 50%–80% KV cache reduction at equivalent quality vs. 1D SnapKV (3D pooling produces better importance maps)
- **End-to-end decode speedup**: 1.4x–2.2x for video generation at 50% KV budget

### Benchmark Metrics

| Metric | Tool |
|--------|------|
| Pooling kernel time (µs) | `nsys profile` |
| FID / PSNR quality vs. full KV | video quality evaluation |
| KV budget at iso-quality | quality vs. compression curve |
| 3D vs. 1D pooling quality difference | ablation study |
| L1 cache hit rate for neighborhood accesses | `ncu --metrics l1tex__t_sector_hit_rate` |

### Implementation Complexity

**Low-Medium** — The 3D pooling loop is simple; the main complexity is handling arbitrary spatial layouts (non-square, non-power-of-2 T/H/W) and integrating with the video model's patch position encoding.

---

## 16. `cuda_graph_pipeline`

### What It Does

Captures an entire multi-step inference pipeline (e.g., noise scheduler step + DiT forward pass + VAE decode) as a **CUDA Graph**, then executes it with **triple-buffered CUDA Streams** to pipeline CPU scheduling, memory transfers, and GPU compute. CUDA Graph eliminates the per-kernel CPU launch overhead (typically 5–20 µs per kernel) and enables the driver to optimize the entire DAG of kernel launches holistically. Stream triple-buffering overlaps computation on step N with memory prep for step N+1.

### Applicable Models / Architectures

Any iterative inference pipeline: diffusion model denoising loops (SD, FLUX, CogVideoX — typically 20–50 steps), speculative decoding (draft + verify loop), multi-turn LLM chat (session-level graph), video generation pipelines with multiple stages (VAE + transformer + scheduler), any pipeline where the same sequence of operations repeats many times.

### Key Algorithm / Technique

**CUDA Graph capture**:
```cuda
// 1. Warm-up run to stabilize allocations
model.forward(dummy_input);

// 2. Begin graph capture on a dedicated capture stream
cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal);

// 3. Run one complete forward pass — all kernels are recorded, not executed
model.forward(graph_input_buffer);

// 4. End capture: build the graph
cudaGraph_t graph;
cudaStreamEndCapture(capture_stream, &graph);

// 5. Instantiate executable graph
cudaGraphExec_t graph_exec;
cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);

// Inference loop:
for (int step = 0; step < num_steps; step++) {
    // Update graph input buffer (noise/conditioning) in place
    update_graph_inputs(graph_input_buffer, step, stream[step % 3]);
    // Launch graph — near-zero CPU overhead
    cudaGraphLaunch(graph_exec, stream[step % 3]);
}
```

**Triple-buffer stream pipeline**:
```
Stream 0: [prep step 0] [compute step 0]
Stream 1:     [prep step 1] [compute step 1]
Stream 2:         [prep step 2] [compute step 2]
```
cudaEvents synchronize stream dependencies (compute step N cannot start until prep N finishes).

### CUDA Primitives Used

- `cudaStreamBeginCapture` / `cudaStreamEndCapture` — CUDA Graph capture API
- `cudaGraphInstantiate` / `cudaGraphLaunch` — Graph execution
- `cudaGraphExecUpdate` — Update graph parameters without re-capture (e.g., update conditioning input)
- `cudaEventRecord` / `cudaStreamWaitEvent` — Cross-stream synchronization for pipeline stages
- `cudaMemcpyAsync` with streams for pipelined H2D/D2D transfers

### H100 Hardware Features Exploited

- **Persistent CUDA Graph execution**: H100's command processor is optimized for Graph launches; kernel scheduling overhead is ~1–2 µs (vs. 5–20 µs per kernel on CPU-scheduled launches)
- **Hardware-level stream concurrency**: H100's multiple hardware engines (copy engine, compute engine) enable true overlap of memcpy and compute in different streams
- **NVLink for multi-GPU pipelines**: Cross-GPU tensor parallelism within a CUDA Graph is supported via NVLink peer-to-peer in H100 NVL configurations

### Expected Speedup

- **10%–30%** end-to-end latency reduction for 20-step diffusion loops (kernel launch overhead eliminated for ~500–1000 kernel launches)
- **5%–15%** for single-step LLM decode (fewer repetitive launches)
- More pronounced for small-batch/low-latency scenarios where CPU overhead is relatively large
- Reference: NVIDIA reports 20–30% speedup for ResNet inference with CUDA Graphs

### Benchmark Metrics

| Metric | Tool |
|--------|------|
| CPU-side scheduling overhead (µs) | `nsys` CPU timeline |
| GPU idle time between kernels | `nsys` GPU timeline gaps |
| End-to-end latency per denoising loop | wall-clock |
| Stream overlap efficiency | `nsys` concurrent kernel visualization |
| Graph update overhead (re-capture vs. update) | profiler |

### Implementation Complexity

**Medium** — CUDA Graph capture is straightforward for static graphs; complexity arises from handling dynamic shapes (variable sequence lengths), conditional operations (scheduler branching), and correctly updating graph inputs without re-capture via `cudaGraphExecUpdate`.

---

## 17. `int2_kv_pack_unpack`

### What It Does

A pair of CUDA kernels — `int2_kv_pack` and `int2_kv_unpack` — that compress KV cache entries from FP16 to **INT2** (2 bits per value, 4 values packed per byte) using the PTX `lop3.b32` instruction for fast nibble manipulation. INT2 provides 8x compression over FP16, enabling extremely long contexts or very large batch sizes within a fixed VRAM budget. Per-group scaling (group size G=16 for INT2) preserves the critical scale information.

### Applicable Models / Architectures

Any autoregressive LLM or DiT requiring maximum KV cache compression for ultra-long contexts or high batch sizes: LLaMA 3 at 128K context length, GPT-4 class models, any model served at high concurrency where KV cache is the VRAM bottleneck. Most applicable to the oldest tier in a tiered KV cache scheme.

### Key Algorithm / Technique

INT2 quantization uses 4 discrete levels {-1.5, -0.5, +0.5, +1.5} scaled by a per-group factor:
```
levels = [-1.5, -0.5, +0.5, +1.5]  ->  encoded as {00, 01, 10, 11}
q_int2 = argmin_i |x/scale - levels[i]|
```

**`lop3.b32` for fast packing**: The PTX `lop3` instruction computes any 3-input bitwise logical function in a single instruction. It is used to assemble 4 INT2 values (each occupying bits of an intermediate register) into a single byte without branching:

```cuda
// Pack four 2-bit values into one byte using lop3.b32
__device__ __forceinline__ uint8_t pack_int2x4(uint8_t b0, uint8_t b1, uint8_t b2, uint8_t b3) {
    // b0, b1, b2, b3 are each in range [0, 3] (2 bits)
    uint32_t result;
    // Using inline PTX lop3 for bit assembly:
    // result = (b0 & 0x03) | ((b1 & 0x03) << 2) | ((b2 & 0x03) << 4) | ((b3 & 0x03) << 6)
    asm volatile(
        "lop3.b32 %0, %1, %2, %3, 0xFE;"  // LUT for OR
        : "=r"(result)
        : "r"((uint32_t)b0 | ((uint32_t)b1 << 2)),
          "r"(((uint32_t)b2 << 4) | ((uint32_t)b3 << 6)),
          "r"(0)
    );
    return (uint8_t)(result & 0xFF);
}

// Unpack: extract 2-bit field i from packed byte
__device__ __forceinline__ int8_t unpack_int2(uint8_t packed, int i) {
    uint32_t shifted;
    asm volatile(
        "lop3.b32 %0, %1, 0x00000003, 0x00000000, 0xCA;"  // AND LUT
        : "=r"(shifted)
        : "r"((uint32_t)packed >> (i * 2))
    );
    return (int8_t)shifted;  // in [0, 3]
}
```

Scale computation uses the same warp-level absmax as the INT4 kernel but with group size G=16.

### CUDA Primitives Used

- `lop3.b32` PTX instruction with configurable 8-bit LUT for arbitrary 3-input boolean functions — the key primitive for INT2 packing
- `__shfl_xor_sync(0xffffffff, ...)` for warp-level absmax over groups of 16
- `prmt.b32` PTX instruction for byte permutation during unpack
- `bfe.u32` (bit field extract) PTX for unpacking individual INT2 fields
- `uint4` for vectorized 128-bit writes of packed INT2 data

### H100 Hardware Features Exploited

- **INT4 tensor core capability**: After unpacking INT2 → INT4 (with zero-padding of high 2 bits), H100's INT4 tensor cores can be used for matmul without full dequantization
- **L2 cache**: INT2 compressed KV entries are 8x smaller than FP16; more of the KV cache fits in H100's 50 MB L2, dramatically improving cache hit rates
- **Memory compression**: H100's HBM3 has hardware delta-compression; INT2 data (more regular bit patterns) compresses additionally at the hardware level

### Expected Speedup

- **8x KV memory reduction** (FP16 → INT2)
- **2x–4x decode throughput** at very long contexts (>32K tokens) where KV bandwidth is the bottleneck
- Quality: ~3–5% accuracy degradation at INT2 (acceptable only for oldest cache tier in tiered scheme)
- Reference: KIVI (Liu et al., 2024) reports 2-bit KV cache at acceptable quality for LLaMA-2 70B

### Benchmark Metrics

| Metric | Tool |
|--------|------|
| Pack/unpack throughput (GB/s) | kernel microbenchmark |
| KV memory at 128K context | `torch.cuda.memory_allocated()` |
| Perplexity at INT2 KV | WikiText-103 benchmark |
| lop3 instruction throughput | `ncu --metrics smsp__inst_executed` |
| Decode latency at various compression levels | throughput sweep |

### Implementation Complexity

**High** — Correct `lop3` LUT computation, group-size-16 scale management, and accurate dequantization on the unpack path require significant bit-manipulation expertise. The interaction with downstream tensor core GEMMs (INT2 → INT4 expansion) adds additional complexity.

---

## 18. `wgmma_int4_gemm`

### What It Does

Implements a **weight-only INT4 GEMM** for linear layers using H100's INT4 tensor cores via the `mma.m16n8k64.s4` PTX instruction (upgraded to WGMMA via `wgmma.mma_async.sync.aligned.m64n128k64.s32.s4.s4`). Weights are stored in INT4 (packed 2-per-byte); activations remain in FP16. Before the GEMM, INT4 weights are loaded into registers and the GEMM produces INT32 accumulators, which are immediately rescaled (dequantized) to FP16/FP32 outputs using per-group scale factors. This is the fastest known path for serving quantized LLM linear layers on H100.

### Applicable Models / Architectures

**Any linear layer in any neural network**: LLM attention projections (Q, K, V, O), LLM FFN layers (gate, up, down projections), DiT MLP layers, LoRA base weight GEMMs, embedding projections, any `nn.Linear` module. Specifically enables 4-bit weight-only quantization (GPTQ, AWQ, SqueezeLLM, QMoE) at maximum speed.

### Key Algorithm / Technique

**INT4 tensor core path on H100**:

H100 supports `mma.m16n8k64.row.col.s32.s4.s4.s32` which computes a 16×8 output tile from a 16×64 INT4 A tile and 8×64 INT4 B tile, producing a 16×8 INT32 accumulator. For weight-only quant, weights W are INT4 and activations X are quantized on-the-fly.

The WGMMA version scales this up:
```
wgmma.mma_async.sync.aligned.m64n128k64.s32.s4.s4
```
This computes a 64×128 output tile from 64×64 INT4 A and 128×64 INT4 B tiles using a full warpgroup (128 threads), giving 4× the tile area of single-warp MMA.

```cuda
// Epilogue: INT32 accumulator -> FP16 output with per-group dequantization
__device__ void dequant_int4_epilogue(
    int32_t* __restrict__ acc_int32,  // [M_tile, N_tile] INT32 accumulators
    __half*  __restrict__ output,     // FP16 output
    const __half* __restrict__ scales,  // per-group scales [N/G]
    const __half* __restrict__ zeros,   // per-group zero points
    int M_tile, int N_tile, int N, int G)
{
    for (int m = 0; m < M_tile; m++) {
        for (int n = threadIdx.x; n < N_tile; n += blockDim.x) {
            int global_n = blockIdx.y * N_tile + n;
            float scale = __half2float(scales[global_n / G]);
            float zero  = __half2float(zeros[global_n / G]);
            float val = (float)acc_int32[m * N_tile + n] * scale + zero;
            output[m * N + global_n] = __float2half(val);
        }
    }
}
```

**CUTLASS 3.x provides reference**: The `cutlass::gemm::kernel::GemmUniversal` with `Int4` element types is the reference implementation; custom kernels extend this with fused dequantization and activation functions.

### CUDA Primitives Used

- `mma.m16n8k64.row.col.s32.s4.s4.s32` — H100/A100 INT4 tensor core MMA instruction
- `wgmma.mma_async.sync.aligned.m64n128k64.s32.s4.s4` — H100 WGMMA INT4 (128-thread tile)
- `cp.async.bulk.tensor` (TMA) — load INT4 weight tiles from HBM with 2D strided access
- `mbarrier` — pipeline stage synchronization between TMA producer and WGMMA consumer
- `lop3.b32` — INT4 weight extraction and packing in the data path
- `__ldmatrix.x4` — efficient 4-matrix load from shared memory into registers for MMA

### H100 Hardware Features Exploited

- **INT4 tensor cores**: H100 doubles INT8 tensor core throughput for INT4 operations — theoretical peak ~1958 TOPS INT4 (vs. 979 TOPS INT8 and 494 TFLOPS FP16)
- **WGMMA with TMA**: The killer combination — TMA handles all address calculation and async copies for the weight tiles while WGMMA fills the tensor cores at peak utilization
- **Large register file**: WGMMA allocates 512 bytes of accumulator registers per warpgroup; H100's 256 KB register file per SM allows 8 concurrent WGMMA operations
- **NVLink for tensor parallelism**: INT4 weights reduce inter-GPU communication 4x for tensor-parallel deployments

### Expected Speedup

- **2x–4x GEMM TFLOPS** vs. FP16 tensor core GEMM (INT4 has 2x higher tensor core throughput + 2x memory reduction)
- **3x–5x end-to-end linear layer speedup** vs. FP16 for memory-bound decoding (batch=1–4)
- **1.5x–2.5x** for compute-bound prefill (batch≥32)
- Reference: GPTQ paper: 3.25x LLaMA-65B throughput at INT4; ExLlama: 4.2x vs. HuggingFace FP16 on single A100

### Benchmark Metrics

| Metric | Tool |
|--------|------|
| INT4 tensor core TFLOPS | `ncu --metrics sm__ops_path_tensor_src_int8` (H100 INT4 uses INT8 pipeline) |
| TMA bandwidth utilization | `ncu --metrics l1tex__data_pipe_lsu_wavefronts_mem_shared_op_atom` |
| WGMMA warpgroup utilization | `ncu --metrics sm__inst_executed_pipe_tensor_op_imma` |
| GEMM efficiency (% of peak TOPS) | `actual_TOPS / 1958_peak_TOPS` |
| Perplexity vs. FP16 baseline | WikiText-103 |
| Linear layer latency sweep (M, N, K) | microbenchmark grid |

### Implementation Complexity

**High** — Requires Hopper-specific PTX knowledge (WGMMA, TMA with 2D tensor descriptors), careful register allocation (WGMMA accumulators must be statically sized at compile time), and correct INT4 packing/unpacking in the hot path. The fused dequantization epilogue must handle per-group scaling without introducing accumulation errors. CUTLASS 3.x provides the scaffolding but customization requires deep expertise.

---

## Summary Table

| # | Kernel | Category | H100-Specific | Speedup | Complexity |
|---|--------|----------|---------------|---------|------------|
| 1 | `fused_adaln` | Normalization | Partial (BW) | 1.8x–2.5x | Medium |
| 2 | `int4_fp8_attention` | Attention | Yes (FP8 TC) | 1.8x–3.1x | High |
| 3 | `fa3_warp_specialized_attention` | Attention | Yes (TMA+WGMMA) | 1.5x–2.6x | High |
| 4 | `tiered_kv_quantization` | KV Cache | Yes (FP8 native) | 2x–4x memory | Medium |
| 5 | `h2o_token_eviction` | KV Cache | Partial | 1.5x–2.0x | Medium |
| 6 | `tome_token_merge` | Token Pruning | Partial | 1.4x–2.0x | Medium |
| 7 | `per_group_int4_quantize` | Quantization | No (portable) | 4x memory | Low |
| 8 | `per_group_fp8_quantize` | Quantization | Yes (FP8 native) | 2x memory | Low |
| 9 | `fused_conv_norm_act` | Conv/Norm | Partial (TMA) | 1.5x–2.2x | High |
| 10 | `warp_topk_selection` | Utility | No (portable) | 3x–8x vs sort | Low |
| 11 | `indexed_gather_scatter` | Utility | No (portable) | ~peak BW | Low |
| 12 | `streaming_kv_ringbuffer` | KV Cache | Partial | 1.5x–2.5x | Low-Med |
| 13 | `pyramidkv_score_extraction` | KV Cache | Yes (FA3) | <5% overhead | High |
| 14 | `fused_rope_embedding` | Embedding | Partial (BW) | 2x–3x | Low-Med |
| 15 | `snapkv_3d_pooling` | KV Cache | Partial | 1.4x–2.2x | Low-Med |
| 16 | `cuda_graph_pipeline` | Pipeline | Yes (Graph) | 10%–30% | Medium |
| 17 | `int2_kv_pack_unpack` | KV Cache | Partial (lop3) | 8x memory | High |
| 18 | `wgmma_int4_gemm` | GEMM | Yes (WGMMA+TMA) | 2x–4x | High |

---

## Key References

- FlashAttention-3: Tri Dao & Jay Shah, "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision," arXiv:2407.08608 (2024)
- SageAttention2: Liu et al., "SageAttention2: Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization," arXiv:2411.10958 (2024)
- H2O: Zhang et al., "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models," NeurIPS 2023
- ToMe: Bolya et al., "Token Merging: Your ViT But Faster," ICLR 2023
- StreamingLLM: Xiao et al., "Efficient Streaming Language Models with Attention Sinks," ICLR 2024
- PyramidKV: Cai et al., "PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling," arXiv:2406.02069 (2024)
- SnapKV: Li et al., "SnapKV: LLM Knows What You are Looking for Before Generation," arXiv:2404.14469 (2024)
- KIVI: Liu et al., "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache," arXiv:2402.02750 (2024)
- GPTQ: Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers," ICLR 2023
- KVQuant: Hooper et al., "KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization," NeurIPS 2024
- NVIDIA H100 Architecture Whitepaper, NVIDIA 2022
- CUTLASS 3.x Documentation and Examples, NVIDIA GitHub
