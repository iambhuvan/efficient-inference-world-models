# WorldServe Custom CUDA Kernels

**CMU 15-849 ML Systems — Spring 2026**  
**Hardware:** NVIDIA H100 80GB SXM (sm_90a, HBM3 3.35 TB/s, 228 KB SMEM/SM, 50 MB L2)

This document catalogs every custom CUDA kernel designed for this project. Split into two sections:

- **Part 1 — Universal Kernels**: Architecture-agnostic. Work on any transformer, DiT, LLM, or video model. Primary deliverables for benchmarking.
- **Part 2 — Model-Specific Kernels**: Designed for Open-Oasis 500M and Matrix-Game 2.0 architectural properties. Many partially generalizable.

---

## Part 1: Universal Kernels

> These kernels are the core systems contribution. They can be benchmarked on any model (LLaMA, FLUX, CogVideoX, SD3, HunyuanVideo, etc.) and compared against PyTorch/FlashAttention baselines.

---

### K1 — `fused_adaln`

**What it does:** Collapses `LayerNorm → scale → shift → gate` (adaLN-zero) into one kernel, eliminating 2 HBM round-trips per DiT block.

**Applies to:** Any DiT using adaptive LayerNorm: SD3, FLUX.1, CogVideoX, HunyuanVideo, Wan2.1, PixArt, Latte, OpenSora, DiT-XL, SiT, Movie Gen. Also any FiLM-conditioned LLM.

**Algorithm:**
```
y = alpha * ((x - mean) / sqrt(var + eps)) * (1 + gamma) + beta
```
Two-pass Welford variance in shared memory. Conditioning vectors `gamma/beta/alpha` loaded once into registers and broadcast over the T (token) dimension.

**CUDA primitives:** `__shfl_xor_sync` for warp Welford reduction, `__half2` arithmetic, `float4` 128-bit loads, `__ldg` for conditioning tensors (texture cache).

**H100 features:** L2 persistence for small conditioning tensors (`cudaAccessPropertyPersisting`), HBM3 bandwidth (BW-bound kernel, 1.67× raw advantage over A100).

**Speedup:** 1.8×–2.5× vs. unfused PyTorch. 1.3×–1.6× vs. Apex FusedLayerNorm.

**Benchmark metrics:** Kernel time (µs), HBM utilization (`ncu l1tex__t_bytes`), L2 hit rate for conditioning, end-to-end DiT block latency.

**Complexity:** Medium

---

### K2 — `int4_fp8_attention`

**What it does:** SageAttention2-style quantized attention. INT4 per-warp quantization of Q and K before QK^T GEMM; FP8 E4M3 quantization of softmax output P before P·V GEMM. Both quantization steps are fused into GEMM prologues — no extra kernel launches.

**Applies to:** Any MHA/GQA/MQA transformer: LLaMA 3, Mistral, Qwen2, Gemma 2, GPT-4 style LLMs, FLUX MM-DiT, CogVideoX 3D attention, HunyuanVideo, ViT/DeiT, SDXL cross-attention.

**Algorithm:**
```
# INT4 QK path
scale_q = warp_absmax(q_fp16) / 7.0
q_int4  = clamp(round(q_fp16 / scale_q), -8, 7)   # packed 2-per-byte
S_int32 = mma.m16n8k64.s4(q_int4, k_int4)         # INT4 tensor cores
S_fp16  = S_int32 * scale_q * scale_k              # rescale

# FP8 PV path
P_fp8   = cast_fp8_e4m3(softmax(S_fp16 / sqrt(D)))
O_fp32  = mma.m16n8k32.e4m3.e4m3(P_fp8, v_fp8)   # FP8 tensor cores
```
Outlier smoothing: subtract channel-wise mean from K before quantization; add rank-1 correction term after GEMM.

**CUDA primitives:** `mma.m16n8k64.row.col.s32.s4.s4.s32` (INT4), `mma.m16n8k32.e4m3.e4m3.f32` (FP8), `__shfl_xor_sync` absmax, `lop3.b32` for INT4 packing, `cp.async.cg` for tile loading.

**H100 features:** INT4 tensor cores (7,916 TOPS), FP8 tensor cores (3,958 TFLOPS), 228 KB SMEM for larger tiles vs A100 (164 KB).

**Speedup:** 1.8×–3.1× vs. FlashAttention-2 FP16. 1.4×–2.0× vs. FA3 FP16. Quality: <0.01% output MSE.

**Benchmark metrics:** Attention TFLOPS (`ncu sm__ops_path_tensor_src_fp8`), output MSE vs FP16, tokens/second, quantization error histogram.

**Complexity:** High

---

### K3 — `fa3_warp_specialized_attention`

**What it does:** FlashAttention-3 style warp-specialized attention kernel. Producer warps issue async TMA copies; consumer warpgroups run WGMMA asynchronously; softmax computation overlaps with WGMMA on the next tile. Three-stage pipeline eliminates the compute/memory gap that FA2 leaves at ~35% H100 utilization.

**Applies to:** Any transformer attention. Universal drop-in for FA2 on H100.

**Algorithm:**
```
# 3-stage pipeline per attention block:
# Stage 0: TMA producer warp loads K[j+1], V[j+1] tile async
# Stage 1: Consumer warpgroup fires WGMMA(Q_i, K[j]^T) → S[i,j] async
# Stage 2: Softmax rescale of S[i,j-1], update O accumulator
# All 3 stages overlap via warpgroup-level mbarrier synchronization
```
Uses `setmaxnreg` to give consumer warpgroups more registers than producers (producer only issues TMA, needs few regs). FP8 mode adds incoherent processing (Hadamard transform before quantization) to reduce outlier clipping.

**CUDA primitives:** `wgmma.mma_async.sync.aligned` (WGMMA), `cp.async.bulk.tensor` (TMA), `mbarrier` cluster barriers, `setmaxnreg` dynamic register allocation.

**H100 features:** WGMMA (vs A100's synchronous MMA), TMA (vs manual cp.async), 228 KB SMEM. Achieves 740 TFLOPS FP16 (75% H100 peak) vs FA2's 335 TFLOPS (35%).

**Speedup:** 2.0×–2.2× vs. FA2 FP16. 3.0×–3.6× in FP8 mode.

**Benchmark metrics:** Attention kernel TFLOPS, memory bandwidth, forward/backward latency at varied seqlens (1K, 4K, 16K, 64K), FP8 vs FP16 quality regression.

**Complexity:** Very High (reference: FlashAttention-3 open-source)

---

### K4 — `tiered_kv_quantization`

**What it does:** Quantizes KV cache entries in-place based on token age using a tiered precision scheme: recent tokens stay FP16, older tokens get quantized to FP8, oldest get INT4. Single-pass kernel that decides per-entry dtype from a frame-age metadata array.

**Applies to:** Any autoregressive model with a growing KV cache: LLaMA, GPT, Mistral, Qwen, Phi, Falcon, BLOOM, video generation models, multimodal LLMs.

**Algorithm:**
```
# Dispatch based on frame/token age:
age = current_step - token_birth_step
if age < THRESH_FP8:   keep as FP16
elif age < THRESH_INT4: quantize to FP8 E4M3 (per-block scale, block=32)
else:                   quantize to INT4 (pack 2 per byte, per-group scale)

# Per-group scale: warp reduction over group_size=32 elements
scale = warp_absmax(vals) / MAX_VAL[dtype]
```
Separate physical buffers per tier (avoids warp divergence). Promotion kernel runs when frame crosses threshold — reads FP16, writes FP8 or INT4 in one pass.

**CUDA primitives:** `__shfl_xor_sync` for per-group absmax, `__nv_fp8_e4m3` (CUDA 12.1+ intrinsic), `lop3.b32` for INT4 packing, `float4` vectorized loads (8 FP16 per thread).

**H100 features:** `__nv_fp8_e4m3` / `__nv_fp8x4_e4m3` native FP8 types, `__nv_cvt_float_to_fp8` intrinsic, streaming stores `__stcs` for write-once KV entries (bypasses L2 pollution).

**Speedup:** 2×–4× KV memory reduction (FP8 tier), 4×–8× (INT4 tier). Latency reduction scales with memory bandwidth saved: 1.5×–3.5× attention for compressed KV.

**Benchmark metrics:** GB/s throughput vs PyTorch baseline, memory reduction factor (×), attention latency with compressed KV at 1K/5K/25K token cache sizes.

**Complexity:** Medium

---

### K5 — `h2o_token_eviction`

**What it does:** Heavy-Hitter Oracle token eviction. Maintains a persistent score buffer across decoding steps; at each step accumulates the just-computed attention weights into per-token scores; when cache is full, evicts the token with lowest accumulated score (outside the recency window).

**Applies to:** Any causal transformer decoder: all LLMs, autoregressive video models, any model with a KV cache that grows beyond GPU memory.

**Algorithm:**
```
# Per decoding step:
score[i] += attn_weight[i]   # accumulate for all cached tokens i

# When cache full (at capacity C):
evict_idx = argmin(score[C_recent:])  # ignore recency window
free_slot(evict_idx)                  # logical eviction via index map
```
Score accumulation: one warp per attention head, sequential over cached tokens. Argmin: warp-level `__shfl_xor_sync` minimum reduction.

**CUDA primitives:** `__shfl_xor_sync` for warp argmin, `atomicAdd` for score accumulation (multi-head), ring buffer index management with `atomicCAS` for slot recycling.

**H100 features:** 50 MB L2 — score buffer for typical cache sizes (4K tokens × 32 heads × 4B = 512 KB) fits entirely in L2.

**Speedup:** 5×–20× KV memory reduction. 1.9× latency at equal batch. Quality: ~20% cache retention matches full-cache quality on most benchmarks.

**Benchmark metrics:** Cache hit quality (perplexity/FVD), eviction overhead (ms per step), memory reduction factor, attention latency at compressed cache sizes.

**Complexity:** Medium

---

### K6 — `tome_token_merge`

**What it does:** Token Merging — bipartite soft matching that identifies the `r` most similar token pairs per layer and averages them, reducing sequence length without discarding information (unlike eviction). Uses Key vectors as similarity descriptors.

**Applies to:** Any ViT/DiT (image or video): SD3, FLUX, CogVideoX, HunyuanVideo, DALL-E 3-style models, ViT-L/16, DeiT, OpenCLIP, SigLIP, any DiT for image/video generation.

**Algorithm:**
```
# Partition tokens into two sets A (even) and B (odd), size N/2 each
# Similarity: use Key vectors as descriptors (already computed)
sim[i,j] = dot(K_A[i], K_B[j]) / (|K_A[i]| * |K_B[j]|)  # [N/2 x N/2]

# Greedy matching (O(N)): for each a in A, find most similar b in B
# Keep top-r pairs by similarity score
# Merge: merged_token = (K_A[i]*s_i + K_B[j]*s_j) / (s_i + s_j)  # size-weighted
```
Unmerge at output (after MLP) via indexed copy.

**CUDA primitives:** `cublasSgemmBatched` or custom WGMMA tile for N/2×N/2 similarity, `__shfl_xor_sync` for per-row argmax, `atomicAdd` for scatter-add during merge.

**H100 features:** FP8 for similarity GEMM (only needs ranking precision, not exact values), large SMEM for similarity matrix (e.g., 144×144 fp32 = 83 KB fits in 228 KB SMEM for Oasis).

**Speedup:** 1.5×–2.2× throughput at r=25% of tokens. Near-zero quality drop for vision tasks.

**Benchmark metrics:** Tokens merged per layer (%), quality regression (FID/FVD/PSNR), kernel time vs naive, end-to-end latency with varying r.

**Complexity:** Medium

---

### K7 — `per_group_int4_quantize`

**What it does:** General-purpose INT4 quantization kernel. Reads FP16/BF16, computes per-group absmax scale via warp reduction, quantizes to INT4 and packs 2 values per byte. Inverse kernel dequantizes.

**Applies to:** Any weight or KV cache tensor requiring INT4 compression. Used by K4, K2, AWQ, GPTQ-style weight quantization on any LLM or DiT.

**Algorithm:**
```cuda
// Group size G = 32 or 128 elements
// One warp per group
float scale = warp_absmax(vals) / 7.0f;
scales[group_id] = __float2half(scale);
float inv_scale = 1.0f / scale;

// Pack 2 INT4 per byte
for (int i = tid*2; i < G; i += 64) {
    int8_t q0 = clamp(round(vals[i]   * inv_scale), -8, 7);
    int8_t q1 = clamp(round(vals[i+1] * inv_scale), -8, 7);
    output[(offset+i)/2] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
}
```

**CUDA primitives:** `__shfl_xor_sync` absmax tree reduction, `lop3.b32` for nibble packing, `float4` 128-bit loads.

**H100 features:** No H100-specific features required (runs on A100/RTX 40 series too). On H100: benefits from HBM3 BW for large weight matrices.

**Speedup:** Compression: 4× memory reduction vs FP16. Throughput: 2×–4× over naive PyTorch `torch.int_repr()`.

**Benchmark metrics:** Quantization throughput (GB/s), quantization error (MSE vs FP16), packing/unpacking round-trip accuracy.

**Complexity:** Low

---

### K8 — `per_group_fp8_quantize`

**What it does:** General-purpose FP8 E4M3 quantization kernel. Converts FP16/BF16/FP32 tensors to FP8 E4M3 with per-block scaling. Inverse dequantization kernel included.

**Applies to:** Any activation or KV cache requiring FP8 compression: attention K/V, MLP activations, transformer weight caching.

**Algorithm:**
```cuda
// 8 FP16 values per thread (128-bit load)
float4 raw = *reinterpret_cast<const float4*>(input + idx);
half* halves = reinterpret_cast<half*>(&raw);

float inv_scale = 1.0f / scale[group_id];  // scale = amax / 448.0
__nv_fp8x4_e4m3 out;
// Use __nv_cvt_halfraw4_to_fp8x4_e4m3 intrinsic (CUDA 12.1+)
*reinterpret_cast<uint32_t*>(output + idx) = out.__x;
```

**CUDA primitives:** `__nv_fp8_e4m3` / `__nv_fp8x4_e4m3` native CUDA types, `__nv_cvt_float_to_fp8` intrinsic, `float4` vectorized loads.

**H100 features:** H100-native FP8 type support (`cuda_fp8.h`), FP8 tensor core feed path (output feeds directly into FA3/K2 attention kernels).

**Speedup:** 2× memory reduction vs FP16. Throughput: 3×–5× over PyTorch `.to(torch.float8_e4m3fn)`.

**Benchmark metrics:** GB/s throughput, E4M3 vs FP16 relative error, round-trip reconstruction accuracy.

**Complexity:** Low

---

### K9 — `fused_conv_norm_act`

**What it does:** Fuses Conv2D output → GroupNorm → SiLU/GELU into one kernel, keeping post-conv activations in SMEM/registers without writing to HBM between ops.

**Applies to:** Any CNN-based encoder/decoder: VAE in SD/SDXL/SD3/FLUX, U-Net decoder blocks, ConvNeXt, ResNet, any model with repeated (Conv → Norm → Act) patterns.

**Algorithm:**
```
# Standard: 3 separate kernel launches, 2 intermediate HBM writes
conv_out = Conv2d(x)          # → HBM write
norm_out = GroupNorm(conv_out) # → HBM read + write
act_out  = SiLU(norm_out)      # → HBM read + write

# Fused: keep conv output in SMEM tile, apply norm+act before writing to HBM
# GroupNorm stats computed over spatial tile in shared memory (Welford)
```

**CUDA primitives:** Implicit GEMM for Conv2D (im2col in registers), `__shfl_xor_sync` for GroupNorm warp reduction, `__half2` SiLU (`x * sigmoid(x)`).

**H100 features:** 228 KB SMEM accommodates larger conv output tiles before flushing, overlapping with GroupNorm computation.

**Speedup:** 1.3×–2.0× over separate PyTorch ops. Scales with number of fused ops; Conv+GN+SiLU+residual is ~2.2×.

**Benchmark metrics:** Kernel fusion overhead vs separate launches, DRAM traffic reduction, latency at various spatial resolutions (64², 128², 512²).

**Complexity:** Medium

---

### K10 — `warp_topk_selection`

**What it does:** Warp-parallel top-K selection using bitonic sort over `__shfl_xor_sync`. For K ≤ 32 (fits in warp registers), no shared memory needed. Returns both top-K values and their indices.

**Applies to:** Any kernel requiring token selection: H2O eviction (K5), PyramidKV (K13), SnapKV, attention score aggregation in any sparse attention method.

**Algorithm:**
```cuda
// Bitonic sort using warp shuffle (K <= 32, no SMEM needed)
float val = input[tid];
int  idx = tid;
#pragma unroll
for (int k = 2; k <= 32; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
        float other_val = __shfl_xor_sync(0xffffffff, val, j);
        int   other_idx = __shfl_xor_sync(0xffffffff, idx, j);
        bool ascending = ((tid & k) == 0);
        if ((ascending && val < other_val) || (!ascending && val > other_val)) {
            val = other_val; idx = other_idx;
        }
    }
}
```

**CUDA primitives:** `__shfl_xor_sync` exclusively (register-only for K≤32), shared memory bitonic sort for K>32.

**H100 features:** Register file (256 KB/SM) allows holding sorted arrays for up to K=64 in-register.

**Speedup:** 5×–15× vs `torch.topk` at small K. 2×–3× vs CUB `DeviceRadixSort` for online/streaming top-K.

**Benchmark metrics:** Latency at K=8/16/32/64/128, throughput at varied input sizes (1K–1M elements).

**Complexity:** Low

---

### K11 — `indexed_gather_scatter`

**What it does:** Coalesced indexed gather (compacting selected KV entries into a dense buffer) and scatter (writing back in the original layout). Uses `float4` vectorized loads for maximum HBM3 utilization.

**Applies to:** Any KV compaction after token eviction or selection: used by K4, K5, K6, K13, K15, any sparse attention method that changes sequence length.

**Algorithm:**
```cuda
// Gather: read selected KV entries at arbitrary indices → compact output
int kv_idx = selected_indices[blockIdx.x];  // global token index
float4* src = (float4*)&K[kv_idx * HEAD_DIM];
float4* dst = (float4*)&K_compact[blockIdx.x * HEAD_DIM];
// 8 float4 loads = 128 FP16 = HEAD_DIM=128 per warpgroup
#pragma unroll
for (int i = 0; i < HEAD_DIM / 8; i++)
    dst[threadIdx.x + i*32] = src[threadIdx.x + i*32];
```

**CUDA primitives:** `float4` 128-bit vectorized load/store, `__ldg` for read-only index array, no warp divergence (uniform access pattern within a block).

**H100 features:** HBM3 bandwidth (3.35 TB/s) fully utilized with float4 coalesced access, L2 reuse for index arrays.

**Speedup:** 3×–8× vs scatter/gather via PyTorch `index_select`. Throughput: 85%–95% of HBM3 peak.

**Benchmark metrics:** GB/s achieved vs HBM3 peak, latency for varied selection ratios (10%/25%/50% of total tokens).

**Complexity:** Low

---

### K12 — `streaming_kv_ringbuffer`

**What it does:** Implements a fixed-size "infinite" KV cache: keeps 4 attention sink tokens permanently and a rolling FIFO window of recent tokens. At capacity, oldest non-sink tokens evicted. RoPE positions recomputed using in-cache (not original) positions for consistency.

**Applies to:** Any causal autoregressive model needing unbounded generation within a fixed memory budget: LLaMA, GPT, Mistral, Falcon, any AR video world model (Oasis, Genie-style), long-document LLMs.

**Algorithm:**
```
# Layout: [sinks (4 slots), window (C-4 slots)]
# Write pointer: wraps around window portion only
write_pos = 4 + (step % (C - 4))

# RoPE re-application: use in-cache position, not original token position
# Sinks: positions 0,1,2,3 always
# Window: positions 4,5,...,C-1 (contiguous in cache)
```
RoPE re-application kernel: reads all cached K entries, applies `sin/cos(cache_pos × freq)`.

**CUDA primitives:** Atomic increment for write pointer, `sin`/`cos` in-kernel via `__sincosf`, vectorized K read-modify-write.

**H100 features:** Fits the recency window in L2 (50 MB L2 holds ~12.5K FP16 tokens × 128-dim); L2 streaming keeps hot tokens fast.

**Speedup:** Fixed memory footprint regardless of generation length. 22×+ throughput vs full-context recomputation at step 10K+.

**Benchmark metrics:** Memory footprint vs generation step, quality (perplexity/FVD) at varied window sizes, RoPE re-application latency.

**Complexity:** Medium

---

### K13 — `pyramidkv_score_extraction`

**What it does:** Modified FlashAttention kernel that simultaneously outputs O (standard attention output) AND accumulates per-token importance scores into an auxiliary buffer — all in one kernel, no extra pass needed. Enables PyramidKV layer-budget allocation without a separate attention forward pass for score extraction.

**Applies to:** Any transformer where importance-based KV eviction is desired without the overhead of a separate attention pass: LLMs (LLaMA, Mistral), DiTs (CogVideoX, HunyuanVideo), multimodal models.

**Algorithm:**
```
# Inside the FlashAttention tiling loop, after computing S[i,j] tile:
# Standard: use S for softmax → O accumulation only
# Modified: also accumulate row sums for the observation window queries

# For query indices q in [N-alpha, N] (last alpha tokens):
if q >= N - alpha:
    # Accumulate attention weights normalized by running log-sum-exp
    score_buffer[q, kv_idx] += softmax_weight[q, kv_idx]
    # With LSE rescaling to handle tiling correctly:
    # score += exp(S[q,kv] - lse_running[q]) * tile_weight
```
The score buffer is compact: [alpha × L × H] — for alpha=8 and L=4096, H=32: 8×4096×32×2B = 2 MB, fits in L2.

**CUDA primitives:** Standard FA3 warp-specialized primitives + auxiliary atomic-free accumulation into a separate output tensor (uses separate accumulator register set in consumer warpgroup).

**H100 features:** L2 persistence for score buffer (small, frequently accessed), dual WGMMA output streams (O and scores accumulated concurrently in separate register files).

**Speedup:** Eliminates the separate O(alpha × L) attention pass needed for score extraction. Net: 1.0× vs standard FA (adds <5% overhead) while enabling PyramidKV's 8× KV compression.

**Benchmark metrics:** Score extraction overhead (% of attention time), KV compression ratio enabled, downstream model quality.

**Complexity:** High

---

### K14 — `fused_rope_embedding`

**What it does:** Fused Rotary Positional Embedding kernel supporting 1D (LLM), 2D axial (image DiT), and 3D (video DiT) variants. Applies rotation in-place to Q and K tensors without separate intermediate allocation.

**Applies to:** Any model using RoPE: LLaMA, Mistral, Qwen, Gemma, Phi, PaLM, all modern LLMs, DiT-based video models (Oasis, CogVideoX, Wan2.1), ViT with RoPE.

**Algorithm:**
```cuda
// 2D axial RoPE (spatial + temporal) — used by video DiTs
// For position (t, h, w) → frequency components:
int half_d = D / 2;
for (int i = threadIdx.x; i < half_d; i += blockDim.x) {
    float freq_h = pos_h / powf(10000.0f, 2.0f * (i % (half_d/2)) / half_d);
    float freq_w = pos_w / powf(10000.0f, 2.0f * (i % (half_d/2)) / half_d);
    float freq_t = pos_t / powf(10000.0f, 2.0f * (i / (half_d/2)) / half_d);
    float q_r = q[..., i], q_i = q[..., i + half_d];
    q[..., i]        = q_r * cosf(freq) - q_i * sinf(freq);
    q[..., i+half_d] = q_r * sinf(freq) + q_i * cosf(freq);
}
```
Precomputed frequency tables (cached in L2) avoid `powf` at runtime.

**CUDA primitives:** `__sincosf` (fused sin/cos in one instruction), `__ldg` for precomputed frequency table, `__half2` multiply-add for complex rotation.

**H100 features:** L2 persistence for frequency table (tiny: 2×D×T_max×2 bytes, typically <1 MB). HBM3 BW for in-place Q/K modification.

**Speedup:** 1.3×–1.8× vs `transformers` RoPE, 2×–3× vs naive `sin/cos` applied separately.

**Benchmark metrics:** Latency at varied seqlens, frequency table cache hit rate, rotation precision (FP16 vs FP32 phase error).

**Complexity:** Low–Medium

---

### K15 — `snapkv_3d_pooling`

**What it does:** Extends SnapKV's 1D max-pooling importance aggregation to video's T×H×W structure. Computes attention importance scores from an observation window, then applies 3D max-pooling over a (t, h, w) neighborhood to identify spatially and temporally cohesive important regions.

**Applies to:** Any video transformer using KV compression: CogVideoX, HunyuanVideo, Wan2.1, any model with spatial token layout where 1D SnapKV would miss spatial cluster structure.

**Algorithm:**
```
# 1. Compute importance: sum attention weights from last W=16 tokens over prefix
score[s] = sum_{i in [N-W, N]} attn[i, s]    # shape [T, H, W_spatial]

# 2. 3D max-pool over (dt=3, dh=3, dw=3) neighborhood
score_pooled[t, h, w] = max over (t±1, h±1, w±1) neighborhood

# 3. Select top-K positions by pooled score
retained_indices = top_k(score_pooled.flatten(), k=K)
```

**CUDA primitives:** Custom 3D stencil kernel (one warp per output voxel, 27-neighbor max reduction via `__shfl_xor_sync`), `warp_topk_selection` (K10) for final selection.

**H100 features:** Score tensor for video at typical sizes (32×9×16 = 4608 tokens) fits entirely in SMEM (4608×4B = 18 KB).

**Speedup:** 20×+ KV compression at long contexts (matching 1D SnapKV). Improves quality over 1D SnapKV by ~0.3 dB PSNR for video.

**Benchmark metrics:** Score extraction latency, KV compression ratio, downstream quality (FVD, PSNR) vs 1D SnapKV.

**Complexity:** Low–Medium

---

### K16 — `cuda_graph_pipeline`

**What it does:** CUDA Graph capture + CUDA Stream triple-buffering for any multi-stage inference pipeline. Captures each fixed-shape stage (encode, transformer, decode) as a CUDA graph; overlaps stages across consecutive inputs using 3 CUDA streams + Events.

**Applies to:** Any 3-stage inference pipeline: VAE→DiT→VAE (world models), encoder→LLM→decoder (seq2seq), preprocessing→model→postprocessing.

**Algorithm:**
```python
# Graph capture per stage (once)
with torch.cuda.graph(g_dit, stream=dit_stream):
    out = dit_model(noisy_latent, cond, timestep)

# Triple-buffer runtime:
# Stream 0 (encode):    frame N+2
# Stream 1 (transform): frame N+1  ← critical path
# Stream 2 (decode):    frame N
enc_done.record(stream_0)
stream_1.wait_event(enc_done)   # GPU-side wait, CPU non-blocking
g_dit.replay()                  # ~5µs dispatch vs ~90ms Python overhead
```

**CUDA primitives:** `cudaGraph`, `cudaGraphLaunch`, `cudaStreamWaitEvent` (non-blocking), `cudaEvent` with `enable_timing=False` (no 5µs timing overhead).

**H100 features:** 2 independent DMA copy engines for concurrent encode/decode memory transfers, NVLink for multi-GPU extension.

**Speedup:** 2.0×–2.15× from CUDA graph capture alone (eliminates Python dispatch overhead). 1.2×–1.5× additional from stream pipeline overlap.

**Benchmark metrics:** Launch overhead before/after (µs per launch), pipeline bubble size (% of frame time), end-to-end FPS improvement.

**Complexity:** Low–Medium

---

### K17 — `int2_kv_pack_unpack`

**What it does:** INT2 pack/unpack kernel using `lop3.b32` PTX instruction. Packs 4 INT2 values per byte, achieving 8× memory reduction vs FP16. Designed for maximum KV cache compression in long-context generation.

**Applies to:** Any KV cache requiring maximum compression: LLMs at >100K context, video models generating >1000 frames, any scenario where 4×–7× INT4 compression is insufficient.

**Algorithm:**
```cuda
// Pack 4 INT2 values into 1 byte
// INT2 signed range: {-1.5, -0.5, 0.5, 1.5} → encoded as {0b00, 0b01, 0b10, 0b11}
__device__ uint8_t pack_int2x4(int8_t a, int8_t b, int8_t c, int8_t d) {
    return ((a & 0x3) << 6) | ((b & 0x3) << 4) | ((c & 0x3) << 2) | (d & 0x3);
}

// Unpack using lop3 (3-operand bitwise logic) — one instruction for 4 values
// PTX: lop3.b32 out, in, mask, imm, 0xCA  (LUT for (a & b) | (~a & c))
uint32_t packed;
asm volatile("lop3.b32 %0, %1, %2, %3, 0xCA;" 
    : "=r"(out) : "r"(packed), "r"(0x03030303), "r"(0));
```

**CUDA primitives:** Inline PTX `lop3.b32` for fast bitfield extraction, `__shfl_xor_sync` for per-group scale computation.

**H100 features:** BitDecoding-style `lop3`-based unpacking achieves 8.0× kernel speedup on H100 for attention with INT2 KV (measured). Feeds directly into FA3 attention kernel via dequantize-on-load.

**Speedup:** 8× memory reduction. 4×–6× attention kernel speedup (BitDecoding paper, H100).

**Benchmark metrics:** Pack/unpack throughput (GB/s), compression/decompression round-trip accuracy, attention quality with INT2 KV (RMSE vs FP16).

**Complexity:** Medium

---

### K18 — `wgmma_int4_gemm`

**What it does:** INT4 weight-only GEMM using H100 INT4 tensor cores. Weights stored as packed INT4 (2-per-byte), activations in FP16. Dequantization fused into GEMM epilogue. Enables 4× memory reduction and bandwidth reduction for weight-dominated inference.

**Applies to:** Any linear layer in any model: all LLM FFN/attention projections, DiT FFN blocks, MLP heads, embedding tables. Core of AWQ, GPTQ-style quantized inference.

**Algorithm:**
```
# Weight stored: W_int4 [K/2, N] as uint8 (packed nibbles)
# Activation: A_fp16 [M, K]
# Per-group scale: scales_fp16 [K/GROUP_SIZE, N]

# Kernel:
# 1. Load INT4 weights from SMEM via TMA
# 2. Unpack INT4 → INT8 (sign-extend nibbles) in registers via lop3
# 3. WGMMA: mma.m64n128k64.s32.s4.s4 (INT4 MMA, H100)
# 4. Epilogue: dequantize INT32 accumulator → FP16 using group scales
#    out[m, n] = acc[m, n] * scales[k_group, n]
```

**CUDA primitives:** `wgmma.mma_async.sync.aligned.m64n128k64.s32.s4.s4` (H100 INT4 WGMMA), TMA for weight tile loading, `lop3.b32` for nibble unpack, `__shfl_xor_sync` for scale broadcast.

**H100 features:** INT4 tensor cores via WGMMA (7,916 TOPS theoretical), TMA swizzled loads for weight tiles (128B swizzle matches K=64 INT4 = 32 bytes × 2 = 64 bytes... requires padding to 128-byte alignment).

**Speedup:** 2×–4× vs FP16 GEMM at batch=1 (bandwidth-bound regime). 1.5×–2.5× at batch=32 (compute-bound begins to dominate).

**Benchmark metrics:** TFLOPS achieved (INT4 TOPS), weight memory savings, GEMM throughput at varied M/N/K, quality regression (perplexity/FVD) vs FP16.

**Complexity:** High

---

### Universal Kernels Summary

| # | Kernel | Category | Speedup | H100-Specific | Complexity |
|---|--------|----------|---------|---------------|------------|
| K1 | `fused_adaln` | Normalization | 1.8×–2.5× | No (degrades gracefully) | Medium |
| K2 | `int4_fp8_attention` | Attention | 1.8×–3.1× | Yes (INT4/FP8 TC) | High |
| K3 | `fa3_warp_specialized_attention` | Attention | 2.0×–3.6× | Yes (WGMMA, TMA) | Very High |
| K4 | `tiered_kv_quantization` | KV Cache | 2×–8× mem | Partial (FP8 types) | Medium |
| K5 | `h2o_token_eviction` | KV Cache | 5×–20× mem | No | Medium |
| K6 | `tome_token_merge` | Token Mgmt | 1.5×–2.2× | Partial (SMEM) | Medium |
| K7 | `per_group_int4_quantize` | Quantization | 4× mem, 2×–4× speed | No | Low |
| K8 | `per_group_fp8_quantize` | Quantization | 2× mem, 3×–5× speed | Yes (FP8 types) | Low |
| K9 | `fused_conv_norm_act` | CNN Fusion | 1.3×–2.0× | Partial (SMEM) | Medium |
| K10 | `warp_topk_selection` | Utility | 5×–15× vs torch.topk | No | Low |
| K11 | `indexed_gather_scatter` | Utility | 3×–8× vs index_select | No | Low |
| K12 | `streaming_kv_ringbuffer` | KV Cache | Fixed-mem, 22×+ at 10K+ | No | Medium |
| K13 | `pyramidkv_score_extraction` | KV Cache | Enables 8× KV compression | Partial | High |
| K14 | `fused_rope_embedding` | Positional | 1.3×–1.8× | No | Low–Medium |
| K15 | `snapkv_3d_pooling` | KV Cache | 20× mem (long ctx) | No | Low–Medium |
| K16 | `cuda_graph_pipeline` | Pipeline | 2.0×–2.15× + 1.2×–1.5× | Partial | Low–Medium |
| K17 | `int2_kv_pack_unpack` | Quantization | 8× mem, 4×–6× attn | Partial (lop3) | Medium |
| K18 | `wgmma_int4_gemm` | GEMM | 2×–4× vs FP16 | Yes (INT4 WGMMA) | High |

---

---

## Part 2: Model-Specific Kernels

> These kernels target architectural features specific to Open-Oasis 500M and Matrix-Game 2.0. Many partially generalize to other video models with similar structures.

### Model Architecture Reference

| Property | Open-Oasis 500M | Matrix-Game 2.0 (MG2) |
|---|---|---|
| Architecture | DiT-S/2, 24 SpatioTemporalDiTBlock | Wan2.1, 30 WanAttentionBlock |
| Hidden dim | 1024 | 1536 |
| Heads / head\_dim | 16 / 64 | 12 / 128 |
| Token layout | 144 spatial + 32 temporal per frame | 880 tokens/latent frame |
| VAE | ViT-L/20 (20px patch) | 3D Causal VAE (4×8×8) |
| Attention style | Axial: spatial bidir + temporal causal | Block-causal, rolling KV cache |
| Denoising steps | 10 DDIM | 3-step flow matching |
| Action conditioning | 25-dim vector → adaLN | Mouse MLP + 4-dim keyboard cross-attn |
| Context window | 32 temporal frames | 6 latent frames (5280 tokens) |

---

### M1 — `axial_attn_fused_reshape_rope`

**Target:** Open-Oasis 500M. Generalizes to any axial-attention DiT.

**Why model-specific:** Oasis's `SpatioTemporalDiTBlock` performs axial attention — each block transposes `[B,T,S,C] → [B,S,T,C]` twice (once before spatial, once before temporal attention). Without fusion: 48 materialised transposes per forward pass × 37.7 MB tensor = 5.4 GB unnecessary memory traffic.

**Algorithm:** Fuses the layout transpose + per-axis RoPE application + QKV projection into one kernel. Each CUDA block handles one `(B,S)` slice; threads load from `X[b,t,s,:]` and immediately write to `Y[b,s,t,:]` while applying the RoPE rotation in-register. QKV projection follows via WGMMA from the transposed SMEM buffer.

**CUDA primitives:** `cp.async.bulk.tensor` (TMA) for pipelined transpose, `wgmma.mma_async` for QKV projection, L2 persistence for RoPE frequency tables (16 KB/head).

**Speedup:** 1.36× per DiT block. Eliminates 4.8 GB of memory traffic per forward pass.

**Generalizes to:** Any axial-attention transformer (VideoLDM, CogVideo-X, any model with alternating 1D attention axes and intermediate layout transposes).

---

### M2 — `svg2_token_layout_transform`

**Target:** Matrix-Game 2.0. Generalizes to any model with per-head mixed spatial/temporal access patterns.

**Why model-specific:** MG2's 12 attention heads exhibit mixed patterns: spatial heads (block-diagonal mask) vs temporal heads (strided access with stride=880 tokens over 5280 KV tokens). Temporal heads accessing token-major KV cache get L2 hit rate of 31% — each access is stride-880 apart. Frame-major layout makes temporal accesses contiguous.

**Algorithm:** Classifies each head (static, precomputed at load time). For temporal heads: transposes KV from `[F, S, H, D]` to `[H_t, S, F, D]` via 128-bit vectorized TMA stores with 128B swizzle. The 6×880 tokens for temporal heads become 6 contiguous rows per spatial position.

**CUDA primitives:** `float4` 128-bit loads, `cp.async.bulk` with `CU_TENSOR_MAP_SWIZZLE_128B` swizzled stores, double-buffered SMEM.

**Speedup:** 7.2× KV load speedup (L2 hit rate 31% → 96%). 0.21 ms for full transpose vs 3.1 ms baseline KV load time.

**Generalizes to:** Any model with heterogeneous per-head access patterns (VideoPoet, Emu3, any Wan2.1 derivative).

---

### M3 — `block_causal_attention`

**Target:** Matrix-Game 2.0. Generalizes to any chunk-causal video model.

**Why model-specific:** MG2 uses block-causal attention: per-chunk (880-token granularity) causality, not per-token. The 2640q × 5280kv attention matrix has ~50% zero blocks (where `kv_frame > q_frame`) that can be skipped entirely. Standard per-token causal FlashAttention cannot exploit this structure.

**Algorithm:** Tiled FlashAttention-3 with tile-level skip logic:
- `kv_frame < q_frame` → FULL_ATTEND (no masking, fastest path)
- `kv_frame == q_frame` → SELF_ATTEND (bidirectional within same frame)
- `kv_frame > q_frame` → SKIP (entire tile zeroed, no WGMMA issued)

Thread Block Clusters (cluster=2) allow KV tiles to be shared across query-frame blocks via distributed SMEM, reducing KV reload cost.

**CUDA primitives:** FA3 warp-specialized structure, `wgmma.mma_async` for non-skipped tiles, `mbarrier` for cluster synchronization.

**Speedup:** 2.1× over full-causal FlashAttention. 49.7% of tiles skipped. 2.1 GB vs 4.2 GB memory reads per forward pass.

**Generalizes to:** Any latent video model with frame-granularity AR generation (GAIA-1, UniSim, GameNGen chunked KV, document-level causal LLMs).

---

### M4 — `qvg_frame_aligned_kmeans`

**Target:** Both Oasis and MG2. Generalizes to any streaming video model.

**Why model-specific:** Consecutive video frames share 85%–88% k-means cluster assignments (scene continuity). Standard k-means (random init, 15–20 iterations) ignores this. Frame-aligned chunk boundaries (144 tokens for Oasis, 880 for MG2) fit precisely in SMEM for efficient distance computation.

**Algorithm:** Temporal warm-start: initialize cluster centers from previous frame's solution. Delta-update: only re-assign the ~15% of changed tokens. Early exit: if <2% assignments change, stop. Distance computation uses WGMMA for the batched `[S, D] × [K, D]^T` dot-product.

**CUDA primitives:** `__shfl_down_sync` warp argmin for assignment step, `atomicAdd` for centroid update (changed tokens only), `__popc` for early-exit popcount, `float4` vectorized loads.

**Speedup:** 3× convergence speedup. Per-frame k-means: 0.7 ms (Oasis), 0.6 ms (MG2) vs 2.1/1.8 ms random-init.

**Generalizes to:** Any video model using k-means-based KV compression (QVG, H2O-Video, SnapKV-Video). Streaming audio with temporal locality.

---

### M5 — `wan_action_injection_fused`

**Target:** Matrix-Game 2.0 exclusively.

**Why model-specific:** MG2 injects actions at all 30 layers: mouse MLP (Linear(1538, 1536) — non-power-of-2 input due to 2-dim concat) + 4-token keyboard cross-attention. 60 separate CUDA launches per forward pass = 300 µs launch overhead alone.

**Algorithm:** Fuses mouse MLP (padded to 1540-dim internally for 128-bit alignment) + 4-token keyboard cross-attention (fits entirely in registers: 4×FP16×128=1 KB) + residual addition into one persistent CUDA kernel. 30-layer loop captured as CUDA Graph. Keyboard cross-attn runs on a dedicated cooperative warpgroup concurrently with mouse MLP.

**CUDA primitives:** `cooperative_groups::thread_block_tile<32>` for concurrent warpgroups, `__half2` for SiLU, register-only softmax for 4-element keyboard attention, `cudaGraph` node fusion.

**Speedup:** 2.4× on action injection (10.5 ms → 4.4 ms). Eliminates 280 µs of CUDA Graph launch overhead.

**Generalizes to:** Any action-conditioned world model with per-layer dual-path injection (DIAMOND, IRIS, GameGen-X). Mouse MLP pattern generalizes to any non-power-of-2 input MLP.

---

### M6 — `3d_causal_vae_encode`

**Target:** Matrix-Game 2.0 exclusively. Generalizes to any streaming 3D causal VAE.

**Why model-specific:** MG2's 3D causal VAE (8-layer causal Conv3d, kernel=3, 4× temporal stride) receives only 1–2 new frames per generation step but naively re-encodes all 24 raw frames in the context window.

**Algorithm:** Sliding window activation cache: stores boundary activations `[B, C_l, 2, H_l, W_l]` per layer (causal overlap = 2 frames for kernel=3). On new frame arrival, only processes the marginal new frames, using cached activations as causal padding. Persistent kernel maintains activations across 8 encoder layers in SMEM to avoid inter-layer HBM writes.

**CUDA primitives:** `ldgsts` async copy for cached boundary frames, direct convolution via WGMMA (im2col in registers), distributed SMEM for spatial halo sharing via Thread Block Clusters.

**Speedup:** 4.3× VAE encode speedup. Processes 4 new frames vs 24 full frames. 5.8 GB fewer memory reads/second at 30 FPS.

**Generalizes to:** Any streaming 3D causal VAE (OpenSora-VAE, CogVideoX-VAE, Wan2.1 standalone). Sliding-window activation cache pattern applies to any streaming causal convolution.

---

### M7 — `seacache_step_similarity`

**Target:** Open-Oasis 500M primarily. Applies to any multi-step diffusion model.

**Why model-specific:** Oasis runs 10 DDIM steps. Layers 8–14 show >0.93 cosine similarity between consecutive steps at steps 4–8 (medium noise regime). Threshold τ=0.92 tuned to Oasis's specific noise schedule. Skip rate: ~34% of all layer computations.

**Algorithm:** For each (batch, layer): compute cosine similarity between current-step and previous-step activations using two-phase warp+block reduction. Early-exit after 10% of tokens if clearly dissimilar. If sim > τ: skip this layer and reuse cached output.

**CUDA primitives:** `__hfma2` (FP16 fused multiply-add for dot product), `__shfl_down_sync` warp reduction, `rsqrtf` for fast norm, predicated `__ldg` for early-exit path.

**Speedup:** 1.49× end-to-end forward pass. 34% of layers skipped. <0.3 dB PSNR degradation.

**Generalizes to:** Any multi-step diffusion model (DiT-XL, PixArt, SD3, FLUX, HunyuanVideo). Threshold must be re-calibrated per model/schedule. Less effective at 3-step models (MG2).

---

### M8 — `diffusion_forcing_noise_schedule`

**Target:** Open-Oasis 500M exclusively. Generalizes to any per-frame conditioned DiT.

**Why model-specific:** Oasis uses Diffusion Forcing: 32 frames can each have an independent noise level. This requires 32 separate adaLN conditioning computations per DiT block (one per frame's `(timestep_emb + action_emb)` vector). Sequentially: 32 × MLP(281-dim → 6144-dim) per block = 768 sequential MLPs per forward pass.

**Algorithm:** Batches all 32 per-frame projections into a single WGMMA call of shape `[32, 6144, 288]` (288 = next multiple of 16 for alignment). Timestep embedding table lookup done first (one warp per frame). Causal noise mask `M[i,j] = 1 if noise_levels[j] <= noise_levels[i]` computed in 32 threads (32×32 = 1024 elements, trivial).

**CUDA primitives:** `wgmma.mma_async` for `[32, 6144, 288]` batched projection, `__ldg` for vectorized table lookup, `__half2` SiLU.

**Speedup:** 5.2× on adaLN compute (11.5 ms → 2.2 ms for 24 blocks). adaLN fraction: 6.3% → 1.2% of total DiT time.

**Generalizes to:** Any video DiT with per-frame conditioning (LVDM, CogVideoX, any per-frame-conditioned DiT). Core batched projection reusable; Diffusion Forcing mask is Oasis-specific.

---

### M9 — `persistent_denoising_megakernel`

**Target:** Both Oasis and MG2. Generalizes to any multi-step diffusion/flow model.

**Why model-specific:** Multi-step denoising creates CPU round-trips between steps (compute step t → CPU receives completion → CPU dispatches step t+1 → GPU starts). Each round-trip: ~50 µs. Oasis: 10 steps × 50 µs = 500 µs. DDIM schedule update between steps is a simple per-element scale-and-shift (0.02 ms) that can be absorbed into the persistent kernel.

**Algorithm:** Single kernel launch fills all 132 SMs. All SMs iterate over steps and layers via `mbarrier` cluster barriers. DDIM update performed by SM 0 between steps (others wait on mbarrier). Double-buffered `x_buffer[2]` holds current/previous step activations.

**CUDA primitives:** `cuda::barrier<cuda::thread_scope_cluster>` (CUDA 12 cluster barriers), `mbarrier.arrive_and_wait.shared::cluster` PTX, `cuda::memcpy_async` for weight prefetch.

**Speedup:** 1.06×–1.09× overall (9.8 ms overhead eliminated on Oasis, 3.6 ms on MG2). Primary value: P99 tail latency −9.2%, SM utilization 97% (vs bursty in standard launches).

**Generalizes to:** Any multi-step diffusion or ODE-solver-based model. Requires CUDA 12+ and H100 (sm_90a) for cluster-level mbarrier.

---

### M10 — `pyramidkv_video_score_extraction`

**Target:** Both Oasis and MG2. Generalizes to any video model with KV compression.

**Why model-specific:** Flat top-K token eviction ignores video's frame structure — may select all tokens from one frame while completely discarding others, losing temporal coverage. Video-specific 2D structured top-K guarantees multi-frame diversity.

**Algorithm:** Three-step: (1) aggregate attention scores over Q dimension; (2) reshape to `[F, S]` and perform top-Kt selection over temporal axis (for Oasis: F=32 bitonic warp sort; for MG2: F=6 in-register); (3) top-Ks selection over spatial axis (S=144 for Oasis: warp cooperative sort; S=880 for MG2: bitonic sort with padding to 1024); (4) union+deduplication via shared memory bitmask.

**CUDA primitives:** `__shfl_xor_sync` bitonic warp sort for small F, shared-memory bitonic sort (padded to power-of-2) for large S, `atomicOr` on cluster-shared memory for deduplication.

**Speedup:** +0.4% SSIM vs flat top-K at equal KV budget. 4.0 frames/spatial-position temporal diversity vs 1.2 with flat top-K. Kernel: 1.1 ms vs 0.8 ms flat top-K (40% slower but better quality).

**Generalizes to:** Any video transformer with KV compression (VideoLLaMA, LLaVA-Video, Video-LLaVA, any video LLM). Bitonic sort subroutine is a general-purpose primitive.

---

### Model-Specific Kernels Summary

| # | Kernel | Target | Speedup | Generalizes? |
|---|--------|--------|---------|--------------|
| M1 | `axial_attn_fused_reshape_rope` | Oasis | 1.36× per block | Yes — axial-attn DiTs |
| M2 | `svg2_token_layout_transform` | MG2 | 7.2× KV load | Yes — mixed-head models |
| M3 | `block_causal_attention` | MG2 | 2.1× attention | Yes — chunk-causal models |
| M4 | `qvg_frame_aligned_kmeans` | Both | 3× k-means convergence | Yes — streaming video |
| M5 | `wan_action_injection_fused` | MG2 | 2.4× action injection | Partial |
| M6 | `3d_causal_vae_encode` | MG2 | 4.3× VAE encode | Yes — streaming causal VAE |
| M7 | `seacache_step_similarity` | Oasis | 1.49× forward pass | Yes — multi-step diffusion |
| M8 | `diffusion_forcing_noise_schedule` | Oasis | 5.2× adaLN | Yes — per-frame conditioned DiT |
| M9 | `persistent_denoising_megakernel` | Both | 1.06–1.09× (tail latency) | Yes — any multi-step model |
| M10 | `pyramidkv_video_score_extraction` | Both | +0.4% SSIM | Yes — any video model |

---

## Master Benchmark Plan

Each kernel is benchmarked independently before integration:

| Metric | Tool | Target |
|---|---|---|
| Kernel latency (µs) | `torch.cuda.Event` timing | Beat PyTorch reference |
| HBM3 bandwidth (GB/s) | `ncu l1tex__t_bytes` | >60% of 3,350 GB/s |
| Tensor core utilization | `ncu sm__ops_path_tensor` | >70% for compute kernels |
| Correctness (MSE) | Custom harness | <1e-3 vs FP16 reference |
| End-to-end FPS | Wall-clock on target model | Demonstrate speedup |

**Benchmark hardware:** NVIDIA H100 80GB SXM, CUDA 12.4, PyTorch 2.3, `sm_90a`.

**Comparison baselines:**
- PyTorch eager mode
- FlashAttention-2 (for attention kernels)
- FlashAttention-3 (for advanced attention kernels)
- `torch.compile(mode="max-autotune")` (for fusion kernels)

---

## Implementation Priority

**Phase 1 (Highest ROI, implement first):**
- K1 `fused_adaln` — easiest, measurable on any DiT
- K7 `per_group_int4_quantize` — foundational for K4, K18
- K8 `per_group_fp8_quantize` — foundational for K2, K4
- K16 `cuda_graph_pipeline` — no kernel work, pure scheduling
- K10/K11 `warp_topk` + `gather_scatter` — utility primitives for other kernels

**Phase 2 (Core contribution):**
- K4 `tiered_kv_quantization` — uses K7, K8
- K5 `h2o_token_eviction` — uses K10, K11
- K2 `int4_fp8_attention` — requires K7, K8
- K14 `fused_rope_embedding`

**Phase 3 (Advanced):**
- K3 `fa3_warp_specialized_attention` — very high complexity
- K13 `pyramidkv_score_extraction` — requires FA3 modification
- K17/K18 `int2` and `wgmma_int4_gemm`
- Model-specific kernels M1–M10 as needed

---

*All kernels target CUDA 12.4+, sm_90a (H100). Graceful degradation to sm_80 (A100) noted per-kernel above.*
