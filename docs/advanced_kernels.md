# Advanced CUDA Kernels — WorldServe
## CMU 15-849 ML Systems — Spring 2026
**Hardware:** NVIDIA H100 80GB SXM (sm_90a) | Source: 40+ papers, 2024–2026

**Scope:** Custom CUDA kernels, no model training required. Each kernel replaces a PyTorch/FA2 op in an existing pretrained ViT/DiT/world model and is benchmarked for wall-clock speedup and quality on H100.

---

## Target Models for Testing

| Model | Size | Architecture | Primary kernels to test |
|-------|------|-------------|------------------------|
| **DiT-XL/2** | 675M | DiT, 1000-step DDIM, ImageNet | D1, D4, D5, C1, C2, A1 |
| **FLUX.1-schnell** | 12B | MM-DiT, 4-step flow, text→image | D1, D3, C1, A1, A4 |
| **CogVideoX-2B** | 2B | 3D DiT, video generation | A1, A2, B1, B3, C3 |
| **HunyuanVideo** | 13B | 3D full-attn DiT, video | A1, A2, C3, C5, D1 |
| **Open-Oasis 500M** | 500M | DiT-S/2, 10-step DDIM, world model | All kernels — primary target |
| **Wan2.1-T2V-1.3B** | 1.3B | Wan DiT, flow matching | A1, B3, C2, C5, D4 |

**Integration method per model:**
```python
# Pattern A — monkey-patch (no model code changes needed)
model.transformer_blocks[i].attn.forward = my_kernel_forward

# Pattern B — custom attention processor (diffusers)
model.set_attn_processor(MyFP8AttnProcessor())

# Pattern C — module swap
model.transformer_blocks[i].norm1 = FusedAdaLNModule(...)
```

---

## Category 1: Sparse Attention

### A1 — `sliding_tile_attention`
**Source:** arXiv 2502.04507, ICML 2025

**Problem:** Standard sliding-window attention creates "mixed blocks" — tiles where some Q-K pairs are in-window and some out. These tiles must compute the full matrix product then mask, yielding zero actual speedup for ~30% of tiles.

**Solution:** Align tile boundaries with window boundaries. Every tile is either 100% attend or 100% skip — zero masked tiles.

**CUDA design:**
- Producer warpgroup: consults `valid[q_tile, kv_tile]` boolean table before `cp.async.bulk.tensor` (TMA). Non-valid tiles never enter SMEM.
- Consumer warpgroup: fires WGMMA only on loaded tiles. No masking, 100% tensor-core utilization.
- Table for Oasis (32, 12, 12) grid: 576 bytes → L1-resident permanently.

**H100 primitives:** TMA `cp.async.bulk.tensor` with predicated issue; `wgmma.mma_async.sync.aligned.m64n128k16.f16`

**Kernel speedup:** 10.45× at 91% sparsity | **End-to-end:** 3.53× HunyuanVideo 720p
**Test on:** HunyuanVideo, CogVideoX-2B, Wan2.1, Oasis temporal attention

---

### A2 — `radial_decay_static_mask`
**Source:** arXiv 2506.19852, NeurIPS 2025

**Problem:** Video attention decays with spatiotemporal distance but this is exploited only empirically. Without a theoretical bound, skip thresholds are conservatively set.

**Solution (mathematically proven):**
```
p_{j,l} ≤ C · exp(-α|Δt| - β|Δs|)
```
For Δt ≥ 8 frames and Δs ≥ 16 tokens, bound < 0.001. Calibrate α, β on 10 inference passes → **static mask, zero runtime overhead**.

**CUDA design:**
- Calibration: 10 forward passes, fit α, β via least-squares on per-token attention weights.
- Build `valid[q_frame, kv_frame, Δs]` table: 2 KB for Oasis → L1-resident.
- Composes with A1's validity table and M3's block-causal mask: skip rate 50% → **68-72%**.

**End-to-end:** 3.7× HunyuanVideo, 9× attention reduction for 500-frame 720p
**Test on:** Oasis (temporal attention), HunyuanVideo, CogVideoX

---

### A3 — `tempache_ann_kv_dedup`
**Source:** arXiv 2602.01801, Feb 2026

**Theorem (Duplicate Attention Lemma):** Merging duplicate key entries is mathematically exact — not approximate:
```
Attn(Q, K_dedup, V_dedup)  is exactly equal to  Attn(Q, K, V)
where V_dedup[merged] = mean(V[bucket])  and  logit_bias[merged] = log(count)
```

**CUDA design:**
```cuda
// 1. FP8 LSH projection: K → [4608, 32] sign bits
// 2. cub::DeviceRadixSort by bucket
// 3. Merge same-bucket entries: V_merged = mean, count per bucket
// 4. FA3 with log-count logit bias added in softmax
scores[i, merged_j] += log(count[merged_j]);  // fused into softmax
```
H100: FP8 WGMMA for projection `[4608,1024]×[1024,32]`. Count table 18 KB → L2-persistent.

**End-to-end:** 5–10× on world model rollouts. Memory flat across 3000-frame rollouts vs linear growth.
**Test on:** Oasis (AR generation), Wan2.1 Self-Forcing

---

### A4 — `triattention_prerope_score`
**Source:** arXiv 2604.04921, April 2026

**Problem:** Post-RoPE attention scores are distorted by position rotation → poor KV importance estimates for distant tokens, causing good KV to be evicted.

**Solution:** Q/K vectors in pre-RoPE space concentrate around fixed centers. Using center vectors, predict salience analytically:
```
score(position p) = Re[ Σ_k A_k · exp(i · ω_k · p) ]   // no inference-time sampling
```

**CUDA design:**
- One-time calibration: `center_k[H, D]` = mean over 10 inference passes. 2 KB total → L2-persistent.
- Per step: `score[N]` = `__sincosf` trig eval + `[D, 1]×[1, N]` WGMMA per head. Trivial for 4608 tokens.
- TopK via `warp_topk` (`__shfl_xor_sync` bitonic) → evict low-score KV.

**Result:** 10.7× KV compression at **full-attention accuracy**. 2.5× throughput. Replaces H2O/PyramidKV scoring.
**Test on:** Oasis, FLUX.1, LLaMA 3.1 8B (as baseline comparison)

---

## Category 2: KV Cache Compression

### B1 — `ca2vdm_invariant_kv_prefix`
**Source:** arXiv 2411.16375, ICML 2025

**Problem (specific to AR video DiT):** Conditioning frames receive `t=0` timestep embedding at every denoising step → their KV vectors are **identical across all 10 DDIM steps**. Standard code recomputes them 10×, wasting 9 of 10 computations.

**CUDA design:**
```python
# Before denoising loop — compute once:
K_prefix, V_prefix = dit.compute_kv(clean_frames)          # one-time
cudaMemAdvise(K_prefix, cudaMemAdviseSetAccessedBy, device) # L2 hint

# Inside denoising loop — use cached:
out = flash_attn(Q_noisy, K_prefix, V_prefix)               # no recompute
```
For Oasis: 432 MB prefix KV. Per-layer = 18 MB. Mark recent 4 frames (2.25 MB INT4) as `cudaAccessPropertyPersisting` → L2 at 12 TB/s.

**Speedup:** 2.5× on prefix overhead. Overall: 2.5× on conditioning-heavy AR models.
**Test on:** Oasis, CogVideoX (both have clean-frame conditioning)

---

### B3 — `qvg_temporal_residual_kv`
**Source:** arXiv 2602.02958, Feb 2026 (MIT/MIT-IBM/Song Han)

**Insight:** Consecutive frames share semantics → KV residuals `K_t - K_{t-1}` have 5× smaller dynamic range than raw `K_t`. Enables INT2 compression that direct INT4 cannot achieve.

**Two-stage kernel:**
```cuda
// Stage 1: semantic smoothing (EWA mean, in registers)
mean_K = 0.9f * mean_K + 0.1f * K_curr;
residual = K_curr - mean_K;

// Stage 2: progressive residual quant (PRQ)
q1    = quantize_int4(residual, scale1);        // INT4 first pass
r1    = dequantize(q1, scale1) - residual;       // residual error
q2    = quantize_int2(r1, scale2);               // INT2 correction
packed = __byte_perm(q1, q2, perm_mask);         // lop3.b32 pack
cp.async.cg(kv_buf + offset, &packed, 1);        // bypass L1
```
H100: With 2-bit KV, all 24 layers × 31 old frames = 13.4 MB → fits in L2. Temporal residuals at 12 TB/s.

**Speedup:** 7× KV compression, <4% latency overhead. Tested on Wan2.1-based Self-Forcing.
**Test on:** Oasis (32-frame window), Wan2.1, MG2

---

## Category 3: Step/Layer Computation Caching

### C1 — `predi_t_ab_feature_predict`
**Source:** arXiv 2602.18093, Feb 2026

**Insight:** Instead of reusing cached features unchanged (zero-order), predict them using Adams-Bashforth second-order extrapolation. 30× cheaper per skipped step than recomputation.

**Adaptive horizon:** `J = floor(τ / (δ_n + ε)^(1/(p+1)))` — auto-adjusts based on dynamics. Skips up to J=8 steps in smooth mid-noise regions; J=1 at high-dynamics edges.

**Kernel (20 lines):**
```cuda
__global__ void ab2_predict(__half* F_pred,
                             const __half* F_t, const __half* F_t1,
                             float alpha, int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __half2* o = (__half2*)F_pred;
    const __half2* a = (const __half2*)F_t;
    const __half2* b = (const __half2*)F_t1;
    __half2 al = __float2half2_rn(alpha);              // alpha = 3*Δt/2
    o[idx] = __hfma2(al, __hsub2(a[idx], b[idx]), a[idx]);
}
// Cost: 32 MB memory traffic vs ~500µs full attention = 30× cheaper
```

**Speedup:** FLUX.1 **5.54×**, HunyuanVideo **3.28×**, DiT-XL/2 **2.48×**. Outperforms TaylorSeer at aggressive compression (FID 2.24 vs 10.12).
**Test on:** FLUX.1-schnell, DiT-XL/2, HunyuanVideo, Oasis (replaces SeaCache M7)

---

### C2 — `tea_cache_emb_distance`
**Source:** arXiv 2411.19108, CVPR 2025

**Insight:** Timestep-embedding-modulated input tracks output similarity better than raw inputs. Accumulated L1 distance drives adaptive cache reuse:
```
L1_rel(t) = ||O_t - O_{t+1}||_1 / ||O_{t+1}||_1
// reuse cached output until Σ L1_rel > threshold δ
```
Polynomial rescaling `y = a₀ + a₁x + ... + a₄x⁴` corrects proxy bias — 5 floats fit in constant memory.

**Kernel:** `fused_subtract_l1_norm` → `__shfl_down_sync` warp reduction → scalar compare → branch to cache-reuse path (`cp.async.ca`) or full forward. Sub-microsecond decision overhead.

**Speedup:** **4.41×** on OpenSora-Plan 65-frame at -0.07% VBench.
**Test on:** OpenSora, DiT-XL/2, CogVideoX, Oasis

---

### C3 — `clusca_cluster_sparse_attn`
**Source:** arXiv 2509.10312, ACM MM 2025

**Insight:** K-Means cluster tokens at anchor steps. At non-anchor steps, only 1 representative per cluster runs full attention (90% token reduction), others blend with the representative output.

**Three-kernel pipeline:**
```
1. gather_representatives:  rep[k] = tokens[cluster_centroid_idx[k]]
                             — indexed gather, K11 pattern
2. mini_attention:           K×K dense FA3 (K = N/10, 100× cheaper)
3. blend_scatter:            out[i] = γ*rep_out[cluster_id[i]] + (1-γ)*cache[i]
                             — fully coalesced if cluster_ids sorted
```
Calibration: run K-Means once at inference time (~100 forward passes, no gradients).

**Speedup:** HunyuanVideo **6.21×**, FLUX **4.96×**, DiT-XL/2 **3.97×**. Clustering overhead ≤5%.
**Test on:** HunyuanVideo, FLUX.1, DiT-XL/2, CogVideoX

---

### C5 — `liteattn_skiplist_persistent`
**Source:** arXiv 2511.11062, Nov 2025

**Insight:** The tile-level Skip-Mask (which attention tiles contribute negligibly) is stable across denoising steps. Compute once at step 3, propagate through steps 4-10.

**FA3 integration:**
- Skip-List: run-length (start, end) pairs for contiguous skippable ranges, stored in persistent SMEM.
- Producer warpgroup reads Skip-List before TMA issue → non-listed tiles never touch SMEM or WGMMA.
- Skip predicate evaluated in consumer epilogue (not prologue) to avoid sync stalls.
- Composable with C1/C2: LiteAttention catches tile-level redundancy within layers not skipped at layer level.

**Speedup:** **1.9×** on Wan2.1-14B at 42% sparsity. Zero quality degradation vs SVG's visible artifacts at 66%.
**Test on:** Wan2.1, HunyuanVideo, Oasis

---

## Category 4: Quantization

### D1 — `sage_attn2_fp22_accumulator_fix` ⚠️ MANDATORY
**Source:** arXiv 2411.10958, ICML 2025

**Critical hardware bug:** H100's FP8 MMA accumulator is **FP22** (1 sign + 8 exp + 13 mantissa), NOT FP32. Any kernel using `wgmma.mma_async.e4m3` without this fix silently degrades at long sequences — affects ALL existing FP8 attention kernels including the original K2.

**Fix — two-level accumulation:**
```cuda
// Level 1: FP22 hardware accumulator within each WGMMA tile (fast)
wgmma.mma_async.sync.aligned.m64n128k32.e4m3.e4m3.f32 ...

// Level 2: before cross-tile accumulation — explicit FP32 rescale
// acc_fp32 += exp(m_prev - m_curr) * acc_fp22_tile   ← rescale then add
// Without this: FP22 partial sum is treated as FP32 → silent drift
```

**Full SageAttention2 scheme (complete replacement for K2):**
- Q, K: per-thread INT4 quantization aligned to `mma.m16n8k64.s4` layout (zero dequant overhead)
- Outlier smoothing: `Q_smooth = Q - q̄`, correction `ΔS = q̄ · γ(K)^T` fused into single kernel (one activation read)
- P, V: FP8 E4M3 with the two-level accumulation fix

**Speedup:** 481 TOPS on RTX 4090 (3× over FA2). H100: matches FA3-FP8 speed with substantially better accuracy.
**Test on:** All models — required baseline for any FP8 attention benchmarking

---

### D3 — `svdquant_joint_int4_lora_gemm`
**Source:** arXiv 2411.05007, MIT Han Lab, Nov 2024

**Insight:** W4A4 fails with standard smoothing because outlier channels can't be simultaneously handled in 4-bit weights AND activations. SVDQuant absorbs outliers offline via SVD:
1. `torch.linalg.svd(W)` → keep rank-32/64 branch R = UΣV^T at FP16 (absorbs outliers)
2. Quantize residual `W - R` to INT4
3. No training — pure offline matrix decomposition

**Nunchaku fusion kernel:** Standard approach reads activations twice (once for INT4 GEMM, once for FP16 low-rank GEMM). Nunchaku fuses both into one kernel with a single activation read:
```cuda
// Single __half2 load of x, feeds two concurrent WGMMA paths:
// Path 1: x @ W_int4     (WGMMA INT4 tensor cores, 7,916 TOPS)
// Path 2: x @ (U·Σ·V^T) (WGMMA FP16 tensor cores)
// output = path1 + path2  (element-wise fused in epilogue)
```

**Speedup:** 3.0× vs 4-bit weight-only baseline. 3.5× memory reduction FLUX.1 12B. Works with existing LoRA adapters without re-quantization.
**Test on:** FLUX.1-schnell (12B), HunyuanVideo, DiT-XL/2

---

### D4 — `taqdit_emb_quant_schedule`
**Source:** arXiv 2411.14172 (TaQ-DiT) + arXiv 2503.06564 (TR-DQ)

**Insight:** The timestep embedding already computed for adaLN conditioning encodes the noise level. Reuse it to derive quantization scales at zero extra cost. Confirmed across 5 papers: high-noise steps (1-3) tolerate INT4; low-noise steps (8-10) require FP8 minimum.

**Three-group schedule:**
```
Steps 1-3  (high noise):  INT4 Q/K/V + INT4 weights  → 7,916 TOPS path
Steps 4-7  (mid noise):   INT4 Q/K  + FP8 V + INT8 W
Steps 8-10 (low noise):   FP8  Q/K/V + INT8 weights
```
Scale lookup table: 10 steps × 24 layers × FP16 = 960 bytes → constant memory, zero HBM read.

**CUDA design:** Per-step scale factor loaded from constant memory in GEMM prologue. The dynamic dispatch between INT4/FP8/FP16 paths uses PTX `@p` predicated execution — no branch divergence (all threads in a warp see the same step index).

**Speedup:** 1.38–1.89× over static quantization, 1.97–2.58× memory reduction.
**Test on:** Oasis (10-step DDIM), DiT-XL/2 (1000-step), FLUX (4-step) — ablate across step counts

---

### D5 — `vidit_q_per_token_dynamic`
**Source:** arXiv 2406.02540, ICLR 2025

**Insight:** Video DiT activation ranges vary across timesteps AND prompts/actions. Static calibration always under- or over-quantizes. Dynamic per-token scale at runtime with negligible overhead:
```cuda
// Fused into GEMM prologue — not a separate kernel launch
// One __shfl_xor_sync warp reduction per token row = ~10 cycles
float scale = warp_absmax(hidden[token, :]) / 127.0f;  // INT8
// or / 448.0f for FP8 E4M3
```
Scale table `[N_tokens, N_layers]` = 18 KB for Oasis → L2-resident, no HBM penalty.

**Speedup:** 1.4–1.7× end-to-end at W8A8, negligible quality loss vs static calibration.
**Test on:** Oasis, Wan2.1, CogVideoX — specifically test with diverse action inputs (varies activation range)

---

## Category 5: Pipeline & Speculative

### E4 — `stream_diffusion_stagger_pipeline`
**Source:** arXiv 2312.12491, ICCV 2025

**Insight:** Stack multiple frames at different denoising stages into one batch. At steady state, one DiT forward produces one finished frame (k× throughput):
```
Batch at time T:
  [frame_n   @ step 1,  frame_{n-1} @ step 2, ..., frame_{n-9} @ step 10]
→ one forward pass → frame_{n-9} is finished (completed step 10)
→ shift window: drop frame_{n-9}, add new frame_{n+1} @ step 1
```

**R-CFG:** Cache unconditional CFG forward and reuse for m steps — eliminates one full DiT forward every m steps:
```cuda
__device__ __half2* uncond_cache;   // persistent buffer
if (step % m == 0) {
    uncond_cache = run_dit(noisy, null_cond);
}
output = cond_out + cfg_scale * (cond_out - uncond_cache);
```
K16 CUDA Graph captures steady-state pipeline as a replay-able graph.

**Speedup:** R-CFG alone: **2.05×**. Full StreamDiffusion (with TensorRT): 91 FPS on RTX 4090.
**Test on:** Oasis (10-step), CogVideoX, DiT-XL/2 — throughput benchmark (frames/second)

---

### E5 — `nfd_speculative_action_repeat`
**Source:** arXiv 2506.01380, Jun 2025

**Insight (world-model-specific):** Game actions repeat ~80% of the time. Generate N=2-3 frames speculatively assuming action repeats; discard on mismatch.

**Kernel:**
```cuda
__global__ void action_verify(
    const float* pred_actions, const float* true_actions,
    int* accept_count, int N, int D, float threshold) {
    for (int i = 0; i < N; i++) {
        float mse = 0.0f;
        for (int d = 0; d < D; d++) {
            float diff = pred_actions[i*D+d] - true_actions[i*D+d];
            mse += diff * diff;
        }
        if (mse / D > threshold) { *accept_count = i; return; }
    }
    *accept_count = N;
}
// KV rollback: decrement frame_counter pointer — no memcpy needed
```
**Speedup:** 31.14 FPS on A100. 1.19–1.26× from speculation alone. Composable with all other kernels.
**Test on:** Oasis (action-conditioned), MG2 — test with replay of recorded game sessions

---

## Upgrade Map: Existing K1-K18 Critical Fixes

| Original Kernel | Upgrade | Source | Impact |
|----------------|---------|--------|--------|
| K2 `int4_fp8_attention` | Fix FP22 accumulator → two-level accumulation | SageAttention2 | **Correctness** — silent degradation at long sequences |
| K2 | Per-thread INT4 outlier smoothing: `Q = q̄ + γ(Q)` | SageAttention2 | +15-20% quality at same bit-width |
| K3 `fa3_warp_specialized_attn` | Add block-pair validity table: STA window + A2 radial decay | A1 + A2 | Skip rate 50% → 68-72% |
| K3 | LiteAttention Skip-List in persistent SMEM, propagate across steps | C5 | +34% tile-level skip within non-cached layers |
| K4 `tiered_kv_quant` | Temporal residual smoothing before quant: `K_smooth = K - mean_K` | B3 | Enables INT2 (7× vs 4×) |
| K5 `h2o_token_eviction` | Replace post-RoPE with TriAttention pre-RoPE center scoring | A4 | 10.7× compression at full accuracy vs 5× at -2% |
| K7/K8 quantize | Timestep-group schedule: INT4/FP8/FP16 by noise level | D4 | +20-30% quality at same average bit-width |
| M3 `block_causal_attention` | Extend validity table with radial decay bounds | A2 | 50% → 70% skip rate |
| M7 `seacache_step_similarity` | Replace cosine similarity with AB-predictor | C1 | 5.54× FLUX vs 1.49× SeaCache |

---

## Ablation Study Design

### Ablation 1: Sparse Attention Stack (A1 → A1+A2 → A1+A2+C5)
**Model:** HunyuanVideo or Oasis temporal attention
**Measure:** Attention kernel time (µs), FID/SSIM, NSight tensor-core utilization %
```
Baseline:  FA2 full attention
+A1 only:  STA tiled window (no radial mask)
+A2:       STA + radial decay static mask
+C5:       STA + radial decay + LiteAttention persistent skip-list
```

### Ablation 2: KV Compression Stack (A4 → A4+B3 → A4+B3+B1)
**Model:** Oasis (32-frame AR generation)
**Measure:** Peak KV memory (GB), per-frame latency (ms), SSIM vs baseline
```
Baseline:  Full FP16 KV cache
+A4:       TriAttention scoring → evict bottom 50% KV
+B3:       A4 + temporal residual INT2 for evicted-not-evicted mid tier
+B1:       A4+B3 + Ca2-VDM prefix reuse (skip 9/10 prefix recomputes)
```

### Ablation 3: Step Caching (C2 → C2+C1 → C2+C1+C3)
**Model:** FLUX.1-schnell or DiT-XL/2
**Measure:** Steps actually computed (%), wall-clock time (ms), FID
```
Baseline:  All 10 steps computed
+C2:       TeaCache embedding distance (adaptive step skip)
+C1:       PrediT AB predictor for skipped steps (predict instead of reuse)
+C3:       ClusCa cluster token reduction on non-skipped steps
```

### Ablation 4: Quantization Precision Stack
**Model:** DiT-XL/2 (known FID baseline), Oasis (SSIM)
**Measure:** FID/SSIM vs bit-width, throughput (tokens/sec), memory (GB)
```
Baseline:  FP16 attention + FP16 weights
+D5:       Per-token dynamic INT8 activations
+D4:       Timestep-adaptive schedule (INT4 high-noise, FP8 low-noise)
+D1:       Full SageAttention2 (INT4 Q/K, FP8 P·V, two-level accumulator)
+D3:       SVDQuant W4A4 (offline SVD decomposition)
```

### Ablation 5: Full Stack — Oasis End-to-End
**Model:** Open-Oasis 500M, 20fps generation
**Measure:** FPS, SSIM, peak memory, NSight roofline position
```
Baseline:       PyTorch FA2 full attention, FP16
+Sparse:        A1+A2 sparse attention
+KV:            A4+B3+B1 KV compression
+StepCache:     C2+C1 adaptive step caching
+Quantization:  D1+D4 adaptive precision
+Pipeline:      E4 stagger + E5 speculative action
COMBINED:       All above
```

---

## Benchmark Metrics Per Kernel

| Metric | Tool | Kernel | Baseline comparison |
|--------|------|--------|-------------------|
| Kernel time (µs) | `torch.cuda.Event` timing | All | PyTorch/FA2 equivalent |
| HBM bytes read/written | `ncu dram__bytes_read.sum` | All | Bytes per call |
| Tensor core utilization % | `ncu sm__pipe_tensor_cycles_active` | D1, D3, D4 | FA3 baseline |
| L2 hit rate | `ncu l2_global_hit_rate` | A3, B1, B3, D4 | Uncached baseline |
| Attention sparsity achieved | tile skip count / total tiles | A1, A2, C5 | 0% for FA2 |
| KV memory peak (GB) | `torch.cuda.max_memory_allocated` | A3, A4, B1, B3 | Full FP16 cache |
| FID (image DiTs) | ADM evaluator | C1, C2, C3, D4 | FP16 baseline |
| SSIM (video DiTs) | `torchmetrics.SSIM` | B1, B3, A2 | FP16 baseline |
| FPS (world models) | wall-clock / frame count | E4, E5, B1 | Sequential baseline |

---

## Implementation Priority

### Phase 1 — Foundation (implement these first, validate each independently)
1. **D1** FP22 accumulator fix — correctness baseline for all FP8 work
2. **A1** Sliding tile attention — flagship sparse kernel, most H100-native
3. **B1** Ca2-VDM prefix KV — pure Python change, immediate 2.5× win
4. **C1** PrediT AB predictor — 20-line kernel, fastest path to step-caching result
5. **C2** TeaCache embedding distance — composable dispatch layer

### Phase 2 — KV Compression + Calibrated Sparse
6. **A4** TriAttention pre-RoPE scoring — replace H2O scoring in K5
7. **A2** Radial decay static mask — calibrate then compose with A1
8. **B3** QVG temporal residual INT2 — upgrade K4

### Phase 3 — Full Stack + Ablations
9. **C3** ClusCa cluster sparse attention
10. **C5** LiteAttention Skip-List persistent
11. **D4** Timestep-adaptive quantization schedule
12. **D5** Per-token dynamic quantization
13. **D3** SVDQuant INT4+FP16 joint kernel
14. **E4** StreamDiffusion stagger pipeline
15. **E5** NFD+ speculative action repeat

---

## Papers Reference

| ID | Kernel | Paper | Venue | arXiv |
|----|--------|-------|-------|-------|
| A1 | sliding_tile_attention | Fast Video Generation with Sliding Tile Attention | ICML 2025 | 2502.04507 |
| A2 | radial_decay_static_mask | Radial Attention | NeurIPS 2025 | 2506.19852 |
| A3 | tempache_ann_kv_dedup | Fast AR Video Diffusion with Temporal Cache | Feb 2026 | 2602.01801 |
| A4 | triattention_prerope_score | TriAttention | Apr 2026 | 2604.04921 |
| B1 | ca2vdm_invariant_kv_prefix | Ca2-VDM Causal Cross-View Attention | ICML 2025 | 2411.16375 |
| B3 | qvg_temporal_residual_kv | Quant VideoGen 2-Bit KV Cache | Feb 2026 | 2602.02958 |
| C1 | predi_t_ab_feature_predict | PrediT Adams-Bashforth Predictor | Feb 2026 | 2602.18093 |
| C2 | tea_cache_emb_distance | TeaCache Timestep-Embedding Distance | CVPR 2025 | 2411.19108 |
| C3 | clusca_cluster_sparse_attn | ClusCa Cluster Caching | ACM MM 2025 | 2509.10312 |
| C5 | liteattn_skiplist_persistent | LiteAttention Skip-List Persistent | Nov 2025 | 2511.11062 |
| D1 | sage_attn2_fp22_fix | SageAttention2 FP22 Accumulator Fix | ICML 2025 | 2411.10958 |
| D3 | svdquant_joint_int4_lora | SVDQuant / Nunchaku W4A4 | Nov 2024 | 2411.05007 |
| D4 | taqdit_emb_quant_schedule | TaQ-DiT Timestep-Aware Quantization | 2024 | 2411.14172 |
| D5 | vidit_q_per_token_dynamic | ViDiT-Q Per-Token Dynamic Quantization | ICLR 2025 | 2406.02540 |
| E4 | stream_diffusion_stagger | StreamDiffusion Stagger Pipeline | ICCV 2025 | 2312.12491 |
| E5 | nfd_speculative_action | NFD+ Next-Frame Diffusion 30+ FPS | Jun 2025 | 2506.01380 |
