# WorldServe Benchmark Plan — Per-Model

Each section below is **self-contained**: read only the section for the model you are benchmarking. Every section lists (a) what the model is, (b) the continuous-frame dataset(s) we will feed it, (c) the optimizations expected to deliver measurable speedup or memory reduction, and (d) the ones not worth running and why.

Hardware: NVIDIA H100 80GB SXM. All numbers below are projections grounded in our own measured ablations or published benchmarks; verify on hardware before reporting.

---

## 1. Oasis-500M (`Etched/oasis-500m`)

### Model snapshot
- DiT-S/2 + axial attention, 500M params, MIT
- 24 SpatioTemporalDiTBlock — `SpatialAxialAttention` (144 tokens) + `TemporalAxialAttention` (32-frame history)
- VAE: ViT-L-20-shallow (360×640 → 18×32 latent, 16 channels)
- Sampler: 10-step v-prediction DDIM, sigmoid β-schedule
- Conditioning: 25-dim Minecraft action (WASD + mouse deltas + buttons) → adaLN
- KV cache: ~30 MB/frame; trivial scale

### Continuous frame datasets
| Dataset | Format | What we use | Wiring |
|---|---|---|---|
| **open-oasis sample_data** (in repo) | 1 prompt frame + 1 action stream `.pt` | Already used in `oasis_quality_eval_modal.py`. Single 32-frame rollout — too short for cache-aging studies | Already loaded via `utils.load_prompt`, `utils.load_actions` |
| **OpenAI VPT contractor data** | 4.7K hrs Minecraft @ 20 Hz with synced keyboard+mouse | Subset 10 sessions × 100 frames. Map raw VPT actions → Oasis 25-dim action vocabulary (WASD, jump, attack, camera dx/dy, hotbar) | `huggingface.co/datasets/aliangdw/VPT`; convert with VPT's `actions_mapping.py` |
| **MineRL v1.0 / BASALT** | Labeled survival/build-house episodes, action-frame aligned | Cleaner than VPT but smaller. Use as quality-stress dataset for long rollouts | `minerl.io` |
| **Decart / Etched public Oasis demos** | Short Minecraft clips Etched released with the paper | First-frame prompt only, free-roll generation — no ground-truth rollout for PSNR | Dropbox links in Etched repo |

**Recommended primary eval:** VPT 10 sequences × 100 frames each. Long enough to measure cache-aging (FP16 → FP8 → INT4 thresholds at frames 30, 90), short enough to fit each rollout in <60s.

### Optimizations expected to work

| Technique | Expected impact | Justification |
|---|---|---|
| `torch.compile(mode="max-autotune")` | **2.0×–2.2× FPS** | Already measured 2.15×. Eliminates Python dispatch over 24 blocks × 10 steps × N frames |
| TaylorSeer (feature prediction, AB-2) | **1.4×–1.6×** | Measured 1.53×. 144/464 blocks predicted (31% skip) |
| SeaCache step similarity | **1.4×–1.5×** | Measured 1.49×. Skips middle 60% of blocks at steps 4–8 (cos_sim > 0.92) |
| INT4 weight-only quant (torchao) | **1.3×–1.6×** | Batch=1 DiT is bandwidth-bound (AI ≈ 1 FLOP/byte ≪ 590 ridge); weight BW dominates |
| MagCache | **1.2×–1.4×** | Step-caching family, complements TaylorSeer |
| FlashAttention-3 (1D for axial) | **1.05×–1.15×** | Marginal at 144 tokens — kernel launch overhead dominates |

**Stacked target:** ~30–60 FPS (vs 22 baseline). Compose order: compile → step-caching → INT4-wo. Don't double-stack two step-caching variants — they fight each other.

### Optimizations NOT expected to pay off
| Technique | Why not |
|---|---|
| KV-Quant Speculative Decoding | **Measured 0.68×.** 144-token cache too small; wire-hook overhead and 2× draft+verify forwards dominate |
| Tiered KV quantization (FP16→FP8→INT4) | Cache is 30 MB total — saving 70% of 30 MB is irrelevant |
| Spatial token eviction (H2O, SnapKV) | Only 144 spatial tokens — eviction granularity too coarse |
| 2-bit KV (Quant VideoGen) | Same — savings invisible at this scale |
| Sparse attention (STA, SVG2) | Spatial axis already factorized; 144 tokens below H100 sparse-attn threshold |
| TreeActionBrancher | **Measured 0.28×.** Sequential single-GPU execution, no parallelism gain |
| ParaDiGMS | **Measured 0.60×.** Picard iteration overhead > step-batching benefit |

### Quality metrics
- Per-frame latent MSE, pixel PSNR, pixel L1 vs. baseline at same seed (already implemented)
- FVD over the full 100-frame rollout vs. VPT ground-truth
- Action consistency: re-encode generated frame, compare action embedding to input action

### Suggested experiments (priority order)
1. Wire VPT loader into `oasis_quality_eval_modal.py` → 10×100 frame baseline
2. Stack: baseline → +compile → +TaylorSeer → +INT4-wo → measure FPS, PSNR, FVD at each step
3. Long-rollout sweep: 100, 200, 500 frames — confirm step-caching doesn't degrade after frame 50

---

## 2. Matrix-Game 2.0 1.8B (`Skywork/Matrix-Game-2.0`)

### Model snapshot
- Wan2.1-derived DiT, 1.8B params, MIT
- 30 WanAttentionBlock, hidden 1536, 12 heads × 128 head_dim
- 880 tokens/latent frame at 352×640; 6-frame rolling KV → 5,280 KV tokens
- Sampler: 3-step flow matching (ODE) per latent block
- Conditioning: mouse MLP (Linear(1538, 1536), non-power-of-2) + 4-token keyboard cross-attention at every layer
- KV cache: ~118 MB/frame — real KV optimization target

### Continuous frame datasets
| Dataset | Format | What we use | Wiring |
|---|---|---|---|
| **MG2 official demo set** (Skywork repo) | Short Minecraft + GTA gameplay clips with action logs | First-frame prompt + 60-frame continuation, action stream from clip | Skywork's `inference_demo.py` reference |
| **VPT (Minecraft subset)** | Same as Oasis | Re-use the VPT subset; project Oasis 25-dim → MG2 mouse+keyboard format | Convert `(dx, dy)` mouse deltas, keep WASD bits |
| **GTA-V clip dataset** (community-curated, e.g., `GTA-V-gameplay` on HF) | Driving + on-foot footage with synthetic action labels | First-frame prompt + driving continuation. Tests cross-game generalization | HF mirror or GamePhysics dataset |
| **OpenAI Diamond Atari frames** (downsample to 352×640) | Atari-style action+frame pairs | Sanity check only — out-of-domain | Public release |

**Recommended primary eval:** MG2 demo set (matches training domain) + VPT subset (cross-checks vs. Oasis).

### Optimizations expected to work

| Technique | Expected impact | Justification |
|---|---|---|
| Tiered KV quantization (FP16 0–30 / FP8 30–90 / INT4 90+) | **2.5×–4× KV mem, 1.3×–1.6× latency** | 118 MB/frame × 6 frames = 708 MB cache; meaningful absolute savings |
| Spatial token eviction (4×4 block, low-attn drop) | **1.5×–2× KV mem** with <0.3 dB PSNR drop | 880 spatial tokens — H2O/SnapKV granularity actually works here |
| Temporal token merging (ToMe along time, τ=0.95) | **2.5×–3.5× KV mem** | Measured 2.90× in earlier ablation; static-scene tokens (sky, ground) merge readily |
| Block-causal sparse attention (M3) | **1.8×–2.1×** attention | 49.7% of `(q_frame, kv_frame)` tile pairs are zero — skippable in FA3 tile loop |
| SVG2 sparse attention (M2 head classification) | **1.4×–1.7×** | Per-head spatial vs temporal patterns; KV layout transpose recovers L2 hits (31% → 96%) |
| SageAttention2 (INT4 Q/K + FP8 P·V) | **1.5×–1.8×** attention | 5,280 KV × 2,640 Q is firmly in attention-bound regime |
| `torch.compile` | **1.4×–1.7×** | Measured ~baseline due to CausalWanModel dispatch quirks; needs partial compile around block forwards |
| Action injection fusion (M5: mouse MLP + keyboard cross-attn) | **2.0×–2.4× on action path** | 60 launches/forward → 1 fused kernel; 280 µs launch overhead eliminated |
| 3D causal VAE encode (M6, sliding-window) | **3×–4× VAE encode** | Only 1–2 new frames/step but naively re-encodes full 24-frame window |
| INT4 weight-only quant | **1.3×–1.5×** | 1.8B model is BW-bound at batch=1 |

**Stacked target:** ~10–14 FPS (vs 4.4 baseline). Compose: KV compression first (memory headroom), then sparse attention, then compile.

### Optimizations NOT expected to pay off
| Technique | Why not |
|---|---|
| Step caching (TaylorSeer / SeaCache / TeaCache / MagCache) | **Measured 0.96× (MixCache).** Only 3 ODE steps per block — caching schedule shorter than warmup. Rule: step-caching needs ≥10 steps |
| LayerSkip Speculative Decoding | **Measured 0.99×.** CausalWanModel blocks need rotary embeddings + cross-attn context — cannot split at block level |
| KV-Quant Speculative Decoding | Same draft-overhead pathology as Oasis but worse — 30-block draft ≈ verify cost |
| ParaDiGMS | 3 ODE steps too few to overlap |
| DPM-Solver++ | Already a 3-step ODE — no headroom to reduce further |

### Quality metrics
- FVD over 60-frame rollout vs. MG2 demo ground-truth
- Action-conditional consistency: feed same prompt + 2 different action sequences, measure divergence
- Memory footprint over generation horizon (peak GB at frames 30, 60, 100)

### Suggested experiments (priority order)
1. Re-implement model loader (currently absent from `worldserve/models/`)
2. Verify baseline reproduces 4.4 FPS — confirm MG2 demo rollout works
3. KV compression sweep (temporal merge → progressive quant → spatial eviction); each in isolation, then combined
4. Sparse attention overlay (block-causal + SVG2); verify <2% FVD regression
5. Action injection fusion (M5) — measurable as standalone block-level test

---

## 3. CogVideoX-5B (`zai-org/CogVideoX-5b`)

### Model snapshot
- 3D Full DiT, 5B params, Apache-2.0
- ~17,550 tokens at 480p × 49 frames; 51% attention fraction
- 50 denoising steps (default)
- T2V natively; for continuous-frame eval we use the I2V variant `THUDM/CogVideoX-5b-I2V`
- Attention: `CogVideoXAttnProcessor2_0` → direct `F.sdpa` (clean injection via `set_attn_processor()`)

### Continuous frame datasets
| Dataset | Format | What we use | Wiring |
|---|---|---|---|
| **OpenVid-1M** | 1M captioned video clips with first-frame and full-clip pairs | First-frame I2V prompt + 49-frame continuation; ground truth from clip | HF: `nkp37/OpenVid-1M` |
| **Panda-70M** | Captioned web video | First-frame conditioning + caption | HF: `tsujuifu/panda-70m` |
| **VBench** | Standardized benchmark suite (16 dimensions: motion smoothness, subject consistency, etc.) | Run all VBench prompts; report per-dimension scores | github.com/Vchitect/VBench |
| **Kinetics-600** | 600 action classes, 10s clips | First-frame I2V; classify generated video to measure action faithfulness | HF mirror |
| **WebVid-10M** (deprecated but accessible) | 10M short web clips | Backup if other datasets are slow | HF mirrors |

**Recommended primary eval:** VBench (publication-comparable scores) + OpenVid first-frame continuation (continuous-frame fidelity).

### Optimizations expected to work

| Technique | Expected impact | Justification |
|---|---|---|
| SageAttention2 (`_sage_qk_int8_pv_fp8_cuda_sm90`) | **1.5×–1.8×** | 17.5K tokens × 51% attention fraction = direct bandwidth win |
| FlashAttention-3 | **1.4×–1.6×** | WGMMA + TMA over FA2 |
| TeaCache / TaylorSeer (`TaylorSeerCacheConfig`) | **2.0×–2.5×** | 50 denoising steps — large amortization headroom |
| `torch.compile(mode="max-autotune")` | **1.3×–1.5×** | Most CogVideoX blocks compile cleanly |
| INT4 weight-only quant (torchao) | **1.4×–1.7×** | 5B params, batch=1, BW-bound |
| Token merging (ToMe, K6) at r=25% | **1.4×–1.8×** | Vision tokens merge with near-zero quality drop |
| Tiered KV quantization | **1.2×–1.4×** | Per-frame KV is meaningful at 50-step×17.5K-token scale |
| STA sparse attention (3D, T=12 H=30 W=45 window=8) | **1.3×–1.6×** | Dense full attention has spatial-temporal locality |

**Stacked target:** baseline ~108s → ~35–50s/generation. Compose: SageAttn2 + TeaCache + INT4-wo + compile.

### Optimizations NOT expected to pay off
| Technique | Why not |
|---|---|
| LayerSkip Spec Decoding | Similar block-coupling issues; no published wins on CogVideoX |
| KV-Quant Spec Decoding | Diffusion model, not autoregressive — speculation paradigm doesn't apply cleanly |
| TreeActionBrancher | No discrete action conditioning |
| 3D causal VAE sliding window (M6) | Single-shot generation, no streaming |

### Quality metrics
- VBench 16-dim breakdown (publication-grade)
- FVD vs ground-truth (OpenVid clips)
- LPIPS first-frame consistency
- CLIP score for caption alignment

### Suggested experiments (priority order)
1. Add CogVideoX loader to `worldserve/models/` (mirror `oasis.py` structure)
2. Baseline VBench run (publishable number)
3. Single-kernel ablations: SageAttn2, TeaCache, INT4-wo separately
4. Stacked all-on run; compare against published CogVideoX-5B numbers
5. VBench delta sweep — confirm <2% regression on each dimension

---

## 4. Wan2.1-T2V-1.3B / 14B (`Wan-AI/Wan2.1-T2V-1.3B`, `Wan-AI/Wan2.1-T2V-14B`)

### Model snapshot
- 3D DiT, 1.3B (fast iteration) or 14B (publication-scale), Apache-2.0
- ~20K tokens (1.3B, 480p, 81 frames) / ~27K tokens (14B, 720p)
- ~50% attention fraction
- 50 denoising steps
- Attention: `WanAttnProcessor` + `dispatch_attention_fn` (universal diffusers backend injection)

### Continuous frame datasets
Same datasets as CogVideoX (OpenVid-1M first-frame, VBench, Panda-70M). Wan also supports I2V (`Wan2.1-I2V-14B-480P`) — use the I2V variant for continuous-frame benchmarks.

**Recommended primary eval:** VBench on 1.3B (fast, publishable) → repeat top-3 ablations on 14B (scale validation).

### Optimizations expected to work

| Technique | Expected impact | Justification |
|---|---|---|
| SageAttention2 via `set_attention_backend("_sage_qk_int8_pv_fp8_cuda_sm90")` | **1.6×–1.9×** on 14B | 27K tokens, 50% attention — strong absolute win |
| FlashAttention-3 (`_flash_3_hub`) | **1.5×–1.7×** | WGMMA over FA2 |
| TeaCache / TaylorSeer | **1.8×–2.4×** | 50 steps; all Wan blocks supported by `TaylorSeerCacheConfig` |
| `torch.compile` | **1.3×–1.5×** | Wan compiles cleanly with `mode="reduce-overhead"` |
| INT4 weight-only quant | **1.6×–2.0×** on 14B | 14B params at batch=1 → weight memory dominates |
| Token merging (ToMe) at r=25% | **1.4×–1.7×** | Same as CogVideoX |
| Tiered KV quantization | **1.2×–1.4×** | 27K tokens × 50 steps; KV mem footprint meaningful |
| Radial attention | **1.6×–1.9×** | Sliding-window pattern fits Wan's attention |

**Stacked target (14B, 480p, 81f):** baseline ~10 min → ~4–5 min. Compose: SageAttn2 + TeaCache + INT4-wo + compile.

### Optimizations NOT expected to pay off
| Technique | Why not |
|---|---|
| LayerSkip Spec / KV-Quant Spec | Diffusion model, no autoregressive draft path |
| TreeActionBrancher | No action conditioning |
| Step caching at very low step counts | If using a distilled 4-step variant, caching margin disappears |

### Quality metrics
- VBench full suite
- I2V continuation FVD on OpenVid clips
- Compare 1.3B vs 14B speedup ratios — confirms scaling behavior

### Suggested experiments (priority order)
1. 1.3B baseline + compile + SageAttn2 + TeaCache (fast iteration)
2. 14B baseline + same stack (publication numbers)
3. INT4-wo on 14B specifically (memory-bandwidth story is strongest at 14B)

---

## 5. HunyuanVideo 13B (`tencent/HunyuanVideo`)

### Model snapshot
- 3D DiT, 13B params, gated free
- ~115,200 tokens at 720p × 129 frames; **>80% attention fraction**
- 50 denoising steps
- `HunyuanVideoAttnProcessor2_0` + `dispatch_attention_fn`
- Largest absolute speedup target — every attention kernel µs compounds

### Continuous frame datasets
- **VBench** (priority: publication-comparable)
- **OpenVid-1M** first-frame I2V via the `HunyuanVideo-I2V` variant
- **Panda-70M**
- **HD-VG-130M** for high-res evaluation

**Recommended primary eval:** VBench at 720p × 129f. Single config, single number — cost prohibits sweeps.

### Optimizations expected to work

| Technique | Expected impact | Justification |
|---|---|---|
| FlashAttention-3 | **1.8×–2.1×** | 115K-token attention at FP16 — closest to 740 TFLOPS WGMMA peak |
| SageAttention2 (FP8) | **2.5×–3.0×** | INT4 Q/K + FP8 P·V at this scale; quality drop <0.5% in published benchmarks |
| TeaCache | **1.8×–2.2×** | 50 steps × 13B model — cache amortization peaks here |
| STA sparse attention (3D, window=8) | **1.5×–2.0×** | Spatial-temporal locality strongest at long rollouts |
| INT4 weight-only quant | **1.7×–2.1×** | 13B at batch=1 is weight-BW-bound |
| Tiered KV quantization (FP8 path) | **1.3×–1.5×** | 115K-token KV is largest in our model set |

**Stacked target:** baseline ~30 min → ~10–12 min/generation. Compose: SageAttn2 + STA + TeaCache + INT4-wo.

### Optimizations NOT expected to pay off
| Technique | Why not |
|---|---|
| `torch.compile(max-autotune)` | Compilation time often exceeds savings on 13B; use `reduce-overhead` instead |
| Token merging (ToMe) | Risk of FVD regression at 115K tokens; defer until other kernels stable |
| Spec decoding family | Diffusion — not applicable |

### Quality metrics
- VBench full suite (16 dimensions)
- FVD on subset of HD-VG-130M
- Per-frame LPIPS on I2V continuation

### Suggested experiments (priority order)
1. Single baseline run (~30 min) — sanity check
2. SageAttn2 + TeaCache stack (most impactful pair)
3. Add INT4-wo if memory headroom needed
4. STA only if budget remains

---

## 6. Cosmos-1.0-Diffusion-7B-Video2World (`nvidia/Cosmos-1.0-Diffusion-7B-Video2World`)

### Model snapshot
- 3D DiT, 7B params, NVIDIA gated (free with HF account)
- ~56,320 tokens at 720p × 121 frames
- 50-step flow matching
- Cosmos Continuous Video Tokenizer (8× spatial, 4× temporal)
- Conditioning: T5-XXL text + video clip — true Video2World, not pure T2V
- Diffusers port supports `set_attention_backend()`

### Continuous frame datasets
| Dataset | Format | What we use | Wiring |
|---|---|---|---|
| **DROID** (Stanford) | 76K robot manipulation episodes, 564 scenes, multi-cam | First 2 seconds as conditioning video → predict next 5 seconds | `huggingface.co/datasets/KarlP/droid` |
| **OpenX-Embodiment / RT-X** | 1M+ robot trajectories, 22 embodiments | Cosmos's intended workload — Physical AI eval | `huggingface.co/datasets/jxu124/OpenX-Embodiment` |
| **AgiBot World Alpha** | Humanoid manipulation video | Frame-aligned actions for multi-task | HF: `agibot-world/AgiBotWorld-Alpha` |
| **nuScenes** | Driving with ego-motion as "action" | First N frames + ego pose → predict continuation | nuscenes.org |
| **Waymo Open** | Driving with full sensor suite | Same — driving domain | waymo.com/open |
| **Cosmos eval split** (released with paper) | NVIDIA's curated physical-AI test clips | Reproduce paper's reported numbers | NVIDIA Cosmos repo |

**Recommended primary eval:** DROID (10 episodes × 121 frames) for robotics + nuScenes (10 clips × 121 frames) for driving. Two domains exercise different distribution.

### Optimizations expected to work

| Technique | Expected impact | Justification |
|---|---|---|
| SageAttention2 | **1.7×–2.1×** | 56K tokens, 50 steps — strong attention regime |
| FlashAttention-3 | **1.5×–1.8×** | Same |
| STA sparse attention (3D, T=31 H=60 W=80 window=8) | **1.6×–2.0×** | Already implemented; physical scenes have strong spatial locality |
| TeaCache | **1.7×–2.1×** | 50 steps amortize step-caching well |
| `torch.compile(reduce-overhead)` | **1.3×–1.5×** | Diffusers Cosmos port compiles cleanly |
| INT4 weight-only quant | **1.5×–1.7×** | 7B params at batch=1 |
| Tiered KV quantization | **1.2×–1.4×** | 56K-token KV is meaningful |

**Stacked target:** baseline ~18 min → ~7–9 min/generation. Compose: SageAttn2 + STA + TeaCache + INT4-wo.

### Optimizations NOT expected to pay off
| Technique | Why not |
|---|---|
| Token merging (ToMe) | Physical AI quality bar is high; defer |
| Spec decoding | Diffusion — not applicable |
| Action injection fusion (M5) | Cosmos conditions on text+video, not per-layer action vectors |

### Quality metrics
- FVD vs DROID held-out clips
- Physical consistency: optical flow continuity across generated frames
- VBench-Physical subset
- 3D-VAE round-trip PSNR for the Cosmos tokenizer

### Suggested experiments (priority order)
1. Baseline DROID 121-frame rollout (~18 min) — confirm reproduction
2. SageAttn2 + STA stack (already implemented in repo)
3. + TeaCache + INT4-wo for full stack
4. Domain-shift check: nuScenes 10 clips with same kernel stack

---

## 7. LTX-Video (`Lightricks/LTX-Video`)

### Model snapshot
- 3D DiT, ~2B params, "other" license but free
- ~14,400 tokens at 720p × 121 frames; ~45% attention fraction
- 20–50 denoising steps (configurable)
- `LTXVideoAttnProcessor` + `dispatch_attention_fn`
- Cleanest diffusers DiT — fast iteration target

### Continuous frame datasets
- **VBench** (primary)
- **OpenVid-1M** first-frame continuation
- **Distilled variant** `Lightricks/LTX-Video-0.9.7-distilled` for 4–8 step ablations

**Recommended primary eval:** VBench full suite + OpenVid 50-clip continuation. Cheap enough to sweep.

### Optimizations expected to work

| Technique | Expected impact | Justification |
|---|---|---|
| SageAttention2 | **1.5×–1.7×** | 14.4K tokens × 45% attention — solid attention regime |
| FlashAttention-3 | **1.4×–1.6×** | Standard |
| STA sparse attention | **1.4×–1.7×** | LTX has strong spatial structure |
| TeaCache | **1.8×–2.2×** | Per-step caching scales with denoising count |
| `torch.compile(max-autotune)` | **1.4×–1.6×** | LTX compiles cleanly |
| INT4 weight-only quant | **1.3×–1.5×** | 2B at batch=1 |
| Token merging (ToMe) at r=25% | **1.3×–1.5×** | Same |

**Stacked target:** baseline ~90s → ~30–40s/generation. Compose: SageAttn2 + TeaCache + compile.

### Optimizations NOT expected to pay off
| Technique | Why not |
|---|---|
| Spec decoding | Diffusion |
| Heavy KV compression | KV is small at 14.4K tokens × 50 steps |

### Quality metrics
- VBench 16 dimensions
- FVD on OpenVid continuation
- Comparison: distilled (8 steps) vs full (50 steps) under same kernel stack

### Suggested experiments (priority order)
1. Add LTX loader to `worldserve/models/`
2. Baseline VBench
3. SageAttn2 + TeaCache + compile (fastest path to publishable numbers)
4. Distilled variant: confirm step-caching dies at low step count (negative result is also a result)

---

## Cross-Model Summary

### Which optimization wins on which model
| Optimization | Oasis | MG2 | CogVideoX-5B | Wan2.1-14B | HunyuanVideo | Cosmos-7B | LTX |
|---|---|---|---|---|---|---|---|
| `torch.compile` | ✅ 2.15× | ⚠️ partial | ✅ | ✅ | ⚠️ slow | ✅ | ✅ |
| Step caching (TeaCache/TaylorSeer/SeaCache) | ✅ 1.5× | ❌ 3 steps | ✅ 2.5× | ✅ 2.4× | ✅ 2.2× | ✅ 2.1× | ✅ 2.2× |
| SageAttention2 | ⚠️ small | ✅ | ✅ | ✅ | ✅ best | ✅ | ✅ |
| FlashAttention-3 | ⚠️ small | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| STA sparse attention | ❌ 144 tok | ✅ block-causal | ✅ | ✅ | ✅ | ✅ | ✅ |
| Tiered KV quant | ❌ 30 MB | ✅ | ⚠️ marginal | ⚠️ marginal | ✅ | ⚠️ marginal | ❌ |
| Spatial KV eviction (H2O) | ❌ 144 tok | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Temporal token merge (time-axis ToMe) | ❌ 32 frames | ✅ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ |
| INT4 weight-only quant | ✅ | ✅ | ✅ | ✅ best | ✅ | ✅ | ✅ |
| Token merging ToMe (spatial r=25%) | ❌ | ⚠️ | ✅ | ✅ | ⚠️ risk | ⚠️ risk | ✅ |
| LayerSkip Spec | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| KV-Quant Spec | ❌ 0.68× | ❌ | ❌ diff | ❌ diff | ❌ diff | ❌ diff | ❌ diff |
| ParaDiGMS | ❌ 0.60× | ❌ 3 steps | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ |
| TreeActionBrancher | ❌ 0.28× | ⚠️ multi-GPU | N/A | N/A | N/A | N/A | N/A |
| 3D causal VAE sliding window | N/A | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Action injection fusion (M5) | ❌ | ✅ | N/A | N/A | N/A | N/A | N/A |

### Two model/dataset pairs to start with
1. **Oasis-500M + VPT 100-frame rollouts** — small token regime, step-caching + compile + INT4-wo story. Continuous AR with real action streams.
2. **CogVideoX-5B + OpenVid first-frame continuation** — large token regime, attention/KV/sparse story. Publishable VBench numbers.

Add **Cosmos-7B + DROID** as a third when the first two are stable — gives the physical-AI / video-conditioned narrative.

---

## GPU Budget — Ascending Order (run cheap models first)

Estimates are wall-clock H100 SXM time per **single benchmark** (1 warmup + 1 timed generation, weights cached after first download). "Full ablation budget" assumes ~6 kernel variants per model (baseline + 5 optimizations).

| # | Model | Tokens | Steps | 1 generation (baseline) | 1 ablation cell (warmup+timed) | Full ablation budget (~6 variants) | Risk |
|---|---|---|---|---|---|---|---|
| 1 | **Oasis-500M** | 144 spatial + 32 temporal | 10 DDIM | ~10–20s | ~3–5 min | **~20–30 min** | Low — already implemented |
| 2 | **Matrix-Game 2.0 1.8B** | 880 / latent frame, 6-frame KV | 3 ODE | ~15–30s | ~3–6 min | **~25–40 min** | Medium — loader needs work |
| 3 | **FLUX.1-schnell** (image, reference) | 16,384 | 4 (schnell) | ~30–60s | ~4–8 min | **~30–50 min** | Low — for kernel sanity-checks |
| 4 | **LTX-Video** | ~14,400 | 20–50 | ~60–120s | ~5–10 min | **~40–60 min** | Low — clean diffusers DiT |
| 5 | **CogVideoX-5B** | ~17,550 | 50 | ~108s ✅ confirmed | ~7–10 min | **~50–70 min** | Low — Apache-2.0, working |
| 6 | **Wan2.1-T2V-1.3B** | ~20,000 | 50 | ~2–4 min | ~8–12 min | **~60–80 min** | Low — Apache-2.0 |
| 7 | **Cosmos-1.0-Diffusion-7B-V2W** | ~56,320 | 50 | ~10–25 min | ~25–55 min | **~3–5 hours** | High — NVIDIA gated, long runs |
| 8 | **Wan2.1-T2V-14B** | ~27,000 | 50 | ~5–15 min | ~15–35 min | **~2–4 hours** | Low — Apache-2.0 but slow |
| 9 | **HunyuanVideo 13B** | ~115,200 | 50 | ~20–45 min | ~45–95 min | **~5–10 hours** | High — biggest model, single config only |

### Recommended GPU spend strategy

**Phase 1 — Cheap, high signal (~1 hour total):**
Run Oasis + MG2 baselines and full ablation stacks. These exercise the small-token / KV-cache stories and use almost no budget. If anything is broken in the harness, you find out here, not at hour 5 of HunyuanVideo.

**Phase 2 — Mid-cost, primary publishable numbers (~2 hours total):**
LTX + CogVideoX-5B. Both have VBench support and 50-step caching headroom — this is where the SageAttn2 / TeaCache / INT4-wo story gets its strongest curves.

**Phase 3 — Expensive, single-config validation (~4–6 hours total):**
Wan2.1-14B (one full ablation) + Cosmos-7B-V2W on DROID (one full ablation). These confirm the small-model wins scale up.

**Phase 4 — Skip or single-shot only (~1 hour):**
HunyuanVideo — one baseline + one stacked-optimization run only. Don't ablate individual kernels at 13B; the cost-to-information ratio is bad.

### Hard rules to stay within budget
- **Always** run the 15-frame config first (vs 121-frame) — same code path, ~8× faster, catches bugs cheaply.
- Cache model weights in a Modal volume — first download is one-time per model.
- Skip multi-iteration timed runs (`num_iters=3`) until kernels are stable. Use `num_iters=1` during development.
- Never run two ablations in the same container — the JIT cache pollution masks regressions.
- Defer HunyuanVideo until the kernel stack is locked on CogVideoX-5B. Same kernels, same diffusers backend — the speedup ratio transfers.

---

## Newly Discovered World Models on Hugging Face (Apr 2026 deep research)

35+ public checkpoints found beyond what was in `worldmodels.md`. Listed by impact tier — not all need to run; pick by what gap each fills in our optimization story.

### Tier S — Run these first (best fit for optimization story)

| Model | HF Repo | Params | Tokens / Ctx | Conditioning | License | Why it fits WorldServe |
|---|---|---|---|---|---|---|
| **MineWorld** | `microsoft/mineworld` | 300M / 700M / 1.2B | 256 tok/frame, 128-frame ctx | image + Minecraft action vector | MIT | Pure AR transformer with discrete action tokens — ideal KV-cache + spec-decoding target. Three sizes for scaling sweeps. |
| **Hunyuan-GameCraft-1.0** | `tencent/Hunyuan-GameCraft-1.0` | ~13B (HunyuanVideo backbone) + distilled | ~1.4K tok/frame, 33–129f ctx | image + KB+mouse → camera latent | Tencent custom (commercial OK) | Flagship interactive game DiT; distilled checkpoint runs faster. 488 likes — the most-watched 2026 release. |
| **HY-WorldPlay (HY-World 1.5)** | `tencent/HY-WorldPlay` | TBD | ~1.5K tok/frame, sliding ctx | image/text + KB+mouse | Tencent custom | **Three variants** (bidirectional, AR, distilled) shipped together — perfect controlled ablation for KV strategies. |
| **LongLive-1.3B** | `Efficient-Large-Model/LongLive-1.3B` | 1.3B (Wan2.1 base) | ~1.5K tok/frame, 240-sec ctx | text prompt switching mid-stream | CC-BY-NC-SA-4.0 | NVIDIA's frame-level AR with KV-recache + frame-level attention sink. **20.7 FPS on H100, INT8 path** — direct competitor to your optimization stack. |
| **Vista** | `OpenDriveLab/Vista` | 1.4B (SVD backbone) | ~1K tok/frame, 25f rollout | image + ego trajectory + low-level command | Apache-2.0 | Most-cited driving world model. UNet-based — different arch profile than DiT models. |
| **Yume-5B-720P** | `stdstu123/Yume-5B-720P` | ~5B (Wan2.2 base) | ~2K tok/frame, infinite ctx | image + quantized camera (KB-mapped) | Apache-2.0 | Has explicit memory module; benchmarks long-horizon. **PKU's flagship.** |
| **GR00T-N1.6-3B** | `nvidia/GR00T-N1.6-3B` | 3B | ~256 vis tok × 16 ctx | image + text + robot proprio + action | NVIDIA OSS | 33K downloads — the de-facto humanoid robot world model. Action-conditioned VLM. |
| **Dreamerv4-MC** | `IamCreateAI/Dreamerv4-MC` | 1.7B DiT + 430M tokenizer | latent ~256 tok/frame | image + Minecraft action | Apache-2.0 | Only public Dreamer 4 reproduction. Shortcut-forcing AR — different paradigm from Oasis. |

### Tier A — Strong candidates for ablation depth

| Model | HF Repo | Params | Domain | Why it matters |
|---|---|---|---|---|
| **Self-Forcing** | `gdhe17/Self-Forcing` | 1.3B (Wan2.1 base) | Open-domain | Apache-2.0, **142.7K downloads, 128 likes**. Canonical "train-test gap" AR baseline; many derivatives hang off it. |
| **RollingForcing** | `TencentARC/RollingForcing` | 1.3B | Open-domain | MIT — direct comparison point to Self-Forcing for KV-rolling strategies |
| **SANA-Video LongLive** | `Efficient-Large-Model/SANA-Video_2B_480p_LongLive_diffusers` | 2B | Open-domain | **Linear-attention** world model variant — very different KV profile, exposes whether your kernels generalize beyond softmax attention |
| **MagicDriveDiT** | `flymin/MagicDriveDiT-stage3-40k-ft` | DiT 6-view | Driving (nuScenes) | Multi-view 60-frame generation — heaviest KV-cache target (~3K tok/view × 6 views × 60 frames). GPL-3.0 |
| **GigaWorld-0-GR1-2B** | `open-gigaai/GigaWorld-0-Video-GR1-2b` | 2B | Robot (GR1) | Apache-2.0, action-conditioned, 93×480×768 (~1.2K tok/frame) |
| **GigaWorld-0-Pretrain-2B** | `open-gigaai/GigaWorld-0-Video-Pretrain-2b` | 2B | Embodied/robot | Apache-2.0, base ckpt for ablations |
| **WorldVLA** | `Alibaba-DAMO-Academy/WorldVLA` | 7B (Chameleon base) | Robot | Pure AR, unified action+image tokens — discrete VQ tokens (~1K/frame). Apache-2.0 |
| **AetherV1** | `AetherWorldModel/AetherV1` | 5B (CogVideoX base) | Sim-to-real | MIT, geometry-aware unified world model. Same backbone as CogVideoX-5B — kernel transfer is direct |
| **AdaWorld** | `Little-Podi/AdaWorld` | small | Cross-env | Apache-2.0, AR transformer + latent action autoencoder. Small enough for fast iteration on action-token KV-cache study |
| **Yume-I2V-540P** | `stdstu123/Yume-I2V-540P` | ~5B | Sekai | Apache-2.0, smaller resolution variant — easier to bench than 720P |
| **Cosmos-Predict2-14B-GR00T-Dreams-GR1** | `nvidia/Cosmos-Predict2-14B-Sample-GR00T-Dreams-GR1` | 14B | Robot dreams | NVIDIA gated-free, action-conditioned, **121-frame ctx, ~3K tok/frame** — biggest KV stress test in this list |
| **Cosmos-Predict2-14B-GR00T-Dreams-DROID** | `nvidia/Cosmos-Predict2-14B-Sample-GR00T-Dreams-DROID` | 14B | Robot (DROID) | Same as above, DROID arms |
| **Waypoint-1.5-1B** | `Overworld/Waypoint-1.5-1B` | 1B | Open-domain | Real-time AR (>16 FPS, <1s latency). The "small action-conditioned WM" reference point |

### Tier B — Useful for breadth but lower priority

| Model | HF Repo | Notes |
|---|---|---|
| **MagicDrive-t-60f** | `flymin/MagicDrive-t-60f-224x400-80k` | UNet variant of MagicDrive — UNet vs DiT comparison |
| **WorldVLA-LIBERO-10-256** | `jcenaa/WorldVLA-ActionModel-LIBERO-10-256` | Small-context (256-tok) finetune for fast tests |
| **RT-1 World Model** | `thuml/rt1-world-model-multi-step-base` (+rlvr/single-step) | 127M Llama-style — tiny, lots of variants |
| **Yume-1.5 MLX** | `ckurasek/Yume-1.5-5B-720P-MLX` | Apple Silicon port — only useful as quantized reference |
| **GenEx World-Initializer** | `genex-world/World-Initializer-image-to-panorama` | FLUX.1-Fill finetune; first stage of GenEx WM |
| **Ctrl-World** | `yjguo/Ctrl-World` | AR DiT, control-conditioned |
| **HY-World-2.0** | `tencent/HY-World-2.0` | Successor to HY-WorldPlay; **monitor for full ckpt drop** |
| **GameFactory** | `KlingTeam/GameFactory-Dataset` | Kuaishou Kling — code+dataset live, model ckpt promised |
| **Cosmos-Predict2-2B-Video2World** | `nvidia/Cosmos-Predict2-2B-Video2World` | 2B sibling of 7B Cosmos — cheaper baseline |
| **Cosmos-Predict2.5-2B** | `nvidia/Cosmos-Predict2.5-2B` | 619K downloads — highest-traffic Cosmos variant. T2V/I2V/V2V unified, NATTEN+FA3 baseline |
| **Cosmos-Predict1 family** (4B/5B/7B/12B/13B/14B) | `nvidia/Cosmos-Predict1-*` | Full Cosmos-1 zoo. AV-Multiview + WorldInterpolator are uniquely interesting |
| **Matrix-Game-3.0** | `Skywork/Matrix-Game-3.0` | Apache-2.0, Wan2.2-TI2V-5B base — direct upgrade target over MG2 |
| **Matrix-Game v1** | `Skywork/Matrix-Game` | MIT, predecessor — useful for ablation lineage |
| **FastVideo MG2 ports** | `FastVideo/Matrix-Game-2.0-{Base,GTA,TempleRun}-Diffusers` | Cleaner diffusers ports than upstream Skywork; **GTA + TempleRun heads** are new domains |

### Tier C — Skip for now (paper-only or unclear weights)

PAN (2511.09057), RELIC (2512.04040), VideoSSM (2512.04519), BlockVid (2511.22973), Deep Forcing (2512.05081), LIVE (2602.03747), Astra (2512.08931 — code only), DrivingWorld (2412.19505 — code only), Drive-WM, DriveDreamer/DriveDreamer-2, GAIA-1/2 (Wayve closed), MIDAS (2508.19320), TeleWorld, StableWorld, LiveWorld, UCM, Memorize-and-Generate, ViD-GPT, MineWorld-RL, Hunyuan-GameCraft-2, Genie. **Monitor — many will drop weights.**

### Recommended 8-model bench-set (covers all WorldServe optimization categories)

Pick these to maximize coverage with minimum H100 hours:

1. **MineWorld 700M** — small AR + discrete action tokens (KV-cache, spec-decoding gold)
2. **Oasis-500M** (already have) — axial attention baseline
3. **LongLive-1.3B** — direct competitor; their reported 20.7 FPS sets the bar
4. **Self-Forcing** — most-downloaded open AR; community comparison point
5. **Vista** — driving domain, UNet arch — checks generalization beyond DiT
6. **GR00T-N1.6-3B** — robot domain, VLM action head
7. **MagicDriveDiT** — multi-view 6-camera DiT — biggest KV stress test under 14B
8. **Yume-5B-720P** — explicit memory module + camera conditioning, long-horizon

This 8-model set spans: DiT × UNet × pure AR transformer; Minecraft × driving × robot × game × open-domain; discrete action × camera trajectory × keyboard+mouse × continuous robot proprio. Every WorldServe kernel finds at least one model where it wins.

### Optimization-method papers to read before re-running ours

- **TempCache + AnnCA + AnnSA** (2602.01801) — temporal cache compression + sparse attention for AR video diffusion. Direct competitor; combine or compare.
- **Salience-based KV policy** (2601.21896) — past+future-informed eviction
- **KV Cache Quantization for Self-Forcing — 33-method empirical study** (2603.27469) — read this *before* publishing your KV ablations.
- **Quant VideoGen** (2602.02958) — 2-bit KV quant; you already cite this
- **Deep Forcing** (2512.05081) — Deep Sink + Participative Compression for KV pruning

---

## Conditioning Datasets — Per-Model Mapping (verified Apr 2026)

Every entry below names: (a) the **primary dataset** that exactly matches the model's training distribution and conditioning format (highest-fidelity quality eval), (b) **backups** for similar-domain alternatives, (c) the **specific subset** to download to fit under ~100 GB and produce statistically meaningful FVD. All HF URLs verified via `hub_repo_search` / `paper_search`. Entries marked "off-HF" require download from project sites (URLs given inline).

### Tier S — Game / action-conditioned world models

| Model | Primary dataset | Backup | Size / License | Conditioning compatibility | Recommended bench subset |
|---|---|---|---|---|---|
| **microsoft/mineworld** | **OpenAI VPT contractor data** (off-HF: `github.com/openai/Video-Pre-Training` → `contractor_recordings.json` index → S3 mp4+jsonl) | `Skywork/Matrix-Game-MC` (gated request); MineRL v1.0 (`minerl.io/dataset`) | VPT: ~270K hr 20 Hz / MIT | Direct match — MineWorld trained on VPT-style 20 Hz `(frame, action_dict)` pairs; same KB+mouse_dx/dy schema | First 100 contractor episodes "9.x" (cave/forest, ~80 GB at 360p) → FVD on 16-frame windows |
| **tencent/Hunyuan-GameCraft-1.0** | No public training set (proprietary 1M+ AAA gameplay) | `Lixsp11/Sekai-Project` (FPV walking with quantized cameras); `nkp37/OpenVid-1M` for free-form text eval | Sekai 5K hr / non-commercial research | Approximate — GameCraft uses unified KB+mouse → camera embedding; Sekai's 8-token directional must remap to GameCraft's continuous repr | Sekai walking subset, 200 clips × 5s (~30 GB) |
| **tencent/HY-WorldPlay** | Same gap as GameCraft (proprietary) | `Lixsp11/Sekai-Project`; `nkp37/OpenVid-1M` | Sekai 5K hr | Conditioning mismatch — HY-WorldPlay expects KB+mouse; use Sekai's camera-trajectory JSON as proxy KB sequence | Sekai 100 FPV clips for I2V continuation |
| **Skywork/Matrix-Game-3.0** | **Matrix-Game-MC** (referenced in arxiv 2506.18701, gated request to authors) | `Lixsp11/Sekai-Project`; VPT contractor for action-labeled | MC: 2,700 hr unlabeled + 1,000 hr labeled / gated | Direct match for Minecraft (KB+mouse JSON identical to VPT format) | Request 50 labeled clips; fall back to 50 VPT contractor episodes |
| **Skywork/Matrix-Game-2.0** | MG2 demo set (Skywork repo `inference_demo.py`) | VPT contractor; FastVideo MG2-GTA / -TempleRun port demo data | small — domain match for MG2 | KB+mouse, packed into MG2's mouse-MLP+keyboard-cross-attn format | MG2 demo + 50 VPT clips with action remap |
| **IamCreateAI/Dreamerv4-MC** | **OpenAI VPT contractor data** | `Skywork/Matrix-Game-MC`; MineRL v1.0 | Same as MineWorld row | Dreamer-4 paper trains on VPT 20 Hz directly; identical KB+mouse schema | Same 100 VPT episodes as MineWorld (paired comparison) |
| **Etched/oasis-500m** | Local `sample_data/` (already in repo) | **OpenAI VPT contractor data** for held-out conditioning | n/a + VPT 270K hr | Oasis 25-dim vocab is a remap of VPT KB buttons + mouse-dx/dy 8-bin quantization → **needs `vpt_to_oasis_25d` adapter** | sample_data/ first 200 frames + 50 fresh VPT clips for cross-eval |

### Tier S — Open-domain AR (text / prompt switching)

| Model | Primary dataset | Backup | Size / License | Conditioning compatibility | Recommended bench subset |
|---|---|---|---|---|---|
| **Efficient-Large-Model/LongLive-1.3B** | **`Vchitect/VBench_sampled_video`** + VBench-Long prompt set (gated) | LongLive's author-released "interactive prompt sequence" set (`github.com/NVlabs/LongLive`, 100 multi-prompt sequences); `nkp37/OpenVid-1M` | VBench: 1K–10K clips / MIT (gated); OpenVid: 1M / CC-BY-4.0 | Direct — LongLive's reported numbers use VBench short+long; for prompt-switching specifically use the NVlabs/LongLive 100-sequence set | 50 VBench prompts × 240s (baseline) + 30 multi-prompt switching sequences (streaming) |
| **Efficient-Large-Model/SANA-Video LongLive** | Same as LongLive | Same | Same | Same — SANA-LongLive shares prompt-switch eval protocol | Same 30 switching sequences |
| **gdhe17/Self-Forcing** | `nkp37/OpenVid-1M` | `Vchitect/VBench_sampled_video`; `tianweiy/CausVid-VBench` | OpenVid 1M / CC-BY-4.0 | Pure text→video, no action conditioning | VBench standard 946 prompts (paired with sampled videos for FVD-vs-real) |
| **TencentARC/RollingForcing** | Same as Self-Forcing | Same | Same | Pure text-conditioned | VBench-Long 100 prompts |

### Tier S — Driving world models

| Model | Primary dataset | Backup | Size / License | Conditioning compatibility | Recommended bench subset |
|---|---|---|---|---|---|
| **OpenDriveLab/Vista** | **OpenDV-YouTube** (clip list at `github.com/OpenDriveLab/DriveAGI`) + **`OpenDriveLab/OpenDV-YouTube-Language`** annotations | nuScenes (motional.com gated, ~360 GB); Waymo Open (waymo.com gated) | OpenDV 1,747 hr / CC-BY-NC-SA-4.0 | Direct — Vista paper pretrains on OpenDV + evals on nuScenes; OpenDV-Language JSON has image+trajectory+command | OpenDV first 500 clips (~80 GB at 512p) — has both ego command annotations and ground-truth future frames |
| **flymin/MagicDriveDiT-stage3-40k-ft** | **nuScenes v1.0 trainval** (motional.com gated; 6-cam + 3D bbox + map + ego pose) | `orangewen/Gen-nuScenes` HF mirror (~1M imgs, partial annotations) | nuScenes 1000 scenes / 5.5 hr / CC-BY-NC-SA-4.0 | Exact — MagicDriveDiT's conditioning *is* the nuScenes annotation format | nuScenes val first 150 scenes (~20 GB) → 6,019 6-camera samples for FVD/multi-view consistency |

### Tier S — Robot world models

| Model | Primary dataset | Backup | Size / License | Conditioning compatibility | Recommended bench subset |
|---|---|---|---|---|---|
| **nvidia/GR00T-N1.5/1.6/1.7** | **`agibot-world/AgiBotWorld-Beta`** (gated) | `agibot-world/AgiBotWorld-Alpha`; `physical-intelligence/libero` | AgiBot-Beta 2,976 hr / 1M+ trajectories / gated CC-BY-NC-SA | Match — GR00T expects humanoid proprio + dual-arm action vec; AgiBot ships per-frame proprio + action JSON | AgiBot Beta first 50 episodes from `task_327` family (~30 GB) |
| **nvidia/Cosmos-Predict2-14B-GR00T-Dreams-GR1** | NVIDIA's GR00T Dreams sample episodes (shipped with model, ~50 episodes) | `agibot-world/AgiBotWorld-Beta` (different humanoid but compatible 7-DoF arm proprio) | Cosmos sample / NVIDIA OSS | Direct — Cosmos exposes 32-dim GR1 action; sample episodes embed proprio | Use shipped sample (no download); validate on 20 AgiBot dual-arm episodes for OOD |
| **nvidia/Cosmos-Predict2-14B-GR00T-Dreams-DROID** | **`cadene/droid_1.0.1`** | `lerobot-raw/droid_100_raw`; `EDiRobotics/droid_low_resolution` | DROID 95.6K episodes / 27.6M frames / Apache-2.0 / 15 fps Franka | Exact — DROID 7-DoF Franka end-effector + gripper matches Cosmos-DROID action head | First 100 episodes from Stanford camera (chunk-000, ~25 GB) — paper-cited subset |
| **open-gigaai/GigaWorld-0-Video-GR1-2b** | NVIDIA Cosmos-GR1 sample episodes | AgiBot Beta as humanoid proxy | n/a | GR1 robot proprio | 20 AgiBot dual-arm episodes |
| **Alibaba-DAMO-Academy/WorldVLA** | **`jxu124/OpenX-Embodiment`** | `physical-intelligence/libero` | OXE 1M+ trajectories, 22 embodiments / CC-BY-4.0 | Match — WorldVLA trains on OXE's normalized 7-DoF action; OXE has the canonical action+language schema | OXE `bridge` + `fractal20220817_data` first 1K episodes (~50 GB) |
| **jcenaa/WorldVLA-LIBERO** | **`physical-intelligence/libero`** | `HuggingFaceVLA/libero`; `openvla/modified_libero_rlds` | LIBERO 1,693 episodes / 273K frames / CC-BY-4.0 / Panda 7-DoF | Exact — LIBERO-10 task suite, same Panda action vec | LIBERO-10 split = 50 episodes (~5 GB) |
| **Little-Podi/AdaWorld** | `jxu124/OpenX-Embodiment` (cross-env latent action self-supervised) | `physical-intelligence/libero` for downstream | OXE / CC-BY-4.0 | Latent action — no native action labels needed; just frames | OXE diverse-embodiment 500 clips (top 20 embodiments × 25 each) |
| **thuml/ivideogpt-oxe-\*** | **`jxu124/OpenX-Embodiment`** | `physical-intelligence/libero`; BAIR pushing (UC Berkeley site) | OXE / CC-BY-4.0 | Exact — iVideoGPT-OXE checkpoints trained on this exact repo | OXE `bridge_dataset` 200 episodes (~15 GB) |
| **thuml/rt1-world-model-\*** | RT-1 / Fractal20220817 (subset of `jxu124/OpenX-Embodiment` → `fractal20220817_data`) | Bridge V2 (`jxie/bridge_data_v2`) | OXE subset / CC-BY-4.0 | Exact — RT-1 7-DoF action format | Fractal first 500 episodes (~40 GB) |

### Tier S — Camera / exploration world models

| Model | Primary dataset | Backup | Size / License | Conditioning compatibility | Recommended bench subset |
|---|---|---|---|---|---|
| **stdstu123/Yume-5B-720P** | **`Lixsp11/Sekai-Project`** (Yume's training corpus, paper-confirmed) | `DavidYan2001/RealEstate10K` | Sekai 5,000 hr / non-commercial research | Exact — Yume uses Sekai's quantized camera-trajectory tokens (8 KB-aligned bins); per-frame trajectory JSON ships with dataset | Sekai-Real walking subset, 100 clips × 10s @ 720p (~50 GB) |
| **stdstu123/Yume-I2V-540P** | Same as above | Same | Same | Same | Same 100 clips at 540p (~25 GB) |
| **AetherWorldModel/AetherV1** | `DavidYan2001/RealEstate10K` (camera-pose annotated) | `Lixsp11/Sekai-Project` | RE10K 80K clips / CC-BY-4.0 | Match — Aether expects 4D-aware camera pose + goal-frame; RE10K's COLMAP poses are the canonical input | RE10K test split first 500 clips (~10 GB) |
| **genex-world/World-Initializer** | **`TaiMingLu/GenEx-DB-Panorama-World`** (declared in model card) | RE10K panorama-extracted views | GenEx-DB / CC-BY-4.0 | Exact — model card declares this dataset | Full GenEx-DB val (~5 GB) |
| **Overworld/Waypoint-1.5-1B** | `Lixsp11/Sekai-Project` (waypoint-style camera) | `DavidYan2001/RealEstate10K` | Sekai 5K hr | Approximate — Waypoint expects goal-waypoint conditioning; Sekai's location annotations enable waypoint construction | Sekai 100 walking clips with location metadata |
| **yjguo/Ctrl-World** | **`cadene/droid_1.0.1`** (model card declares dataset) | `physical-intelligence/libero` | DROID / Apache-2.0 | Exact — Ctrl-World fine-tuned on DROID 1.0.1 directly | Same DROID Stanford subset as Cosmos-DROID for paired bench |

### Existing T2V / V2V models (datasets confirmed)

| Model | Primary dataset | Backup | Size / License | Conditioning compatibility | Recommended bench subset |
|---|---|---|---|---|---|
| **nvidia/Cosmos-Predict2.5-2B** | `nkp37/OpenVid-1M` (T2V); RE10K for I2V | `Vchitect/VBench_sampled_video` | OpenVid 1M / CC-BY-4.0 | Match — T2V/I2V/V2V via prompt + first frame | VBench 946 prompts + RE10K 200 clips for I2V continuation |
| **nvidia/Cosmos-1.0-Diffusion-7B-V2W** | **`cadene/droid_1.0.1`** + `jxu124/OpenX-Embodiment` (Physical AI focus) | nuScenes mini for driving domain | DROID + OXE | Match — Video2World takes 2 s of conditioning video; DROID gives clean robot trajectories | DROID 10 episodes × 121 frames + nuScenes mini 10 clips |
| **zai-org/CogVideoX-5b** | `nkp37/OpenVid-1M` | `Vchitect/VBench_sampled_video`; `tianweiy/CausVid-VBench` | OpenVid 1M / CC-BY-4.0 | Pure text; for "world-model" eval extend last frame as I2V continuation | VBench-Long 100 prompts × 6s |
| **Wan-AI/Wan2.1-T2V-1.3B/14B** | `nkp37/OpenVid-1M` | `Vchitect/VBench_sampled_video` | OpenVid 1M / CC-BY-4.0 | Pure text | VBench 946 |
| **tencent/HunyuanVideo** | `nkp37/OpenVid-1M` | VBench | OpenVid 1M / CC-BY-4.0 | Pure text | VBench 100 dynamic-action prompts |
| **Lightricks/LTX-Video** | `nkp37/OpenVid-1M` | VBench | OpenVid 1M / CC-BY-4.0 | Pure text + I2V | VBench 100 + RE10K 50 I2V clips |

### Cross-cutting dataset notes

- **VPT contractor data** (off-HF, `github.com/openai/Video-Pre-Training/tree/main/data`): index files `all_6xx.json`, `all_7xx.json`, `all_8xx.json`, `all_9xx.json` point to AWS S3 `.mp4` + `.jsonl`. Each episode 50–200 MB. No HF mirror exists. Pull 100 episodes (~80 GB) directly. Use this as the **shared Minecraft conditioning source** for Oasis + MineWorld + Dreamerv4-MC + Matrix-Game-3.0 cross-comparisons.
- **nuScenes** (motional.com, registration required): mini split (10 scenes, 4 GB) is enough for FVD smoke-test on MagicDriveDiT. Trainval (~360 GB) only if running statistical FVD.
- **Sekai** (`Lixsp11/Sekai-Project`): non-commercial research license — fine for course paper, not for any commercial release.
- **Action-format unifier**: build `worldserve/datasets/action_adapter.py` with three converters: `vpt_to_oasis_25d`, `vpt_to_gamecraft_camera`, `oxe_to_droid_7dof`. **This is the single piece of glue code that makes the same 100-clip benchmark drive every Tier-S model** — without it, each model needs its own data pipeline.
- **Statistical FVD**: standard is 2,048 clips; **500 clips gives ±5% FVD CI** which is acceptable for relative baseline-vs-WorldServe comparison. Use full 2,048 only for the publication number on your top 2–3 models.
- **Quality-eval framework already in repo**: `benchmarks/optimised/oasis_quality_eval_modal.py` (per-frame latent MSE, pixel PSNR, pixel L1) is the right scaffolding — extend it to load VPT episodes via the new `action_adapter.py` and the same per-frame metric harness applies to MineWorld, Dreamerv4-MC, MG2, MG3.

### Dataset download priority (matches GPU-budget order)

For Phase 1 (Oasis + MG2 cheap ablation): **VPT contractor 100 episodes (~80 GB)** + existing `sample_data/`. One download covers Oasis, MineWorld, Dreamerv4-MC, MG2, MG3 cross-comparison.

For Phase 2 (LTX + CogVideoX mid-tier): **`nkp37/OpenVid-1M` first 1,000 clips (~5 GB)** + **`Vchitect/VBench_sampled_video`** subset. Single OpenVid download covers all 5 T2V/V2V models.

For Phase 3 (Cosmos + Wan-14B expensive): **`cadene/droid_1.0.1` first 100 episodes (~25 GB)** + nuScenes mini (4 GB). DROID covers Cosmos-DROID + Ctrl-World + Video2World story; nuScenes mini covers MagicDriveDiT smoke test.

For Tier-S new-model expansion (when first 3 phases complete): **AgiBot-Beta 50 episodes (~30 GB)** for GR00T family + **Sekai-Project walking subset (~30 GB)** for Yume + **`jxu124/OpenX-Embodiment` bridge subset (~15 GB)** for iVideoGPT/WorldVLA/AdaWorld.

Total disk budget across all phases: **~200 GB** — fits on a single Modal volume.
