# KV Cache Compression for Open-Oasis 500M

This is the working notebook for Component 2 of the WorldServe proposal:
spatial-temporal KV cache compression on Etched/oasis-500m. The goal is to
keep recent frames in FP16 while progressively quantizing older frames to
FP8 / INT4, augmented with cross-frame token merging and (optionally)
QVG semantic smoothing.

---

## Why Oasis Needs This (and Why It's Different from Text LLMs)

A text LLM appends 1 token per decode step; an autoregressive video world
model appends an **entire frame** (144 spatial tokens after `patch_size=2`
patchify, plus T temporal tokens per spatial location). For a 30 s session
at 24 fps with 256 tokens/frame the KV cache hits 30+ GB. Even at our
benchmark size (60 frames × 144 spatial × 16 layers × FP16) we already
have a non-trivial cache.

Two properties make this workload different from text LLMs:

1. **Massive temporal redundancy.** Frame `t` and frame `t-1` look almost
   identical in static scenes. Empirically (`TemporalTokenMerger` measurement
   at threshold 0.95) we observe **95% of K vectors at the same spatial
   position across frames are mergeable** in our 16-frame test runs.
2. **Aging is naturally meaningful.** Older frames are less useful for the
   current frame's attention, so they tolerate aggressive quantization that
   recent frames don't. This is the "PyramidKV-but-along-time" intuition.

---

## Techniques Implemented

All four live under `worldserve/optimizations/system_level/kv_cache/`.

### 1. `ProgressiveKVQuantizer` — Age-Based Tier Demotion

`progressive_quant.py`. The headline mechanic: each KV entry has an `age`
counter (frames since insertion). At thresholds, it gets demoted:

```
FP16  →  FP8  (at age >= fp8_age_threshold)
FP16  →  FP8  →  INT4  (at age >= int4_age_threshold)
```

Defaults from the proposal: `fp8_age=30, int4_age=90`. **For 16-frame Oasis
runs, those thresholds are unreachable**, so the benchmark uses
`fp8_age=3, int4_age=10` instead — frames 0-2 stay FP16, 3-9 see FP8, 10+
see INT4.

**K/V asymmetry (DiffKV insight):** values tolerate quantization better than
keys (which directly steer attention routing). When `key_value_asymmetry=True`,
V thresholds are 70% of K thresholds.

**Quantization formats:**
- FP8: `torch.float8_e4m3fn` per-tensor scale = `amax / 448`
- INT4: per-group-128 with separate `scale` (range/15) and `zero_point` (min),
  packed two nibbles per byte

### 2. `TemporalTokenMerger` — Cross-Frame Token Dedup

`temporal_merge.py`. For each spatial position, compare the K vector at frame
`t` to the K vector at the same position in frame `t-1` via cosine similarity.
If above `similarity_threshold` (default 0.95):

- **Weighted average** strategy: `merged = 0.6 * current + 0.4 * previous`
- **Keep-latest** strategy: `merged = current`

If >95% of tokens in a frame merge, the entire previous frame is dropped
and the merged frame replaces both. This is what gives the headline
**95% merge rate** on Oasis at 16 frames.

### 3. `QVGSmoothingQuantizer` — Centroid + 2-bit Residual

`qvg_smoothing.py`. Reproducing Quant VideoGen (Feb 2026) for 2-bit KV
compression. Per-token compression:

1. Run k-means (`n_clusters=16` default) on the KV tokens
2. Store: centroids (FP16, small), per-token cluster ID (4 bits), residual
   `(token - centroid)` quantized to 2 bits per element (group-64)

Effective per-element cost: `4 / D + 2 + 16/group_size_bytes ≈ 2.3 bits`
for D=64 and group_size=64. Reconstruction: `centroid + dequantized_residual`.

### 4. `SpatialBlockEvictor` — Spatial Token Pruning (MG2-tuned)

`spatial_eviction.py`. Group spatial tokens into `block_size × block_size`
patches; for old frames (`min_age_frames` threshold), evict the bottom
`eviction_ratio` blocks by attention score, while protecting top-10%
"attention sink" blocks.

Tuned for Matrix-Game 2.0's ~15,360 spatial tokens/frame. Oasis only has
**144 spatial tokens (9×16 grid)**, which is too small for spatial eviction
to meaningfully help — we run with `block_size=2` for completeness but
expect minimal benefit.

---

## Orchestration: `KVCacheManager`

`manager.py` (~900 lines). Composes the four techniques behind a small API:

```python
manager = KVCacheManager(config={
    "num_layers": 16,
    "max_frames": 64,
    "temporal_merge":     {"similarity_threshold": 0.95, ...},
    "progressive_quant":  {"fp8_age_threshold": 3, "int4_age_threshold": 10, ...},
    "qvg_smoothing":      {"n_clusters": 16, "residual_bits": 2},
    "spatial_eviction":   False,
    "enable_quant_substitution": True,   # see "Substitution path" below
})
n_wired = manager.wire_to_model(dit)     # patches all attention forwards
manager.set_current_frame_idx(frame_i)   # called per video frame
manager.create_cache(num_layers=16, max_frames=64)  # reset between iterations
report = manager.get_memory_report()     # FP16 baseline vs effective bytes
```

### How the hooks work (`wire_to_model`)

Iterates `model.named_modules()` for class names in
`{SpatialAxialAttention, TemporalAxialAttention, Attention, WanAttention,
FlashSelfAttention}` and replaces each module's `forward` with a wrapper:

```
1.  Save the live F.scaled_dot_product_attention.
2.  Monkey-patch F.scaled_dot_product_attention with `_capture_sdpa`.
3.  Call orig_forward(*args, **kwargs).
4.  Restore F.scaled_dot_product_attention.
5.  manager.update({l_idx: {"key": captured_k, "value": captured_v}},
                    frame_idx=manager._current_frame_idx)
```

The `_capture_sdpa` inner closure does **four things in order** before
calling the real SDPA:

1. **Context KV injection** (Diffusion Forcing, optional): if
   `manager.has_context_kv(l_idx)`, prepend stored context K/V to live
   K/V along the sequence dim.
2. **Speculative inject mode**: when `_inject_mode` is armed, swap live
   K/V for stored quantized cache (used by KVQuantSpec draft passes).
3. **K/V capture**: `captured["k"] = key.detach()` (once per forward).
4. **Live quant substitution** (the new path, see below): if enabled,
   apply quantize-dequantize roundtrip on K/V at a precision derived from
   `manager._current_frame_idx`.

### Substitution path (the change that makes numerics honest)

`_capture_sdpa` ends with:

```python
if manager._enable_quant_substitution:
    prec = manager._precision_for_frame(manager._current_frame_idx)
    if prec != "fp16":
        key   = manager._quant_roundtrip(key,   prec)   # FP8 or INT4 roundtrip
        value = manager._quant_roundtrip(value, prec)
return orig_sdpa(query, key, value, ...)
```

`_precision_for_frame(frame_idx)` selects:
- `frame_idx >= int4_age` → `"int4"`
- `frame_idx >= fp8_age`  → `"fp8"`
- else                    → `"fp16"` (no-op)

`_quant_roundtrip(t, "fp8")` calls
`ProgressiveKVQuantizer.quantize_to_fp8` then `dequantize_fp8`. Same for
INT4 with the static methods on the quantizer. **The PyTorch eager-mode
implementation of these roundtrips is the dominant runtime cost** (and a
known target for replacement with the existing CUDA kernels).

### Manager `update()` pipeline

When `_capture_sdpa` finishes, the wrapper calls
`manager.update(new_kv, frame_idx)` which runs:

```
1. Append new K/V to _raw_cache (per-layer, rolling window of max_frames)
2. Temporal merging (latest frame only — partial vs apply_to_cache)
3. Importance tracking (if attention_weights provided — not yet wired)
4. Progressive quantization (parallel store in _cache_k / _cache_v with ages)
4b. QVG semantic smoothing (parallel store in _qvg_compressed)
5. Spatial eviction (if attention_weights provided — MG2-tuned)
```

### Key bug fixes already shipped

| Bug | Cause | Fix |
|---|---|---|
| Frame ages were inflated 10× | Used `_wire_step_counter` (denoising step) as `frame_idx` for ages → entries crossed thresholds way too early | Added `set_current_frame_idx()`; `_capture_sdpa` prefers it over the step counter |
| `qvg_smoother` was dead code | Instantiated when config has `qvg_smoothing` but `update()` never called it | Added Step 4b in `update()` that calls `qvg.semantic_smooth(kv["key"])` per layer per frame |
| Iter-to-iter slowdown of 25% | `temporal_merger._stats.per_frame_merge_rate` list and `spatial_evictor._block_score_ema` accumulated across iterations | `create_cache()` now also clears merger stats and EMA |
| `get_memory_report` ignored QVG | QVG store had no accountant entry | Added QVG bytes calculation: `centroids_fp16 + assignments_4bit + residuals_2bit + scales_fp32` |

---

## Run Files

| File | What it runs |
|---|---|
| `benchmarks/baseline/oasis_modal.py` | Native SDPA, no manager hooks, `n_prompt=1`, `ddim_steps=10`. CLI: `--num-frames N` |
| `benchmarks/optimised/oasis_kvcache_modal.py` | Same generation loop, but installs `KVCacheManager.wire_to_model(dit)` and calls `manager.set_current_frame_idx(frame_i)` per frame. CLI flags: `--use-temporal-merge`, `--use-progressive-quant`, `--use-qvg`, `--use-spatial-eviction`, `--enable-quant-substitution`, `--fp8-age-threshold N`, `--int4-age-threshold N`, `--num-frames N` |

Both write JSON results to `benchmarks/runs/{baseline,optimised_kernels}/`.

---

## Measured Results (16-frame Oasis on H100 SXM)

| Variant | Latency mean | FPS | VRAM | Iter stability |
|---|---:|---:|---:|---|
| **Baseline (native SDPA)** | **8,250 ms** | **1.94** | **1.31 GB** | n/a |
| KVCache accountant only | 12,634 ms | 1.27 | 1.31 GB | unstable (+25% iter 2) |
| **KVCache + substitution** | **13,818 ms** | **1.16** | **2.93 GB** | stable (0.7% drift) |

**Memory report (substitution run, 16 frames):**
```
total_fp16_bytes:    91 MB
total_actual_bytes:  91 MB    ← parallel store, doesn't free FP16 originals
savings_ratio:       2.78%

temporal_merge:
  merge_rate:        95.2%      ← 14.6M of 15.4M tokens flagged duplicates
  memory_savings:    0%         ← max_frames clamps before merge can shrink

progressive_quant precision_counts: fp16: 136, fp8: 8, int4: 0
  ↳ even at fp8_age=3, only 8 entries crossed (counting bug
    being investigated; main story: cache is too small at 16 frames)
```

### What this tells us

- **Plumbing works** — 32 attention modules wired (16 spatial + 16 temporal),
  no crashes, model produces output.
- **Substitution is honestly active** — frames 3+ attend through FP8
  roundtripped K/V; frames 10+ through INT4. Numerics differ from baseline.
- **Iter-to-iter is now stable** (13.87 / 13.77 ms — 0.7% drift).
- **Latency is +67% over baseline** because per-call quantize-dequantize
  in eager mode adds ~344 ms / frame. 5,120 SDPA calls × eager-mode
  quant work dominates.
- **VRAM more than doubled** because the parallel `_cache_k` store keeps
  every entry alongside the live FP16 K/V — exactly the opposite of what
  KV compression should do.
- **Memory savings ratio is small** because the accountant operates on
  the parallel store, never frees the live FP16 entries.

### One real positive finding

`temporal_merge.merge_rate = 95.2%` is **the strongest validation we have
of the proposal's premise**: Oasis K-vectors at the same spatial position
are nearly identical across consecutive frames. This is what would justify
aggressive temporal merging in a properly-wired implementation.

---

## 60-Frame Comparison (just completed)

The cache size grows linearly with frame count, so KV compression should
start paying off at longer generations. Both runs at 60 frames:

| Variant | Latency mean | FPS | VRAM | Iter stability | Merge rate |
|---|---:|---:|---:|---|---:|
| **Baseline (60f)** | **71,183 ms** | **0.84** | **2.88 GB** | 0.3% drift | n/a |
| **KVCache + sub (60f)** | **93,353 ms** | **0.64** | **3.83 GB** | 3.9% drift | **94.3%** |

**Takeaways:**

1. **The relative gap shrinks as cache grows.** At 16f the KVCache path was
   +67% slower than baseline (8.25 → 13.82 s); at 60f it's only **+31% slower**
   (71.18 → 93.35 s). The fixed per-call overhead amortizes over more useful
   compute, but the gap is still negative — substitution is too expensive in
   eager mode.
2. **VRAM gap also shrinks.** At 16f it was 1.31 → 2.93 GB (+124%); at 60f
   it's 2.88 → 3.83 GB (**+33%**). Same reason: as the legitimate cache
   grows, the parallel-store overhead becomes proportionally smaller.
3. **Temporal merger reports 94.3% mergeable tokens** — same headline finding
   as 16f, holding up at 60f. Strongest signal that compression is real.
4. **Memory savings ratio still tiny: 3.95%.** Reflects the parallel-store
   accountant, not actual VRAM saved.

The trend is clear: **the longer the generation, the smaller the relative
KVCache penalty.** Extrapolating, parity with baseline would need
generations ~150-200 frames or aggressive optimization of the substitution
path (CUDA fp8 kernel instead of eager-mode roundtrip).

For the proposal's 30-second target (~720 frames at 24 fps) the KVCache
overhead would likely be a small constant on top of baseline, with real
VRAM savings IF we fix the parallel-store double-allocation.

### How to reproduce

```bash
# Baseline at 60 frames
PYTHONPATH=benchmarks modal run benchmarks/baseline/oasis_modal.py \
    --num-frames 60 --num-iters 2 --num-warmup 1

# KVCache + substitution at 60 frames
PYTHONPATH=benchmarks modal run benchmarks/optimised/oasis_kvcache_modal.py \
    --num-frames 60 --num-iters 2 --num-warmup 1 \
    --use-temporal-merge --use-progressive-quant --enable-quant-substitution
```

---

## Known Limitations / TODO

1. **The accountant's parallel `_cache_k` is never freed.** It coexists
   with the live FP16 cache, doubling VRAM. Fix: in `_capture_sdpa`, after
   roundtrip, write back into `_raw_cache` and drop the original.
2. **Eager-mode quant-dequant is too slow.** Replace
   `quantize_to_fp8 / dequantize_fp8` in `ProgressiveKVQuantizer` with the
   existing CUDA kernel `worldserve/kernels/fp8_quantize.cu` (already
   validated in `oasis_custom_modal`). Expected: cuts ~1.2 s of substitution
   overhead to ~100 ms.
3. **`precision_counts` undercount needs investigation.** Either
   `progressive_quant.update_cache` is being called fewer times than
   expected (manager-level filtering somewhere?) or `get_memory_report`
   is summing wrong. The substitution path itself ignores `_cache_k`, so
   the discrepancy is purely cosmetic for runtime, but the memory report
   is misleading.
4. **No quality measurement yet.** We have no MSE / SSIM / FVD numbers
   comparing baseline vs `enable_quant_substitution=True` outputs at the
   same seed. Without that, we can't claim quantization is "lossless enough."
5. **Importance tracking and layer-aware quant are wired in `manager.update()`
   conditionally on `attention_weights` being passed — but no benchmark
   currently passes attention weights, so those code paths are inactive.**
6. **`max_frames` rolling window invalidates per-frame ages.** When the
   raw cache rolls over, the corresponding `_frame_ages` list also rolls,
   but the progressive quantizer's parallel `_ages` list does NOT — they
   can desynchronize for long generations.

---

## Architecture Diagram (text)

```
                       benchmarks/baseline/oasis_modal.py
                                  │
                                  │ baseline path
                                  ▼
                     ┌─────────────────────────┐
                     │   Etched/oasis-500m     │
                     │   DiT-S/2 + ViT-L VAE   │
                     │ 16 SpatioTemporalDiT    │
                     │   blocks (32 attn       │
                     │   modules total)        │
                     └─────────────────────────┘

                       benchmarks/optimised/oasis_kvcache_modal.py
                                  │
                                  ▼
                  ┌──────────────────────────────────────┐
                  │ KVCacheManager(config)               │
                  │  ┌────────────────────────────────┐  │
                  │  │ wire_to_model(dit)             │  │
                  │  │  for each attn module:         │  │
                  │  │    mod.forward = _kv_forward   │  │
                  │  └────────────────────────────────┘  │
                  └──────────────────────────────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              ▼                   ▼                   ▼
   set_current_frame_idx   _capture_sdpa hook   manager.update()
   (called per frame in    (replaces F.sdpa     (after each forward)
    generate loop)         per-call):           runs:
                              ① context inject  1. append _raw_cache
                              ② inject_cache    2. temporal_merge
                              ③ capture K/V     3. importance (no-op)
                              ④ QUANT ROUNDTRIP 4. progressive_quant
                              ⑤ orig_sdpa       4b. qvg_smoothing
                                                 5. spatial_eviction
```

---

## File Map

| File | Purpose |
|---|---|
| `worldserve/optimizations/system_level/kv_cache/__init__.py` | Lazy registry; `apply_kv_cache_compression(config)` factory |
| `worldserve/optimizations/system_level/kv_cache/manager.py` | `KVCacheManager` orchestrator, `wire_to_model`, `_capture_sdpa`, substitution helpers |
| `worldserve/optimizations/system_level/kv_cache/temporal_merge.py` | `TemporalTokenMerger` |
| `worldserve/optimizations/system_level/kv_cache/progressive_quant.py` | `ProgressiveKVQuantizer` + static FP8/INT4 quantize/dequantize methods |
| `worldserve/optimizations/system_level/kv_cache/qvg_smoothing.py` | `QVGSmoothingQuantizer` |
| `worldserve/optimizations/system_level/kv_cache/spatial_eviction.py` | `SpatialBlockEvictor` (MG2-tuned, mostly unused for Oasis) |
| `worldserve/optimizations/system_level/kv_cache/importance_tracker.py` | `ImportanceTracker` (EMA attention scores; not wired) |
| `worldserve/optimizations/system_level/kv_cache/layer_aware_quant.py` | `LayerAwareQuantizer` (per-layer entropy precision; not wired) |
| `benchmarks/baseline/oasis_modal.py` | Pure baseline; no KV compression |
| `benchmarks/optimised/oasis_kvcache_modal.py` | KVCache benchmark; CLI toggles for each technique |
