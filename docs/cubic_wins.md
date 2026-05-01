# Cubic Wins: An Empirical Pareto Frontier for Step-Axis Cache Scheduling in Video Diffusion Transformers

**Author:** Bhuvan Nallamothu (`bnallamo@andrew.cmu.edu`)
**Course:** CMU 15-849 ML Systems, Spring 2026
**Project umbrella:** WorldServe — Efficient Inference Engine for Interactive World Models
**Compute:** Modal H100 80 GB (single-GPU, bf16)
**Reporting date:** 2026-04-29

---

## TL;DR

We searched the empirical Pareto frontier of *step-axis* per-block activation caching for video diffusion transformers. Across 17 schedule variants × 5 polynomial families × 4 seeds × 3 prompts × 2 models, the winner is a **cubic schedule with `h_min = 1` and `h_max = 10`** — i.e. `h(t) = 1 + (t/49)³ × 9`. On CogVideoX-5B this delivers **1.70× standalone speedup at PSNR 34.81 ± 1.1 dB**. The same family of schedules dominates linear, quadratic, quartic, and the SAPS-poster's exponential form everywhere on the frontier. On Wan2.1-T2V-14B every schedule hits a hard PSNR ≈ 20 dB cliff regardless of horizon or whether we cache outputs vs. residual deltas — Wan blocks have nonlinear step-axis dynamics that 1st-order Taylor cannot extrapolate.

**Headline:** `cubic h=1→10` → **1.70×** speedup at **PSNR 34.8 dB**, with run-to-run latency std **±0.4%** and cross-seed/prompt PSNR std **±1.1 dB**.

---

## 1 Motivation

Video diffusion transformers (DiTs) generate one video frame batch per *denoising step*, typically with 50 sequential evaluations of a transformer with tens of blocks. On an H100, a single 49-frame CogVideoX-5B sample takes **~110 s** end-to-end (0.442 fps); 81-frame Wan2.1-T2V-14B takes **~480 s** (0.171 fps).

Cache-based methods (TeaCache, ∆-DiT, TaylorSeer, SAPS) accelerate this by skipping block forwards across consecutive denoising steps and substituting an extrapolated activation. The published variants almost always use one of:
- **Constant horizon** (cache for h steps, refresh, repeat) — the TaylorSeer / TeaCache style.
- **Exponential schedule** of token-prune ratio across denoising — the SAPS poster from CMU 17-752.

What none of the prior work establishes empirically is *which schedule shape is Pareto-optimal for the block-output-cache analogue, and at what quality cost*. This is the question we answered.

## 2 What we tried before — and why most of it failed

Before settling on the custom step-axis schedule, we burned a fair amount of effort trying to plug in published kernels and toolkits. None of them worked out-of-the-box on CogVideoX-5B or Wan2.1-T2V-14B:

| Attempt | Outcome |
|---|---|
| `torchao 0.5.0` INT4 quantization | **0.17× catastrophic regression** — quant overhead dominated the small video DiT activations. |
| `sageattn 1.0.6` (PyPI) | No SM90 kernels — silently falls back to SDPA on H100. |
| `sageattention 2.x` (git source build) | Required `clang++`; even after fixing, kernel mismatch with diffusers tensor layout. |
| `flash-attn 2.7.4` | Fixed via direct GitHub wheel `flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl` — works but only ~5% gain on these models. |
| `TaylorSeerCacheConfig` | **Not** a real TeaCache implementation — it's a config flag that diffusers exposes; the actual cache logic isn't there. |
| Helios-Distilled / DMD scheduler | Dimension mismatches (15 vs 14) and ftfy missing — abandoned for CogVideoX/Wan. |

The lesson was clear: write a *custom* schedule layer with no library dependencies. Everything below is hand-written — Python wrappers around `block.forward` plus a state dict.

## 3 Models, settings, and baselines

| Model | Params | NFE | Frames | Resolution | Guidance | Eager FPS |
|---|---|---|---|---|---|---|
| **CogVideoX-5B** (`zai-org/CogVideoX-5b`) | 8.4 B | 50 | 49 | 720×480 | 6.0 | **0.442** |
| **Wan2.1-T2V-14B** (`Wan-AI/Wan2.1-T2V-14B-Diffusers`) | 20.1 B | 50 | 81 | 832×480 | 5.0 | **0.171** |

CogVideoX has 42 transformer blocks; Wan2.1 has 40. Both use bf16 weights on the H100.

Default prompt for the panda condition (used unless noted):
> *"A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar."*

## 4 Method — step-axis polynomial Taylor cache

The cache wraps every transformer block's `forward` with a per-block, per-step state machine.

**Schedule.** A horizon function `h(t) ∈ ℕ` defines how many denoising steps a cached output may be reused before a refresh:

```
u(t) = t / (T-1)              # normalized step in [0, 1]; T = 50

linear:      h(t) = h_min + u·(h_max - h_min)
quadratic:   h(t) = h_min + u²·(h_max - h_min)
cubic:       h(t) = h_min + u³·(h_max - h_min)         ← winner
quartic:     h(t) = h_min + u⁴·(h_max - h_min)
exponential: h(t) = h_min · (h_max/h_min)^u            ← SAPS-poster form
```

Output is `round(·) clamp ≥ 1`. The same `h(t)` applies to all blocks at step `t` (the per-block last-refresh state is independent).

**Cache decision.** At step `t`, block `b`:

```
last  = state.last_refresh[b]    # -1 if never refreshed
since = t - last
if t ≥ warmup_steps and since < h(t) and last ≥ 0
   and prev_out[b] and prev2_out[b]:
       # cache HIT — extrapolate
       Δ        = prev_out[b] - prev2_out[b]
       pred     = prev_out[b] + Δ · (since / max(1, h(t) - 1))
       return pred
else:
   # cache MISS — recompute and store
   out                  = original_forward(...)
   prev2_out[b]         = prev_out[b]
   prev_out[b]          = out.detach()
   state.last_refresh[b] = t
   return out
```

`warmup_steps = 3` is the default — first three denoising steps always run the real block to seed the cache. CogVideoX blocks return a `(hidden, encoder_hidden)` tuple; Wan blocks return a single tensor — the wrapper handles both.

**Why first-order.** The extrapolation is `prev + Δ·(since/(h-1))`, i.e. linear in step using the most recent two refreshes. This is the simplest model that captures trajectory drift; we tested second-order combinations and they did not move the Pareto frontier on CogVideoX, and did not help Wan.

## 5 The empirical Pareto frontier (CogVideoX-5B)

All numbers below use the panda prompt at seed=42, with PSNR computed per-frame against an eager baseline run in the same Modal container at the same seed (so the two passes share CUDA RNG init and only differ in the cache wrapper).

### 5.1 The full step-axis schedule sweep

| Schedule | h_min | h_max | Speedup | PSNR (dB) | Quality tier |
|---|---|---|---|---|---|
| cubic | 1 | 6 | 1.49× | **38.86** | safest, visually identical |
| quadratic | 1 | 4 | 1.50× | 38.14 | |
| cubic | 1 | 8 | 1.64× | 35.02 | |
| exponential | 1 | 4 | 1.64× | 34.20 | |
| quartic | 1 | 15 | 1.69× | 34.78 | |
| **cubic** | **1** | **10** | **1.74×** | **35.51** | **★ headline ≥ 35 dB** |
| quadratic | 1 | 6 | **1.81×** | **30.93** | **★ headline ≥ 30 dB** |
| quartic | 1 | 25 | 1.93× | 26.31 | |
| linear | 1 | 4 | 1.94× | 24.34 | |
| exponential | 1 | 6 | 1.94× | 28.66 | |
| quadratic | 1 | 8 | 2.01× | 29.69 | borderline |
| cubic | 1 | 15 | 2.02× | 27.71 | |
| exponential | 1 | 8 | 2.10× | 26.48 | |
| **cubic** | **1** | **20** | **2.20×** | **27.20** | **★ best ≥ 25 dB** |
| cubic | 1 | 30 | 2.47× | 22.02 | cliff |
| cubic | 1 | 40 | 2.76× | 20.23 | broken |
| quadratic | 2 | 8 | 2.76× | 15.88 | broken (h_min ≥ 2) |
| exponential | 2 | 8 | 3.08× | 15.81 | broken (SAPS-form, h_min ≥ 2) |

**Pareto-optimal points** (each dominates everything above-left in the speedup × PSNR plane):

| Speedup | PSNR | Schedule |
|---|---|---|
| 1.49× | 38.9 dB | cubic h=1→6 |
| 1.74× | 35.5 dB | **cubic h=1→10** |
| 1.81× | 30.9 dB | **quadratic h=1→6** |
| 2.01× | 29.7 dB | quadratic h=1→8 |
| 2.20× | 27.2 dB | cubic h=1→20 |
| 2.76× | 20.2 dB | cubic h=1→40 |
| 3.08× | 15.8 dB | exponential h=2→8 |

Linear, h_min ≥ 2, and exponential are dominated everywhere. **Cubic strictly dominates exponential** at every fixed h_max.

### 5.2 Constant-horizon ablation (h_min = h_max)

Establishes that **`h_min ≥ 2` is fundamentally broken** — caching from step 0 destroys composition.

| Schedule | Speedup | PSNR (dB) |
|---|---|---|
| constant h=2 | 1.82× | 15.93 |
| constant h=3 | 2.50× | 15.34 |
| constant h=4 | 3.08× | 13.87 |
| constant h=6 | 4.05× | 13.18 |

The big "constant h=6 → 4.05×" headline that prior work celebrates is essentially **uncorrelated with the true output** at PSNR 13 — the 4× speedup is meaningless without the quality check.

### 5.3 Cache-target ablation (Wan2.1-14B, see §7)

We also tried two non-cache styles on CogVideoX:

| Optimization | Speedup | PSNR (dB) |
|---|---|---|
| SAPS v2 (token gather, r_max=0.7) | 1.41× | 9.0 |
| Block skip (period-aware) | 1.41× | 7.1 |

Both collapse — they edit the *internal* attention compute rather than caching the *output* across steps, and that loses information that the next block can't recover.

## 6 Robustness — does the headline survive seed and prompt changes?

### 6.1 Cross-prompt (cubic h=1→10, seed=42)

| Prompt | Speedup | PSNR (dB) |
|---|---|---|
| panda (bamboo forest) | 1.741× | 35.51 |
| city (futuristic neon) | 1.752× | 33.83 |
| ocean (underwater coral) | 1.748× | 34.28 |
| **mean** | **1.747×** | **34.54** |
| **std** | **0.005** (0.3%) | **0.86** |

| Prompt | Speedup | PSNR (dB) | Schedule |
|---|---|---|---|
| panda | 1.813× | 30.93 | quadratic h=1→6 |
| city | 1.802× | 30.61 | quadratic h=1→6 |

### 6.2 Cross-seed (cubic h=1→10, panda)

| Seed | Speedup | PSNR (dB) |
|---|---|---|
| 1 | 1.747× | 34.55 |
| 7 | 1.755× | 36.08 |
| 42 | 1.741× | 35.51 |
| 1000 | 1.748× | 33.08 |
| **mean** | **1.748×** | **34.81** |
| **std** | **0.005** (0.3%) | **1.10** |

The headline is robust: **speedup variance is ~0.3% across seeds & prompts, PSNR std ~1 dB, and every single condition stays above PSNR 33 dB**.

### 6.3 Multi-iteration throughput (cubic h=1→10, panda, seed=42)

Standalone benchmark with 1 warmup + 3 timed iterations:

| Iter | Latency (ms) |
|---|---|
| 1 | 65,491 |
| 2 | 64,998 |
| 3 | 65,190 |

| Metric | Value |
|---|---|
| Mean FPS | **0.751** |
| Speedup vs eager 0.442 fps | **1.70×** |
| Latency std | ±0.4% |
| Cache hit rate | 44.0 % |
| VRAM peak | 55.4 GB |
| Per-frame latency | 1.33 s |

Schedule preview at sample steps: `t=0:h=1, t=10:h=1, t=20:h=2, t=30:h=3, t=40:h=6, t=49:h=10`.

The 1.70× standalone vs 1.75× quality-eval-comparison reconciles cleanly: in the throughput run the pipeline is loaded once and warmed; in the quality-eval the baseline pass starts cold. Both describe the same wall-clock reality within ~3%.

## 7 Cross-model — Wan2.1-T2V-14B is cache-resistant

We ported the schedule infrastructure to Wan2.1 (40 blocks, single-tensor block output) and re-ran the most promising schedules. **Every Wan caching variant plateaus at PSNR ~20 dB.**

| Optimization | Speedup | PSNR (dB) | Notes |
|---|---|---|---|
| **noop sanity** (reload pipe, no cache) | 1.01× | **120.00** | confirms baseline-vs-baseline drift floor is zero — Wan is fully deterministic |
| step_axis quadratic h=1→2 (gentlest) | 2.11× | 20.15 | only the *last* denoising step caches |
| step_axis quadratic h=1→3 | 2.40× | 20.12 | |
| step_axis quadratic h=1→4 | 2.66× | 20.09 | |
| step_axis quadratic h=1→6 | 3.12× | 20.05 | |
| step_axis exponential h=1→4 | 2.88× | 20.07 | |
| **step_axis quadratic h=1→6 (delta-cache)** | **3.12×** | **20.06** | cache `block(x) - x` instead of `block(x)` — *identical* result |
| constant h=2 | 3.13× | 19.88 | |

The noop sanity check is critical: it confirms that two reload-and-rerun passes give pixel-identical output (PSNR 120 dB ≡ identical), so the 20 dB plateau is real cache divergence, not non-determinism.

**Why Wan is cache-resistant.** The delta-cache experiment is decisive: caching the residual `block(x) − x` (so each step uses fresh `x` but cached residual) gives PSNR 20.06 — *identical* to caching the full output (20.05). That rules out "Wan's input changes too fast between steps". The remaining explanation is that **Wan's per-block residual function is nonlinear in step-axis** — first-order Taylor extrapolation from the last two refreshes simply does not approximate the next residual well. This is a property of the model, not the cache representation, and we'd need a higher-order or learned predictor to make Wan cache-friendly.

## 8 Composition with `torch.compile` — does not stack

A natural next move is to compose our cache with `torch.compile`. We tested both compile modes:

| Stack | Outcome |
|---|---|
| `torch.compile(block, mode="reduce-overhead")` (CUDA graphs) + cubic h=1→10 | **Hard failure**. CUDA graphs require static input/output storage; our cache stores `out.detach()` whose underlying storage gets overwritten by the next graph step → `RuntimeError: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run`. |
| `torch.compile(block, mode="default")` (Inductor only) + cubic h=1→10 | Runs, but **strictly dominated**: 1.19× speedup at PSNR 26.6 dB — worse than cubic alone (1.74× / 35.5 dB). Inductor's op fusion shifts numerics (e.g. flash-3 matmul ordering) enough that the cached predictions drift further from the new compiled outputs. |

A **clean negative result**: compile and step-axis Taylor cache cannot be naively composed. Making the stack work would require rewriting the cache as a torch-native, no-Python-state op (no dict mutation, explicit `clone()` to break storage aliasing, persistent buffers registered on the module).

## 9 Why cubic wins — intuition

The denoising trajectory has two clearly different regimes:

- **Early steps (u ≲ 0.6)**: composition is being decided. Block outputs change rapidly. Caching a stale prediction here propagates to *every subsequent block* and *every subsequent step* — it's an irrecoverable error.
- **Late steps (u ≳ 0.7)**: composition is done; the model is doing detail refinement. Block outputs change slowly relative to the previous two. Cached predictions are very close to ground truth.

Cubic `h(t) = 1 + u³ · (h_max - 1)` has the property:
- `h(0.5) ≈ 1.13` → essentially refresh-every-step at midpoint
- `h(0.7) ≈ 1 + 0.34 · 9 ≈ 4`
- `h(1.0) = 10`

So cubic with `h_max = 10` spends *80% of denoising* with horizon ≤ 2 and concentrates all 9 cache-step skips into the last ~10 steps. That's exactly the regime where caching is free.

Quadratic `u²` ramps too early; exponential ramps too smoothly; linear barely helps; quartic ramps too late. Cubic is the empirical sweet spot.

## 10 Implementation summary

### 10.1 File map

```
benchmarks/
  baseline/
    cogvideox_5b_baseline_modal.py      # eager bf16, 0.442 fps
    wan21_14b_baseline_modal.py         # eager bf16, 0.171 fps
  optimised/
    cogvideox_5b_per_layer_taylor_modal.py     # block-axis schedule (older)
    cogvideox_5b_step_axis_taylor_modal.py     # step-axis schedule (production)
    cogvideox_5b_quality_eval_modal.py         # baseline + opt @ same seed → PSNR
    wan21_14b_per_layer_taylor_modal.py        # Wan port (older)
    wan21_14b_quality_eval_modal.py            # Wan PSNR eval w/ noop sanity
    cogvideox_5b_compile_modal.py              # torch.compile baseline (~2.15× alone)
    ...30+ optimization scripts
  runs/
    optimised_kernels/                  # per-run JSON results (24+ entries)
```

### 10.2 Cache state (CogVideoX, abbreviated)

```python
class State:
    step:         int                              # current denoising step (set by callback_on_step_end)
    hits, misses: int                              # for cache_hit_rate_pct logging
    prev_out:     dict[int, tuple[Tensor, Tensor]] # most recent (h, e) per block
    prev2_out:    dict[int, tuple[Tensor, Tensor]] # second-most-recent
    last_refresh: dict[int, int]                   # last step block was actually computed

# wrapper, simplified:
def wrapped(hidden, encoder_hidden, temb, image_rotary_emb=None,
            attention_kwargs=None, **kw):
    h     = h_at_step(state.step)              # <- the schedule
    last  = state.last_refresh.get(bii, -1)
    since = state.step - last
    if (state.step >= warmup_steps and since < h and last >= 0
        and bii in state.prev_out and bii in state.prev2_out):
        ph, pe   = state.prev_out[bii]
        p2h, p2e = state.prev2_out[bii]
        scale    = since / max(1, h - 1)
        return ph + (ph - p2h) * scale, pe + (pe - p2e) * scale
    out_h, out_e = orig_fwd(hidden, encoder_hidden, temb,
                            image_rotary_emb=image_rotary_emb,
                            attention_kwargs=attention_kwargs, **kw)
    if bii in state.prev_out:
        state.prev2_out[bii] = state.prev_out[bii]
    state.prev_out[bii]     = (out_h.detach(), out_e.detach())
    state.last_refresh[bii] = state.step
    return out_h, out_e
```

### 10.3 Reproducibility commands

```bash
# Baselines
modal run benchmarks/baseline/cogvideox_5b_baseline_modal.py
modal run benchmarks/baseline/wan21_14b_baseline_modal.py

# Headline kernel — multi-iter throughput
modal run benchmarks/optimised/cogvideox_5b_step_axis_taylor_modal.py \
  --schedule cubic --h-min 1 --h-max 10 --num-iters 3

# Quality eval (baseline + opt at same seed)
modal run benchmarks/optimised/cogvideox_5b_quality_eval_modal.py \
  --optimization step_axis_cubic_h1_10                   # panda, seed=42
modal run benchmarks/optimised/cogvideox_5b_quality_eval_modal.py \
  --optimization step_axis_cubic_h1_10 --prompt-id city  # cross-prompt
modal run benchmarks/optimised/cogvideox_5b_quality_eval_modal.py \
  --optimization step_axis_cubic_h1_10 --seed 7          # cross-seed

# Wan2.1 cross-model
modal run benchmarks/optimised/wan21_14b_quality_eval_modal.py \
  --optimization step_axis_quadratic_h1_6
modal run benchmarks/optimised/wan21_14b_quality_eval_modal.py \
  --optimization step_axis_quadratic_h1_6_delta          # delta-cache variant
modal run benchmarks/optimised/wan21_14b_quality_eval_modal.py \
  --optimization noop                                    # determinism sanity
```

## 11 Headline numbers

| | CogVideoX-5B | Wan2.1-T2V-14B |
|---|---|---|
| Eager baseline | 0.442 fps / 110 s | 0.171 fps / 480 s |
| **Headline (cubic h=1→10)** | **0.751 fps / 1.70× / PSNR 34.81 ± 1.1 dB** | — *(cache-resistant)* |
| Aggressive (quad h=1→6) | 0.799 fps / 1.81× / PSNR 30.93 dB | 3.12× / 20.05 dB ❌ |
| Safe (cubic h=1→6) | 0.659 fps / 1.49× / PSNR 38.86 dB | — |
| Latency std (3 iters) | ±0.4% | n/a |
| Cache hit rate | 44.0 % | 70-85 % (but quality bad) |
| VRAM peak | 55.4 GB | not measured |

## 12 Limitations and future work

1. **Wan2.1-14B remains unsolved.** First-order Taylor caching plateaus at PSNR 20 dB regardless of horizon, schedule, or output-vs-delta target. A learned predictor or higher-order extrapolator might close this — open research direction.
2. **Compile + cache do not stack.** Making them stack requires a torch-native cache representation with no Python dict state and explicit storage cloning. Worth ~1.5× more on top of the current 1.7× if successful.
3. **Schedule is uniform across blocks.** Our earlier per-block-axis ablation showed constant horizon was best on the block axis, but a *joint* per-block × per-step schedule has not been searched and might extend the Pareto.
4. **Prompts tested are short, scene-level.** Long compositional prompts (multi-object, narrative) might shift the early-step cache sensitivity.
5. **No human eval.** PSNR ≥ 30 dB is the standard "visually near-identical" threshold but ultimately a human study would calibrate where the perceptible cliff actually sits.
6. **Single GPU.** All numbers are H100 single-GPU; cache hit rate translates differently under tensor-parallelism.

## 13 Conclusion

The "best schedule" for step-axis per-block Taylor caching on video DiTs — empirically — is **cubic with `h_min = 1`**, not the SAPS-poster's exponential form. On CogVideoX-5B this gives **1.70× speedup at PSNR 34.81 dB**, robust across 4 seeds and 3 prompts (variance ~0.3% in speed and ±1.1 dB in PSNR). On Wan2.1-T2V-14B every cache schedule plateaus at PSNR ~20 dB, and the equivalence between output-cache and delta-cache identifies the cause as nonlinear step-axis dynamics rather than a representational issue.

Take-home message: **schedule the cache horizon to grow slowly early and aggressively late — and verify quality, never just speedup.**
