# Action Aware Step Scheduling for Interactive World Model Inference

**Bhuvan Nallamothu** · CMU 15-849 ML Systems · Spring 2026 · H100 SXM · Open Oasis 500M

## Result

**3.54x speedup at -0.02 dB self-coherence on 950-frame real Minecraft generation.**
9,500 model forwards reduced to 2,683. Validated against canonical DDIM-10 sampler.

| Metric | Baseline DDIM-10 | This Work | Delta |
|---|---:|---:|---:|
| Speedup | 1.00x | **3.54x** | +254% |
| Latency on 950 frames | 426,869 ms | 120,583 ms | -71.7% |
| Forwards per frame (avg) | 10.0 | 2.82 | -71.8% |
| Self-coherence (vs_prev mean) | 41.16 dB | 41.14 dB | **-0.02 dB** |

![headline](figures/headline_breakdown.png)

## Recipe

Two orthogonal step-count optimizations stacked. Both reduce **how many** model forwards execute per frame; neither substitutes any per-step state.

1. **DPM-Solver++ 2M at 5 base steps** replaces DDIM-10. Multistep correction `D = (3*eps_t - eps_{t-1})/2` matches DDIM-10 quality at half the steps.
2. **Action-magnitude difficulty schedule** picks 2/3/5 steps per frame from the `||a||_1` of the 25-dim Minecraft action vector (free signal, no retraining).

```
||a||_1 < 0.5     → 2 steps  (idle: camera still, no keys)
||a||_1 < 1.5     → 3 steps  (moderate)
||a||_1 ≥ 1.5     → 5 steps  (high motion)
```

## Why This Works When Others Don't

![per family](figures/per_family_pareto.png)

The empirical rule across 66 measured runs:

> **Step count reduction is robust at length. Per step substitution compounds and breaks.**

KV-cache quantization, action-conditional v reuse, and prior-frame anchor blending all looked acceptable at 32 frames and **catastrophic at 950 frames** (action-KV: -21 dB; warm-start: +12 dB freezing).

![length](figures/length_robustness.png)

## Distribution Dominates Skip Count

In step caching, **how skips are arranged matters more than how many** there are. At identical 50% skip rate, interleaved (`alternate`) preserves coherence; clustered (`skip_late`, `skip_early`) collapses cross-PSNR by 5-7 dB. The pattern from text-diffusion (SAPS) does not transfer to video.

![distribution](figures/skip_pattern_distribution.png)

## Speedup vs Quality (All 66 Runs)

![scatter](figures/speedup_vs_quality.png)

## Negative Findings (also research output)

| Tried | Why it failed |
|---|---|
| KV-cache FP8/INT4 substitution | Autoregressive feedback compounds quantization noise. Eager-mode quant ops also add latency on 608M / batch=1. |
| Action-conditional v-reuse via slice math | Constant -3.4 dB substitution cost compounds to -21 dB at 950f. |
| Warm-start with prior-frame anchor blending | Any blend > 0 freezes output. blend=0 collapses to step skip. |
| ROI denoising via post-hoc latent blend | No compute saved. Real ROI needs attention-mask injection (~100 line rewrite). |
| Custom kernels stacked with stepcache | F.sdpa monkey patch overhead grows proportionally as call count drops. |
| SageAttention2 / Sliding tile attention | Designed for S >> 4K. Oasis spatial S=144 too small. |
| INT4 weight quantization on 608M | Dequantize overhead exceeds bandwidth saving at batch=1. |
| torch.compile | rotary_embedding_torch mutates SymInt buffers, incompatible with AOT autograd. |

## Reproduce

```bash
PYTHONPATH=benchmarks modal run \
    benchmarks/optimised/oasis_difficulty_dpmpp_modal.py \
    --num-frames 950 --seed 42 \
    --solver-full-steps 5 --solver-mid-steps 3 --solver-reduced-steps 2
```

Full results CSV: `benchmarks/results/all_runs.csv` (66 runs, 15 families).
Full report: `docs/action_aware_step_scheduling.md`.
