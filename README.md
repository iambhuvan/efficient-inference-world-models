# Efficient Inference Engine for Interactive World Models

CMU **15-442 / 15-642** Machine Learning Systems, Spring 2026 — Bhuvan Nallamothu (`bnallamo@andrew.cmu.edu`)

**Action-Driven Compute Adaptation for Autoregressive World Models.** A training-free **3.54× speedup** recipe for [Open-Oasis 500M](https://huggingface.co/Etched/oasis-500m) inference on H100 SXM, validated across **950 real Minecraft frames** at preserved self-coherence (Δ vs_prev = **−0.02 dB**).

The recipe stacks two orthogonal step-count cuts:

1. **DPM-Solver++ 2M** at 5 base steps replaces canonical 10-step DDIM (halves the per-frame compute ceiling at no quality cost).
2. **Action-magnitude difficulty schedule** picks 2, 3, or 5 model forwards per frame from the free 25-dim Minecraft action vector (idle frames need fewer steps; active frames need more).

Per generation, total model forwards drop from **9,500 to 2,683** (71.8% reduction).

---

## Headline numbers (950 frames, real Minecraft data)

| Method                       | Speedup    | Coherence Δ        | Forwards |
|------------------------------|-----------:|-------------------:|---------:|
| DDIM-10 baseline             | 1.00×      | +0.00 dB           | 9,500    |
| DPM-Solver++ 2M (5-step)     | 1.98×      | +4.55 dB           | 4,750    |
| Difficulty 2/4/10            | 2.69×      | +0.76 dB           | 3,656    |
| TaylorSeer order=2           | 2.52×      | 0/0 val failures   | 5,592    |
| Stepcache interval-3         | 2.47×      | −1.56 dB           | 3,134    |
| **DPM++ 2M × Difficulty**    | **3.54×**  | **−0.02 dB**       | **2,683** |

Across 72 measured runs spanning 16 optimization families, our central empirical finding is that **step-count reduction is autoregressive-safe**, while every per-step substitution method we tested (KV-cache quantization, action-conditional v reuse, anchor blending, mega-stack triple stacking) compounds its per-step error across 950 frames and crashes 5 to 22 dB below baseline coherence.

---

## Reproduce the headline in one command

```bash
PYTHONPATH=benchmarks modal run \
    benchmarks/optimised/oasis_difficulty_dpmpp_modal.py \
    --num-frames 950 --seed 42 \
    --solver-full-steps 5 --solver-mid-steps 3 --solver-reduced-steps 2
```

Requires a Modal account with H100 access and an HF token for the gated `Etched/oasis-500m` model. Wall clock is roughly 7 min baseline plus 2 min stacked.

Other reproductions:

```bash
# DDIM-10 baseline alone
PYTHONPATH=benchmarks modal run benchmarks/baseline/oasis_modal.py \
    --num-frames 950 --num-iters 1

# Difficulty schedule alone (2/4/10 buckets)
PYTHONPATH=benchmarks modal run \
    benchmarks/optimised/oasis_difficulty_steps_modal.py \
    --num-frames 950 --seed 42 \
    --reduced-steps 2 --mid-steps 4 --full-steps 10

# TaylorSeer alone (order-2, threshold 0.15)
PYTHONPATH=benchmarks modal run \
    benchmarks/optimised/oasis_taylorseer_modal.py \
    --num-frames 950 --order 2 --threshold 0.15
```

---

## Repository layout

```
.
├── report/                        Final paper (LaTeX + figures)
│   ├── main.tex                   8-page MLSys-format report
│   ├── refs.bib                   24 references
│   └── figures/                   5 publication-quality plots
├── worldserve/                    The package
│   ├── models/                    Oasis sampler (canonical v-pred autoregressive)
│   ├── kernels/                   Custom CUDA + Triton kernels (Phase 1, deprecated)
│   └── optimizations/
│       ├── model_level/
│       │   ├── feature_caching/   TaylorSeer port (496 lines, per-frame reset)
│       │   └── samplers/          DPM-Solver++ 2M
│       └── system_level/
│           ├── kv_cache/          KV-cache compression manager (Phase 2, abandoned)
│           ├── speculative/       Speculative decoding scaffolding
│           └── step_caching/      Stepcache patterns
├── benchmarks/                    Modal H100 SXM benchmarks
│   ├── baseline/oasis_modal.py    Canonical DDIM-10 sampler
│   ├── optimised/                 28 optimization variants (DPM++, Difficulty, TaylorSeer, megastack, ...)
│   ├── results/all_runs.csv       72-row aggregated results
│   └── runs/                      Raw per-experiment JSON outputs
├── docs/
│   ├── action_aware_step_scheduling.md   Comprehensive project journal (~1,400 lines)
│   ├── poster_prompt.md                  CMU SCS poster template prompt
│   └── figures/                          Source PNGs
└── scripts/plot_results.py        Regenerate report figures from CSV
```

---

## What's in the report

The 8-page MLSys-format report (`report/main.tex`) covers:

1. **Introduction** with three contributions
2. **Problem** definition, baseline (Table 1), goals, workload-specific constraints
3. **Related work** spanning ODE solvers, feature caching, KV-cache compression, compute scheduling, and world model inference
4. **System overview** with a single pipeline figure
5. **Method** with DPM-Solver++ 2M math, the action-magnitude bucket equation, stacking analysis, and TaylorSeer with per-frame reset
6. **Evaluation** with the headline table, length-robustness analysis, per-family Pareto, distribution-dominates-skip-count ablation, all-configurations scatter, and failed-direction discussion
7. **Conclusion** in one paragraph

Compile in Overleaf, or locally with `pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`.

---

## Acknowledgements

Builds on the upstream [open-oasis](https://github.com/etched-ai/open-oasis) repository (Etched AI / Decart) for the Oasis DiT, ViT-L VAE, sigmoid beta schedule, and sliding-window temporal context. The TaylorSeer block-feature predictor is adapted from [arXiv:2503.06923](https://arxiv.org/abs/2503.06923) with our per-frame state reset addition. DPM-Solver++ multistep formula from [arXiv:2211.01095](https://arxiv.org/abs/2211.01095).

## Citation

If you reference this work:

```
@misc{nallamothu2026oasis,
  author = {Bhuvan Nallamothu},
  title  = {Efficient Inference Engine for Interactive World Models:
            Action-Driven Compute Adaptation for Autoregressive World Models},
  year   = {2026},
  note   = {CMU 15-442/15-642 Machine Learning Systems final project},
  howpublished = {\url{https://github.com/iambhuvan/efficient-inference-world-models}}
}
```
