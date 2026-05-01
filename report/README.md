# Final Course Project Report. Overleaf upload guide

This directory holds a self-contained 8-page MLSys-style report ready to drop into Overleaf.

## Files

```
report/
├── main.tex          The paper (compiles standalone)
├── refs.bib          BibTeX references (DPM-Solver++, TaylorSeer, etc.)
├── figures/          5 PNG figures (also live in docs/figures/)
│   ├── headline_breakdown.png
│   ├── length_robustness.png
│   ├── per_family_pareto.png
│   ├── skip_pattern_distribution.png
│   └── speedup_vs_quality.png
└── README.md         (this file)
```

## Upload to Overleaf in three minutes

1. Go to https://www.overleaf.com → **New Project** → **Blank Project**.
2. Name it `MLSys-Final-Report-Bhuvan`.
3. Delete the default `main.tex` Overleaf creates.
4. Click **Upload** (top toolbar) and drag in:
   - `main.tex`
   - `refs.bib`
   - The entire `figures/` folder (Overleaf accepts folder upload, or upload each PNG)
5. In the Overleaf project root, the file tree should look like:
   ```
   main.tex
   refs.bib
   figures/headline_breakdown.png
   figures/length_robustness.png
   figures/per_family_pareto.png
   figures/skip_pattern_distribution.png
   figures/speedup_vs_quality.png
   ```
6. Click **Recompile** (top right). The first compile should produce the 8-page PDF.

## Compile settings (Overleaf top-right gear)

- **Compiler:** pdfLaTeX (default)
- **TeX Live version:** 2024 or later (default)
- **Main document:** `main.tex`

The bibliography uses `natbib` + `abbrvnat`. Overleaf will run BibTeX automatically on first compile; if citations show as `[?]`, click Recompile a second time.

## What's in the report

8 pages following the MLSys 2025 conference structure required by the course logistics PDF:

1. **Abstract**, a 150 word summary of the recipe, headline numbers, and key findings.
2. **Introduction**, motivation and three contributions.
3. **Problem**, Oasis architecture, baseline numbers (16/32/75/950 frames), goals, and workload constraints.
4. **Related Work**, four paragraphs covering ODE solvers, feature caching, KV cache compression, compute scheduling, and world model inference.
5. **System Overview**, a single TikZ pipeline figure (Stage 1 action vector to Stage 2 bucket selector to Stage 3 DPM-Solver++ 2M).
6. **Method**, DPM-Solver++ 2M math, action magnitude bucket schedule, stacking analysis, TaylorSeer with per-frame reset, and implementation details.
7. **Evaluation**, headline result table, length robustness analysis, per-family Pareto, skip pattern ablation, all configurations scatter, and failed direction discussion.
8. **Conclusion**, a single paragraph.

Total: 5 PNG figures, 1 inline TikZ figure, 2 tables, ~25 BibTeX references.

## After upload, quick sanity checks

After the first successful compile, verify:

- [ ] Page count is **≤ 8** (the spec says "up to 8 pages without references"). References are allowed to spill onto page 9+.
- [ ] All figures render at full column width with crisp text.
- [ ] BibTeX citations resolve (no `[?]` markers).
- [ ] Author block shows `Bhuvan Nallamothu` / `bnallamo` / Carnegie Mellon University.
- [ ] Title on page 1 reads *Efficient Inference Engine for Interactive World Models* with the italic subtitle *Action-Driven Compute Adaptation for Autoregressive World Models* below it.
- [ ] The system overview figure (Section 4) renders correctly via TikZ.

## If you want the exact MLSys 2025 style file

The current `main.tex` uses standard LaTeX packages to approximate the MLSys two-column conference layout. If you want the literal MLSys 2025 style:

1. Download `mlsys2025.sty` from the conference repo (search `mlsys2025` on GitHub).
2. Drop it into the Overleaf project root.
3. In `main.tex`, replace the preamble (everything from `\documentclass` through `\predate{}\postdate{}\date{}`) with:
   ```latex
   \documentclass{article}
   \usepackage{mlsys2025}
   \mlsysauthor{Bhuvan Nallamothu}{Carnegie Mellon University}
   \mlsystitle{Efficient Inference Engine for Interactive World Models}
   ```
4. Recompile.

The content sections will work without any changes.

## Submission checklist (Autolab, due May 3 11:59 PM EST)

- [ ] Final PDF (Overleaf → Menu → Download → PDF)
- [ ] Source code zip from this repo (the `worldserve/` and `benchmarks/` directories)
- [ ] Confirm file size limit on Autolab (typically 25 MB)

## Reproducibility one-liner cited in the report

```bash
PYTHONPATH=benchmarks modal run \
    benchmarks/optimised/oasis_difficulty_dpmpp_modal.py \
    --num-frames 950 --seed 42 \
    --solver-full-steps 5 --solver-mid-steps 3 --solver-reduced-steps 2
```

This produces the headline 3.54× / -0.02 dB / 950 frames result on a Modal H100 SXM in roughly 7 minutes (baseline) plus 2 minutes (stacked).
