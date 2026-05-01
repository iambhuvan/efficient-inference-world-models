# Poster Generation Prompt — CMU SCS Style

Paste the block below into Gemini (Imagen / 2.5 Pro), DALL-E, Midjourney, or any
image-gen model that accepts long structured prompts. The poster uses the
standard CMU School of Computer Science red-banner header with white text,
red section bars, and a clean professional chart grid in the body.

All charts use uniform dimensions and a consistent visual style — no variable
sizes, no ad-hoc decoration. The hero results table, three giant numbers, and
four uniform professional charts carry the impact.

---

```
Create a single-page academic research poster in portrait orientation, sized for 30 x 40 inches at 300 DPI (9000 x 12000 pixels, aspect ratio 3:4).

OVERALL STYLE
- Standard CMU School of Computer Science research poster style.
- Solid CMU red (#c41230) full-width header banner at the top, ~12% of poster height.
- Body below the banner is clean white background.
- Section labels rendered as thin solid red bars (#c41230, 80px tall) with white bold ALL CAPS text — mimics CMU SCS section header style.
- Body text near-black (#1a1a1a), 18pt for paragraphs, 16pt for bullets.
- Single accent color for OURS row in the results table and the three giant numbers: warm crimson (#c41230, same as the header).
- Sans-serif throughout (Inter or Helvetica).
- 2-column body grid below the header with sharp alignment.
- Generous whitespace, hairline dividers, no decorative gradients.

CMU-RED HEADER BANNER (top, full width, ~12% of height)
- Solid red (#c41230) background with all text in white
- Top-left (22pt regular white): "Carnegie Mellon University"
- Top-right (22pt regular white, right-aligned): "Poster #134  ·  Bhuvan Nallamothu"
- Centered title (80pt bold serif or strong sans, white): "Efficient Inference Engine for Interactive World Models"
- Centered subtitle directly below (30pt regular white, lighter weight): "Action-Driven Compute Adaptation for Autoregressive World Models — 3.54× speedup on Open-Oasis 500M, validated across 950 real Minecraft frames on H100"
- Thin white hairline at the bottom of the banner separating it from the body

SECTION HEADER STYLE (applies to every section label below the banner)
- Each header is a solid red (#c41230) bar, full column width, 80px tall
- White bold ALL CAPS text, 26pt, left-aligned with 30px left padding
- Examples: "THE PROBLEM", "KEY INSIGHT", "METHOD", "RESULTS", "WHY IT WORKS", "ABLATIONS", "FUTURE WORK", "TAKEAWAY"

CHART STYLE (UNIFORM across all charts — this is critical)
- All charts rendered at the same canvas size: 2400 px wide × 1500 px tall (moderate, NOT poster-dominating)
- All charts use the same plotting framework look: matplotlib-style, white background, no top/right spines, light gray (#cccccc) gridlines on Y axis only
- Same font family and sizes across all charts: 16pt axis labels, 14pt tick labels, 18pt chart title in deep navy (#1f2a44)
- Consistent color palette across all charts:
  - OURS / headline: warm crimson (#c41230)
  - DPM-Solver++ / Difficulty / TaylorSeer family: steel blue (#2c5282)
  - Stepcache / drift_cache: navy (#1f2a44)
  - Failure modes (action_kv, warmstart, megastack): gray (#777777)
  - Baseline reference: black (#000000)
- Every chart has axis labels, a one-line caption below, and a legend if more than 3 series
- No 3D effects, no shadows, no gradients

(1) THE PROBLEM (left column, top)
Body (2 sentences, 18pt): Open-Oasis 500M runs 10 DDIM steps per frame regardless of whether the player is still or sprinting. On H100 this caps interactive generation at 2 fps, an order of magnitude below the 30 fps real-time target.
Visual: A horizontal action timeline strip (full column width × 600 px tall, no chart axes). Twelve frame slots in a row, each showing the same flat block of "10 steps" labeled below. Above the strip, a varied action stream rendered as small dots for idle frames and large arrows for camera turns or jumps. A red label connects to the strip reading "every frame pays the same compute regardless of motion."

(2) KEY INSIGHT (left column, middle)
Body (2 sentences): The 25-dim Minecraft action vector that drives the world model also tells you how much new content the next frame must generate. Free per-frame compute knob, no retraining required.
Bullets:
  - idle frame · ‖a‖_1 < 0.5 → 2 steps suffice
  - moderate frame · ‖a‖_1 < 1.5 → 3 steps
  - active frame · ‖a‖_1 ≥ 1.5 → 5 steps (full ceiling)

CHART 1 — "Action magnitude distribution across 950 real Minecraft frames" (uniform 2800×2000 canvas)
- Type: histogram + overlay
- X axis 0 to 4 labeled "Action magnitude ||a||_1"
- Y axis 0 to 200 labeled "Frame count"
- Three vertical shaded zones: blue (idle, <0.5), light blue (moderate, 0.5-1.5), crimson (active, >=1.5)
- Histogram bars colored by zone
- Annotation: "60% idle  ·  30% moderate  ·  10% active"
- Caption: "Free per-frame difficulty signal: action magnitude correlates with how much new content the next frame must generate."

(3) METHOD (left column, bottom)
Body (3 sentences): DPM-Solver++ 2M halves the per-frame ceiling from 10 to 5 with no quality cost. An action-magnitude bucket schedule then picks 2, 3, or 5 steps within that ceiling per frame. Two orthogonal step-count cuts, stacked.
Equation (monospace, navy, 18pt, centered):
    steps(frame_i) = 2 if ||a_i||_1 < 0.5
                   = 3 if ||a_i||_1 < 1.5
                   = 5 otherwise
Visual: ONE pipeline schematic, 2800×1400 px (slightly shorter than chart canvas height since it is a diagram). Three connected stages with arrows. Stage 1: a 25-bar action vector icon with a magnitude meter. Stage 2: a bucket selector showing "2 / 3 / 5 steps" with the meter pointing into one box. Stage 3: a denoising loop labeled "DPM-Solver++ 2M (5-step ceiling)" feeding into "next frame latent."
Below the schematic, a single text line in monospace, 16pt navy:
    9,500 forwards (DDIM-10)  →  4,750 (+DPM++)  →  2,683 (+Difficulty)   = 71.8% saved

(4) RESULTS (full-width hero band, center of poster)
Body (1 sentence, 18pt): At 950 frames our recipe delivers 3.54× speedup while staying within 0.1 dB of baseline self-coherence — every other family of optimization either delivers less speedup or compounds quality loss across length.

Hero results table (full poster width, the highlighted row in solid crimson with white bold text — the crimson background alone signals this is the headline recipe; do NOT add any "OURS:" prefix, "(Ours)" suffix, "Mine" badge, or other marker):

  Method                       Speedup    Coherence Δ          Forwards    Verdict
  DDIM-10 baseline             1.00×      +0.00 dB             9,500       reference
  DPM-Solver++ 2M              1.98×      +4.55 dB             4,750       stable smoothing
  Difficulty 2/4/10            2.69×      +0.76 dB             3,656       length robust
  TaylorSeer order=2           2.52×      0/0 val failures     5,592       length robust
  Stepcache interval_3         2.47×      -1.56 dB             3,134       stable
  DPM-Solver++ × Difficulty    3.54×      -0.02 dB             2,683       project headline   ← crimson row

Below the table, three giant number callouts in 220pt bold crimson, evenly spaced across the full width, each with a 24pt navy label below it:
   3.54×       -0.02 dB         71.8%
   speedup     coherence        forwards saved

Below the callouts, render TWO charts side by side, both at the uniform 2800×2000 canvas size:

CHART 2 — "Length robustness: 32 vs 950 frames" (uniform canvas)
- Type: scatter with connecting lines
- X axis: Δ vs_prev (dB) from -25 to +15, labeled "Δ vs_prev (dB)  ·  closer to zero is better"
- Y axis: speedup from 1.0 to 4.0, labeled "Speedup over DDIM-10"
- Pale green vertical band from -2 to +2 dB labeled "quality preservation zone"
- Each variant: a colored line connecting an open circle (32-frame measurement) to a filled square (950-frame measurement)
- Three named annotations: (a) "Ours: DPM++ × Difficulty" with circle and square BOTH in the green band near (0, 3.5), star marker, crimson; (b) "TaylorSeer order=2" with circle and square overlapping at (0, 2.55), steel blue; (c) "Megastack collapse" with the connecting line dragged 11 dB to the left in red, illustrating length failure
- Caption: "Variants whose square stays inside the green band are length robust."

CHART 3 — "Speedup vs quality preservation, all 16 optimization families" (uniform canvas)
- Type: horizontal bar chart
- Y axis (top to bottom by speedup descending): OURS DPM++×Difficulty, Difficulty aggressive, TaylorSeer o=2, Stepcache interval_3, Drift_Cache τ=0.20, DPM-Solver++ 2M, Warmstart k=6, Action-KV sim=0.95, ROI denoise, baseline DDIM-10
- X axis 0 to 4.0 labeled "Speedup over DDIM-10 (best variant per family at 950 frames)"
- Bars colored by quality verdict: crimson for OURS, steel blue for within-2-dB-band variants, gray for quality-compromised variants
- To the right of each bar a label "Nx (Δ dB)" with a small ✓ or ✗ symbol in green or red
- Vertical dotted gray reference at X = 1.0
- Caption: "Crimson = headline. Blue = quality preserved (|Δ| < 2 dB). Gray = quality compromised."

(5) WHY IT WORKS (right column, top)
Body (3 sentences): Cutting how many forwards run per frame introduces no per-step error. Methods that substitute *what* runs per step (quantized KV, action-conditional v reuse, anchor blends) all compound their substitution noise across the autoregressive feedback loop. By 950 frames every substitution method has crashed 5 to 22 dB below baseline coherence; pure step count reduction holds within 0.1 dB.

CHART 4 — "Per-step substitution compounds; step-count reduction does not" (uniform canvas)
- Type: line chart, two series
- X axis: frames generated, 0 to 950, labeled "Frames generated"
- Y axis: Δ vs_prev coherence (dB) from -25 to +5, labeled "Δ vs_prev (dB) vs DDIM-10 baseline"
- Series 1 (red, "per-step substitution mean"): starts near 0 at frame 32, drops sharply to -22 dB by frame 950 (action_kv style)
- Series 2 (steel blue, "step-count reduction mean"): hovers near 0 across the full range, stays within ±1 dB
- Horizontal pale green band -2 to +2 labeled "quality preservation zone"
- Caption: "Substitution noise compounds across the autoregressive feedback loop; step-count reduction does not."

(6) ABLATIONS (right column, middle)
Render as three stacked findings cards (each with a thin red left border accent, 22pt bold red header, 16pt body):

CARD 1 — DISTRIBUTION > COUNT
At identical 50% skip rate, interleaved skip preserves coherence (-0.64 dB) but clustered skip collapses (-7.17 dB).

CARD 2 — BUCKET > SMOOTH
Hard cutoff sends idle frames to the 2-step floor for 45% savings. Linear and exponential schedules leak compute and lose 30% of the speedup.

CARD 3 — STACK ONLY WHEN ORTHOGONAL
DPM++ × Difficulty composes cleanly (3.54× at -0.02 dB). DPM++ × stepcache_alternate compounds blur (-6.72 dB). DPM++ × Difficulty × TaylorSeer collapses at length (-11.29 dB).

(7) FUTURE WORK (right column, bottom)
Body (1 sentence): The headline number is conservative; three orthogonal directions could push it higher.
Bullets (small, plain text):
  - CG-Taylor (arXiv 2508.02240) — confidence-gated block prediction with first-block error gate
  - Persistent KV cache architecture for autoregressive video DiTs (~200-line attention rewrite)
  - Attention-mask injection for region-of-interest spatial sparsification (~100 lines)

(8) TAKEAWAY (bottom, full width, moderate whitespace above and below — proportional, NOT a giant block)
Render in 50pt bold navy, one or two lines max, center aligned, no decorative box:

   "The action signal is the schedule.  3.54× faster Open-Oasis inference at preserved coherence on 950 real Minecraft frames."

No footer line below the takeaway. The takeaway block is the final visual element on the poster.

FINAL DESIGN ENFORCEMENT
- Top of poster is dominated by the CMU-red header banner (white title, white subtitle, white author name, white "Carnegie Mellon University" and "Poster #134")
- Section headers below the banner are red bars with white bold ALL CAPS text — uniform style
- All FOUR charts use the SAME canvas dimensions (2400×1500 px), the SAME font sizes, the SAME color palette, and the SAME chart-style spines/gridlines — no variable chart sizes, and charts are moderate-sized (NOT poster-dominating)
- Hero results table is full poster width with the highlighted row in solid crimson and white bold text — the row label reads "DPM-Solver++ × Difficulty" with no "OURS:" prefix, "(Ours)" suffix, "Mine" badge, or any other marker; the crimson background IS the marker
- Three giant numbers (3.54×, -0.02 dB, 71.8%) are 220pt bold crimson, the largest non-title elements
- Takeaway block at the bottom is 50pt bold navy on one or two lines, proportional to body — not poster-dominating
- Three findings cards in section 6 are uniform width and height with the same red left border accent
- No paragraph longer than 3 sentences anywhere
- No clip art, no stock icons, no decorative gradients
```

---

## Iteration tips

If the model renders something off, target the specific element rather than re-prompting from scratch.

- **Header banner doesn't look like CMU**: "redo only the top header. Make it solid CMU red #c41230, full width, 12% of poster height, with all white text. Title centered in 80pt bold, subtitle below in 30pt regular, 'Carnegie Mellon University' top-left and 'Poster #134 · Bhuvan Nallamothu' top-right, both 22pt white."
- **Charts have different sizes**: "redo all four charts at the exact same canvas dimensions: 2800 px wide × 2000 px tall. Same font sizes (16pt axis labels, 14pt ticks, 18pt title), same color palette (crimson #c41230 for ours, steel blue #2c5282 for in-band variants, gray #777 for compromised variants), same minimal spine style."
- **Table looks weak**: "redo the results table only, with the OURS row in solid crimson background and 18pt white bold text. Other rows white background with thin gray hairline borders. Header row in navy."
- **Numbers not big enough**: "make the three callouts (3.54×, -0.02 dB, 71.8%) at least 1.5x larger than the section headers, in solid crimson, with 220pt bold sans-serif."
- **Section headers don't match CMU style**: "redo every section header as a solid red (#c41230) horizontal bar, 80px tall, with white bold ALL CAPS text at 26pt, left-aligned with 30px padding."

## Source data

The poster claims are backed by `benchmarks/results/all_runs.csv` (72 rows) and the following project artifacts:

- `docs/action_aware_step_scheduling.md` — full report with 17 bug fixes catalog
- `docs/figures/*.png` — five publication-quality charts already rendered (regenerate with `python3 scripts/plot_results.py`)
- `benchmarks/runs/optimised_kernels/*.json` — raw measurement JSONs

The headline 3.54× is from `oasis_diff_dpmpp_r2m3_f950.json`. The TaylorSeer 2.52× is from `oasis_taylorseer_o2_thr0.15_f950.json`. The megastack collapse is from `oasis_megastack_f950.json`.
