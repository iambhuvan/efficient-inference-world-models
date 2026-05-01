# Poster Generation Prompt (template-style)

Paste the block below into Gemini (Imagen / 2.5 Pro) or any image-gen model.
Designed for chart-light visual hierarchy: the results table and three giant
numbers carry the impact, with one method schematic and one length-robustness
scatter as the only large visuals.

---

## Prompt:

Create a single-page academic research poster in portrait orientation, sized for 30 x 40 inches at 300 DPI (9000 x 12000 pixels, aspect ratio 3:4).

Style requirements:

CMU-red header banner at the top spanning the full width, taking ~12% of the poster height. The header is solid CMU red (#c41230) with all text in white, matching standard CMU SCS poster style.
Body area below the header is clean white background.
Section headers below the banner are small red bars (#c41230, ~80px tall) with white bold text, ALL CAPS, mimicking the CMU SCS poster section header style.
Body text in dark gray near-black (#1a1a1a) for readability.
Single accent color: warm crimson (#c0392b) used only for the OURS row in the results table and the three giant numbers.
Minimalist layout with sharp alignment and a 2-column grid.
Subtle dividers, clean hairlines, soft section separation.
Use schematic diagrams plus 2–4 charts. Bar charts, histograms, and scatters are welcome when professionally styled and UNIFORMLY SIZED. Skip pie charts (they rarely add value on a research poster).
Prioritize legibility from 3–6 feet away.
Maintain generous whitespace for a polished look.
Use concise, punchy text (max 2–3 sentences per section).
Emphasize key numbers visually with 200pt+ bold typography.
Ensure all text is clean, professional, typo-free.

CMU-RED HEADER BANNER (top of poster, ~12% of height, full width)

Layout inside the red banner:
- Top-left corner (small white text, 22pt regular): "Carnegie Mellon University"
- Top-right corner (small white text, 22pt regular, right-aligned): "Poster #134  ·  Bhuvan Nallamothu"
- Centered title (white, 80pt bold serif or strong sans-serif, two lines if needed):
    "Efficient Inference Engine for Interactive World Models"
- Centered subtitle directly below the title (white, 30pt regular, slightly lighter weight):
    "Action-Driven Compute Adaptation for Autoregressive World Models — 3.54× speedup on Open-Oasis 500M, validated across 950 real Minecraft frames on H100"
- Bottom edge of the banner: a thin white hairline separating header from body
- Optional: a small CMU shield / wordmark in the top-left of the banner if available; otherwise just the "Carnegie Mellon University" text is sufficient

The banner sets the entire poster's identity. Everything below is on white.

Authors: Bhuvan Nallamothu (already shown in red banner top-right)

Affiliation: Carnegie Mellon University · 15-849 ML Systems · Spring 2026 (already represented by the banner; if a second affiliation line is desired, place under the subtitle in the banner in 22pt white regular)

Poster ID: #134 (in red banner, top-right)

(1) — Problem

Text: Open-Oasis 500M runs 10 DDIM steps per frame regardless of whether the player is still or sprinting. On H100 this caps interactive generation at 2 fps, an order of magnitude below the 30 fps real-time target.

Visual: A horizontal action timeline strip — twelve Minecraft-style frame thumbnails in a row, each showing actual Minecraft scene content (sky, grass, tree) that VISIBLY CHANGES as the player acts. Frames during idle stretches look nearly identical to each other (same tree position, same sky). Frames during camera-turn or jump arrows show the scene rotating: the tree shifts left or right across consecutive frames, the horizon tilts, the sky-ground ratio changes — you can SEE the camera turning by reading left to right. Above the strip, a varied action stream rendered as small dots for idle frames and large bold arrows for camera turns or jumps, with a "camera turns" mini-label spanning the section where arrows cluster. Below each frame, a uniform flat block of "10 steps" labeled. Below the entire strip, a red caption reading "every frame pays the same compute regardless of motion."

(2) — Key Insight

Text: The 25-dim Minecraft action vector that drives the world model also tells you how much new content the next frame must generate. A free per-frame compute knob, no retraining required.

Bullet points:
- idle frame · ‖a‖₁ < 0.5 → 2 steps suffice
- moderate frame · ‖a‖₁ < 1.5 → 3 steps
- active frame · ‖a‖₁ ≥ 1.5 → 5 steps (full ceiling)

Visual: NONE. Use only the bullets and crisp typography.

(3) — Method / Approach

Text: DPM-Solver++ 2M halves the per-frame ceiling from 10 to 5 with no quality cost. An action-magnitude bucket schedule then picks 2, 3, or 5 steps within that ceiling per frame. Two orthogonal step-count cuts, stacked.

Equation:
```
steps(frame_i) = 2  if ||a_i||_1 < 0.5
               = 3  if ||a_i||_1 < 1.5
               = 5  otherwise
```

Visual: ONE pipeline schematic — three connected stages with arrows.
Stage 1: a 25-bar action vector icon with a magnitude meter underneath.
Stage 2: a bucket selector showing the three boxes "2 / 3 / 5 steps" with the meter pointing into one box.
Stage 3: a denoising loop labeled "DPM-Solver++ 2M (5-step ceiling)" feeding into "next frame latent".

DO NOT add a second Minecraft frame strip in this section. The Problem section already has the only Minecraft strip on the entire poster — the Method section uses just the pipeline schematic above plus the cumulative-forwards line below.

Below the pipeline schematic, a single text line in monospace, no bar chart:
`9,500 forwards (DDIM-10)  →  4,750 (+DPM++)  →  2,683 (+Difficulty)   = 71.8% saved`

(4) — Results

Table (the highlighted row in solid crimson background with white bold text — no "OURS:" prefix; the crimson highlight itself signals our recipe):

| Method                            | Speedup   | Coherence Δ      | Forwards  |
|-----------------------------------|-----------|------------------|-----------|
| DDIM-10 baseline                  | 1.00x     | +0.00 dB         | 9,500     |
| DPM-Solver++ 2M                   | 1.98x     | +4.55 dB         | 4,750     |
| Difficulty 2/4/10                 | 2.69x     | +0.76 dB         | 3,656     |
| TaylorSeer order=2                | 2.52x     | 0/0 val failures | 5,592     |
| Stepcache interval_3              | 2.47x     | -1.56 dB         | 3,134     |
| **DPM-Solver++ × Difficulty** | **3.54x** | **-0.02 dB**     | **2,683** |

Text: At 950 frames on real Minecraft action data, our recipe delivers 3.54× speedup while staying within 0.1 dB of baseline self-coherence — every other family of optimization either delivers less speedup or compounds quality loss across length.

Visual emphasis:
- The highlighted row "DPM-Solver++ × Difficulty" uses solid crimson background with white bold text. The crimson highlight alone signals this is the headline recipe. Do NOT add any "OURS:" prefix, "(Ours)" suffix, "Mine" badge, decorative banner, or other marker — the crimson row IS the marker.
- Three giant metric callouts in 220pt bold crimson, evenly spaced across the RIGHT COLUMN width (within the Results section ONLY), each with a 24pt navy label below:
  **3.54×** (speedup) · **-0.02 dB** (coherence) · **71.8%** (forwards saved)
- THESE THREE CALLOUTS APPEAR EXACTLY ONCE ON THE ENTIRE POSTER — only inside the Results section, between the table and the scatter chart. DO NOT duplicate them anywhere else (not under the Method section, not under Why It Works, not above the Takeaway). They are the single hero numerical block.
- Below the callouts, ONE chart — a length-robustness scatter at COMPACT size (~1800 px wide × 1100 px tall, occupying roughly half the column width and ~25% of the column height — clearly smaller than the giant number callouts above it, NOT poster-dominating):
  X axis: Δ vs_prev (dB) from -25 to +15
  Y axis: speedup from 1.0 to 3.8
  Pale green vertical band from -2 to +2 dB labeled "quality preservation zone"
  Plot EXACTLY THREE LINES on this chart, NO MORE, NO LESS. Each line is a single colored line connecting a 32-frame open circle to a 950-frame filled square. Every line has its own annotation label visible on the chart. Do NOT add any extra variants, helper lines, sample points, gridline points, or unlabeled markers.
  The three lines are:
    (a) "Ours: DPM++ × Difficulty" - crimson line, circle near (-1.5, 3.05) and square near (0, 3.5), both inside the green band, marked with a small gold star next to the square
    (b) "TaylorSeer order=2" - steel blue line, circle and square nearly overlapping at (0, 2.55), both inside the green band
    (c) "Megastack collapse" - red line, circle near (+1.5, 3.5) inside the green band, square crashed to (-11, 3.8) far outside the band on the left, illustrating length failure
  Caption: "Circle = 32 frames, square = 950 frames. Variants whose square stays inside the green band are length robust."

(5) — Why It Works

Text: Cutting how many forwards run per frame introduces no per-step error. Methods that substitute *what* runs per step (quantized KV, action-conditional v reuse, anchor blends) all compound their substitution noise across the autoregressive feedback loop. By 950 frames every substitution method has crashed 5 to 22 dB below baseline coherence; pure step count reduction holds within 0.1 dB.

Additional note: The action signal in the input pipe doubles as a free per-frame difficulty knob — no retraining, no extra inference cost.

(6) — Ablations & Future Work

Render as three stacked findings cards (each with a navy left border accent, 22pt bold header, single sentence body):

Card 1 — Distribution > count
At identical 50% skip rate, interleaved skip preserves coherence (-0.64 dB) but clustered skip collapses (-7.17 dB).

Card 2 — Bucket > smooth
Hard cutoff sends idle frames to the 2-step floor for 45% savings; linear and exponential schedules leak compute and lose 30% of the speedup.

Card 3 — Stack only when orthogonal
DPM++ × Difficulty composes cleanly (3.54× at -0.02 dB). DPM++ × stepcache_alternate compounds blur (-6.72 dB). DPM++ × Difficulty × TaylorSeer collapses at length (-11.29 dB).

Future work (small bullets, plain text):
- CG-Taylor (arXiv 2508.02240) — confidence-gated block prediction with a runtime first-block error gate
- Persistent KV cache architecture for autoregressive video DiTs (~200-line attention rewrite)
- Attention-mask injection for region-of-interest spatial sparsification (~100 lines)

(7) — Takeaway

Place near the bottom, centered and visually emphasized in 50pt bold navy across two lines with moderate whitespace above and below (NOT a giant block — keep it proportional to body section headers):

"The action signal is the schedule.  3.54× faster Open-Oasis inference at preserved coherence on 950 real Minecraft frames."

Design:
- Center aligned
- Bold but proportional typography (50pt — about half the title size, not poster-dominating)
- One or two lines maximum, single visual block
- No decorative elements, no underlines, no boxes — only spacing and contrast carry the emphasis

Section header style (applies to all section labels: "THE PROBLEM", "KEY INSIGHT", "METHOD", "RESULTS", "WHY IT WORKS", "ABLATIONS & FUTURE WORK", "TAKEAWAY"):
- Each section header is a thin solid CMU-red bar (#c41230) about 80px tall and full column width
- Inside the red bar: white bold text, ALL CAPS, ~26pt, left-aligned with ~30px left padding
- This matches the CMU SCS poster style shown in the reference

Final design enforcement:
- POSTER SIZE IS NON-NEGOTIABLE: portrait orientation, 30 inches wide × 40 inches tall, 9000×12000 pixels at 300 DPI, aspect ratio EXACTLY 3:4. If the tool offers a different default size, override it. The layout assumes this canvas — section spacing, chart sizes, and font sizes are all calibrated for a 30×40 inch poster viewed from 3-6 feet away.
- Top of poster is dominated by the CMU-red header banner (title + subtitle + author + Carnegie Mellon University + Poster #134 in white text)
- Section headers below the banner are red bars with white bold ALL-CAPS text (CMU SCS style)
- Each section is clearly separated by ample whitespace (~80px vertical gap) and the bold red section header bar — sections must NOT visually blur into each other
- The three giant numbers (3.54×, -0.02 dB, 71.8%) appear EXACTLY ONCE on the entire poster, inside the Results section only. DO NOT render this callout block under any other section.
- The Method section contains ONE visual only: the pipeline schematic. Do NOT add any extra Minecraft frame strip in the Method section.
- The Problem section is the ONLY place that contains a Minecraft frame strip on the entire poster.
- Total large charts: ONE (the length-robustness scatter in Results)
- Total schematic diagrams: ONE (the method pipeline in section 3) plus ONE Minecraft frame strip (in the Problem section only) — those are illustrative diagrams, not "charts"
- Hero results table: full poster width with the highlighted row in solid crimson and white bold text
- No paragraph longer than 3 sentences anywhere
- Bar charts and histograms are fine when professionally rendered and ALL CHARTS share the SAME canvas size (2400×1500 px), font sizes, and color palette — no variable sizing. Skip pie charts.

No footer line below the takeaway. The takeaway is the final visual element on the poster.
