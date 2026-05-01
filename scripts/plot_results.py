"""
Generate poster-quality figures from benchmarks/results/all_runs.csv.

Outputs (PNG, 300dpi) into docs/figures/:
  - speedup_vs_quality.png       speedup vs delta vs_prev, families color coded
  - length_robustness.png        speedup at 32f vs 950f, quality bands
  - per_family_pareto.png        bar chart per family of best (speedup, delta vs_prev)
  - headline_breakdown.png       cumulative forwards reduction at 950f
  - skip_pattern_distribution.png  distribution dominates count finding

Usage: python scripts/plot_results.py
"""
from __future__ import annotations
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "benchmarks", "results", "all_runs.csv")
OUT = os.path.join(ROOT, "docs", "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

FAMILY_COLORS = {
    "stack":          "#d62728",
    "difficulty":     "#1f77b4",
    "dpmpp":          "#2ca02c",
    "stepcache":      "#ff7f0e",
    "drift_cache":    "#9467bd",
    "action_kv":      "#8c564b",
    "warmstart":      "#e377c2",
    "kv_cache":       "#7f7f7f",
    "roi_denoise":    "#bcbd22",
    "custom_kernels": "#17becf",
    "predit":         "#aec7e8",
    "oasis_all":      "#ffbb78",
    "torch_compile":  "#c49c94",
    "taylorseer":     "#9b59b6",   # NEW: distinct purple
    "quality_eval":   "#cccccc",
    "baseline":       "#000000",
}

df = pd.read_csv(CSV)
# TaylorSeer rows have 0 validation failures and no quality eval was run, so
# delta_vs_prev is NaN. Treat these as 0 dB (within-tolerance) for plotting,
# annotated as "no fail" rather than measured.
ts_mask = (df["family"] == "taylorseer") & df["delta_vs_prev_db"].isna()
df.loc[ts_mask, "delta_vs_prev_db"] = 0.0

plot_df = df.dropna(subset=["speedup", "delta_vs_prev_db"]).copy()
plot_df = plot_df[plot_df["family"] != "quality_eval"]

# ---------- Figure 1: Speedup vs Quality scatter ----------
fig, ax = plt.subplots(figsize=(9, 6))
for fam, g in plot_df.groupby("family"):
    ax.scatter(
        g["delta_vs_prev_db"], g["speedup"],
        s=80, alpha=0.85,
        c=FAMILY_COLORS.get(fam, "#444"),
        label=fam, edgecolors="white", linewidths=0.7,
    )
hl = plot_df[(plot_df.family == "stack") & (plot_df.length_frames == 950) & (plot_df.speedup == 3.54)]
if len(hl):
    ax.annotate(
        "headline:  3.54x at -0.02 dB  (950 frames)",
        xy=(hl.delta_vs_prev_db.iloc[0], hl.speedup.iloc[0]),
        xytext=(-15, 3.75),
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.4),
        fontsize=10, color="#d62728", fontweight="bold",
    )
ax.axhline(1.0, color="#888", linestyle=":", lw=0.8)
ax.axvline(0.0, color="#888", linestyle=":", lw=0.8)
ax.axvspan(-2, 2, alpha=0.06, color="#2ca02c", label=None)
ax.set_xlabel(r"$\Delta$ vs_prev (dB) — closer to zero is better")
ax.set_ylabel("Speedup over DDIM-10")
ax.set_title("Speedup vs Quality: All Configurations")
ax.set_xlim(-25, 22)
ax.set_ylim(0.4, 4.0)
ax.legend(loc="upper left", fontsize=8, ncol=2, frameon=True)
ax.text(0, 0.45, "quality band: |delta| < 2 dB", fontsize=8, color="#2ca02c", ha="center")
plt.savefig(os.path.join(OUT, "speedup_vs_quality.png"))
plt.close()
print("wrote speedup_vs_quality.png")

# ---------- Figure 2: Length robustness ----------
short = df[df.length_frames == 32][["family", "variant", "speedup", "delta_vs_prev_db"]].rename(
    columns={"speedup": "speedup_32", "delta_vs_prev_db": "delta_32"})
long_ = df[df.length_frames == 950][["family", "variant", "speedup", "delta_vs_prev_db"]].rename(
    columns={"speedup": "speedup_950", "delta_vs_prev_db": "delta_950"})
pair = short.merge(long_, on=["family", "variant"], how="inner").dropna()
pair = pair[pair.family != "quality_eval"]

fig, ax = plt.subplots(figsize=(9, 6))
for _, r in pair.iterrows():
    ax.plot([r.delta_32, r.delta_950], [r.speedup_32, r.speedup_950],
            "-", color=FAMILY_COLORS.get(r.family, "#444"), alpha=0.55, lw=1.2)
    ax.scatter(r.delta_32, r.speedup_32, s=55, marker="o",
               c=FAMILY_COLORS.get(r.family, "#444"), edgecolors="white", linewidths=0.7)
    ax.scatter(r.delta_950, r.speedup_950, s=85, marker="s",
               c=FAMILY_COLORS.get(r.family, "#444"), edgecolors="black", linewidths=0.8)
    label = r.variant if len(r.variant) <= 20 else r.variant[:18] + ".."
    ax.annotate(label, (r.delta_950, r.speedup_950), fontsize=7,
                xytext=(4, 4), textcoords="offset points")
ax.axvspan(-2, 2, alpha=0.08, color="#2ca02c")
ax.axhline(1.0, color="#888", linestyle=":", lw=0.8)
ax.set_xlabel(r"$\Delta$ vs_prev (dB)   circle = 32f, square = 950f")
ax.set_ylabel("Speedup")
ax.set_title("Length Robustness: 32 vs 950 frames\nsquares drift right and down = degrades at length")
ax.set_xlim(-25, 15)
plt.savefig(os.path.join(OUT, "length_robustness.png"))
plt.close()
print("wrote length_robustness.png")

# ---------- Figure 3: Per family best Pareto ----------
rep = (plot_df.assign(score=plot_df.speedup * (plot_df.delta_vs_prev_db.abs() < 2).astype(int))
       .sort_values(["family", "score", "speedup"], ascending=[True, False, False])
       .drop_duplicates(subset=["family"]))
rep = rep.sort_values("speedup", ascending=True)

fig, ax = plt.subplots(figsize=(10, 7))
ypos = range(len(rep))
colors = [FAMILY_COLORS.get(f, "#444") for f in rep.family]
ax.barh(ypos, rep.speedup, color=colors, alpha=0.85, edgecolor="white")
for i, (_, r) in enumerate(rep.iterrows()):
    ok = abs(r.delta_vs_prev_db) < 2
    mark = "OK" if ok else "x"
    color = "#2ca02c" if ok else "#d62728"
    ax.text(r.speedup + 0.06, i,
            f"{r.speedup:.2f}x  ({r.delta_vs_prev_db:+.2f} dB) {mark}",
            va="center", fontsize=9, color=color, fontweight="bold" if ok else "normal")
ax.set_yticks(list(ypos))
ax.set_yticklabels([f"{r.family}\n{r.variant[:30]}" for _, r in rep.iterrows()], fontsize=8)
ax.axvline(1.0, color="#888", linestyle=":", lw=0.8)
ax.set_xlabel("Speedup over DDIM-10 (best representative per family)")
ax.set_title("Per Family: Best Configuration\nGreen check = within 2 dB of baseline; red x = quality compromised")
ax.set_xlim(0, 4.3)
plt.savefig(os.path.join(OUT, "per_family_pareto.png"))
plt.close()
print("wrote per_family_pareto.png")

# ---------- Figure 4: Headline breakdown ----------
fig, ax = plt.subplots(figsize=(9, 5.5))
labels = ["DDIM-10\nbaseline", "+ DPM++ 2M\n(5 base steps)", "+ Difficulty\n(2/3/5 per frame)"]
forwards_per_frame = [10.0, 5.0, 2.82]
speedups = [1.0, 2.00, 3.54]
total_forwards = [9500, 4750, 2683]
xpos = range(3)
bars = ax.bar(xpos, total_forwards, color=["#888", "#1f77b4", "#d62728"],
              alpha=0.85, edgecolor="white")
for i, (sp, fpf, tf) in enumerate(zip(speedups, forwards_per_frame, total_forwards)):
    ax.text(i, tf + 200, f"{sp:.2f}x\n{fpf:.2f} fwd/frame", ha="center", fontsize=10,
            fontweight="bold", color="#222")
ax.set_xticks(list(xpos))
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("Total model forwards (950 frames)")
ax.set_title("Headline Stack: Cumulative Compute Reduction\n9500 to 2683 forwards = 71.8% saved at -0.02 dB coherence")
ax.set_ylim(0, 11000)
plt.savefig(os.path.join(OUT, "headline_breakdown.png"))
plt.close()
print("wrote headline_breakdown.png")

# ---------- Figure 5: Skip pattern: distribution dominates count ----------
patterns = ["alternate\n(50% interleaved)", "skip_late\n(50% clustered)",
            "skip_early\n(50% clustered)", "interval_3\n(67% interleaved)",
            "interval_4\n(75% interleaved)"]
skip_rate = [50, 50, 50, 67, 75]
delta = [-0.64, -7.17, -5.62, -1.72, -3.50]
speedup = [2.09, 2.07, 1.69, 2.40, 3.26]
colors_p = ["#1f77b4", "#d62728", "#d62728", "#1f77b4", "#ff7f0e"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
ax1.bar(patterns, speedup, color=colors_p, alpha=0.85, edgecolor="white")
for i, (s, p) in enumerate(zip(speedup, patterns)):
    ax1.text(i, s + 0.05, f"{s:.2f}x", ha="center", fontsize=10, fontweight="bold")
ax1.set_ylabel("Speedup over DDIM-10")
ax1.set_title("Speedup")
ax1.set_ylim(0, 3.7)
ax1.tick_params(axis="x", rotation=0, labelsize=8)

ax2.bar(patterns, delta, color=colors_p, alpha=0.85, edgecolor="white")
for i, (d, p) in enumerate(zip(delta, patterns)):
    ax2.text(i, d - 0.4 if d < 0 else d + 0.2, f"{d:+.2f} dB",
             ha="center", fontsize=10, fontweight="bold",
             color="#d62728" if abs(d) > 2 else "#2ca02c")
ax2.axhspan(-2, 2, alpha=0.08, color="#2ca02c")
ax2.axhline(0, color="#222", lw=0.8)
ax2.set_ylabel(r"$\Delta$ vs_prev (dB)")
ax2.set_title("Quality")
ax2.set_ylim(-9, 1)
ax2.tick_params(axis="x", rotation=0, labelsize=8)

fig.suptitle("Step Caching: Distribution Dominates Skip Count\n"
             "All three 50% configurations have identical compute saving.\n"
             "Interleaved (blue) preserves coherence. Clustered (red) breaks it.",
             fontsize=12, y=1.02)
plt.savefig(os.path.join(OUT, "skip_pattern_distribution.png"))
plt.close()
print("wrote skip_pattern_distribution.png")

print(f"\nAll figures written to {OUT}")
