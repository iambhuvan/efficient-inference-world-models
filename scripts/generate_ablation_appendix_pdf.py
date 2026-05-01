"""
Generate a printable PDF appendix for Oasis ablations not emphasized on poster.

Outputs:
  docs/figures/ablation_appendix_tables_charts.pdf
"""

from __future__ import annotations

import os
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "benchmarks", "results", "all_runs.csv")
OUT = os.path.join(ROOT, "docs", "figures", "ablation_appendix_tables_charts.pdf")
os.makedirs(os.path.dirname(OUT), exist_ok=True)


def _fmt(x):
    if pd.isna(x):
        return "—"
    if isinstance(x, float):
        return f"{x:.2f}"
    return str(x)


def add_title_page(pdf: PdfPages, title: str, subtitle: str) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=24, fontweight="bold")
    ax.text(0.5, 0.48, subtitle, ha="center", va="center", fontsize=14)
    ax.text(0.5, 0.35, "Source: benchmarks/results/all_runs.csv", ha="center", va="center", fontsize=11, color="#444")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_table_page(pdf: PdfPages, title: str, df: pd.DataFrame, columns: List[str], note: str = "") -> None:
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.set_title(title, fontsize=16, pad=16, fontweight="bold")
    if df.empty:
        ax.text(0.5, 0.5, "No rows found for this section.", ha="center", va="center", fontsize=13)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        return

    show = df[columns].copy()
    for c in show.columns:
        show[c] = show[c].map(_fmt)
    table = ax.table(
        cellText=show.values.tolist(),
        colLabels=show.columns.tolist(),
        cellLoc="center",
        colLoc="center",
        bbox=[0.02, 0.14, 0.96, 0.78],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1, 1.45)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f0f0f0")
        cell.set_edgecolor("#555")
    if note:
        ax.text(0.02, 0.06, note, transform=ax.transAxes, fontsize=10.5, color="#333")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_bar_page(pdf: PdfPages, title: str, df: pd.DataFrame, xcol: str, ycol: str, huecol: str | None = None, y2col: str | None = None) -> None:
    fig, ax1 = plt.subplots(figsize=(11, 8.5))
    if df.empty:
        ax1.axis("off")
        ax1.set_title(title, fontsize=16, pad=16, fontweight="bold")
        ax1.text(0.5, 0.5, "No rows found.", ha="center", va="center", fontsize=13)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        return

    ax1.set_title(title, fontsize=16, pad=12, fontweight="bold")
    labels = [str(v) for v in df[xcol].tolist()]
    x = list(range(len(labels)))
    vals = df[ycol].astype(float).tolist()
    colors = None
    if huecol is not None:
        fam = df[huecol].tolist()
        palette = {
            "stable": "#2ca02c",
            "degrades_at_length": "#d62728",
            "catastrophic": "#d62728",
            "stable_at_short": "#ff7f0e",
            "winner": "#1f77b4",
            "clean_win": "#1f77b4",
            "project_headline": "#1f77b4",
            "blur_visible": "#9467bd",
            "blur_compounds": "#8c564b",
            "length_collapse": "#d62728",
            "acceptable_at_short": "#ff7f0e",
        }
        colors = [palette.get(str(f), "#4c78a8") for f in fam]
    bars = ax1.bar(x, vals, color=colors if colors else "#4c78a8")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=22, ha="right", fontsize=10)
    ax1.set_ylabel(ycol)
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    ax1.grid(axis="y", alpha=0.25)

    if y2col is not None and y2col in df.columns:
        ax2 = ax1.twinx()
        y2 = pd.to_numeric(df[y2col], errors="coerce").fillna(0).tolist()
        ax2.plot(x, y2, color="#111", marker="o", linewidth=2, label=y2col)
        ax2.set_ylabel(y2col)
        for xi, yi in zip(x, y2):
            ax2.text(xi, yi, f"{yi:.2f}", fontsize=9, ha="center", va="bottom")
        ax2.axhline(0.0, color="#777", linestyle="--", linewidth=0.8, alpha=0.5)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(CSV)

    with PdfPages(OUT) as pdf:
        add_title_page(
            pdf,
            "Oasis Ablation Appendix",
            "Sections 4.3, 4.4, 4.5, 5.2, 5.3 (+ related stack ablations)",
        )

        # 4.3 Step cache (32f)
        step32 = df[(df.family == "stepcache") & (df.length_frames == 32)].copy()
        step32 = step32.sort_values("speedup", ascending=False)
        add_table_page(
            pdf,
            "Section 4.3 — Step Caching Patterns (32 frames)",
            step32,
            ["variant", "config", "speedup", "delta_vs_prev_db", "cross_psnr_db", "verdict"],
            note="Takeaway: distribution of skips matters more than raw skip rate.",
        )
        add_bar_page(
            pdf,
            "Section 4.3 — Step Caching: Speedup vs Coherence Delta (32 frames)",
            step32,
            xcol="variant",
            ycol="speedup",
            y2col="delta_vs_prev_db",
        )

        step_len = df[(df.family == "stepcache") & (df.length_frames.isin([75, 950]))].copy()
        add_table_page(
            pdf,
            "Section 4.3 — Step Caching Length Validation (75/950 frames)",
            step_len.sort_values(["variant", "length_frames"]),
            ["variant", "length_frames", "speedup", "delta_vs_prev_db", "verdict", "notes"],
        )

        # 4.4 DPM++
        dpm = df[df.family == "dpmpp"].copy().sort_values("length_frames")
        add_table_page(
            pdf,
            "Section 4.4 — DPM-Solver++ 2M Results",
            dpm,
            ["variant", "length_frames", "speedup", "delta_vs_prev_db", "cross_psnr_db", "verdict", "notes"],
        )

        # 4.5 Difficulty
        diff32 = df[(df.family == "difficulty") & (df.length_frames == 32)].copy()
        add_table_page(
            pdf,
            "Section 4.5 — Difficulty Schedule Buckets / Formula Variants (32 frames)",
            diff32.sort_values("speedup", ascending=False),
            ["variant", "config", "speedup", "delta_vs_prev_db", "cross_psnr_db", "verdict", "notes"],
        )
        diff_len = df[(df.family == "difficulty") & (df.length_frames == 950)].copy()
        add_table_page(
            pdf,
            "Section 4.5 — Difficulty Length Validation (950 frames)",
            diff_len,
            ["variant", "length_frames", "speedup", "delta_vs_prev_db", "verdict", "notes"],
        )

        # 5.2 DPM++ + stepcache
        s52 = df[(df.family == "stack") & (df.variant.str.contains("stepcache_alternate", na=False))].copy()
        add_table_page(
            pdf,
            "Section 5.2 — Stack: DPM++5 + Alternate Stepcache",
            s52.sort_values("length_frames"),
            ["variant", "length_frames", "speedup", "delta_vs_prev_db", "cross_psnr_db", "verdict", "notes"],
        )

        # 5.3 DPM++ + Difficulty
        s53 = df[(df.family == "stack") & (df.variant.str.contains("DPM\\+\\+5 \\+ Difficulty r=2/m=3/f=5", regex=True, na=False))].copy()
        add_table_page(
            pdf,
            "Section 5.3 — Stack: DPM++5 + Difficulty (Headline Pair)",
            s53.sort_values("length_frames"),
            ["variant", "length_frames", "speedup", "delta_vs_prev_db", "cross_psnr_db", "verdict", "notes"],
        )

        # Related: megastack (often asked in Q&A)
        mega = df[(df.family == "stack") & (df.variant.str.contains("megastack", case=False, na=False))].copy()
        add_table_page(
            pdf,
            "Related Stack Ablation — DPM++5 + Difficulty + TaylorSeer (Megastack)",
            mega.sort_values("length_frames"),
            ["variant", "length_frames", "speedup", "delta_vs_prev_db", "cross_psnr_db", "verdict", "notes"],
            note="Included for presentation Q&A since it degrades strongly at long horizon.",
        )

        # Related not-on-poster: compile, predit, drift, action_kv, warmstart
        extra = df[df.family.isin(["torch_compile", "predit", "drift_cache", "action_kv", "warmstart"])].copy()
        add_table_page(
            pdf,
            "Additional Not-on-Poster Ablations (Compile / Predit / Drift / Action-KV / Warmstart)",
            extra.sort_values(["family", "length_frames", "variant"]),
            ["family", "variant", "length_frames", "speedup", "delta_vs_prev_db", "cross_psnr_db", "verdict", "notes"],
        )

    print(f"Wrote: {OUT}")


if __name__ == "__main__":
    main()
