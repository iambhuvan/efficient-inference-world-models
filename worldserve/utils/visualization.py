"""
Visualization utilities for WorldServe profiling results.

All functions write to files (PNG / MP4) and optionally return
matplotlib Figure objects for interactive use.  Only depends on
matplotlib + numpy + torch (no heavy video libs for plotting).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for headless servers
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


# ── Attention Heatmap ─────────────────────────────────────────────────────

def plot_attention_heatmap(
    attn_weights: torch.Tensor,
    save_path: Optional[str | Path] = None,
    title: str = "Attention Weights",
    head_idx: int = 0,
    layer_name: str = "",
    figsize: Tuple[int, int] = (10, 8),
) -> Figure:
    """
    Visualize a single attention head's weight matrix as a heatmap.

    Parameters
    ----------
    attn_weights : (num_heads, seq_len, seq_len) or (seq_len, seq_len).
    head_idx : which head to plot when multi-head tensor is provided.
    save_path : if given, save PNG to this path.
    """
    w = attn_weights.detach().float().cpu()
    if w.ndim == 3:
        w = w[head_idx]
    elif w.ndim > 3:
        # (B, H, S, S) — take first batch, selected head
        w = w[0, head_idx]

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(w.numpy(), aspect="auto", cmap="viridis")
    full_title = f"{title}"
    if layer_name:
        full_title += f" [{layer_name}]"
    full_title += f" (head {head_idx})"
    ax.set_title(full_title)
    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


# ── KV Cache Size Over Time ──────────────────────────────────────────────

def plot_kv_cache_over_time(
    records: List[Dict],
    save_path: Optional[str | Path] = None,
    title: str = "KV Cache Size Over Time",
    figsize: Tuple[int, int] = (12, 5),
) -> Figure:
    """
    Line plot of KV-cache memory over time.

    Parameters
    ----------
    records : list of dicts with keys ``timestamp``, ``size_mb``,
        and optionally ``label`` / ``num_entries``.
        (matches ``ProfileSummary.kv_cache_records``).
    """
    if not records:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(title + " (no data)")
        return fig

    # Normalize timestamps to start at 0
    t0 = records[0]["timestamp"]
    times = [r["timestamp"] - t0 for r in records]
    sizes = [r["size_mb"] for r in records]

    fig, ax1 = plt.subplots(figsize=figsize)
    color_mem = "#2196F3"
    ax1.plot(times, sizes, marker="o", markersize=3, color=color_mem, linewidth=1.5)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("KV Cache Size (MB)", color=color_mem)
    ax1.tick_params(axis="y", labelcolor=color_mem)
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)

    # Secondary y-axis for num_entries if available
    if "num_entries" in records[0]:
        entries = [r.get("num_entries", 0) for r in records]
        ax2 = ax1.twinx()
        color_ent = "#FF9800"
        ax2.plot(times, entries, marker="s", markersize=3, color=color_ent,
                 linewidth=1.0, linestyle="--")
        ax2.set_ylabel("Num Entries", color=color_ent)
        ax2.tick_params(axis="y", labelcolor=color_ent)

    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


# ── Per-Layer Latency Bar Chart ──────────────────────────────────────────

def plot_layer_latency(
    region_stats: Dict[str, Dict[str, float]],
    save_path: Optional[str | Path] = None,
    title: str = "Per-Region Latency",
    top_n: int = 40,
    figsize: Optional[Tuple[int, int]] = None,
) -> Figure:
    """
    Horizontal bar chart of mean latency per profiled region.

    Parameters
    ----------
    region_stats : mapping ``region_name -> {mean_ms, total_ms, ...}``.
        (matches ``ProfileSummary.region_stats``).
    top_n : only show the top N regions by total time.
    """
    # Sort by total_ms descending
    items = sorted(region_stats.items(), key=lambda kv: kv[1]["total_ms"], reverse=True)
    items = items[:top_n]
    names = [it[0] for it in items][::-1]  # reverse for bottom-up bar
    means = [it[1]["mean_ms"] for it in items][::-1]

    if figsize is None:
        figsize = (12, max(4, len(names) * 0.35))

    fig, ax = plt.subplots(figsize=figsize)

    colors = []
    for n in names:
        nl = n.lower()
        if "attn" in nl or "attention" in nl:
            colors.append("#E53935")
        elif "ffn" in nl or "mlp" in nl or "feedforward" in nl:
            colors.append("#43A047")
        else:
            colors.append("#1E88E5")

    ax.barh(range(len(names)), means, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Mean Latency (ms)")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#E53935", label="Attention"),
        Patch(facecolor="#43A047", label="FFN / MLP"),
        Patch(facecolor="#1E88E5", label="Other"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


# ── Side-by-Side Frame Comparison ────────────────────────────────────────

def plot_frame_comparison(
    frames_a: torch.Tensor,
    frames_b: torch.Tensor,
    save_path: Optional[str | Path] = None,
    labels: Tuple[str, str] = ("Baseline", "Optimized"),
    frame_indices: Optional[Sequence[int]] = None,
    max_frames: int = 8,
    figsize_per_frame: Tuple[float, float] = (4.0, 3.0),
) -> Figure:
    """
    Side-by-side comparison of selected frames from two videos.

    Parameters
    ----------
    frames_a, frames_b : (T, C, H, W) or (B, T, C, H, W) in [0, 1].
        If 5-D, uses the first batch element.
    frame_indices : which frames to show; defaults to evenly spaced.
    """
    # Normalize to (T, C, H, W)
    a = frames_a.detach().cpu().float()
    b = frames_b.detach().cpu().float()
    if a.ndim == 5:
        a = a[0]
    if b.ndim == 5:
        b = b[0]

    T = min(a.shape[0], b.shape[0])
    if frame_indices is None:
        step = max(1, T // max_frames)
        frame_indices = list(range(0, T, step))[:max_frames]

    n = len(frame_indices)
    fig, axes = plt.subplots(
        2, n,
        figsize=(figsize_per_frame[0] * n, figsize_per_frame[1] * 2),
        squeeze=False,
    )

    for col, fi in enumerate(frame_indices):
        for row, (tensor, label) in enumerate([(a, labels[0]), (b, labels[1])]):
            img = tensor[fi].permute(1, 2, 0).clamp(0, 1).numpy()
            axes[row, col].imshow(img)
            axes[row, col].set_title(f"{label} t={fi}", fontsize=9)
            axes[row, col].axis("off")

        # Difference map in bottom... actually keep 2 rows: A and B
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


# ── Save Video as MP4 ────────────────────────────────────────────────────

def save_video_mp4(
    frames: torch.Tensor,
    save_path: str | Path,
    fps: int = 20,
) -> Path:
    """
    Write a (T, C, H, W) or (B, T, C, H, W) tensor to MP4.

    Tries ``imageio`` with ffmpeg backend; falls back to OpenCV.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    f = frames.detach().cpu().float()
    if f.ndim == 5:
        f = f[0]
    # (T, C, H, W) -> (T, H, W, C) uint8
    f = f.permute(0, 2, 3, 1).clamp(0, 1).mul(255).byte().numpy()

    # Try imageio first
    try:
        import imageio.v3 as iio  # type: ignore[import-untyped]

        iio.imwrite(
            str(save_path),
            f,
            fps=fps,
            codec="libx264",
            plugin="pyav",
        )
        return save_path
    except (ImportError, Exception):
        pass

    # Fallback: OpenCV
    try:
        import cv2  # type: ignore[import-untyped]

        T, H, W, C = f.shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(save_path), fourcc, fps, (W, H))
        for i in range(T):
            frame_bgr = cv2.cvtColor(f[i], cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        out.release()
        return save_path
    except ImportError:
        pass

    # Last resort: save individual frames as PNGs
    frame_dir = save_path.parent / (save_path.stem + "_frames")
    frame_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    for i in range(f.shape[0]):
        Image.fromarray(f[i]).save(frame_dir / f"frame_{i:04d}.png")

    raise RuntimeError(
        f"Could not write MP4 (no imageio or cv2). Saved {f.shape[0]} "
        f"PNG frames to {frame_dir} instead."
    )
