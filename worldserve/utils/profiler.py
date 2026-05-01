"""
WorldServe GPU profiler built on torch.cuda.Event timing with NVTX markers.

Tracks per-region (layer, attention, FFN) latency, CUDA memory snapshots,
KV-cache sizes, and FPS.  Exports JSON and CSV reports.
"""

from __future__ import annotations

import csv
import json
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

# ── optional NVTX import (works in NSight-compatible builds) ──────────────
try:
    from torch.cuda import nvtx  # type: ignore[attr-defined]

    _HAS_NVTX = True
except (ImportError, AttributeError):
    _HAS_NVTX = False


# ── data containers ───────────────────────────────────────────────────────
@dataclass
class RegionRecord:
    """Timing record for a single profiled region."""

    name: str
    start_event: Optional[torch.cuda.Event] = None
    end_event: Optional[torch.cuda.Event] = None
    wall_start: float = 0.0
    wall_end: float = 0.0


@dataclass
class MemorySnapshot:
    """Point-in-time CUDA memory snapshot."""

    timestamp: float
    allocated_mb: float
    reserved_mb: float
    max_allocated_mb: float
    active_allocs: int
    label: str = ""


@dataclass
class KVCacheRecord:
    """KV-cache size at a point in time."""

    timestamp: float
    num_entries: int
    size_mb: float
    label: str = ""


@dataclass
class ProfileSummary:
    """Aggregated profiling results."""

    region_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    memory_snapshots: List[Dict[str, Any]] = field(default_factory=list)
    kv_cache_records: List[Dict[str, Any]] = field(default_factory=list)
    total_wall_time_s: float = 0.0
    total_frames: int = 0
    fps: float = 0.0
    attention_total_ms: float = 0.0
    ffn_total_ms: float = 0.0
    attention_pct: float = 0.0
    ffn_pct: float = 0.0
    device_name: str = ""
    peak_memory_mb: float = 0.0


class WorldServeProfiler:
    """
    GPU-aware profiler for world model inference.

    Usage::

        profiler = WorldServeProfiler()
        profiler.start_region("dit_block_0.attn")
        # ... run attention ...
        profiler.end_region("dit_block_0.attn")
        summary = profiler.get_summary(total_frames=32)
        profiler.save_report("benchmarks/baseline.json")
    """

    def __init__(self, enable_nvtx: bool = True, device: Optional[str] = None):
        self._enabled = torch.cuda.is_available()
        if not self._enabled:
            warnings.warn(
                "CUDA not available — WorldServeProfiler will collect wall-clock "
                "times only (no GPU events or memory stats)."
            )
        self._enable_nvtx = enable_nvtx and _HAS_NVTX and self._enabled
        self._device = device or ("cuda" if self._enabled else "cpu")

        # Storage
        self._open_regions: Dict[str, RegionRecord] = {}
        self._completed: List[RegionRecord] = []
        self._memory_snapshots: List[MemorySnapshot] = []
        self._kv_cache_records: List[KVCacheRecord] = []
        self._wall_start: Optional[float] = None
        self._wall_end: Optional[float] = None

    # ── region timing ─────────────────────────────────────────────────
    def start_region(self, name: str) -> None:
        """Begin timing a named region (layer, attn, ffn, etc.)."""
        if name in self._open_regions:
            warnings.warn(f"Region '{name}' already started — overwriting.")

        record = RegionRecord(name=name, wall_start=time.perf_counter())

        if self._enabled:
            start_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record(torch.cuda.current_stream())
            record.start_event = start_evt

        if self._enable_nvtx:
            nvtx.range_push(name)

        self._open_regions[name] = record

        # Record the very first region start as wall start for the run
        if self._wall_start is None:
            self._wall_start = record.wall_start

    def end_region(self, name: str) -> None:
        """End timing a previously-started region."""
        if name not in self._open_regions:
            warnings.warn(f"Region '{name}' was never started — ignoring end_region.")
            return

        record = self._open_regions.pop(name)
        record.wall_end = time.perf_counter()

        if self._enabled:
            end_evt = torch.cuda.Event(enable_timing=True)
            end_evt.record(torch.cuda.current_stream())
            record.end_event = end_evt

        if self._enable_nvtx:
            nvtx.range_pop()

        self._completed.append(record)
        self._wall_end = record.wall_end

    # ── memory tracking ───────────────────────────────────────────────
    def snapshot_memory(self, label: str = "") -> MemorySnapshot:
        """Take a CUDA memory snapshot (no-op on CPU)."""
        ts = time.perf_counter()
        if not self._enabled:
            snap = MemorySnapshot(timestamp=ts, allocated_mb=0, reserved_mb=0,
                                  max_allocated_mb=0, active_allocs=0, label=label)
            self._memory_snapshots.append(snap)
            return snap

        stats = torch.cuda.memory_stats(self._device)
        snap = MemorySnapshot(
            timestamp=ts,
            allocated_mb=stats.get("allocated_bytes.all.current", 0) / (1024 ** 2),
            reserved_mb=stats.get("reserved_bytes.all.current", 0) / (1024 ** 2),
            max_allocated_mb=stats.get("allocated_bytes.all.peak", 0) / (1024 ** 2),
            active_allocs=stats.get("num_alloc_retries", 0),
            label=label,
        )
        self._memory_snapshots.append(snap)
        return snap

    # ── KV-cache tracking ─────────────────────────────────────────────
    def record_kv_cache(
        self, num_entries: int, size_mb: float, label: str = ""
    ) -> None:
        """Record a KV-cache size measurement."""
        self._kv_cache_records.append(
            KVCacheRecord(
                timestamp=time.perf_counter(),
                num_entries=num_entries,
                size_mb=size_mb,
                label=label,
            )
        )

    # ── summary computation ───────────────────────────────────────────
    def get_summary(self, total_frames: int = 0) -> ProfileSummary:
        """
        Synchronize GPU, compute aggregate stats, return a ``ProfileSummary``.

        Parameters
        ----------
        total_frames : int
            Number of frames generated, used for FPS calculation.
        """
        if self._enabled:
            torch.cuda.synchronize(self._device)

        # Per-region latency (ms)
        region_latencies: Dict[str, List[float]] = defaultdict(list)
        for rec in self._completed:
            if self._enabled and rec.start_event and rec.end_event:
                lat = rec.start_event.elapsed_time(rec.end_event)  # ms
            else:
                lat = (rec.wall_end - rec.wall_start) * 1000.0
            region_latencies[rec.name].append(lat)

        region_stats: Dict[str, Dict[str, float]] = {}
        for name, lats in region_latencies.items():
            region_stats[name] = {
                "count": len(lats),
                "total_ms": sum(lats),
                "mean_ms": sum(lats) / len(lats),
                "min_ms": min(lats),
                "max_ms": max(lats),
            }

        # Attention vs FFN split
        attn_ms = sum(
            v["total_ms"]
            for k, v in region_stats.items()
            if "attn" in k.lower() or "attention" in k.lower()
        )
        ffn_ms = sum(
            v["total_ms"]
            for k, v in region_stats.items()
            if "ffn" in k.lower() or "mlp" in k.lower() or "feedforward" in k.lower()
        )
        compute_total = attn_ms + ffn_ms if (attn_ms + ffn_ms) > 0 else 1.0

        # Wall time & FPS
        wall = 0.0
        if self._wall_start is not None and self._wall_end is not None:
            wall = self._wall_end - self._wall_start
        fps = total_frames / wall if wall > 0 and total_frames > 0 else 0.0

        # Peak memory
        peak = 0.0
        if self._enabled:
            peak = torch.cuda.max_memory_allocated(self._device) / (1024 ** 2)

        device_name = ""
        if self._enabled:
            device_name = torch.cuda.get_device_name(self._device)

        return ProfileSummary(
            region_stats=region_stats,
            memory_snapshots=[asdict(s) for s in self._memory_snapshots],
            kv_cache_records=[asdict(r) for r in self._kv_cache_records],
            total_wall_time_s=wall,
            total_frames=total_frames,
            fps=fps,
            attention_total_ms=attn_ms,
            ffn_total_ms=ffn_ms,
            attention_pct=attn_ms / compute_total * 100,
            ffn_pct=ffn_ms / compute_total * 100,
            device_name=device_name,
            peak_memory_mb=peak,
        )

    # ── export ────────────────────────────────────────────────────────
    def save_report(self, path: str | Path, total_frames: int = 0) -> ProfileSummary:
        """
        Compute summary and save as JSON.  If *path* ends with ``.csv``, also
        writes a per-region CSV alongside the JSON.

        Returns the ``ProfileSummary`` for further programmatic use.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        summary = self.get_summary(total_frames=total_frames)
        data = asdict(summary)

        # JSON
        json_path = path.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        # CSV for per-region stats
        csv_path = path.with_suffix(".csv")
        if summary.region_stats:
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                header = ["region", "count", "total_ms", "mean_ms", "min_ms", "max_ms"]
                writer.writerow(header)
                for name, stats in sorted(summary.region_stats.items()):
                    writer.writerow(
                        [name, int(stats["count"]), f"{stats['total_ms']:.3f}",
                         f"{stats['mean_ms']:.3f}", f"{stats['min_ms']:.3f}",
                         f"{stats['max_ms']:.3f}"]
                    )

        return summary

    # ── context manager for convenience ───────────────────────────────
    def region(self, name: str):
        """Context manager wrapping start/end_region."""
        return _RegionCtx(self, name)

    # ── reset ─────────────────────────────────────────────────────────
    def reset(self) -> None:
        """Clear all collected data."""
        self._open_regions.clear()
        self._completed.clear()
        self._memory_snapshots.clear()
        self._kv_cache_records.clear()
        self._wall_start = None
        self._wall_end = None
        if self._enabled:
            torch.cuda.reset_peak_memory_stats(self._device)


class _RegionCtx:
    """Thin context-manager returned by ``WorldServeProfiler.region``."""

    def __init__(self, profiler: WorldServeProfiler, name: str):
        self._profiler = profiler
        self._name = name

    def __enter__(self):
        self._profiler.start_region(self._name)
        return self

    def __exit__(self, *exc):
        self._profiler.end_region(self._name)
        return False
