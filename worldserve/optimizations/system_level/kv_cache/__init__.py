"""
KV cache compression optimizations for WorldServe (Phase 3).

Baseline (must-do):
  - TemporalTokenMerger: TempCache-style temporal merging of redundant KV tokens
  - ProgressiveKVQuantizer: FP16 -> FP8 -> INT4 age-based progressive quantization

Advanced (research extras):
  - ImportanceTracker: EMA-based attention importance scoring for precision decisions
  - LayerAwareQuantizer: Entropy-profiled per-layer precision assignment
  - QVGSmoothingQuantizer: Semantic centroid smoothing + residual quantization
  - SpatialBlockEvictor: Block-level spatial eviction for Matrix-Game 2.0

Manager:
  - KVCacheManager: Orchestrates all KV cache optimizations end-to-end
"""

from typing import Optional, Dict, Any

__all__ = [
    "TemporalTokenMerger",
    "ProgressiveKVQuantizer",
    "ImportanceTracker",
    "LayerAwareQuantizer",
    "QVGSmoothingQuantizer",
    "SpatialBlockEvictor",
    "KVCacheManager",
]


def get_kv_cache_optimizers() -> Dict[str, type]:
    """Return a dictionary of available KV cache optimization classes."""
    optimizers: Dict[str, type] = {}

    try:
        from worldserve.optimizations.system_level.kv_cache.temporal_merge import TemporalTokenMerger
        optimizers["temporal_merge"] = TemporalTokenMerger
    except ImportError:
        pass

    try:
        from worldserve.optimizations.system_level.kv_cache.progressive_quant import ProgressiveKVQuantizer
        optimizers["progressive_quant"] = ProgressiveKVQuantizer
    except ImportError:
        pass

    try:
        from worldserve.optimizations.system_level.kv_cache.importance_tracker import ImportanceTracker
        optimizers["importance_tracker"] = ImportanceTracker
    except ImportError:
        pass

    try:
        from worldserve.optimizations.system_level.kv_cache.layer_aware_quant import LayerAwareQuantizer
        optimizers["layer_aware_quant"] = LayerAwareQuantizer
    except ImportError:
        pass

    try:
        from worldserve.optimizations.system_level.kv_cache.qvg_smoothing import QVGSmoothingQuantizer
        optimizers["qvg_smoothing"] = QVGSmoothingQuantizer
    except ImportError:
        pass

    try:
        from worldserve.optimizations.system_level.kv_cache.spatial_eviction import SpatialBlockEvictor
        optimizers["spatial_eviction"] = SpatialBlockEvictor
    except ImportError:
        pass

    try:
        from worldserve.optimizations.system_level.kv_cache.manager import KVCacheManager
        optimizers["manager"] = KVCacheManager
    except ImportError:
        pass

    return optimizers


def apply_kv_cache_compression(
    config: Optional[Dict[str, Any]] = None,
) -> "KVCacheManager":
    """
    Create and return a configured KVCacheManager.

    Args:
        config: KV cache optimization configuration dict.

    Returns:
        Configured KVCacheManager ready for use during generation.
    """
    from worldserve.optimizations.system_level.kv_cache.manager import KVCacheManager

    return KVCacheManager(config=config or {})
