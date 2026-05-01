"""
Step-caching optimizations for WorldServe (Phase 2).

Caches DiT features across denoising steps within each generation block.
Primary method: SeaCache (CVPR 2026) — spectral-evolution-aware caching for Wan2.1.
Compatibility study: FlowCache — chunk-wise importance-redundancy caching.
Optional comparison: SpeCa — speculative caching with lightweight verification.
"""

from typing import Optional, Dict, Any

__all__ = [
    "SeaCacheOptimizer",
    "FlowCacheOptimizer",
    "SpeCaOptimizer",
    "StepCachingManager",
]


def get_step_caching_optimizers() -> Dict[str, type]:
    """Return a dictionary of available step-caching optimizer classes."""
    optimizers: Dict[str, type] = {}

    try:
        from worldserve.optimizations.system_level.step_caching.seacache import SeaCacheOptimizer
        optimizers["seacache"] = SeaCacheOptimizer
    except ImportError:
        pass

    try:
        from worldserve.optimizations.system_level.step_caching.flowcache import FlowCacheOptimizer
        optimizers["flowcache"] = FlowCacheOptimizer
    except ImportError:
        pass

    try:
        from worldserve.optimizations.system_level.step_caching.speca import SpeCaOptimizer
        optimizers["speca"] = SpeCaOptimizer
    except ImportError:
        pass

    return optimizers


def apply_step_caching(
    model: Any,
    method: str = "seacache",
    config: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Apply step-caching optimization to a DiT model.

    Args:
        model: The DiT model (Wan2.1 backbone) to optimize.
        method: One of "seacache", "flowcache", "speca".
        config: Method-specific configuration dict.

    Returns:
        The optimized model with caching hooks installed.
    """
    from worldserve.optimizations.system_level.step_caching.manager import StepCachingManager

    manager = StepCachingManager(method=method, config=config or {})
    return manager.optimize(model)
