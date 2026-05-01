"""
Sparse attention optimizations for WorldServe (Phase 4).

Primary method: SVG2 — head-aware spatial/temporal sparse attention with
semantic token permutation for Matrix-Game 2.0's 15,360 spatial tokens/frame.

Fallback/ablation: NATTEN — 2D neighborhood attention for spatial heads.

Profiler: AttentionProfiler — extracts and analyzes attention patterns to
guide sparsity decisions.

Note: Open-Oasis (144 spatial tokens/frame) is too small to benefit from
sparse attention. These optimizations target MG2 exclusively.
"""

from typing import Optional, Dict, Any

__all__ = [
    "SVG2SparseAttention",
    "NATTENSparseAttention",
    "AttentionProfiler",
    "SparseAttentionOptimizer",
]


class SparseAttentionOptimizer:
    """
    Stack-compatible wrapper for sparse attention.

    Sparse attention is a structural model modification (it wraps attention
    modules) rather than a generation-loop hook. This class stores the config
    so OptimizationStack can hold it, and exposes apply(model) for modal_app.py
    to call after model loading.

    Usage in modal_app.py:
        if "sparse_attention" in opt_stack.modules:
            opt_stack.modules["sparse_attention"].apply(model["pipeline"])
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.method = config.get("method", "svg2")
        self._applied = False

    def apply(self, model) -> None:
        """Apply sparse attention to the model. Call once after model load."""
        if self._applied:
            return
        apply_sparse_attention(model, method=self.method, config=dict(self.config))
        self._applied = True

    # OptimizationStack protocol — sparse attention has no per-step hooks
    def pre_step(self, step_idx: int, total_steps: int, latents) -> bool:
        return False

    def post_step(self, step_idx: int, total_steps: int, latents) -> None:
        pass

    def get_stats(self) -> Dict[str, Any]:
        return {"method": self.method, "applied": self._applied}


def get_sparse_attention_optimizers() -> Dict[str, type]:
    """Return a dictionary of available sparse attention optimization classes."""
    optimizers: Dict[str, type] = {}

    try:
        from worldserve.optimizations.system_level.sparse_attention.svg2 import SVG2SparseAttention
        optimizers["svg2"] = SVG2SparseAttention
    except ImportError:
        pass

    try:
        from worldserve.optimizations.system_level.sparse_attention.natten_wrapper import NATTENSparseAttention
        optimizers["natten"] = NATTENSparseAttention
    except ImportError:
        pass

    try:
        from worldserve.optimizations.system_level.sparse_attention.attention_profiler import AttentionProfiler
        optimizers["profiler"] = AttentionProfiler
    except ImportError:
        pass

    return optimizers


def apply_sparse_attention(
    model: Any,
    method: str = "svg2",
    config: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Apply sparse attention optimization to a DiT model.

    Args:
        model: The MG2 DiT model (Wan2.1 backbone) to optimize.
        method: One of "svg2" (primary) or "natten" (fallback/ablation).
        config: Method-specific configuration dict.

    Returns:
        The optimized model with sparse attention installed.
    """
    config = config or {}

    if method == "svg2":
        from worldserve.optimizations.system_level.sparse_attention.svg2 import SVG2SparseAttention
        optimizer = SVG2SparseAttention(**config)
        # Classify heads then wrap model
        sample_input = config.pop("sample_input", None)
        if sample_input is not None:
            head_classification = optimizer.classify_heads(model, sample_input)
        else:
            head_classification = None
        optimizer.wrap_model(model, head_classification)
        return model

    elif method == "natten":
        from worldserve.optimizations.system_level.sparse_attention.natten_wrapper import NATTENSparseAttention
        kernel_size = config.get("kernel_size", 7)
        optimizer = NATTENSparseAttention(kernel_size=kernel_size)
        optimizer.wrap_spatial_attention(model, kernel_size)
        return model

    else:
        raise ValueError(f"Unknown sparse attention method: {method!r}. Use 'svg2' or 'natten'.")
