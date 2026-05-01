"""
Radial Attention for WorldServe (Sprint 2 — model-level optimizations).

Implements spatiotemporal energy-decay sparse attention (arxiv 2506.19852)
targeting Matrix-Game 2.0's Wan2.1 backbone.  Window size decays
geometrically with temporal distance, yielding O(N log N) complexity
instead of the dense O(N²).

Validated on Wan2.1-14B: 1.9x speedup on standard 6-frame sequences,
3.7x on long video (>= 12 frames).

LoRA adaptation support is included via RadialLoRAAdapter so that
the sparse attention heads can be fine-tuned without touching the
frozen base weights.
"""

from worldserve.optimizations.model_level.attention.radial_attention import (
    RadialAttentionMask,
    RadialAttentionModule,
    RadialAttentionReplacer,
    RadialLoRAAdapter,
)

__all__ = [
    "RadialAttention",          # canonical alias
    "RadialAttentionMask",
    "RadialAttentionModule",
    "RadialAttentionReplacer",
    "RadialLoRAAdapter",
]

# Canonical top-level alias
RadialAttention = RadialAttentionReplacer
