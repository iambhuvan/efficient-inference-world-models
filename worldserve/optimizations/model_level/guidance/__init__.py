"""
Guidance optimizations for WorldServe.

APG — Adaptive Projected Guidance (arxiv 2410.02416):
    Decomposes the CFG guidance vector into orthogonal and parallel components
    relative to the unconditional prediction, then applies momentum to the
    orthogonal component.  Reduces oversaturation while maintaining sharpness.

NAG — Normalized Attention Guidance (arxiv 2505.21179):
    Applies guidance in attention space and L1-normalizes the guided attention
    weights to keep them on the probability simplex.  Designed for few-step
    distilled models (4-step Oasis, 3-step MG2).
"""

from worldserve.optimizations.model_level.guidance.apg import APGGuidance, APGWrapper
from worldserve.optimizations.model_level.guidance.nag import NAGHook

__all__ = ["APGGuidance", "APGWrapper", "NAGHook"]
