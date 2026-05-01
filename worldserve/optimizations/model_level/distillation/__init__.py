"""
WorldServe distillation optimizations package.

Provides model-level distillation techniques for Oasis (DiT-S/2 DDPM with
v-prediction, sigmoid_beta_schedule, 10-step DDIM at inference) and Matrix-
Game 2.0 (Wan2.1 Flow Matching, 1.8B params).

Sprint 3 / 4 contributions
--------------------------
PCMDistiller         -- Phased Consistency Model (arxiv 2405.18407) for Oasis.
DCMDistiller         -- Dual-Expert Consistency Model (arxiv 2506.03123).
ProReflowDistiller   -- Progressive Reflow + aligned v-prediction (arxiv 2503.04824) for MG2.
ContextForcingDistiller -- Context-forced distillation (WorldPlay arxiv 2512.14614).
rCMDistiller         -- Score-regularised continuous-time CM (arxiv 2510.08431) for MG2.
"""

from __future__ import annotations

from worldserve.optimizations.model_level.distillation.pcm import PCMDistiller
from worldserve.optimizations.model_level.distillation.dcm import DCMDistiller
from worldserve.optimizations.model_level.distillation.proreflow import ProReflowDistiller
from worldserve.optimizations.model_level.distillation.context_forcing import ContextForcingDistiller
from worldserve.optimizations.model_level.distillation.rcm import rCMDistiller

__all__ = [
    "PCMDistiller",
    "DCMDistiller",
    "ProReflowDistiller",
    "ContextForcingDistiller",
    "rCMDistiller",
]
