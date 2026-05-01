"""
Dynamic compute optimizations for WorldServe (Sprint 2 + Sprint 3/4).

DyDiT++ (arxiv 2504.06803)
    Per-timestep dynamic width adaptation (TDW) combined with spatial token
    skipping (SDT) for Matrix-Game 2.0's DiT denoising loop.

DC-DiT (arxiv 2603.06351) — Sprint 3/4
    Dynamic Chunking DiT: learns a TokenRouter that sends important tokens
    through the full DiT and spatially pools background tokens, then
    reconstructs with a TokenDecoder.  Upcycles from a pretrained checkpoint
    in 8× fewer training steps.

``DCDiT`` exported from this package is the DyDiT++ alias (Sprint 2 convention).
Use ``DCDiTChunking`` (or import from ``dc_dit`` directly) for the Sprint 3/4
Dynamic Chunking model to avoid naming conflicts.
"""

from worldserve.optimizations.model_level.dynamic_compute.dydit_plus_plus import (
    DyDiTPlusPlus,
    TimestepWidthPredictor,
    SpatialDynamicToken,
    TDLoRAAdapter,
)

# Sprint 2 alias (kept for backwards compatibility).
DCDiT = DyDiTPlusPlus

# Sprint 3/4: Dynamic Chunking DiT.
from worldserve.optimizations.model_level.dynamic_compute.dc_dit import (
    DCDiT as DCDiTChunking,
    DCDiTDistiller,
    TokenRouter,
    SpatialPooler,
    TokenDecoder,
)

__all__ = [
    # Sprint 2
    "DyDiTPlusPlus",
    "DCDiT",
    "TimestepWidthPredictor",
    "SpatialDynamicToken",
    "TDLoRAAdapter",
    # Sprint 3/4
    "DCDiTChunking",
    "DCDiTDistiller",
    "TokenRouter",
    "SpatialPooler",
    "TokenDecoder",
]
