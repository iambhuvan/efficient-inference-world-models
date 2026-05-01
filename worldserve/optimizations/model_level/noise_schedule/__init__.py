"""
DIAMOND-informed game-video noise schedule for WorldServe (Sprint 2).

Adapts the diffusion noise schedule for Oasis (Open-Oasis world model)
based on findings from the DIAMOND paper (arxiv 2405.12399).

Key insight: game video frames have ~0.95 inter-frame cosine similarity
vs ~0.5 for natural images.  Standard ImageNet-tuned noise schedules
(cosine, from Improved DDPM arxiv 2102.09672) use too much noise at
each step for game video, because the frame-prediction task is inherently
easier (frames are very similar).  Reducing the schedule's s-offset
toward 0.003 shifts the schedule to higher SNR throughout.

The DIAMONDScheduleAdaptor wraps any existing denoising loop so that
the standard linspace(1.0, 0.0, 21) timestep grid (Oasis default for
20 denoising steps) is replaced by an optimized non-uniform grid that
concentrates steps in the 0.2–0.6 range where game-video detail forms.
"""

from worldserve.optimizations.model_level.noise_schedule.diamond_schedule import (
    DIAMONDNoiseSchedule,
    DIAMONDScheduleAdaptor,
)

__all__ = [
    "DIAMONDNoiseSchedule",
    "DIAMONDScheduleAdaptor",
]
