"""
Model-level optimizations for WorldServe.

These optimizations change *what computation is performed* — fewer denoising
steps, smarter ODE solvers, feature reuse via prediction, token compression,
and distilled knowledge. No systems-level hardware tuning.

Sprint 1 (zero retraining):  samplers, guidance, feature_caching
Sprint 2 (LoRA fine-tuning): attention, dynamic_compute, noise_schedule
Sprint 3 (full distillation): distillation/pcm, dcm, proreflow, context_forcing
Sprint 4 (research-grade):   distillation/rcm, dynamic_compute/dc_dit
"""
