"""
Feature-caching optimizations for WorldServe DiT models.

Sprint 1 feature-caching module.  Exports three complementary strategies:

- ERTACache  : Error-corrected Residual Timestep-Aware Cache (arXiv 2508.21091).
               Corrects feature-shift and step-amplification errors via offline
               calibration; validated on Wan2.1 (~2x speedup).

- MagCache   : Magnitude-Law Cache (arXiv 2506.09045).
               Exploits the empirical monotonic decay of per-layer residual
               magnitudes across denoising steps to predict which layers are safe
               to skip; validated on Wan2.1 (~2.68x speedup).

- TaylorSeer : Taylor-series feature predictor (arXiv 2503.06923).
               Instead of reusing stale cached features, predicts future activations
               via finite-difference derivatives; validated on HunyuanVideo (~5x
               speedup for longer denoising schedules).
"""

from worldserve.optimizations.model_level.feature_caching.erta_cache import ERTACache
from worldserve.optimizations.model_level.feature_caching.mag_cache import MagCache
from worldserve.optimizations.model_level.feature_caching.taylor_seer import TaylorSeer

__all__ = ["ERTACache", "MagCache", "TaylorSeer"]
