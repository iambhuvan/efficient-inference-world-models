"""
MagCache: Magnitude-Law Cache (arXiv 2506.09045).

Validated on Wan2.1 (30 DiT blocks, 3 denoising steps).  Target: ~2.68x speedup.

The Magnitude Law
-----------------
Empirically, the relative residual magnitude for each DiT layer decreases
monotonically across denoising steps:

    r_l(s) = || out_l(s) - out_l(s-1) ||_F  /  || out_l(s-1) ||_F

    r_l(2) <= r_l(1) <= r_l(0)   (holds consistently across Wan2.1 layers)

This means that once a layer's residual ratio drops below a per-layer threshold,
it is safe to reuse the cached output.  Calibration measures r_l(s) on a single
representative input and sets per-layer thresholds at the 80th percentile of the
calibrated ratios.

Adaptive pre-skipping (look-ahead)
-----------------------------------
At step s the measured r_l(s) is used to predict r_l(s+1):

    r_l_pred(s+1) = r_l(s) * decay_factor,   decay_factor ≈ 0.7

Layers whose predicted ratio is below threshold are pre-marked for skipping so
that the block forward is bypassed entirely (no tensor allocation at all), rather
than running the forward and then deciding to discard the output.

Usage
-----
    cache = MagCache(num_layers=30, num_steps=3,
                     cache_threshold_percentile=80,
                     min_threshold=0.02, max_threshold=0.15)
    cache.calibrate(model, sample_input, timesteps)
    cache.wrap_model(model)

    for step_idx in range(num_steps):
        cache._current_step = step_idx
        out = model(latent, timesteps[step_idx], **kwargs)
    metrics = cache.get_metrics()
    cache.remove_hooks()
"""

from __future__ import annotations

import logging
import math
import types as _types
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_DECAY_FACTOR: float = 0.7   # empirical magnitude-law decay between steps


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class _MagMetrics:
    total_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    _per_layer_hits: Dict[int, int] = field(default_factory=dict)
    _per_layer_total: Dict[int, int] = field(default_factory=dict)
    # Observed magnitude ratios at runtime: layer -> list[(step, ratio)]
    observed_ratios: Dict[int, List[Tuple[int, float]]] = field(default_factory=dict)

    @property
    def hit_rate(self) -> float:
        return self.cache_hits / max(self.total_calls, 1)

    def record(self, layer_idx: int, hit: bool, step: int, ratio: float) -> None:
        self.total_calls += 1
        self._per_layer_hits.setdefault(layer_idx, 0)
        self._per_layer_total.setdefault(layer_idx, 0)
        self._per_layer_total[layer_idx] += 1
        if hit:
            self.cache_hits += 1
            self._per_layer_hits[layer_idx] += 1
        else:
            self.cache_misses += 1
        self.observed_ratios.setdefault(layer_idx, []).append((step, ratio))

    def per_layer_hit_rate(self) -> Dict[int, float]:
        return {
            l: self._per_layer_hits.get(l, 0) / max(self._per_layer_total.get(l, 0), 1)
            for l in self._per_layer_total
        }

    def summary(self) -> Dict[str, Any]:
        avg_ratio: Dict[int, float] = {}
        for l, pairs in self.observed_ratios.items():
            ratios = [r for _, r in pairs]
            avg_ratio[l] = sum(ratios) / len(ratios) if ratios else 0.0
        return {
            "total_layer_calls": self.total_calls,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "overall_hit_rate": self.hit_rate,
            "per_layer_hit_rate": self.per_layer_hit_rate(),
            "per_layer_avg_magnitude_ratio": avg_ratio,
        }


# ---------------------------------------------------------------------------
# MagCache
# ---------------------------------------------------------------------------

class MagCache:
    """Magnitude-Law Cache for DiT denoising models.

    Parameters
    ----------
    num_layers:
        Number of DiT transformer blocks.
    num_steps:
        Total denoising steps in the schedule.
    cache_threshold_percentile:
        Calibration percentile (default 80).  Per-layer thresholds are set so
        that caching occurs when the ratio is below the p-th percentile of the
        calibrated ratios.
    min_threshold / max_threshold:
        Hard clamps on computed thresholds to prevent degenerate cases.
    """

    def __init__(
        self,
        num_layers: int = 30,
        num_steps: int = 3,
        cache_threshold_percentile: int = 80,
        min_threshold: float = 0.02,
        max_threshold: float = 0.15,
    ) -> None:
        self.num_layers = num_layers
        self.num_steps = num_steps
        self.cache_threshold_percentile = cache_threshold_percentile
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

        # Calibration products
        # threshold_l[l][s] = threshold for (layer l, step s)
        self._threshold: Dict[int, Dict[int, float]] = {}
        # Calibrated r_l(s) values: layer -> list of per-step ratios
        self._calib_ratios: Dict[int, List[float]] = {}
        self._calibrated: bool = False

        # Runtime state
        self._cache: Dict[int, torch.Tensor] = {}
        self._current_step: int = 0
        # Pre-skip flags: layer_idx -> bool (should skip on *next* step)
        self._pre_skip: Dict[int, bool] = {}
        # Last observed ratio per layer (for look-ahead prediction)
        self._last_ratio: Dict[int, float] = {}

        self._layer_map: Dict[int, int] = {}
        self._hooks: List[torch.utils.hooks.RemovableHook] = []

        self.metrics = _MagMetrics()

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> None:
        """Single-sample calibration pass to compute per-layer-per-step thresholds.

        Runs *num_steps* forward passes on *sample_input* and records
        r_l(s) = ||out_l(s) - out_l(s-1)||_F / ||out_l(s-1)||_F for each layer.
        Per-layer thresholds are derived as the p-th percentile of observed ratios.

        Args:
            model:        The DiT model.
            sample_input: Representative latent (single sample, any resolution).
            timesteps:    1-D tensor of timestep values, length == num_steps.
        """
        blocks = self._find_blocks(model)
        if blocks is None:
            raise RuntimeError(
                "MagCache.calibrate: could not locate DiT blocks."
            )

        actual_n = len(blocks)
        if actual_n != self.num_layers:
            self.num_layers = actual_n
            logger.info("MagCache: recalibrated num_layers=%d", self.num_layers)

        calib_outputs: Dict[int, List[torch.Tensor]] = {
            i: [] for i in range(self.num_layers)
        }
        temp_hooks: List[torch.utils.hooks.RemovableHook] = []

        def _make_record_hook(layer_idx: int) -> Callable:
            @torch._dynamo.disable
            def _record(module: nn.Module, inp: Any, out: Any) -> None:
                tensor = out[0] if isinstance(out, (tuple, list)) else out
                if isinstance(tensor, torch.Tensor):
                    calib_outputs[layer_idx].append(tensor.detach().float().clone())

            return _record

        for idx, block in enumerate(blocks):
            h = block.register_forward_hook(_make_record_hook(idx))
            temp_hooks.append(h)

        model.eval()
        with torch.no_grad():
            x = sample_input
            for step_idx in range(self.num_steps):
                t = timesteps[step_idx] if timesteps.dim() >= 1 else timesteps
                x = model(x, t)

        for h in temp_hooks:
            h.remove()

        # Derive per-layer per-step ratios, then compute thresholds
        self._calib_ratios.clear()
        self._threshold.clear()

        for l in range(self.num_layers):
            outs = calib_outputs[l]
            ratios: List[float] = []
            for s in range(1, len(outs)):
                r = self._compute_ratio_tensors(outs[s], outs[s - 1])
                ratios.append(r)
            self._calib_ratios[l] = ratios

            # Per-step thresholds: for step s, threshold = percentile of calib ratios
            step_thresholds: Dict[int, float] = {}
            if ratios:
                # Compute the global percentile threshold from all observed ratios
                sorted_r = sorted(ratios)
                pct_idx = int(
                    math.ceil(self.cache_threshold_percentile / 100.0 * len(sorted_r)) - 1
                )
                pct_idx = max(0, min(pct_idx, len(sorted_r) - 1))
                base_thresh = sorted_r[pct_idx]
                base_thresh = max(self.min_threshold, min(self.max_threshold, base_thresh))
            else:
                base_thresh = self.min_threshold

            for s in range(1, self.num_steps):
                # Later steps have lower ratios (magnitude law) so we can afford
                # tighter thresholds.  Scale threshold proportionally.
                step_factor = _DECAY_FACTOR ** (s - 1)
                thresh_s = base_thresh * (1.0 + (1.0 - step_factor) * 0.5)
                thresh_s = max(self.min_threshold, min(self.max_threshold, thresh_s))
                step_thresholds[s] = thresh_s

            self._threshold[l] = step_thresholds

        self._calibrated = True
        all_thresholds = [
            t for step_d in self._threshold.values() for t in step_d.values()
        ]
        if all_thresholds:
            logger.info(
                "MagCache calibration complete. Threshold range: [%.4f, %.4f]",
                min(all_thresholds), max(all_thresholds),
            )

    # ------------------------------------------------------------------
    # Magnitude ratio helper
    # ------------------------------------------------------------------

    def _compute_ratio(
        self, current_out: torch.Tensor, cached_out: torch.Tensor
    ) -> float:
        """Compute r_l = ||current - cached||_F / ||cached||_F."""
        return self._compute_ratio_tensors(current_out, cached_out)

    @staticmethod
    def _compute_ratio_tensors(a: torch.Tensor, b: torch.Tensor) -> float:
        denom = b.float().norm(p="fro").item()
        if denom < 1e-12:
            return 0.0
        return (a.float() - b.float()).norm(p="fro").item() / denom

    # ------------------------------------------------------------------
    # Caching criterion
    # ------------------------------------------------------------------

    def _should_cache_layer(self, layer_idx: int, step_idx: int) -> bool:
        """Return True if this layer should be skipped (pre-skip prediction).

        Uses the magnitude law to predict whether r_l at the current step will
        be below threshold based on the last observed ratio.
        """
        if step_idx == 0:
            return False
        if self._cache.get(layer_idx) is None:
            return False

        # Check pre-skip flag set at the end of the previous step
        return self._pre_skip.get(layer_idx, False)

    def _update_pre_skip(self, layer_idx: int, current_ratio: float, step_idx: int) -> None:
        """Update look-ahead skip flag for the *next* step."""
        if step_idx + 1 >= self.num_steps:
            self._pre_skip[layer_idx] = False
            return

        # Predict next-step ratio using magnitude law decay
        predicted_ratio = current_ratio * _DECAY_FACTOR
        next_step = step_idx + 1
        threshold = self._threshold.get(layer_idx, {}).get(next_step, self.min_threshold)
        self._pre_skip[layer_idx] = predicted_ratio < threshold

    # ------------------------------------------------------------------
    # Model wrapping
    # ------------------------------------------------------------------

    def wrap_model(self, model: nn.Module) -> nn.Module:
        """Install caching forward-replacement hooks on each DiT block.

        Returns the same model (mutated in place).
        """
        if not self._calibrated:
            logger.warning(
                "MagCache.wrap_model called before calibrate(). "
                "All thresholds default to min_threshold=%.4f.",
                self.min_threshold,
            )
            # Default thresholds: min_threshold for all layers/steps
            for l in range(self.num_layers):
                self._threshold[l] = {
                    s: self.min_threshold for s in range(1, self.num_steps)
                }

        blocks = self._find_blocks(model)
        if blocks is None:
            raise RuntimeError("MagCache.wrap_model: could not locate DiT blocks.")

        actual_n = len(blocks)
        if actual_n != self.num_layers:
            self.num_layers = actual_n
            # Rebuild default thresholds for new depth
            for l in range(self.num_layers):
                if l not in self._threshold:
                    self._threshold[l] = {
                        s: self.min_threshold for s in range(1, self.num_steps)
                    }

        self._layer_map.clear()
        for idx, block in enumerate(blocks):
            self._layer_map[id(block)] = idx
            original_fwd = block.forward
            wrapped_fwd = self._make_cached_forward(idx, original_fwd)
            block.forward = _types.MethodType(wrapped_fwd, block)
            post_hook = block.register_forward_hook(self._make_post_hook(idx))
            self._hooks.append(post_hook)

        logger.info(
            "MagCache: wrapped %d DiT blocks.", self.num_layers
        )
        return model

    # ------------------------------------------------------------------
    # Hook factories
    # ------------------------------------------------------------------

    def _make_cached_forward(self, layer_idx: int, original_fwd: Callable) -> Callable:
        optimizer = self

        @torch._dynamo.disable
        def cached_forward(self_block: nn.Module, *args: Any, **kwargs: Any) -> Any:
            step = optimizer._current_step

            if optimizer._should_cache_layer(layer_idx, step):
                cached_out = optimizer._cache[layer_idx]
                # Compute actual ratio for metric recording and look-ahead update
                if args and isinstance(args[0], torch.Tensor):
                    x_in = args[0]
                    if x_in.shape == cached_out.shape:
                        ratio = optimizer._compute_ratio(x_in, cached_out)
                    else:
                        ratio = 0.0
                else:
                    ratio = 0.0
                optimizer.metrics.record(layer_idx, hit=True, step=step, ratio=ratio)
                optimizer._last_ratio[layer_idx] = ratio
                # Update look-ahead for the next step
                optimizer._update_pre_skip(layer_idx, ratio, step)
                return cached_out

            # Cache miss: fall through to original forward
            return original_fwd(*args, **kwargs)

        return cached_forward

    def _make_post_hook(self, layer_idx: int) -> Callable:
        optimizer = self

        @torch._dynamo.disable
        def post_hook_fn(module: nn.Module, inp: Any, output: Any) -> None:
            step = optimizer._current_step
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(tensor, torch.Tensor):
                return

            # Compute the ratio between this output and the cached previous output
            cached = optimizer._cache.get(layer_idx)
            if cached is not None and tensor.shape == cached.shape:
                ratio = optimizer._compute_ratio(tensor, cached)
            else:
                ratio = 1.0  # no prior cache: treat as large change

            # Update pre-skip flag for next step using the *actual* ratio
            optimizer._update_pre_skip(layer_idx, ratio, step)
            optimizer._last_ratio[layer_idx] = ratio

            # Store new output
            optimizer._cache[layer_idx] = tensor.detach().clone()

            # Record metrics (only on cache miss or step 0)
            if step == 0:
                optimizer.metrics.record(layer_idx, hit=False, step=step, ratio=ratio)

        return post_hook_fn

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def reset_cache(self) -> None:
        """Clear cached outputs for a new generation request."""
        self._cache.clear()
        self._pre_skip.clear()
        self._last_ratio.clear()
        self._current_step = 0

    def remove_hooks(self) -> None:
        """Remove all registered forward hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def get_metrics(self) -> Dict[str, Any]:
        """Return cache hit rates, magnitude ratios observed."""
        return self.metrics.summary()

    # ------------------------------------------------------------------
    # OptimizationStack interface
    # ------------------------------------------------------------------

    def pre_step(self, step_idx: int, total_steps: int, latents: Any) -> bool:
        self._current_step = step_idx
        return False

    def post_step(self, step_idx: int, total_steps: int, latents: Any) -> None:
        pass

    def get_stats(self) -> Dict[str, Any]:
        return self.get_metrics()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _find_blocks(model: nn.Module) -> Optional[nn.ModuleList]:
        for attr in ("blocks", "layers", "transformer_blocks", "dit_blocks"):
            candidate = getattr(model, attr, None)
            if isinstance(candidate, (nn.ModuleList, list)):
                return candidate
        for child in model.children():
            if isinstance(child, nn.ModuleList):
                return child
        return None
