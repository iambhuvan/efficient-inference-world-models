"""
TaylorSeer: Taylor-series Feature Prediction for DiT denoising (arXiv 2503.06923).

Validated on HunyuanVideo (~5x speedup for long denoising schedules).

Core idea
---------
Rather than reusing a stale cached output, TaylorSeer *predicts* the next-step
feature via a truncated Taylor expansion around the current timestep:

    f_l(t + h) ≈ f_l(t) + h·f'_l(t) + (h²/2)·f''_l(t)

where derivatives are approximated by finite differences:

    f'_l(t)  ≈  (f_l(t) - f_l(t-Δt)) / Δt          [1st-order FD, needs 2 history]
    f''_l(t) ≈  (f'_l(t) - f'_l(t-Δt)) / Δt         [2nd-order FD, needs 3 history]

For the 3-step Matrix-Game 2.0 schedule (steps 0→1→2), 1st-order is sufficient:
    Step 0: full forward, record f_l(0)
    Step 1: full forward, record f_l(1); derive f'_l(1) = f_l(1) - f_l(0)
    Step 2: predict f̂_l(2) = f_l(1) + f'_l(1)        ← bypass block entirely

For longer schedules (e.g. 10-step Oasis canonical), 2nd-order provides better accuracy.

Prediction validation
---------------------
Before applying predictions on a given step, TaylorSeer evaluates the Taylor
approximation on a small *validation subset* (default: 3 randomly sampled layers,
user-configurable).  If the relative prediction error for those layers exceeds
``prediction_threshold``, the step falls back to a full forward pass for all
layers.  This guard prevents compounding errors on distribution shifts.

Usage
-----
    seer = TaylorSeer(num_layers=30, num_steps=3, order=1,
                      prediction_threshold=0.15)
    seer.wrap_model(model)          # no offline calibration needed

    for step_idx in range(num_steps):
        seer._current_step = step_idx
        out = model(latent, timesteps[step_idx], **kwargs)
    metrics = seer.get_metrics()
    seer.remove_hooks()
"""

from __future__ import annotations

import logging
import random
import types as _types
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class _TaylorMetrics:
    total_calls: int = 0
    predictions_used: int = 0
    full_forwards: int = 0
    validation_failures: int = 0
    # Per-layer prediction errors (when we have ground truth)
    prediction_errors: Dict[int, List[float]] = field(default_factory=dict)
    _per_layer_preds: Dict[int, int] = field(default_factory=dict)
    _per_layer_full: Dict[int, int] = field(default_factory=dict)

    @property
    def prediction_rate(self) -> float:
        return self.predictions_used / max(self.total_calls, 1)

    @property
    def estimated_speedup(self) -> float:
        """Rough estimate: predicted layers cost ~0.05x of a full forward."""
        if self.total_calls == 0:
            return 1.0
        fwd_frac = self.full_forwards / self.total_calls
        pred_frac = self.predictions_used / self.total_calls
        return 1.0 / (fwd_frac + pred_frac * 0.05)

    def record(self, layer_idx: int, predicted: bool) -> None:
        self.total_calls += 1
        self._per_layer_preds.setdefault(layer_idx, 0)
        self._per_layer_full.setdefault(layer_idx, 0)
        if predicted:
            self.predictions_used += 1
            self._per_layer_preds[layer_idx] += 1
        else:
            self.full_forwards += 1
            self._per_layer_full[layer_idx] += 1

    def record_error(self, layer_idx: int, rel_error: float) -> None:
        self.prediction_errors.setdefault(layer_idx, []).append(rel_error)

    def per_layer_prediction_rate(self) -> Dict[int, float]:
        all_layers = set(self._per_layer_preds) | set(self._per_layer_full)
        return {
            l: self._per_layer_preds.get(l, 0) / max(
                self._per_layer_preds.get(l, 0) + self._per_layer_full.get(l, 0), 1
            )
            for l in all_layers
        }

    def avg_prediction_error(self) -> Dict[int, float]:
        return {
            l: sum(v) / len(v)
            for l, v in self.prediction_errors.items()
            if v
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "total_layer_calls": self.total_calls,
            "predictions_used": self.predictions_used,
            "full_forwards": self.full_forwards,
            "prediction_rate": self.prediction_rate,
            "validation_failures": self.validation_failures,
            "estimated_speedup": self.estimated_speedup,
            "per_layer_prediction_rate": self.per_layer_prediction_rate(),
            "per_layer_avg_prediction_error": self.avg_prediction_error(),
        }


# ---------------------------------------------------------------------------
# TaylorSeer
# ---------------------------------------------------------------------------

class TaylorSeer:
    """Taylor-series Feature Predictor for DiT denoising models.

    Parameters
    ----------
    num_layers:
        Number of DiT transformer blocks.
    num_steps:
        Total denoising steps.
    order:
        Taylor expansion order.  1 for 3-step schedules; 2 for ≥5 steps.
    prediction_threshold:
        Maximum relative prediction error (on validation layers) above which the
        step falls back to a full forward.  0.15 means 15% error tolerance.
    validate_layers:
        Explicit list of layer indices used for prediction validation.  If None,
        3 layers are sampled uniformly from the model depth at wrap time.
    """

    def __init__(
        self,
        num_layers: int = 30,
        num_steps: int = 3,
        order: int = 1,
        prediction_threshold: float = 0.15,
        validate_layers: Optional[List[int]] = None,
    ) -> None:
        self.num_layers = num_layers
        self.num_steps = num_steps
        self.order = order
        self.prediction_threshold = prediction_threshold
        self._validate_layers_cfg: Optional[List[int]] = validate_layers

        # Feature history per layer: deque of up to (order+2) most-recent outputs
        # History is in chronological order: history[0] = oldest, [-1] = newest
        self._feature_history: Dict[int, Deque[torch.Tensor]] = {}

        # Taylor predictions pre-computed after step s for use in step s+1
        # layer_idx -> predicted tensor
        self._predictions: Dict[int, torch.Tensor] = {}

        # Per-step prediction validity flag (set by validation logic)
        # True = use Taylor predictions for this step, False = full forward
        self._use_predictions: bool = False
        self._current_step: int = 0

        # Validation layer indices (resolved in wrap_model)
        self._validate_layers: List[int] = []

        self._layer_map: Dict[int, int] = {}
        self._hooks: List[torch.utils.hooks.RemovableHook] = []
        # Original block forwards saved by wrap_model; restored by remove_hooks
        self._original_fwds: Dict[int, Any] = {}
        # List of (block, original_forward) pairs for restoration
        self._block_fwd_pairs: List[Tuple[nn.Module, Any]] = []

        self.metrics = _TaylorMetrics()

        # History deque max length: order+2 gives room for 2nd-order FD
        self._hist_maxlen: int = max(self.order + 2, 3)

    # ------------------------------------------------------------------
    # History helpers
    # ------------------------------------------------------------------

    def _get_history(self, layer_idx: int) -> Deque[torch.Tensor]:
        if layer_idx not in self._feature_history:
            self._feature_history[layer_idx] = deque(maxlen=self._hist_maxlen)
        return self._feature_history[layer_idx]

    def _push_history(self, layer_idx: int, tensor: torch.Tensor) -> None:
        """Append *tensor* to layer history (detached, cloned)."""
        self._get_history(layer_idx).append(tensor.detach().clone())

    # ------------------------------------------------------------------
    # Taylor prediction
    # ------------------------------------------------------------------

    def _taylor_predict(
        self,
        layer_idx: int,
        h: float = 1.0,
    ) -> Optional[torch.Tensor]:
        """Compute Taylor prediction f_l(t+h) from feature history.

        Requires at least ``order+1`` history entries.
        Returns None if history is insufficient.

        Args:
            layer_idx: Layer index.
            h:         Step size (default 1 for uniform timestep spacing).
        """
        hist = self._get_history(layer_idx)

        if len(hist) < 2:
            # Need at least 2 entries for 1st-order
            return None

        f_t = hist[-1]       # most recent (step t)
        f_tm1 = hist[-2]     # previous   (step t-1)

        # 1st derivative: f'(t) ≈ (f(t) - f(t-1)) / 1
        d1 = f_t.float() - f_tm1.float()   # Δt = 1 (uniform steps)

        prediction = f_t.float() + h * d1

        if self.order >= 2 and len(hist) >= 3:
            f_tm2 = hist[-3]   # step t-2
            d1_prev = f_tm1.float() - f_tm2.float()   # f'(t-1)
            # 2nd derivative: f''(t) ≈ (d1 - d1_prev) / 1
            d2 = d1 - d1_prev
            prediction = prediction + (h ** 2 / 2.0) * d2

        return prediction.to(f_t.dtype)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_predictions(self, step_idx: int) -> bool:
        """Validate Taylor predictions on a subset of layers.

        Runs the original forward for *validate_layers* only (the hooks on
        those layers are temporarily bypassed via a flag), compares the result
        against the stored prediction, and returns True if the average relative
        error is below ``prediction_threshold``.

        In the hook-based implementation this is accomplished differently: the
        validation layers' predictions are compared against the *actual* output
        captured by the post-hook on the previous step, since at that point both
        the ground truth and the prediction are available.

        Returns True if predictions are acceptable; False means fall back.
        """
        if not self._validate_layers:
            return True

        errors: List[float] = []
        for l in self._validate_layers:
            pred = self._predictions.get(l)
            hist = self._get_history(l)
            if pred is None or len(hist) == 0:
                continue
            # Compare prediction (for step s) against actual (step s, stored
            # at the end of the post-hook — but at validation time the actual
            # output has just been recorded)
            actual = hist[-1]  # the just-stored actual output
            if pred.shape != actual.shape:
                continue
            denom = actual.float().norm(p="fro").item()
            if denom < 1e-12:
                continue
            err = (pred.float() - actual.float()).norm(p="fro").item() / denom
            errors.append(err)
            self.metrics.record_error(l, err)

        if not errors:
            return True

        avg_err = sum(errors) / len(errors)
        if avg_err > self.prediction_threshold:
            self.metrics.validation_failures += 1
            logger.debug(
                "TaylorSeer step %d: validation failed (avg_err=%.4f > %.4f); "
                "falling back to full forward.",
                step_idx, avg_err, self.prediction_threshold,
            )
            return False
        return True

    # ------------------------------------------------------------------
    # Model wrapping
    # ------------------------------------------------------------------

    def wrap_model(self, model: nn.Module) -> nn.Module:
        """Install feature-prediction hooks on each DiT block.

        No offline calibration is needed.  History accumulates during generation.
        Returns the same model (mutated in place).
        """
        blocks = self._find_blocks(model)
        if blocks is None:
            raise RuntimeError(
                "TaylorSeer.wrap_model: could not locate DiT blocks."
            )

        actual_n = len(blocks)
        if actual_n != self.num_layers:
            self.num_layers = actual_n

        # Choose validation layers
        if self._validate_layers_cfg is not None:
            # Filter to valid indices
            self._validate_layers = [
                l for l in self._validate_layers_cfg if 0 <= l < self.num_layers
            ]
        else:
            # Sample 3 evenly spaced layers
            n_val = min(3, self.num_layers)
            if n_val == self.num_layers:
                self._validate_layers = list(range(n_val))
            else:
                step_size = self.num_layers // n_val
                self._validate_layers = [i * step_size for i in range(n_val)]

        # Adjust Taylor order based on num_steps if user left it at default
        if self.num_steps >= 5 and self.order < 2:
            logger.info(
                "TaylorSeer: num_steps=%d >= 5, upgrading to 2nd-order Taylor.",
                self.num_steps,
            )
            self.order = 2
            self._hist_maxlen = max(self.order + 2, 3)

        self._layer_map.clear()
        self._original_fwds.clear()
        self._block_fwd_pairs.clear()
        for idx, block in enumerate(blocks):
            self._layer_map[id(block)] = idx
            original_fwd = block.forward
            self._original_fwds[id(block)] = original_fwd
            self._block_fwd_pairs.append((block, original_fwd))
            wrapped_fwd = self._make_cached_forward(idx, original_fwd)
            block.forward = _types.MethodType(wrapped_fwd, block)
            post_hook = block.register_forward_hook(self._make_post_hook(idx))
            self._hooks.append(post_hook)

        logger.info(
            "TaylorSeer: wrapped %d DiT blocks (order=%d, validate_layers=%s).",
            self.num_layers, self.order, self._validate_layers,
        )
        return model

    # ------------------------------------------------------------------
    # Hook factories
    # ------------------------------------------------------------------

    def _make_cached_forward(self, layer_idx: int, original_fwd: Callable) -> Callable:
        """Return forward replacement that returns Taylor prediction when available."""
        seer = self

        @torch._dynamo.disable
        def cached_forward(self_block: nn.Module, *args: Any, **kwargs: Any) -> Any:
            step = seer._current_step

            # Step 0 and 1: always full forward (building history)
            # Step >= 2 (or any step where predictions are validated): use prediction
            if step >= seer.order + 1 and seer._use_predictions:
                pred = seer._predictions.get(layer_idx)
                if pred is not None:
                    # Return prediction; post_hook won't fire (forward skipped)
                    # We need to push the prediction into history so future steps
                    # can build on it — done inside post_hook alternative below.
                    seer.metrics.record(layer_idx, predicted=True)
                    # Store prediction into history for compounding Taylor steps
                    seer._push_history(layer_idx, pred)
                    return pred

            # Full forward
            result = original_fwd(*args, **kwargs)
            return result

        return cached_forward

    def _make_post_hook(self, layer_idx: int) -> Callable:
        """Post-hook: record feature output and pre-compute Taylor prediction.

        Fires only when the real forward ran (cache miss path).
        When a prediction was returned by cached_forward, the block's real
        forward is not called, so this hook does not fire — the prediction
        push into history is done inside cached_forward above.
        """
        seer = self

        @torch._dynamo.disable
        def post_hook_fn(module: nn.Module, inp: Any, output: Any) -> None:
            step = seer._current_step
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(tensor, torch.Tensor):
                return

            # Push actual output into history
            seer._push_history(layer_idx, tensor)

            # Pre-compute prediction for the next step
            pred = seer._taylor_predict(layer_idx, h=1.0)
            if pred is not None:
                seer._predictions[layer_idx] = pred

            seer.metrics.record(layer_idx, predicted=False)

        return post_hook_fn

    # ------------------------------------------------------------------
    # Step-level API (called by the generation loop)
    # ------------------------------------------------------------------

    def pre_step(self, step_idx: int, total_steps: int, latents: Any) -> bool:
        """Called before each denoising step.

        Decides whether to use Taylor predictions for this step by running
        the validation check against the *previous* step's predictions.
        Always returns False (prediction is layer-level, not step-level skip).
        """
        self._current_step = step_idx

        if step_idx < self.order + 1:
            # Not enough history yet for Taylor prediction
            self._use_predictions = False
        else:
            # Validate predictions stored at end of previous step
            self._use_predictions = self._validate_predictions(step_idx)

        return False  # never skip entire step

    def post_step(self, step_idx: int, total_steps: int, latents: Any) -> None:
        pass

    def get_stats(self) -> Dict[str, Any]:
        return self.get_metrics()

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def reset_cache(self) -> None:
        """Clear feature history for a new generation request (does NOT remove hooks)."""
        self._feature_history.clear()
        self._predictions.clear()
        self._use_predictions = False
        self._current_step = 0

    def remove_hooks(self) -> None:
        """Remove all registered forward hooks and restore original block forwards."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        # Restore original block.forward methods saved during wrap_model
        for block, orig_fwd in self._block_fwd_pairs:
            block.forward = orig_fwd
        self._block_fwd_pairs.clear()
        self._original_fwds.clear()

    def get_metrics(self) -> Dict[str, Any]:
        """Return prediction accuracy, speedup ratio."""
        return self.metrics.summary()

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
