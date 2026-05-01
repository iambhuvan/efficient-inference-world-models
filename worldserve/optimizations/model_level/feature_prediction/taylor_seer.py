"""
TaylorSeerOptimizer — benchmark-facing adapter for TaylorSeer.

Wraps ``TaylorSeer`` (feature_caching) with the API expected by
``mrun/benchmarks/taylor_seer.py``:

  - ``TaylorSeerOptimizer(config=cfg)``
  - ``.reset()`` — clear history (does NOT remove hooks)
  - ``.attach(dit)`` — install forward hooks; returns num_hooked blocks
  - ``.detach()`` — remove hooks and restore original block forwards
  - ``.pre_step(step_idx, total_steps, latents)``
  - ``.post_step(step_idx, total_steps, latents)``
  - ``.get_stats()`` — returns dict with predict_count, compute_count,
                       skip_fraction

Config keys (all optional):
  warmup_steps        int   (default 3)  steps to always run full forward
  predict_every_n     int   (default 2)  only predict on every N-th post-warmup step
  correction_threshold float (default 0.05)  max relative prediction error
  max_history         int   (default 3)  feature history deque length
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from worldserve.optimizations.model_level.feature_caching.taylor_seer import TaylorSeer


class TaylorSeerOptimizer:
    """Benchmark-facing wrapper around TaylorSeer."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self._warmup_steps = int(cfg.get("warmup_steps", 3))
        self._predict_every_n = int(cfg.get("predict_every_n", 2))
        correction_threshold = float(cfg.get("correction_threshold", 0.05))
        max_history = int(cfg.get("max_history", 3))

        # Taylor order: need at least warmup_steps-1 history entries before predicting
        order = max(1, self._warmup_steps - 1)

        self._seer = TaylorSeer(
            num_layers=30,          # updated by wrap_model
            num_steps=10,           # canonical Oasis DDIM; updated by wrap_model
            order=order,
            prediction_threshold=correction_threshold,
        )
        # Override history length from config
        self._seer._hist_maxlen = max(max_history, order + 2)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear feature history for a fresh generation run (keeps hooks)."""
        self._seer.reset_cache()
        self._seer.metrics = type(self._seer.metrics)()  # reset metrics

    def attach(self, model: Any) -> int:
        """Install forward hooks on DiT blocks. Returns number of blocks hooked."""
        self._seer.wrap_model(model)
        return self._seer.num_layers

    def detach(self) -> None:
        """Remove hooks and restore original block forwards."""
        self._seer.remove_hooks()

    # ------------------------------------------------------------------
    # Step-level API (called by _TSWrapper inside the benchmark)
    # ------------------------------------------------------------------

    def pre_step(self, step_idx: int, total_steps: int, latents: Any) -> bool:
        """Decide whether to use Taylor predictions for this step.

        Respects warmup_steps and predict_every_n schedule:
          - Steps < warmup_steps: always full forward (building history)
          - Post-warmup steps: predict on every predict_every_n-th step

        Note: We bypass TaylorSeer._validate_predictions() because it compares
        f_hat(step+1) against f(step) — different timesteps — which causes false
        negatives. Instead we simply check that the prediction dict is populated.
        """
        self._seer._current_step = step_idx

        if step_idx < self._warmup_steps:
            self._seer._use_predictions = False
        else:
            steps_past_warmup = step_idx - self._warmup_steps
            is_predict_step = (steps_past_warmup % self._predict_every_n) == 0
            # Enable predictions if it's a predict step and we have stored predictions
            self._seer._use_predictions = (
                is_predict_step and len(self._seer._predictions) > 0
            )

        return False  # never skip entire step

    def post_step(self, step_idx: int, total_steps: int, latents: Any) -> None:
        self._seer.post_step(step_idx, total_steps, latents)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        m = self._seer.metrics
        total = m.total_calls
        predict_count = m.predictions_used
        compute_count = m.full_forwards
        skip_fraction = predict_count / max(total, 1)
        return {
            "predict_count": predict_count,
            "compute_count": compute_count,
            "skip_fraction": skip_fraction,
            "total_layer_calls": total,
            "prediction_rate": m.prediction_rate,
            "estimated_speedup": m.estimated_speedup,
            "validation_failures": m.validation_failures,
        }
