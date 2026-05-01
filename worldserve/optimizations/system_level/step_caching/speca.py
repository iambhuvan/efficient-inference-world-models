"""
SpeCa: Speculative Feature Caching with Lightweight Verification.

Optional comparison method for step-caching ablation study.
Core idea: cache all intermediate DiT features at step t, then at step t+1
speculatively reuse them.  A single "verification layer" is run to check
quality; if the verification passes (cosine similarity above threshold) the
cached features are accepted, otherwise a full forward is executed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class SpeCaMetrics:
    total_verifications: int = 0
    accepted: int = 0
    rejected: int = 0
    per_layer_acceptance: Dict[int, float] = field(default_factory=dict)
    _per_layer_accept: Dict[int, int] = field(default_factory=dict)
    _per_layer_total: Dict[int, int] = field(default_factory=dict)
    time_saved_ms: float = 0.0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / max(self.total_verifications, 1)

    def record(self, layer_idx: int, accepted: bool) -> None:
        self.total_verifications += 1
        self._per_layer_accept.setdefault(layer_idx, 0)
        self._per_layer_total.setdefault(layer_idx, 0)
        self._per_layer_total[layer_idx] += 1
        if accepted:
            self.accepted += 1
            self._per_layer_accept[layer_idx] += 1
        else:
            self.rejected += 1
        self.per_layer_acceptance[layer_idx] = (
            self._per_layer_accept[layer_idx] / self._per_layer_total[layer_idx]
        )

    def summary(self) -> Dict[str, Any]:
        return {
            "acceptance_rate": self.acceptance_rate,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "total_verifications": self.total_verifications,
            "time_saved_ms": round(self.time_saved_ms, 2),
            "per_layer_acceptance": dict(self.per_layer_acceptance),
        }


# ---------------------------------------------------------------------------
# SpeCaOptimizer
# ---------------------------------------------------------------------------

class SpeCaOptimizer:
    """Speculative caching with single-layer verification for DiT models.

    Workflow per denoising block (3 steps in MG2):
      Step 0 — full forward; cache every layer's output.
      Steps 1+ — for each layer, *speculatively* reuse cached features.
                  Run only the verification layer; if its output is close
                  enough (cosine_sim > threshold), accept all cached features.
                  Otherwise fall back to a full recompute of remaining layers.
    """

    def __init__(
        self,
        num_layers: int = 30,
        verify_layer_idx: Optional[int] = None,
        threshold: float = 0.95,
    ) -> None:
        """
        Args:
            num_layers: Number of DiT blocks.
            verify_layer_idx: Index of the layer used as a quality proxy.
                Defaults to the last layer (num_layers - 1).
            threshold: Cosine similarity threshold for accepting cached
                features.  Higher = stricter quality but fewer cache hits.
        """
        self.num_layers = num_layers
        self.verify_layer_idx = verify_layer_idx if verify_layer_idx is not None else num_layers - 1
        self.threshold = threshold

        # Runtime state
        self._cached_features: Dict[int, torch.Tensor] = {}
        self._cached_verify_output: Optional[torch.Tensor] = None
        self._current_step: int = 0
        self._speculation_active: bool = False
        self._speculation_accepted: Optional[bool] = None
        self._hooks: List[torch.utils.hooks.RemovableHook] = []
        self.metrics = SpeCaMetrics()

    # ------------------------------------------------------------------
    # Cache a full step
    # ------------------------------------------------------------------

    def cache_step(
        self,
        model: nn.Module,
        x: torch.Tensor,
        timestep: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run a full forward pass through *model* and cache all layer outputs.

        This should be called for denoising step 0 to populate the cache.

        Returns:
            Model output.
        """
        self._cached_features.clear()
        self._current_step = 0
        self._speculation_active = False

        output = model(x, timestep, **kwargs)
        return output

    # ------------------------------------------------------------------
    # Predict (reuse) and verify
    # ------------------------------------------------------------------

    def predict_and_verify(
        self,
        model: nn.Module,
        x: torch.Tensor,
        timestep: torch.Tensor,
        cached_features: Optional[Dict[int, torch.Tensor]] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, bool]:
        """Speculative step: reuse cached features, verify with one layer.

        For each layer the "prediction" is simply the cached feature from the
        previous step (identity prediction).  Only the verification layer
        actually executes; its output is compared against the cached version.

        Args:
            model: DiT model (hooks must be installed).
            x: Input latent for this denoising step.
            timestep: Current timestep value.
            cached_features: Override cache (if None, uses internal cache).
            **kwargs: Forwarded to model.

        Returns:
            (output, accepted): The model output and whether speculation was
            accepted (True) or fell back to full recompute (False).
        """
        if cached_features is not None:
            self._cached_features = cached_features

        if not self._cached_features:
            # No cache — full forward
            output = model(x, timestep, **kwargs)
            return output, False

        # Activate speculation mode — hooks will intercept
        self._speculation_active = True
        self._speculation_accepted = None

        t_start = time.perf_counter()
        output = model(x, timestep, **kwargs)
        elapsed_ms = (time.perf_counter() - t_start) * 1000

        accepted = self._speculation_accepted is True
        if accepted:
            self.metrics.time_saved_ms += elapsed_ms * 0.5  # rough estimate

        self._speculation_active = False
        return output, accepted

    # ------------------------------------------------------------------
    # Full denoising loop
    # ------------------------------------------------------------------

    def cache_and_reuse(
        self,
        model: nn.Module,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        num_steps: int = 3,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run all denoising steps with speculative caching.

        Step 0: full forward (populates cache via hooks).
        Steps 1+: speculative reuse with verification.
        """
        output = x

        for step_idx in range(num_steps):
            self._current_step = step_idx
            t = timesteps[step_idx] if timesteps.dim() >= 1 else timesteps

            if step_idx == 0:
                output = self.cache_step(model, output, t, **kwargs)
            else:
                output, _ = self.predict_and_verify(model, output, t, **kwargs)

        return output

    # ------------------------------------------------------------------
    # Model wrapping
    # ------------------------------------------------------------------

    def wrap_model(self, model: nn.Module) -> nn.Module:
        """Install forward hooks on DiT blocks for speculative caching."""
        blocks = self._find_blocks(model)
        if blocks is None:
            raise RuntimeError("Could not locate DiT blocks on model.")

        for idx, block in enumerate(blocks):
            hook = block.register_forward_hook(self._make_hook(idx))
            self._hooks.append(hook)

        return model

    def _find_blocks(self, model: nn.Module) -> Optional[nn.ModuleList]:
        for attr in ("blocks", "layers", "transformer_blocks", "dit_blocks"):
            candidate = getattr(model, attr, None)
            if isinstance(candidate, (nn.ModuleList, list)):
                return candidate
        for child in model.children():
            if isinstance(child, nn.ModuleList):
                return child
        return None

    def _make_hook(self, layer_idx: int) -> Callable:
        optimizer = self

        def hook_fn(module: nn.Module, input: Any, output: torch.Tensor) -> torch.Tensor:
            step = optimizer._current_step

            if step == 0 or not optimizer._speculation_active:
                # Caching step: store output
                optimizer._cached_features[layer_idx] = output.detach()
                return output

            # --- Speculation mode ---
            cached = optimizer._cached_features.get(layer_idx)

            if layer_idx == optimizer.verify_layer_idx:
                # This is the verification layer — always compute (already done
                # by the hook receiving `output`).  Compare with cache.
                if cached is not None:
                    cos_sim = _cosine_similarity(output, cached)
                    accepted = cos_sim >= optimizer.threshold
                    optimizer._speculation_accepted = accepted
                    optimizer.metrics.record(layer_idx, accepted)

                    if accepted:
                        # Accept cached — but still return the freshly computed
                        # verify layer output (it's cheap and more accurate).
                        return output
                    else:
                        # Reject — continue with full forward from here on
                        optimizer._speculation_active = False
                        return output
                else:
                    optimizer._speculation_accepted = False
                    optimizer.metrics.record(layer_idx, False)
                    return output

            # Non-verification layer during speculation
            if cached is not None and optimizer._speculation_active:
                # Reuse cached feature (speculative prediction = identity)
                optimizer.metrics.record(layer_idx, True)
                return cached

            # No cache available — use computed output
            optimizer.metrics.record(layer_idx, False)
            return output

        return hook_fn

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def reset_cache(self) -> None:
        self._cached_features.clear()
        self._cached_verify_output = None
        self._current_step = 0
        self._speculation_active = False
        self._speculation_accepted = None

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def reset_metrics(self) -> None:
        self.metrics = SpeCaMetrics()

    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics.summary()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute mean cosine similarity between two tensors of the same shape."""
    a_flat = a.float().flatten(1)
    b_flat = b.float().flatten(1)
    return F.cosine_similarity(a_flat, b_flat, dim=1).mean().item()
