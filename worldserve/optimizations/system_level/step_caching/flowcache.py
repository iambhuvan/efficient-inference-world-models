"""
FlowCache compatibility study for WorldServe / Matrix-Game 2.0.

FlowCache uses chunk-wise caching with joint importance-redundancy optimization,
originally designed for autoregressive video generation.  This module evaluates
whether MG2's generation structure (3 denoising steps per AR block, Wan2.1
backbone, KV cache) is compatible with FlowCache's assumptions.

This is a *compatibility study* — full production integration is gated on the
result of ``check_compatibility()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Compatibility report
# ---------------------------------------------------------------------------

@dataclass
class CompatibilityReport:
    """Result of checking a model against FlowCache assumptions."""
    compatible: bool = False
    has_block_generation: bool = False
    has_ar_structure: bool = False
    has_kv_cache: bool = False
    details: Dict[str, str] = field(default_factory=dict)

    def __repr__(self) -> str:
        status = "COMPATIBLE" if self.compatible else "INCOMPATIBLE"
        lines = [f"FlowCache Compatibility: {status}"]
        lines.append(f"  block generation : {self.has_block_generation}")
        lines.append(f"  AR structure     : {self.has_ar_structure}")
        lines.append(f"  KV cache         : {self.has_kv_cache}")
        for k, v in self.details.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class FlowCacheMetrics:
    total_layer_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    per_layer_importance: Dict[int, float] = field(default_factory=dict)
    per_layer_redundancy: Dict[int, float] = field(default_factory=dict)

    @property
    def hit_rate(self) -> float:
        return self.cache_hits / max(self.total_layer_calls, 1)

    def summary(self) -> Dict[str, Any]:
        return {
            "hit_rate": self.hit_rate,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "total_layer_calls": self.total_layer_calls,
            "per_layer_importance": dict(self.per_layer_importance),
            "per_layer_redundancy": dict(self.per_layer_redundancy),
        }


# ---------------------------------------------------------------------------
# FlowCacheOptimizer
# ---------------------------------------------------------------------------

class FlowCacheOptimizer:
    """Chunk-wise denoising-step cache using importance-redundancy gating.

    FlowCache caches DiT features across denoising steps where:
    - **redundancy** between consecutive steps is high (features barely change), AND
    - **importance** of the layer is low (skipping it has minimal quality impact).

    Layers with *high importance* or *low redundancy* are always recomputed.
    """

    def __init__(
        self,
        chunk_size: int = 3,
        importance_threshold: float = 0.3,
        redundancy_threshold: float = 0.85,
    ) -> None:
        """
        Args:
            chunk_size: Number of denoising steps per AR chunk (MG2 default: 3).
            importance_threshold: Layers with importance below this are eligible
                for caching (normalised to [0, 1]).
            redundancy_threshold: Minimum cosine similarity between consecutive
                step features to consider them redundant (cacheable).
        """
        self.chunk_size = chunk_size
        self.importance_threshold = importance_threshold
        self.redundancy_threshold = redundancy_threshold

        # Runtime state
        self._cache: Dict[int, torch.Tensor] = {}
        self._prev_features: Dict[int, torch.Tensor] = {}
        self._current_step: int = 0
        self._hooks: List[torch.utils.hooks.RemovableHook] = []
        self._compatibility: Optional[CompatibilityReport] = None
        self.metrics = FlowCacheMetrics()

    # ------------------------------------------------------------------
    # Compatibility check
    # ------------------------------------------------------------------

    def check_compatibility(self, model: nn.Module) -> CompatibilityReport:
        """Analyze whether *model* matches FlowCache's chunk-wise AR assumptions.

        Checks performed:
        1. Does the model generate in blocks / chunks?
        2. Does the model have an AR structure with KV cache?

        For Matrix-Game 2.0 with Wan2.1 backbone both should be True.
        """
        report = CompatibilityReport()

        # --- Check 1: block/chunk generation ---
        # MG2 generates in blocks of frames with multiple denoising steps.
        # We look for attributes or config keys indicating chunked generation.
        block_attrs = ("chunk_size", "block_size", "num_steps_per_block",
                       "denoising_steps", "num_denoising_steps")
        for attr in block_attrs:
            if hasattr(model, attr):
                report.has_block_generation = True
                report.details["block_attr"] = attr
                break

        # Heuristic: if model has a ModuleList of transformer blocks it is
        # likely a DiT that processes a block of latents at a time.
        for name, child in model.named_children():
            if isinstance(child, nn.ModuleList) and len(child) > 5:
                report.has_block_generation = True
                report.details.setdefault("block_attr", f"ModuleList({name}, len={len(child)})")
                break

        # --- Check 2: AR structure with KV cache ---
        kv_attrs = ("kv_cache", "past_key_values", "cache", "use_kv_cache",
                     "enable_kv_cache", "kv_cache_enabled")
        for attr in kv_attrs:
            val = getattr(model, attr, None)
            if val is not None:
                report.has_ar_structure = True
                report.has_kv_cache = True
                report.details["kv_attr"] = attr
                break

        # Fallback: scan config
        config = getattr(model, "config", None)
        if config is not None and not report.has_kv_cache:
            for attr in kv_attrs:
                if hasattr(config, attr) or (isinstance(config, dict) and attr in config):
                    report.has_ar_structure = True
                    report.has_kv_cache = True
                    report.details["kv_attr"] = f"config.{attr}"
                    break

        # AR structure can also be inferred from causal masking
        if not report.has_ar_structure:
            for name, mod in model.named_modules():
                cls_name = type(mod).__name__.lower()
                if "causal" in cls_name or "autoregressive" in cls_name:
                    report.has_ar_structure = True
                    report.details["ar_module"] = name
                    break

        report.compatible = report.has_block_generation and report.has_ar_structure
        self._compatibility = report
        return report

    # ------------------------------------------------------------------
    # Importance & redundancy scoring
    # ------------------------------------------------------------------

    @staticmethod
    def compute_importance(features: torch.Tensor) -> float:
        """Per-layer feature importance via L2 norm (normalised).

        Higher values mean the layer contributes more to the output.
        Features: (B, C, H, W) or (B, N, C).
        """
        # Use mean L2 norm across spatial/sequence dims per sample, then average
        if features.dim() == 4:
            # (B, C, H, W) -> per-sample L2 over C, H, W
            norms = features.float().flatten(1).norm(dim=1)
        else:
            # (B, N, C) -> per-sample L2 over N, C
            norms = features.float().flatten(1).norm(dim=1)
        # Normalise to [0, 1] range via sigmoid of log-norm
        mean_norm = norms.mean().item()
        # Empirical scaling: log-norm in [4, 12] maps to importance [0, 1]
        importance = 1.0 / (1.0 + abs(-0.5 * (mean_norm - 8.0)))
        return max(0.0, min(1.0, importance))

    @staticmethod
    def compute_redundancy(
        features_prev: torch.Tensor,
        features_curr: torch.Tensor,
    ) -> float:
        """Cross-step feature similarity (cosine similarity).

        Values close to 1.0 mean the features barely changed between steps.
        """
        f1 = features_prev.float().flatten(1)
        f2 = features_curr.float().flatten(1)
        cos = F.cosine_similarity(f1, f2, dim=1).mean().item()
        return cos

    # ------------------------------------------------------------------
    # Caching decision
    # ------------------------------------------------------------------

    def _should_cache(self, layer_idx: int, features: torch.Tensor) -> bool:
        """Decide whether to reuse cache for this layer at the current step."""
        if self._current_step == 0:
            return False  # first step always full compute

        importance = self.compute_importance(features)
        self.metrics.per_layer_importance[layer_idx] = importance

        prev = self._prev_features.get(layer_idx)
        if prev is None:
            return False

        redundancy = self.compute_redundancy(prev, features)
        self.metrics.per_layer_redundancy[layer_idx] = redundancy

        # Cache when redundancy is high AND importance is low
        return redundancy >= self.redundancy_threshold and importance <= self.importance_threshold

    # ------------------------------------------------------------------
    # cache_and_reuse — full denoising loop
    # ------------------------------------------------------------------

    def cache_and_reuse(
        self,
        model: nn.Module,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run chunk denoising steps with importance-redundancy caching.

        Args:
            model: DiT model.
            x: Noisy input latent.
            timesteps: Tensor of timestep values per step.
            **kwargs: Forwarded to model.

        Returns:
            Denoised output.
        """
        if self._compatibility is not None and not self._compatibility.compatible:
            # Fall back to uncached forward
            output = x
            for step_idx in range(self.chunk_size):
                t = timesteps[step_idx] if timesteps.dim() >= 1 else timesteps
                output = model(output, t, **kwargs)
            return output

        self.reset_cache()
        output = x

        for step_idx in range(self.chunk_size):
            self._current_step = step_idx
            t = timesteps[step_idx] if timesteps.dim() >= 1 else timesteps
            output = model(output, t, **kwargs)

        return output

    # ------------------------------------------------------------------
    # Hook-based integration
    # ------------------------------------------------------------------

    def wrap_model(self, model: nn.Module) -> nn.Module:
        """Install forward hooks on DiT blocks for caching."""
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
            optimizer.metrics.total_layer_calls += 1

            if step == 0:
                # First step: store features, full compute
                optimizer._prev_features[layer_idx] = output.detach()
                optimizer.metrics.cache_misses += 1
                return output

            if optimizer._should_cache(layer_idx, output):
                # Reuse cached features
                cached = optimizer._cache.get(layer_idx)
                if cached is not None:
                    optimizer.metrics.cache_hits += 1
                    return cached

            # Full recompute — update stores
            optimizer._cache[layer_idx] = output.detach()
            optimizer._prev_features[layer_idx] = output.detach()
            optimizer.metrics.cache_misses += 1
            return output

        return hook_fn

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def reset_cache(self) -> None:
        self._cache.clear()
        self._prev_features.clear()
        self._current_step = 0

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def reset_metrics(self) -> None:
        self.metrics = FlowCacheMetrics()

    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics.summary()
