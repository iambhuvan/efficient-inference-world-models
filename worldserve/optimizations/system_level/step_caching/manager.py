"""
Step Caching Manager for WorldServe.

Coordinates the selection and application of denoising-step caching methods
(SeaCache, FlowCache, SpeCa) to DiT models in Matrix-Game 2.0.
Provides a unified interface for optimizing models and running ablation
benchmarks across methods.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from worldserve.optimizations.system_level.step_caching.seacache import SeaCacheOptimizer
from worldserve.optimizations.system_level.step_caching.flowcache import FlowCacheOptimizer
from worldserve.optimizations.system_level.step_caching.speca import SpeCaOptimizer


# Default configs per method
_DEFAULT_CONFIGS: Dict[str, Dict[str, Any]] = {
    "seacache": {
        "num_layers": 30,
        "num_steps": 3,
        "frequency_threshold": 0.5,
        "cache_ratio": 0.6,
    },
    "flowcache": {
        "chunk_size": 3,
        "importance_threshold": 0.3,
        "redundancy_threshold": 0.85,
    },
    "speca": {
        "num_layers": 30,
        "verify_layer_idx": None,  # defaults to last layer
        "threshold": 0.95,
    },
}


class StepCachingManager:
    """Unified manager for denoising-step caching optimizations.

    Usage::

        manager = StepCachingManager(method="seacache")
        model = manager.optimize(model)
        # ... run denoising steps via manager.run(model, x, timesteps)
        print(manager.get_metrics())
    """

    def __init__(
        self,
        method: str = "seacache",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Args:
            method: Primary caching method — "seacache", "flowcache", or "speca".
            config: Override default configuration for the chosen method.
        """
        if method not in _DEFAULT_CONFIGS:
            raise ValueError(
                f"Unknown step caching method '{method}'. "
                f"Choose from: {list(_DEFAULT_CONFIGS.keys())}"
            )
        self.method = method
        self.config = {**_DEFAULT_CONFIGS[method], **(config or {})}
        self._optimizer: Optional[Any] = None
        self._all_optimizers: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Optimizer factory
    # ------------------------------------------------------------------

    def _create_optimizer(self, method: Optional[str] = None) -> Any:
        """Instantiate an optimizer for the given method."""
        method = method or self.method
        cfg = {**_DEFAULT_CONFIGS.get(method, {}), **(self.config if method == self.method else {})}

        if method == "seacache":
            return SeaCacheOptimizer(
                num_layers=cfg["num_layers"],
                num_steps=cfg["num_steps"],
                frequency_threshold=cfg["frequency_threshold"],
                cache_ratio=cfg["cache_ratio"],
            )
        elif method == "flowcache":
            return FlowCacheOptimizer(
                chunk_size=cfg["chunk_size"],
                importance_threshold=cfg["importance_threshold"],
                redundancy_threshold=cfg["redundancy_threshold"],
            )
        elif method == "speca":
            return SpeCaOptimizer(
                num_layers=cfg["num_layers"],
                verify_layer_idx=cfg.get("verify_layer_idx"),
                threshold=cfg["threshold"],
            )
        else:
            raise ValueError(f"Unknown method: {method}")

    # ------------------------------------------------------------------
    # Apply optimization to model
    # ------------------------------------------------------------------

    def optimize(self, model: nn.Module, method: Optional[str] = None) -> nn.Module:
        """Apply step caching to *model* and return the wrapped model.

        Args:
            model: DiT model to optimize (Wan2.1 backbone).
            method: Override the primary method for this call.

        Returns:
            The model with caching hooks installed.
        """
        method = method or self.method
        optimizer = self._create_optimizer(method)

        # FlowCache has a compatibility gate
        if method == "flowcache":
            report = optimizer.check_compatibility(model)
            if not report.compatible:
                print(
                    f"[StepCachingManager] FlowCache compatibility check FAILED:\n{report}\n"
                    "Falling back to SeaCache."
                )
                method = "seacache"
                optimizer = self._create_optimizer("seacache")

        optimizer.wrap_model(model)
        self._optimizer = optimizer
        self._all_optimizers[method] = optimizer
        return model

    # ------------------------------------------------------------------
    # Run denoising steps through the active optimizer
    # ------------------------------------------------------------------

    def run(
        self,
        model: nn.Module,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Execute denoising steps with caching via the active optimizer.

        Args:
            model: The wrapped DiT model.
            x: Noisy latent input.
            timesteps: Timestep schedule tensor.
            **kwargs: Forwarded to model.

        Returns:
            Denoised output.
        """
        if self._optimizer is None:
            raise RuntimeError("No optimizer active. Call optimize() first.")

        return self._optimizer.cache_and_reuse(model, x, timesteps, **kwargs)

    # ------------------------------------------------------------------
    # Benchmarking / ablation
    # ------------------------------------------------------------------

    def benchmark(
        self,
        model: nn.Module,
        test_input: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
        methods: Optional[List[str]] = None,
        num_warmup: int = 2,
        num_runs: int = 5,
        **kwargs: Any,
    ) -> Dict[str, Dict[str, Any]]:
        """Compare step caching methods on the same model and input.

        Runs each method and collects timing + quality metrics.  The model
        is restored to its original state (hooks removed) between methods.

        Args:
            model: DiT model (unhhooked).
            test_input: (B, C, H, W) or (B, N, C) latent tensor.
            timesteps: Timestep schedule.  Defaults to linspace(1, 0, 3).
            methods: List of methods to benchmark (default: seacache, speca).
            num_warmup: Warmup iterations (not timed).
            num_runs: Timed iterations.
            **kwargs: Forwarded to model.

        Returns:
            Dict mapping method name to metrics dict.
        """
        methods = methods or ["seacache", "speca"]
        if timesteps is None:
            num_steps = self.config.get("num_steps", 3)
            timesteps = torch.linspace(1.0, 0.0, num_steps, device=test_input.device)

        results: Dict[str, Dict[str, Any]] = {}

        # Baseline: no caching
        baseline_times: List[float] = []
        with torch.no_grad():
            for i in range(num_warmup + num_runs):
                x = test_input.clone()
                t0 = time.perf_counter()
                for step_idx in range(len(timesteps)):
                    t = timesteps[step_idx]
                    x = model(x, t, **kwargs)
                elapsed = (time.perf_counter() - t0) * 1000
                if i >= num_warmup:
                    baseline_times.append(elapsed)

        baseline_avg = sum(baseline_times) / max(len(baseline_times), 1)
        results["baseline"] = {"avg_time_ms": round(baseline_avg, 2)}

        # Each caching method
        for method_name in methods:
            if method_name not in _DEFAULT_CONFIGS:
                results[method_name] = {"error": f"Unknown method: {method_name}"}
                continue

            optimizer = self._create_optimizer(method_name)

            # Compatibility gate for FlowCache
            if method_name == "flowcache":
                report = optimizer.check_compatibility(model)
                if not report.compatible:
                    results[method_name] = {
                        "error": "Compatibility check failed",
                        "report": str(report),
                    }
                    continue

            optimizer.wrap_model(model)
            cached_times: List[float] = []

            with torch.no_grad():
                for i in range(num_warmup + num_runs):
                    optimizer.reset_cache()
                    if hasattr(optimizer, "reset_metrics"):
                        optimizer.reset_metrics()

                    x = test_input.clone()
                    t0 = time.perf_counter()
                    x = optimizer.cache_and_reuse(model, x, timesteps, **kwargs)
                    elapsed = (time.perf_counter() - t0) * 1000
                    if i >= num_warmup:
                        cached_times.append(elapsed)

            avg_time = sum(cached_times) / max(len(cached_times), 1)
            speedup = baseline_avg / max(avg_time, 1e-6)

            method_metrics = optimizer.get_metrics()
            method_metrics["avg_time_ms"] = round(avg_time, 2)
            method_metrics["baseline_avg_ms"] = round(baseline_avg, 2)
            method_metrics["speedup"] = round(speedup, 3)

            results[method_name] = method_metrics

            # Clean up hooks
            optimizer.remove_hooks()

        return results

    # ------------------------------------------------------------------
    # Metrics aggregation
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        """Return metrics from the active optimizer."""
        if self._optimizer is None:
            return {}
        return self._optimizer.get_metrics()

    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Return metrics from all optimizers that have been used."""
        return {name: opt.get_metrics() for name, opt in self._all_optimizers.items()}

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def remove_hooks(self) -> None:
        """Remove hooks from all active optimizers."""
        for opt in self._all_optimizers.values():
            opt.remove_hooks()
        if self._optimizer is not None:
            self._optimizer.remove_hooks()

    def reset(self) -> None:
        """Full reset: remove hooks, clear caches and metrics."""
        self.remove_hooks()
        for opt in self._all_optimizers.values():
            opt.reset_cache()
            if hasattr(opt, "reset_metrics"):
                opt.reset_metrics()
        self._all_optimizers.clear()
        self._optimizer = None
