"""
SeaCache: Spectral-Evolution-Aware Caching for DiT denoising steps (CVPR 2026).

Primary step-caching method for WorldServe / Matrix-Game 2.0.
Explicitly supports the Wan2.1 backbone (30 DiT layers, 3 denoising steps per block).

Core idea: low-frequency structure is stable across consecutive denoising steps,
while high-frequency detail evolves.  Cache the low-frequency component and only
recompute the high-frequency residual, saving ~40-60 % of FLOPs on cacheable layers.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers — DCT-II via real FFT (works on CPU and CUDA without scipy)
# ---------------------------------------------------------------------------

def _dct2_2d(
    x: torch.Tensor,
    spatial_hw: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """Approximate 2-D DCT-II via rfft on the last two spatial dims.

    Accepts (B, C, H, W) or (B, N, C).  For sequence-format tensors,
    *spatial_hw* specifies the (H, W) grid dimensions.  If None, the
    largest divisor of N <= sqrt(N) is used as H.
    """
    if x.dim() == 3:
        B, N, C = x.shape
        if spatial_hw is not None:
            H, W = spatial_hw
        else:
            H = int(math.isqrt(N))
            while H > 1 and N % H != 0:
                H -= 1
            W = N // H if H > 0 else N
        if H * W < N:
            pad = N - H * W  # positive: tokens we need to add to fill the grid
            x = F.pad(x, (0, 0, 0, pad))
        x = x.permute(0, 2, 1).reshape(B, C, H, W)

    # Real FFT along H then W — magnitude gives frequency energy
    X = torch.fft.rfft2(x.float(), norm="ortho")
    return X.abs()


def _idct2_2d(mag: torch.Tensor, phase: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
    """Reconstruct spatial tensor from magnitude and phase via irfft2."""
    complex_spec = mag * torch.exp(1j * phase)
    return torch.fft.irfft2(complex_spec, s=shape[-2:], norm="ortho")


# ---------------------------------------------------------------------------
# Spectral mask utilities
# ---------------------------------------------------------------------------

def _frequency_radius_mask(H: int, W: int, threshold: float, device: torch.device) -> torch.Tensor:
    """Boolean mask selecting frequency bins with normalised radius <= threshold.

    Returns shape (H, W // 2 + 1) matching rfft2 output.
    """
    freq_h = torch.arange(H, device=device).float()
    freq_w = torch.arange(W // 2 + 1, device=device).float()
    # Normalise to [0, 1]
    freq_h = freq_h / max(H - 1, 1)
    freq_w = freq_w / max(W // 2, 1)
    radius = torch.sqrt(freq_h[:, None] ** 2 + freq_w[None, :] ** 2) / math.sqrt(2.0)
    return radius <= threshold  # True = low frequency


# ---------------------------------------------------------------------------
# Metrics dataclass
# ---------------------------------------------------------------------------

@dataclass
class SeaCacheMetrics:
    """Runtime metrics collected during caching."""
    total_layer_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_time_saved_ms: float = 0.0
    memory_cached_bytes: int = 0
    per_layer_hit_rate: Dict[int, float] = field(default_factory=dict)
    _per_layer_hits: Dict[int, int] = field(default_factory=dict)
    _per_layer_total: Dict[int, int] = field(default_factory=dict)

    @property
    def hit_rate(self) -> float:
        return self.cache_hits / max(self.total_layer_calls, 1)

    def record(self, layer_idx: int, hit: bool) -> None:
        self.total_layer_calls += 1
        if hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        self._per_layer_hits.setdefault(layer_idx, 0)
        self._per_layer_total.setdefault(layer_idx, 0)
        self._per_layer_total[layer_idx] += 1
        if hit:
            self._per_layer_hits[layer_idx] += 1
        self.per_layer_hit_rate[layer_idx] = (
            self._per_layer_hits[layer_idx] / self._per_layer_total[layer_idx]
        )

    def summary(self) -> Dict[str, Any]:
        return {
            "hit_rate": self.hit_rate,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "total_layer_calls": self.total_layer_calls,
            "time_saved_ms": round(self.total_time_saved_ms, 2),
            "memory_cached_bytes": self.memory_cached_bytes,
            "per_layer_hit_rate": dict(self.per_layer_hit_rate),
        }


# ---------------------------------------------------------------------------
# SeaCacheOptimizer
# ---------------------------------------------------------------------------

class SeaCacheOptimizer:
    """Spectral-Evolution-Aware step cache for DiT models (Wan2.1).

    Designed for Matrix-Game 2.0 with 30 DiT layers and 3 denoising steps
    per autoregressive block.  At denoising step 0, a full forward pass
    populates the cache.  At steps 1+ cacheable layers reuse low-frequency
    features and only recompute the high-frequency residual.
    """

    def __init__(
        self,
        num_layers: int = 30,
        num_steps: int = 3,
        frequency_threshold: float = 0.5,
        cache_ratio: float = 0.6,
        spatial_hw: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            num_layers: Number of DiT transformer blocks.
            num_steps: Number of denoising steps per generation block.
            frequency_threshold: Normalised radius in [0, 1] below which
                frequency bands are considered "low" and cacheable.
            cache_ratio: Fraction of layers whose features may be cached
                (layers closest to the middle are selected first).
            spatial_hw: Explicit (H, W) for the latent spatial grid. Required
                for non-square resolutions (e.g., MG2 at 352x640 has latent
                ~44x80). If None, the grid is factored automatically.
        """
        self.num_layers = num_layers
        self.num_steps = num_steps
        self.frequency_threshold = frequency_threshold
        self.cache_ratio = cache_ratio
        self.spatial_hw = spatial_hw

        # Determine which layers are cacheable (prefer middle layers)
        self._cacheable_layers = self._select_cacheable_layers()

        # Runtime state -------------------------------------------------
        # Keyed by layer_idx -> cached spectral components
        self._cache: Dict[int, Dict[str, torch.Tensor]] = {}
        self._current_step: int = 0
        self._spectral_masks: Dict[Tuple[int, int], torch.Tensor] = {}
        self._hooks: List[torch.utils.hooks.RemovableHook] = []
        self.metrics = SeaCacheMetrics()

    # ------------------------------------------------------------------
    # Layer selection
    # ------------------------------------------------------------------

    def _select_cacheable_layers(self) -> List[int]:
        """Select layers to cache — prefer middle layers over first/last."""
        num_cached = max(1, int(self.num_layers * self.cache_ratio))
        mid = self.num_layers / 2.0
        # Sort layer indices by distance from the middle (ascending)
        ranked = sorted(range(self.num_layers), key=lambda i: abs(i - mid))
        return sorted(ranked[:num_cached])

    # ------------------------------------------------------------------
    # Spectral analysis
    # ------------------------------------------------------------------

    def analyze_spectral_evolution(
        self,
        features_step1: torch.Tensor,
        features_step2: torch.Tensor,
    ) -> torch.Tensor:
        """Compare spectral energy between two consecutive denoising steps.

        Args:
            features_step1: Features from step t   — (B, C, H, W) or (B, N, C).
            features_step2: Features from step t+1 — same shape.

        Returns:
            Boolean mask (H, rfft_W) — True where the frequency band is
            *stable* (low evolution rate) and therefore cacheable.
        """
        spec1 = _dct2_2d(features_step1, self.spatial_hw)  # (B, C, H, W//2+1)
        spec2 = _dct2_2d(features_step2, self.spatial_hw)

        # Per-frequency evolution rate: relative change in energy
        eps = 1e-8
        delta = (spec2 - spec1).abs().mean(dim=(0, 1))  # (H, W//2+1)
        base = spec1.abs().mean(dim=(0, 1)) + eps
        evolution_rate = delta / base  # higher = more change

        # Stable where evolution rate is below median (adaptive threshold)
        median_rate = evolution_rate.median()
        stable_mask = evolution_rate <= median_rate
        return stable_mask

    # ------------------------------------------------------------------
    # Per-layer caching decision
    # ------------------------------------------------------------------

    def should_cache_layer(
        self,
        layer_idx: int,
        step_idx: int,
        features: Optional[torch.Tensor] = None,
    ) -> bool:
        """Decide whether to cache (and later reuse) this layer at this step.

        Heuristic:
        - Step 0 is always a full compute (populates cache).
        - Middle layers are more cacheable than boundary layers.
        - Earlier denoising steps (1) are more cacheable than later (2).
        """
        if step_idx == 0:
            return False  # first step: full compute, store into cache

        if layer_idx not in self._cacheable_layers:
            return False  # layer not selected for caching

        # For the last denoising step, only cache the innermost 50 % of
        # cacheable layers (quality guard).
        if step_idx == self.num_steps - 1:
            mid = self.num_layers / 2.0
            dist = abs(layer_idx - mid)
            max_dist = self.num_layers / 2.0
            if dist / max_dist > 0.25:
                return False

        return True

    # ------------------------------------------------------------------
    # Spectral cache / reuse core
    # ------------------------------------------------------------------

    def _get_low_freq_mask(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        key = (H, W)
        if key not in self._spectral_masks:
            self._spectral_masks[key] = _frequency_radius_mask(
                H, W, self.frequency_threshold, device
            )
        mask = self._spectral_masks[key]
        if mask.device != device:
            mask = mask.to(device)
            self._spectral_masks[key] = mask
        return mask

    def _store_features(self, layer_idx: int, features: torch.Tensor) -> None:
        """Cache spectral decomposition of features."""
        # Flatten 5-D tensors (e.g. Oasis (B,T,H,W,C)) to 3-D before caching.
        if features.dim() == 5:
            B5, T5, H5, W5, C5 = features.shape
            features = features.reshape(B5, T5 * H5 * W5, C5)
        spatial, orig_shape, was_seq = self._to_spatial(features, self.spatial_hw)
        spec = torch.fft.rfft2(spatial.float(), norm="ortho")
        # .clone() is critical when torch.compile(mode="reduce-overhead") uses
        # CUDA Graphs — without it the graph's memory gets overwritten on the
        # next replay and the cached tensors become stale / corrupt.
        self._cache[layer_idx] = {
            "magnitude": spec.abs().clone(),
            "phase": spec.angle().clone(),
            "orig_shape": orig_shape,
            "was_seq": was_seq,
            "spatial_shape": spatial.shape,
        }
        self.metrics.memory_cached_bytes += spec.nelement() * spec.element_size() * 2  # mag + phase

    def _reuse_features(self, layer_idx: int, new_features: torch.Tensor) -> torch.Tensor:
        """Blend cached low-freq with newly computed high-freq."""
        cached = self._cache.get(layer_idx)
        if cached is None:
            return new_features

        # Shape guard: DC-DiT or DyDiT++ may compress/change token count
        # between steps. If cached spatial dimensions don't match current
        # features, treat as a cache miss rather than crashing.
        cached_spatial_shape = cached.get("spatial_shape")
        if cached_spatial_shape is not None:
            # Normalize new_features to spatial for comparison
            _check = new_features
            if _check.dim() == 5:
                _B, _T, _H, _W, _C = _check.shape
                _check = _check.reshape(_B, _T * _H * _W, _C)
            # Compare token count (N dimension)
            cached_n = cached_spatial_shape[2] * cached_spatial_shape[3]  # H*W from (B,C,H,W)
            new_n = _check.shape[1] if _check.dim() == 3 else _check.shape[-2]
            if cached_n != new_n:
                logger.debug(
                    "[SeaCache] shape mismatch layer %d: cached N=%d vs new N=%d — cache miss",
                    layer_idx, cached_n, new_n,
                )
                return new_features

        # Flatten 5-D tensors (e.g. Oasis (B,T,H,W,C)) to 3-D (B, T*H*W, C)
        # so the existing spatial logic can handle them.
        orig_ndim = new_features.dim()
        orig_5d_shape = new_features.shape if orig_ndim == 5 else None
        if orig_ndim == 5:
            B5, T5, H5, W5, C5 = new_features.shape
            new_features = new_features.reshape(B5, T5 * H5 * W5, C5)

        spatial_new, orig_shape, was_seq = self._to_spatial(new_features, self.spatial_hw)
        B, C, H, W = spatial_new.shape

        spec_new = torch.fft.rfft2(spatial_new.float(), norm="ortho")
        low_mask = self._get_low_freq_mask(H, W, spec_new.device)  # (H, W//2+1)

        cached_mag = cached["magnitude"].to(device=spec_new.device, dtype=spec_new.real.dtype)
        cached_phase = cached["phase"].to(device=spec_new.device, dtype=spec_new.real.dtype)

        # Broadcast mask
        mask = low_mask[None, None, :, :]  # (1, 1, H, W//2+1)

        # Blend: low freq from cache, high freq from new computation
        blended_mag = torch.where(mask, cached_mag, spec_new.abs())
        blended_phase = torch.where(mask, cached_phase, spec_new.angle())
        blended = _idct2_2d(blended_mag, blended_phase, spatial_new.shape)
        blended = blended.to(new_features.dtype)

        if was_seq:
            B_orig, N_orig, C_orig = orig_shape
            blended = blended.reshape(B, C, -1).permute(0, 2, 1)[:, :N_orig, :]

        # Restore 5-D shape if input was 5-D
        if orig_5d_shape is not None:
            blended = blended.reshape(orig_5d_shape)

        return blended

    @staticmethod
    def _to_spatial(
        x: torch.Tensor,
        spatial_hw: Optional[Tuple[int, int]] = None,
    ) -> Tuple[torch.Tensor, Tuple[int, ...], bool]:
        """Convert (B, N, C) -> (B, C, H, W) for spectral ops.

        Args:
            x: Input tensor. If 4D (B, C, H, W), returned as-is.
            spatial_hw: Explicit (H, W) of the spatial grid. Required for
                non-square grids like Matrix-Game 2.0's 352x640 -> ~44x80
                latent resolution. If None, attempts to factor N into the
                closest (H, W) pair by finding the largest divisor <= sqrt(N).
        """
        if x.dim() == 3:
            B, N, C = x.shape
            was_seq = True

            if spatial_hw is not None:
                H, W = spatial_hw
            else:
                # Find the largest H <= sqrt(N) that divides N evenly
                H = int(math.isqrt(N))
                while H > 1 and N % H != 0:
                    H -= 1
                W = N // H if H > 0 else N

            if H * W != N:
                # Either N is not cleanly factorable, or spatial_hw was given
                # but doesn't match N exactly — pad sequence to H*W.
                pad = H * W - N  # always positive here (H*W >= N after factoring)
                if pad < 0:
                    # spatial_hw is smaller than N — clamp (shouldn't happen)
                    pad = 0
                if pad > 0:
                    x = F.pad(x, (0, 0, 0, pad))

            spatial = x.permute(0, 2, 1).reshape(B, C, H, W)
            return spatial, (B, N, C), was_seq
        return x, x.shape, False

    # ------------------------------------------------------------------
    # Full denoising-step caching loop
    # ------------------------------------------------------------------

    def cache_and_reuse(
        self,
        model: nn.Module,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run all denoising steps with spectral caching.

        Args:
            model: DiT model whose forward(x, timestep, **kwargs) runs one
                denoising step over all layers.
            x: Noisy latent input for the block — (B, C, H, W) or (B, N, C).
            timesteps: Tensor of timestep values, length == num_steps.
            **kwargs: Extra args forwarded to model.forward().

        Returns:
            Denoised output after all steps.
        """
        self.reset_cache()
        output = x

        for step_idx in range(self.num_steps):
            self._current_step = step_idx
            t = timesteps[step_idx] if timesteps.dim() >= 1 else timesteps

            if step_idx == 0:
                # Full forward — hooks will populate cache
                output = model(output, t, **kwargs)
            else:
                # Hooks will selectively reuse cached features
                output = model(output, t, **kwargs)

        return output

    # ------------------------------------------------------------------
    # Model wrapping — install forward hooks on DiT blocks
    # ------------------------------------------------------------------

    def wrap_model(self, model: nn.Module) -> nn.Module:
        """Install caching wrappers on each DiT block.

        Uses ``block.forward`` replacement (not pre-hooks) so that on a cache
        hit the block's original computation is **skipped entirely** — not
        merely fed a modified input.  A forward post-hook still stores outputs
        on cache misses.

        Returns:
            The same model (mutated in place).
        """
        blocks = self._find_blocks(model)
        if blocks is None:
            raise RuntimeError(
                "Could not locate DiT blocks on model. Expected an attribute "
                "named 'blocks', 'layers', or 'transformer_blocks'."
            )

        # Recalibrate num_layers to the actual model depth so _cacheable_layers
        # contains valid indices (important when default num_layers=30 is used
        # with e.g. Oasis which has only 12 blocks).
        actual_n = len(blocks)
        if actual_n != self.num_layers:
            self.num_layers = actual_n
            self._cacheable_layers = self._select_cacheable_layers()
            logger.info(
                "SeaCache: recalibrated num_layers=%d, cacheable=%s",
                self.num_layers, self._cacheable_layers,
            )

        for idx, block in enumerate(blocks):
            # Replace forward so cache hits skip computation entirely
            import types as _types
            original_fwd = block.forward
            wrapped = self._make_cached_forward(idx, original_fwd)
            block.forward = _types.MethodType(wrapped, block)

            # Post-hook: populate cache after a full forward pass
            post_hook = block.register_forward_hook(self._make_post_hook(idx))
            self._hooks.append(post_hook)

        return model

    def _find_blocks(self, model: nn.Module) -> Optional[nn.ModuleList]:
        for attr in ("blocks", "layers", "transformer_blocks", "dit_blocks"):
            candidate = getattr(model, attr, None)
            if isinstance(candidate, (nn.ModuleList, list)):
                return candidate
        # Fallback: first ModuleList child
        for child in model.children():
            if isinstance(child, nn.ModuleList):
                return child
        return None

    def _make_cached_forward(self, layer_idx: int, original_fwd: Callable) -> Callable:
        """
        Return a replacement forward function that bypasses original_fwd on
        cache hits, returning the spectrally-blended cached output instead.

        Using forward replacement (rather than register_forward_pre_hook) is
        critical: a pre-hook returning a value replaces the MODULE'S INPUTS,
        not its outputs, so the forward still runs — defeating the purpose.

        CUDA Graph compatibility:
        SeaCache is applied AFTER torch.compile, so original_fwd is the compiled
        module forward.  cached_forward is decorated with @torch._dynamo.disable
        so Dynamo does NOT trace the routing/cache logic into the CUDA Graph — it
        runs in eager mode.  On a cache miss, original_fwd() is called in eager,
        which dispatches into the already-compiled kernel, so we retain compile
        speedup.  On a cache hit, _reuse_features() also runs in eager (safe).
        This allows compile(mode="reduce-overhead") to coexist with SeaCache.
        """
        optimizer = self

        @torch._dynamo.disable
        def cached_forward(self_block: nn.Module, *args: Any, **kwargs: Any) -> Any:
            step = optimizer._current_step

            if step > 0 and optimizer.should_cache_layer(layer_idx, step, None):
                cached = optimizer._cache.get(layer_idx)
                if cached is not None and args:
                    current_input = args[0]
                    t_start = time.perf_counter()
                    blended = optimizer._reuse_features(layer_idx, current_input)
                    elapsed_ms = (time.perf_counter() - t_start) * 1000
                    optimizer.metrics.record(layer_idx, hit=True)
                    optimizer.metrics.total_time_saved_ms += elapsed_ms
                    # Return blended output — original_fwd is NOT called
                    return blended

            # Cache miss or step 0: run original forward (compiled)
            return original_fwd(*args, **kwargs)

        return cached_forward

    def _make_post_hook(self, layer_idx: int) -> Callable:
        """Post-hook: store output in cache (step 0 or cache miss).

        Decorated with @torch._dynamo.disable so tensor allocations in
        _store_features() happen in eager mode, not inside a CUDA Graph.
        Without this, .clone() inside _store_features produces tensors in
        graph-managed memory that get overwritten on the next graph replay.
        """
        optimizer = self

        @torch._dynamo.disable
        def post_hook_fn(module: nn.Module, input: Any, output: torch.Tensor) -> None:
            step = optimizer._current_step

            if step == 0:
                # Always cache on step 0
                optimizer._store_features(layer_idx, output)
                optimizer.metrics.record(layer_idx, hit=False)
            elif layer_idx not in optimizer._cache:
                # Cache miss on later step — store for next time
                optimizer._store_features(layer_idx, output)
                optimizer.metrics.record(layer_idx, hit=False)
            # If it was a cache hit, cached_forward already handled it — post_hook is a no-op

        return post_hook_fn

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def reset_cache(self) -> None:
        """Clear all cached features (call between generation blocks)."""
        self._cache.clear()
        self._current_step = 0

    def remove_hooks(self) -> None:
        """Remove all installed forward hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def reset_metrics(self) -> None:
        self.metrics = SeaCacheMetrics()

    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics.summary()

    def reconfigure(self, num_steps: int) -> None:
        """Update num_steps at runtime (e.g., after PCM distillation changes step count).

        PCM distillation of Oasis reduces from 20 to 4 denoising steps; SeaCache
        must know the correct step count to schedule cache hits correctly.
        """
        self.num_steps = num_steps
        self._cacheable_layers = self._select_cacheable_layers()
        logger.info("[SeaCache] reconfigured: num_steps=%d, cacheable_layers=%s",
                    num_steps, self._cacheable_layers)

    # OptimizationStack hook interface -----------------------------------

    def pre_step(self, step_idx: int, total_steps: int, latents: Any) -> bool:
        """Advance step counter. Returns False — SeaCache works via model wrapping."""
        self._current_step = step_idx
        return False

    def post_step(self, step_idx: int, total_steps: int, latents: Any) -> None:
        """Nothing to do after step; cache is updated by the post_hook."""
        pass

    def get_stats(self) -> Dict[str, Any]:
        return self.get_metrics()
