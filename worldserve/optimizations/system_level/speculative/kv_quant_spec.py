"""
KV cache quantization self-speculative decoding for WorldServe (Phase 5 — PRIMARY).

Uses INT4-quantized KV cache as a fast draft model and FP16 KV cache for
verification. The key insight is that at early denoising steps (high noise),
the INT4 approximation is almost always sufficient, while at the final step
(low noise, fine detail), FP16 verification catches quality regressions.

Step-adaptive thresholds:
  - Early steps (noisy): relax acceptance threshold (noise dominates anyway)
  - Final step: tighten threshold to preserve fine details

Architecture assumptions (Matrix-Game 2.0):
  - 30 DiT blocks, Wan2.1 backbone
  - 3 denoising steps per generation block
  - Rolling KV window of 6 frames
  - ProgressiveKVQuantizer from Phase 3 provides INT4/FP16 caches
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class SpeculativeStats:
    """Tracks acceptance statistics for speculative decoding."""
    per_step_accepts: Dict[int, int] = field(default_factory=dict)
    per_step_rejects: Dict[int, int] = field(default_factory=dict)
    per_step_auto_accepts: Dict[int, int] = field(default_factory=dict)
    per_step_similarities: Dict[int, List[float]] = field(default_factory=dict)
    total_draft_time_ms: float = 0.0
    total_verify_time_ms: float = 0.0
    total_calls: int = 0

    def record(
        self,
        step_idx: int,
        accepted: bool,
        similarity: float,
        auto_accepted: bool = False,
        draft_time_ms: float = 0.0,
        verify_time_ms: float = 0.0,
    ) -> None:
        if step_idx not in self.per_step_accepts:
            self.per_step_accepts[step_idx] = 0
            self.per_step_rejects[step_idx] = 0
            self.per_step_auto_accepts[step_idx] = 0
            self.per_step_similarities[step_idx] = []

        if accepted:
            self.per_step_accepts[step_idx] += 1
        else:
            self.per_step_rejects[step_idx] += 1
        if auto_accepted:
            self.per_step_auto_accepts[step_idx] += 1

        self.per_step_similarities[step_idx].append(similarity)
        self.total_draft_time_ms += draft_time_ms
        self.total_verify_time_ms += verify_time_ms
        self.total_calls += 1


class KVQuantSpeculativeDecoder:
    """
    KV cache quantization self-speculative decoder.

    Uses INT4 KV cache as draft (fast path) and FP16 KV cache for verification
    (accurate path). Step-adaptive thresholds relax acceptance at early denoising
    steps where noise dominates, and tighten at the final step.
    """

    # MG2 defaults
    DEFAULT_TOTAL_STEPS = 3

    def __init__(
        self,
        config: "Dict[str, Any]" = None,
        # Legacy kwargs kept for backwards compat when called directly (not via build())
        acceptance_threshold: float = 0.95,
        step_adaptive: bool = True,
        auto_accept_early: bool = True,
        early_step_cutoff: int = 0,
    ) -> None:
        """
        Args:
            config: Config dict from default.yaml speculative section.
                    If provided, all other kwargs are ignored.
            acceptance_threshold: Base cosine similarity threshold (legacy).
            step_adaptive: Whether to adapt threshold per denoising step.
            auto_accept_early: If True, auto-accept at high-noise steps.
            early_step_cutoff: Legacy hard step cutoff (overridden by noise_floor).
        """
        if config is not None and isinstance(config, dict):
            acceptance_cfg = config.get("acceptance", {})
            self.int4_threshold       = float(acceptance_cfg.get("int4_threshold", 0.855))
            self.fp8_threshold        = float(acceptance_cfg.get("fp8_threshold",  0.969))
            self.acceptance_threshold = self.int4_threshold  # base for adaptive scaling
            self.step_adaptive        = bool(config.get("step_adaptive", True))
            self.auto_accept_early    = bool(config.get("auto_accept_early", True))
            # Noise-floor: auto-accept when noise_fraction (= 1 - progress) >= floor.
            # 0.80 → accept while noise >= 80% = first 20% of denoising steps.
            # For 10-step Oasis this covers steps 0–1; for 3-step MG2 only step 0.
            self.auto_accept_noise_floor = float(acceptance_cfg.get("auto_accept_noise_floor", 0.80))
            # skip_verify_after_n_accepts: after this many consecutive accepted steps
            # (including auto-accepts), skip the verify forward entirely — use draft
            # output directly.  0 = disabled.  Saves one full forward pass per step
            # once acceptance history is well-established (typical acceptance ~100%).
            self.skip_verify_after_n_accepts = int(
                acceptance_cfg.get("skip_verify_after_n_accepts", 0)
            )
        else:
            # Legacy direct-kwarg path
            self.int4_threshold          = acceptance_threshold
            self.fp8_threshold           = min(acceptance_threshold * 1.15, 0.999)
            self.acceptance_threshold    = acceptance_threshold
            self.step_adaptive           = step_adaptive
            self.auto_accept_early       = auto_accept_early
            self.auto_accept_noise_floor = 0.80  # 80% noise → first 20% of steps
            self.skip_verify_after_n_accepts = 0  # disabled in legacy mode

        self.stats = SpeculativeStats()
        # Runtime counter: consecutive accepted steps (auto-accept counts too).
        self._consecutive_accepts: int = 0
        # Cross-generation bypass flag: set True when a generation had 100%
        # acceptance, so the next generation can disable KV capture hooks
        # entirely and run at pure baseline speed.
        self._all_accepted_last_gen: bool = False
        # Hold a reference to the KV cache manager for cross-generation bypass.
        self.kv_cache_manager: Any = None

    def get_adaptive_threshold(
        self,
        step_idx: int,
        total_steps: int,
    ) -> float:
        """
        Compute step-adaptive acceptance threshold.

        Early steps (noisy): relax threshold by 10% (noise dominates anyway).
        Final step: tighten threshold by 2% (preserve fine details).

        Args:
            step_idx: Current denoising step (0-indexed).
            total_steps: Total number of denoising steps.

        Returns:
            Adjusted acceptance threshold.
        """
        if total_steps <= 1:
            return self.acceptance_threshold

        progress = step_idx / (total_steps - 1)  # 0.0 = first step, 1.0 = last step

        if progress < 0.5:
            # Early steps: use INT4 threshold (most relaxed)
            return self.int4_threshold
        elif progress >= 1.0:
            # Final step: use FP8 threshold (most strict)
            return self.fp8_threshold
        else:
            # Middle steps: linear interpolation between int4 and fp8 thresholds
            t = (progress - 0.5) / 0.5
            return self.int4_threshold + t * (self.fp8_threshold - self.int4_threshold)

    def _plain_forward(
        self,
        model: nn.Module,
        x: torch.Tensor,
        timestep: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Plain model-agnostic forward pass (no KV cache injection).

        Handles Oasis DiT calling convention: model(x, t_BT, external_cond=...)
        where t_BT must be (B, T) float, passed positionally.
        """
        model.eval()
        with torch.no_grad():
            try:
                output = model(x, timestep=timestep, **kwargs)
            except TypeError:
                # Oasis DiT: positional (x, t_BT); remap actions → external_cond
                fwd_kwargs: Dict[str, Any] = {k: v for k, v in kwargs.items()
                                               if k != "actions"}
                if "actions" in kwargs:
                    fwd_kwargs["external_cond"] = kwargs["actions"]
                t_arg = timestep
                if isinstance(t_arg, torch.Tensor):
                    B = x.shape[0]
                    T = x.shape[1] if x.dim() >= 2 else 1
                    if t_arg.dim() == 1 and t_arg.shape[0] == B:
                        t_arg = t_arg.unsqueeze(1).expand(B, T).to(x.dtype).contiguous()
                    elif t_arg.dim() == 2 and t_arg.shape == (B, 1):
                        t_arg = t_arg.expand(B, T).to(x.dtype).contiguous()
                try:
                    output = model(x, t_arg, **fwd_kwargs)
                except TypeError:
                    output = model(x, **fwd_kwargs)
        if isinstance(output, (tuple, list)):
            output = output[0]
        return output

    def draft_with_int4(
        self,
        model: nn.Module,
        x: torch.Tensor,
        kv_cache_int4: Any,
        timestep: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Run forward pass using INT4 KV cache (fast draft).

        The INT4 cache uses less memory bandwidth, enabling faster inference.
        Quality is slightly lower due to quantization, but acceptable for
        noisy early denoising steps.

        Args:
            model: The DiT model.
            x: Input latent tensor.
            kv_cache_int4: INT4-quantized KV cache object from Phase 3's
                ProgressiveKVQuantizer.
            timestep: Current diffusion timestep tensor.
            **kwargs: Additional model arguments (actions, etc.).

        Returns:
            Model output tensor.
        """
        # Inject INT4 cache into model's attention layers
        # The cache manager should expose .get_cache(precision='int4')
        if hasattr(kv_cache_int4, "inject_into_model"):
            kv_cache_int4.inject_into_model(model)

        # Use _plain_forward which handles model-specific calling conventions
        # (e.g. Oasis DiT expects positional t_BT, not keyword timestep=).
        return self._plain_forward(model, x, timestep, **kwargs)

    def verify_with_fp16(
        self,
        model: nn.Module,
        x: torch.Tensor,
        kv_cache_fp16: Any,
        timestep: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Run forward pass using FP16 KV cache (accurate verification).

        This is the full-precision path that serves as the ground truth
        for acceptance decisions.

        Args:
            model: The DiT model.
            x: Input latent tensor.
            kv_cache_fp16: FP16-precision KV cache object.
            timestep: Current diffusion timestep tensor.
            **kwargs: Additional model arguments.

        Returns:
            Model output tensor.
        """
        if hasattr(kv_cache_fp16, "inject_into_model"):
            kv_cache_fp16.inject_into_model(model)

        # Use _plain_forward which handles model-specific calling conventions.
        # _plain_forward already strips tuple/list outputs.
        return self._plain_forward(model, x, timestep, **kwargs)

    def accept_or_reject(
        self,
        draft_output: torch.Tensor,
        verify_output: torch.Tensor,
        step_idx: int,
        total_steps: int,
    ) -> Tuple[bool, float]:
        """
        Decide whether to accept the draft output based on cosine similarity.

        Args:
            draft_output: Output from INT4 draft forward pass.
            verify_output: Output from FP16 verification forward pass.
            step_idx: Current denoising step (0-indexed).
            total_steps: Total denoising steps.

        Returns:
            Tuple of (accepted: bool, similarity: float).
        """
        # Flatten spatial dimensions for global cosine similarity
        draft_flat = draft_output.float().flatten()
        verify_flat = verify_output.float().flatten()

        similarity = F.cosine_similarity(
            draft_flat.unsqueeze(0),
            verify_flat.unsqueeze(0),
        ).item()

        if self.step_adaptive:
            threshold = self.get_adaptive_threshold(step_idx, total_steps)
        else:
            threshold = self.acceptance_threshold

        accepted = similarity >= threshold

        logger.debug(
            "Step %d/%d: similarity=%.4f, threshold=%.4f, accepted=%s",
            step_idx, total_steps, similarity, threshold, accepted,
        )

        return accepted, similarity

    def speculative_forward(
        self,
        model: nn.Module,
        x: torch.Tensor,
        kv_cache_manager: Any,
        timestep: torch.Tensor,
        step_idx: int,
        total_steps: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Full speculative forward pass with INT4 draft and FP16 verification.

        Flow:
          1. Get INT4 and FP16 caches from KV cache manager
          2. Run draft with INT4 cache
          3. If step_adaptive and early step: auto-accept (noise dominates)
          4. Otherwise: run FP16 verification and accept/reject
          5. Track acceptance statistics

        Args:
            model: The DiT model.
            x: Input latent tensor.
            kv_cache_manager: KV cache manager that provides .get_cache(precision=...).
            timestep: Current diffusion timestep tensor.
            step_idx: Current denoising step (0-indexed).
            total_steps: Total denoising steps.
            **kwargs: Additional model arguments (actions, encoder_hidden_states, etc.).

        Returns:
            Output tensor (either draft or verified).
        """
        # Store manager reference for cross-generation bypass tracking.
        if kv_cache_manager is not None:
            self.kv_cache_manager = kv_cache_manager

        # Cross-generation bypass: if the previous generation had 100% acceptance,
        # disable KV capture hooks for this entire generation so it runs at
        # pure baseline speed (no Python-level F.sdpa monkey-patching overhead).
        _bypass_active = (
            self._all_accepted_last_gen
            and self.skip_verify_after_n_accepts > 0
            and kv_cache_manager is not None
            and hasattr(kv_cache_manager, "disable_capture")
        )
        if _bypass_active:
            kv_cache_manager.disable_capture()

        # Get caches from manager
        if kv_cache_manager is None:
            # No KV cache manager: fall back to two plain FP16 passes (draft + verify
            # simulation).  This exercises the full speculative decoding decision logic
            # (accept/reject, adaptive thresholds, stats) without INT4 quantization.
            t0 = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            draft_output = self._plain_forward(model, x, timestep, **kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            draft_time_ms = (time.perf_counter() - t0) * 1000.0

            noise_fraction = 1.0 - (step_idx / max(total_steps - 1, 1))
            if self.step_adaptive and self.auto_accept_early and noise_fraction >= self.auto_accept_noise_floor:
                self._consecutive_accepts += 1
                self.stats.record(step_idx=step_idx, accepted=True, similarity=1.0,
                                  auto_accepted=True, draft_time_ms=draft_time_ms)
                return draft_output

            # Skip verify when consecutive accept streak is long enough.
            if (
                self.skip_verify_after_n_accepts > 0
                and self._consecutive_accepts >= self.skip_verify_after_n_accepts
            ):
                self._consecutive_accepts += 1
                self.stats.record(step_idx=step_idx, accepted=True, similarity=1.0,
                                  auto_accepted=True, draft_time_ms=draft_time_ms)
                logger.debug(
                    "Step %d/%d: skipped verify (streak=%d >= %d).",
                    step_idx, total_steps,
                    self._consecutive_accepts, self.skip_verify_after_n_accepts,
                )
                return draft_output

            t0 = time.perf_counter()
            verify_output = self._plain_forward(model, x, timestep, **kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            verify_time_ms = (time.perf_counter() - t0) * 1000.0

            accepted, similarity = self.accept_or_reject(draft_output, verify_output,
                                                          step_idx, total_steps)
            if accepted:
                self._consecutive_accepts += 1
            else:
                self._consecutive_accepts = 0
            self.stats.record(step_idx=step_idx, accepted=accepted, similarity=similarity,
                              auto_accepted=False, draft_time_ms=draft_time_ms,
                              verify_time_ms=verify_time_ms)
            return draft_output if accepted else verify_output

        # --- Auto-accept short-circuit: skip expensive cache ops on noisy steps ---
        # For early denoising steps where noise dominates, the INT4 draft is
        # always acceptable. We know this BEFORE calling get_cache(), so we can
        # skip the quantize/copy overhead entirely and run a single plain forward.
        noise_fraction = 1.0 - (step_idx / max(total_steps - 1, 1))
        will_auto_accept = (
            self.step_adaptive
            and self.auto_accept_early
            and noise_fraction >= self.auto_accept_noise_floor
        )

        if will_auto_accept:
            t0 = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            draft_output = self._plain_forward(model, x, timestep, **kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            draft_time_ms = (time.perf_counter() - t0) * 1000.0
            self._consecutive_accepts += 1
            self.stats.record(
                step_idx=step_idx, accepted=True, similarity=1.0,
                auto_accepted=True, draft_time_ms=draft_time_ms,
            )
            logger.debug("Step %d/%d: auto-accepted (early step, skipping cache ops).",
                         step_idx, total_steps)
            if _bypass_active:
                kv_cache_manager.enable_capture()
            if step_idx == total_steps - 1:
                self._all_accepted_last_gen = (
                    self._consecutive_accepts >= total_steps
                )
            return draft_output

        # Skip verify when acceptance streak is long enough: run only draft,
        # trust the output without FP16 verification.
        if (
            self.skip_verify_after_n_accepts > 0
            and self._consecutive_accepts >= self.skip_verify_after_n_accepts
        ):
            t0 = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            draft_output = self._plain_forward(model, x, timestep, **kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            draft_time_ms = (time.perf_counter() - t0) * 1000.0
            self._consecutive_accepts += 1
            self.stats.record(
                step_idx=step_idx, accepted=True, similarity=1.0,
                auto_accepted=True, draft_time_ms=draft_time_ms,
            )
            logger.debug(
                "Step %d/%d: skipped verify (streak=%d >= %d, skipping cache ops).",
                step_idx, total_steps,
                self._consecutive_accepts, self.skip_verify_after_n_accepts,
            )
            if _bypass_active:
                kv_cache_manager.enable_capture()
            if step_idx == total_steps - 1:
                self._all_accepted_last_gen = (
                    self._consecutive_accepts >= total_steps
                )
            return draft_output

        if hasattr(kv_cache_manager, "get_cache"):
            kv_cache_int4 = kv_cache_manager.get_cache(precision="int4")
            kv_cache_fp16 = kv_cache_manager.get_cache(precision="fp16")
        elif hasattr(kv_cache_manager, "int4_cache") and hasattr(kv_cache_manager, "fp16_cache"):
            kv_cache_int4 = kv_cache_manager.int4_cache
            kv_cache_fp16 = kv_cache_manager.fp16_cache
        else:
            if _bypass_active:
                kv_cache_manager.enable_capture()
            raise ValueError(
                f"kv_cache_manager ({type(kv_cache_manager).__name__}) has no "
                f"get_cache() method or int4_cache/fp16_cache attributes. "
                f"Cannot perform speculative draft/verify."
            )

        # --- Draft with INT4 ---
        t0 = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Arm KV injection so the wired attention hooks replace K/V with the
        # pre-quantized INT4 cache tensors instead of re-projecting full-precision
        # K/V from scratch.  This is what makes the draft pass genuinely cheaper.
        # disable_inject() is always called in the finally block so a mid-forward
        # exception cannot leave the manager in a stale injection state.
        if hasattr(kv_cache_manager, "inject_into_model"):
            kv_cache_manager.inject_into_model(model, precision="int4")
        try:
            draft_output = self.draft_with_int4(model, x, kv_cache_int4, timestep, **kwargs)
        finally:
            if hasattr(kv_cache_manager, "disable_inject"):
                kv_cache_manager.disable_inject()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        draft_time_ms = (time.perf_counter() - t0) * 1000.0

        # --- Verify with FP16 ---
        t0 = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Arm FP16 injection for the verify pass.
        if hasattr(kv_cache_manager, "inject_into_model"):
            kv_cache_manager.inject_into_model(model, precision="fp16")
        try:
            verify_output = self.verify_with_fp16(model, x, kv_cache_fp16, timestep, **kwargs)
        finally:
            if hasattr(kv_cache_manager, "disable_inject"):
                kv_cache_manager.disable_inject()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        verify_time_ms = (time.perf_counter() - t0) * 1000.0

        # --- Accept or reject ---
        accepted, similarity = self.accept_or_reject(
            draft_output, verify_output, step_idx, total_steps
        )

        if accepted:
            self._consecutive_accepts += 1
        else:
            self._consecutive_accepts = 0

        self.stats.record(
            step_idx=step_idx,
            accepted=accepted,
            similarity=similarity,
            auto_accepted=False,
            draft_time_ms=draft_time_ms,
            verify_time_ms=verify_time_ms,
        )

        if _bypass_active:
            kv_cache_manager.enable_capture()
        if step_idx == total_steps - 1:
            self._all_accepted_last_gen = (
                self._consecutive_accepts >= total_steps
            )

        if accepted:
            return draft_output
        else:
            return verify_output

    def get_stats(self) -> Dict[str, Any]:
        """
        Return acceptance statistics and speedup estimates.

        Returns:
            Dict with:
              - acceptance_rate_per_step: {step_idx: rate}
              - overall_acceptance_rate: float
              - auto_accept_rate: float
              - avg_similarity_per_step: {step_idx: avg_sim}
              - estimated_speedup: float (vs always running FP16)
              - total_draft_time_ms: float
              - total_verify_time_ms: float
        """
        stats = self.stats
        result: Dict[str, Any] = {
            "total_calls": stats.total_calls,
            "acceptance_rate_per_step": {},
            "avg_similarity_per_step": {},
            "auto_accept_rate_per_step": {},
            "total_draft_time_ms": stats.total_draft_time_ms,
            "total_verify_time_ms": stats.total_verify_time_ms,
        }

        total_accepted = 0
        total_auto = 0
        total_total = 0

        for step_idx in sorted(
            set(list(stats.per_step_accepts.keys()) + list(stats.per_step_rejects.keys()))
        ):
            accepts = stats.per_step_accepts.get(step_idx, 0)
            rejects = stats.per_step_rejects.get(step_idx, 0)
            auto = stats.per_step_auto_accepts.get(step_idx, 0)
            total = accepts + rejects

            total_accepted += accepts
            total_auto += auto
            total_total += total

            rate = accepts / total if total > 0 else 0.0
            result["acceptance_rate_per_step"][step_idx] = rate
            result["auto_accept_rate_per_step"][step_idx] = auto / total if total > 0 else 0.0

            sims = stats.per_step_similarities.get(step_idx, [])
            result["avg_similarity_per_step"][step_idx] = (
                sum(sims) / len(sims) if sims else 0.0
            )

        result["overall_acceptance_rate"] = (
            total_accepted / total_total if total_total > 0 else 0.0
        )
        result["auto_accept_rate"] = (
            total_auto / total_total if total_total > 0 else 0.0
        )

        # Speedup estimate:
        # When accepted: cost = draft_time only
        # When rejected: cost = draft_time + verify_time
        # Baseline: verify_time for every call
        # Estimate assuming draft ~= 0.5 * verify (INT4 is ~2x faster)
        if stats.total_verify_time_ms > 0 and stats.total_calls > 0:
            # num_verified = calls that actually ran FP16 verify (not auto-accepted)
            num_verified = stats.total_calls - total_auto
            # avg_verify = average FP16 cost per verified call
            avg_verify = stats.total_verify_time_ms / max(num_verified, 1)
            # avg_draft = average INT4 draft cost per call (all calls draft)
            avg_draft = stats.total_draft_time_ms / stats.total_calls

            # Baseline: if every call ran full FP16 (no speculation)
            baseline_time = avg_verify * stats.total_calls
            # Actual: every call pays draft + only rejected calls also pay verify
            actual_time = stats.total_draft_time_ms + stats.total_verify_time_ms
            result["estimated_speedup"] = baseline_time / actual_time if actual_time > 0 else 1.0
            result["avg_draft_ms"] = avg_draft
            result["avg_verify_ms"] = avg_verify
        else:
            result["estimated_speedup"] = 1.0

        return result

    def reset_stats(self) -> None:
        """Reset all tracked statistics (but not cross-generation behavioral state)."""
        self.stats = SpeculativeStats()
        self._consecutive_accepts = 0
        # NOTE: _all_accepted_last_gen is intentionally NOT reset here.
        # It records whether the previous generation had 100% acceptance,
        # which is cross-generation behavioral state used to bypass KV hooks.
        # Resetting it here would defeat the bypass since reset_stats() is
        # called before every timed run.
        # Ensure KV capture is re-enabled in case a previous generation left it
        # disabled (e.g. due to an exception during a bypassed generation).
        if (
            self.kv_cache_manager is not None
            and hasattr(self.kv_cache_manager, "enable_capture")
        ):
            self.kv_cache_manager.enable_capture()
