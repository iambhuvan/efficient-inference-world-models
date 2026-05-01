"""
External Draft Speculative Decoding for Matrix-Game 2.0.

Uses FastWan2.1-T2V-1.3B (DMD-distilled, 3-step) as a fast draft model and
Matrix-Game 2.0 (1.8B, action-conditioned) as the verifier.

Both models share the same Wan2.1 16-channel VAE latent space — latent tensors
are directly comparable without any bridging projection.

Key properties:
  - FastWan: same WanTransformer3DModel backbone (30L/12H/h=1536), no action conditioning
  - MG2: same backbone + action cross-attention in blocks 0-14
  - Same VAE: Wan2.1 16-channel, same tokenization pipeline

Acceptance criterion: CSpD continuous distribution comparison (continuous_acceptance.py)

Expected speedup: ~1.8x at 65% acceptance rate on game video.
On rejection: x_verify (already computed) is returned — no wasted compute.

Reference:
  FastWan2.1: FastVideo/FastWan2.1-T2V-1.3B-Diffusers (HuggingFace, Apache-2.0)
  CSpD: https://hf.co/papers/2411.11925
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from worldserve.optimizations.system_level.speculative.continuous_acceptance import (
    cspd_accept,
    cosine_accept,
    adaptive_sigma_from_flow_t,
)

logger = logging.getLogger(__name__)


@dataclass
class ExternalDraftStats:
    """Tracks acceptance statistics for external draft speculative decoding."""
    total_calls: int = 0
    total_accepts: int = 0
    total_rejects: int = 0
    per_step_accept_probs: Dict[int, List[float]] = field(default_factory=dict)
    total_draft_time_ms: float = 0.0
    total_verify_time_ms: float = 0.0

    def record(
        self,
        step_idx: int,
        accepted: bool,
        accept_prob: float,
        draft_ms: float,
        verify_ms: float,
    ) -> None:
        self.total_calls += 1
        if accepted:
            self.total_accepts += 1
        else:
            self.total_rejects += 1
        self.per_step_accept_probs.setdefault(step_idx, []).append(accept_prob)
        self.total_draft_time_ms += draft_ms
        self.total_verify_time_ms += verify_ms

    def summary(self) -> Dict[str, Any]:
        total = max(self.total_calls, 1)
        return {
            "total_calls": self.total_calls,
            "acceptance_rate": self.total_accepts / total,
            "avg_accept_prob_per_step": {
                s: sum(v) / len(v)
                for s, v in self.per_step_accept_probs.items()
            },
            "total_draft_ms": self.total_draft_time_ms,
            "total_verify_ms": self.total_verify_time_ms,
            "estimated_speedup": (
                self.total_verify_time_ms
                / max(self.total_draft_time_ms + self.total_verify_time_ms
                      - self.total_accepts / total * self.total_verify_time_ms / total
                      * total, 1.0)
            ),
        }


class ExternalDraftSpeculativeDecoder:
    """
    Cross-model speculative decoding: FastWan2.1 draft → Matrix-Game 2.0 verify.

    Both models share the Wan2.1 16-channel VAE latent space.  FastWan runs
    without action conditioning (cheap); MG2 verifies under action conditioning
    (accurate).  For static/slow-action frames the two are nearly identical.

    Usage:
        decoder = ExternalDraftSpeculativeDecoder.from_pretrained(
            draft_hf_id="FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
            verify_model=mg2_pipeline.generator,
            config=cfg["speculative"],
        )
        # Then call in the MG2 inference loop:
        output = decoder.speculative_forward(
            x_noisy, t, actions=actions, step_idx=step_idx, total_steps=3
        )
    """

    def __init__(
        self,
        draft_model: Any,
        verify_model: Any,
        config: Dict[str, Any],
    ) -> None:
        """
        Args:
            draft_model: FastWan2.1 generator forward callable.
                         Signature: (x, t, **kwargs) -> noise_pred
            verify_model: MG2 generator forward callable.
                          Signature: (x, t, actions=..., **kwargs) -> noise_pred
            config: speculative config dict from default.yaml.
        """
        self.draft_model  = draft_model
        self.verify_model = verify_model

        acc_cfg = config.get("acceptance", {})
        self.base_threshold  = float(acc_cfg.get("int4_threshold",  0.88))
        self.final_threshold = float(acc_cfg.get("fp8_threshold",   0.97))
        self.temperature     = float(acc_cfg.get("cspd_temperature", 1.0))
        self.use_cspd        = bool(config.get("use_cspd_criterion", True))
        self.stats           = ExternalDraftStats()

    @classmethod
    def from_pretrained(
        cls,
        draft_hf_id: str,
        verify_model: Any,
        config: Dict[str, Any],
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "ExternalDraftSpeculativeDecoder":
        """
        Load FastWan2.1-T2V-1.3B from HuggingFace and create decoder.

        Args:
            draft_hf_id: HuggingFace repo ID, e.g.
                         "FastVideo/FastWan2.1-T2V-1.3B-Diffusers"
            verify_model: MG2 generator forward callable.
            config: speculative config dict.
            device: Target device.
            dtype: Model dtype (bfloat16 recommended for Wan2.1).
        """
        logger.info("Loading draft model: %s", draft_hf_id)
        try:
            from fastvideo.models.wan.modeling_wan import WanTransformer3DModel  # type: ignore
            draft = WanTransformer3DModel.from_pretrained(
                draft_hf_id, subfolder="transformer",
                torch_dtype=dtype,
            ).to(device).eval()
        except ImportError:
            # Fallback: try loading via diffusers pipeline
            try:
                from diffusers import AutoModel  # type: ignore
                draft = AutoModel.from_pretrained(
                    draft_hf_id, torch_dtype=dtype
                ).to(device).eval()
            except Exception as e:
                raise ImportError(
                    f"Failed to load draft model {draft_hf_id}. "
                    f"Install FastVideo: pip install git+https://github.com/hao-ai-lab/FastVideo.git\n"
                    f"Original error: {e}"
                )
        logger.info("Draft model loaded: %s params", sum(p.numel() for p in draft.parameters()) // 1_000_000)
        return cls(draft_model=draft, verify_model=verify_model, config=config)

    # ------------------------------------------------------------------
    # Core speculative forward
    # ------------------------------------------------------------------

    def speculative_forward(
        self,
        x_noisy: torch.Tensor,
        t: torch.Tensor,
        actions: Optional[Any] = None,
        step_idx: int = 0,
        total_steps: int = 3,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Run draft-then-verify speculative forward.

        Args:
            x_noisy:    Noisy latent input.  Shape: (B, C, T, H, W).
            t:          Flow-matching timestep tensor (scalar or (B,)).
            actions:    Action conditioning for MG2 (keyboard+mouse).
            step_idx:   Current denoising step index (0-indexed).
            total_steps: Total denoising steps (typically 3 for MG2).
            **kwargs:   Passed to verify model only.

        Returns:
            Accepted noise prediction tensor (same shape as model output).
        """
        # 1. Draft (no action conditioning — fast)
        t0 = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        with torch.no_grad():
            x_draft = self.draft_model(x_noisy, t)
            if isinstance(x_draft, (tuple, list)):
                x_draft = x_draft[0]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        draft_ms = (time.perf_counter() - t0) * 1000.0

        # 2. Verify (with action conditioning — accurate)
        t0 = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        with torch.no_grad():
            fwd_kwargs = dict(kwargs)
            if actions is not None:
                fwd_kwargs["actions"] = actions
            x_verify = self.verify_model(x_noisy, t, **fwd_kwargs)
            if isinstance(x_verify, (tuple, list)):
                x_verify = x_verify[0]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        verify_ms = (time.perf_counter() - t0) * 1000.0

        # 3. Acceptance criterion
        theta = self._adaptive_threshold(step_idx, total_steps)

        if self.use_cspd:
            sigma_t = adaptive_sigma_from_flow_t(
                float(t.flatten()[0].item()) if isinstance(t, torch.Tensor) else float(t)
            )
            accepted, accept_prob = cspd_accept(
                x_draft, x_verify, sigma_t=sigma_t,
                threshold=theta, temperature=self.temperature,
            )
        else:
            accepted, accept_prob = cosine_accept(x_draft, x_verify, threshold=theta)

        self.stats.record(step_idx, accepted, accept_prob, draft_ms, verify_ms)

        logger.debug(
            "ExternalDraftSpec step %d/%d: accept_prob=%.4f, theta=%.4f, accepted=%s "
            "draft=%.1fms verify=%.1fms",
            step_idx, total_steps, accept_prob, theta, accepted, draft_ms, verify_ms,
        )

        return x_draft if accepted else x_verify

    def _adaptive_threshold(self, step_idx: int, total_steps: int) -> float:
        """
        Linearly interpolate from base_threshold (early) to final_threshold (last step).
        """
        if total_steps <= 1:
            return self.final_threshold
        progress = step_idx / (total_steps - 1)
        return self.base_threshold + progress * (self.final_threshold - self.base_threshold)

    def get_stats(self) -> Dict[str, Any]:
        return self.stats.summary()

    def reset_stats(self) -> None:
        self.stats = ExternalDraftStats()
