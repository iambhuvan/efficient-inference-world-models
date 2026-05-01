"""
Importance-aware precision assignment for KV cache tokens.

ADVANCED optimization.

Tracks per-token importance using an exponential moving average (EMA) of the
mean attention weight received by each KV position.  This score is combined
with age to produce per-token precision recommendations that override the
simple age-based thresholds:

  - High importance + old  -> keep at FP8 (protect from INT4 degradation)
  - Low importance + recent -> drop to FP8 early (safe to compress)
  - High importance + recent -> keep FP16 (default, no override)
  - Low importance + old   -> INT4 (default, no override)

CRITICAL: importance scores must be computed on FULL-PRECISION attention
weights, *before* any quantization is applied.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch


class ImportanceTracker:
    """
    EMA-based attention importance scorer for KV cache tokens.

    Maintains per-layer, per-token running importance scores and produces
    per-token precision recommendations that can be fed into
    ``ProgressiveKVQuantizer``.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        num_layers: int = 30,
    ) -> None:
        """
        Args:
            alpha: EMA decay rate.  Higher alpha gives more weight to the
                most recent attention pattern (0 = never update, 1 = no memory).
            num_layers: Number of DiT layers to track.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = alpha
        self.num_layers = num_layers

        # Per-layer EMA scores: layer_idx -> Tensor of shape (num_kv_tokens,)
        self._scores: Dict[int, torch.Tensor] = {}

        # Count of updates per layer (for warm-up diagnostics)
        self._update_count: Dict[int, int] = {}

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def update(
        self,
        layer_idx: int,
        attention_weights: torch.Tensor,
    ) -> None:
        """
        Update importance scores for a layer given full-precision attention weights.

        CRITICAL: call this *before* any quantization of the KV cache for this
        layer.  The attention weights must be the raw softmax output.

        Args:
            layer_idx: DiT layer index (0-based).
            attention_weights: Attention weight tensor of shape
                ``(batch, num_heads, seq_q, seq_kv)`` (standard multi-head
                attention output after softmax).  seq_kv is the total number
                of KV tokens across all cached frames.
        """
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise IndexError(
                f"layer_idx {layer_idx} out of range [0, {self.num_layers})"
            )

        # Importance = mean attention received by each KV position
        # Average over batch, heads, and query positions
        # attention_weights: (B, H, Sq, Skv)
        new_scores = attention_weights.float().mean(dim=(0, 1, 2))  # (Skv,)

        device = new_scores.device

        if layer_idx not in self._scores:
            # First observation — initialise directly
            self._scores[layer_idx] = new_scores.detach()
            self._update_count[layer_idx] = 1
        else:
            old = self._scores[layer_idx]

            # Handle size changes (new tokens appended to KV cache)
            if new_scores.shape[0] > old.shape[0]:
                # Extend old scores with zeros for new positions
                pad_size = new_scores.shape[0] - old.shape[0]
                old = torch.cat([
                    old,
                    torch.zeros(pad_size, device=device, dtype=old.dtype),
                ], dim=0)
            elif new_scores.shape[0] < old.shape[0]:
                # Tokens were evicted; truncate old scores
                old = old[: new_scores.shape[0]]

            # EMA update
            self._scores[layer_idx] = (
                self.alpha * new_scores.detach() + (1.0 - self.alpha) * old
            )
            self._update_count[layer_idx] = self._update_count.get(layer_idx, 0) + 1

    def get_importance_scores(self, layer_idx: int) -> torch.Tensor:
        """
        Return current EMA importance scores for a layer.

        Args:
            layer_idx: DiT layer index.

        Returns:
            Tensor of shape ``(num_kv_tokens,)`` with non-negative scores.
            Higher = more important (receives more attention on average).

        Raises:
            KeyError: If no scores have been recorded for *layer_idx* yet.
        """
        if layer_idx not in self._scores:
            raise KeyError(
                f"No importance scores recorded for layer {layer_idx}. "
                f"Call update() first."
            )
        return self._scores[layer_idx]

    def get_quantization_recommendation(
        self,
        layer_idx: int,
        frame_ages: torch.Tensor,
        k_fp8_age: int = 30,
        k_int4_age: int = 90,
        importance_high_percentile: float = 0.75,
        importance_low_percentile: float = 0.25,
    ) -> torch.Tensor:
        """
        Combine age and importance to recommend per-token precision.

        Override rules (vs pure age-based):
          1. High importance + old (age >= int4_age) -> "fp8" (protect)
          2. Low importance + recent (age < fp8_age) -> "fp8" (compress early)
          3. Otherwise -> follow standard age-based assignment.

        Args:
            layer_idx: DiT layer index.
            frame_ages: Per-token ages, shape ``(num_kv_tokens,)``.
            k_fp8_age: Age threshold for FP8 (from ProgressiveKVQuantizer).
            k_int4_age: Age threshold for INT4.
            importance_high_percentile: Percentile above which a token is
                considered "high importance".
            importance_low_percentile: Percentile below which a token is
                considered "low importance".

        Returns:
            Integer tensor of shape ``(num_kv_tokens,)`` encoding precision:
              - 0 = FP16
              - 1 = FP8
              - 2 = INT4
        """
        scores = self.get_importance_scores(layer_idx)
        num_tokens = frame_ages.shape[0]

        # Align sizes
        if scores.shape[0] < num_tokens:
            scores = torch.cat([
                scores,
                torch.zeros(num_tokens - scores.shape[0],
                            device=scores.device, dtype=scores.dtype),
            ])
        elif scores.shape[0] > num_tokens:
            scores = scores[:num_tokens]

        # Determine percentile thresholds
        if scores.numel() == 0:
            return torch.zeros(num_tokens, dtype=torch.long, device=frame_ages.device)

        high_thresh = torch.quantile(scores, importance_high_percentile)
        low_thresh = torch.quantile(scores, importance_low_percentile)

        is_high_importance = scores >= high_thresh
        is_low_importance = scores <= low_thresh

        # Start with age-based defaults
        precision = torch.zeros(num_tokens, dtype=torch.long, device=frame_ages.device)
        ages_float = frame_ages.float()
        precision = torch.where(ages_float >= k_fp8_age,
                                torch.ones_like(precision), precision)
        precision = torch.where(ages_float >= k_int4_age,
                                torch.full_like(precision, 2), precision)

        # Override 1: high importance + would-be INT4 -> keep at FP8
        protect_mask = is_high_importance & (precision == 2)
        precision = torch.where(protect_mask, torch.ones_like(precision), precision)

        # Override 2: low importance + still FP16 -> drop to FP8 early
        compress_mask = is_low_importance & (precision == 0) & (ages_float > 0)
        precision = torch.where(compress_mask, torch.ones_like(precision), precision)

        return precision

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_layer_summary(self) -> Dict[int, Dict[str, float]]:
        """
        Return per-layer summary statistics.

        Returns:
            Dict mapping layer_idx to {mean, std, min, max, num_tokens, updates}.
        """
        summary = {}
        for layer_idx, scores in self._scores.items():
            summary[layer_idx] = {
                "mean": scores.mean().item(),
                "std": scores.std().item() if scores.numel() > 1 else 0.0,
                "min": scores.min().item(),
                "max": scores.max().item(),
                "num_tokens": scores.numel(),
                "updates": self._update_count.get(layer_idx, 0),
            }
        return summary

    def reset(self) -> None:
        """Clear all tracked scores."""
        self._scores.clear()
        self._update_count.clear()
