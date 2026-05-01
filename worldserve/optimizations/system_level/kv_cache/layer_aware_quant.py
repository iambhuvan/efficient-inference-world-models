"""
Layer-aware quantization: assign per-layer precision based on attention entropy.

ADVANCED optimization.

Intuition: layers with high-entropy (scattered) attention distributions rely on
fine-grained distinctions between KV tokens and therefore need higher precision.
Layers with low-entropy (focused/peaked) attention are robust to quantization
because only a few dominant tokens matter.

Profiling is done once (or periodically) on a representative input batch. The
resulting per-layer precision map is then used throughout inference.

Precision tiers (by entropy rank):
  - Top 1/3 (highest entropy):  FP16
  - Middle 1/3:                 FP8
  - Bottom 1/3 (lowest entropy): INT4
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


class LayerAwareQuantizer:
    """
    Profile attention entropy per DiT layer and assign precision tiers.
    """

    def __init__(self, num_layers: int) -> None:
        """
        Args:
            num_layers: Total number of DiT layers in the model.
        """
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.num_layers = num_layers

        # Per-layer entropy (populated after profiling)
        self._entropy_scores: Dict[int, float] = {}
        # Per-layer precision assignment
        self._layer_precision: Dict[int, str] = {}

    # ------------------------------------------------------------------
    # Profiling
    # ------------------------------------------------------------------

    def profile_layer_entropy(
        self,
        model: Any,
        sample_input: Any,
        attention_hook_fn: Optional[Callable] = None,
    ) -> Dict[int, float]:
        """
        Run the model on a sample input and capture attention entropy per layer.

        This method registers forward hooks on attention layers to capture the
        softmax attention weights, computes their entropy, and stores the result.

        Args:
            model: The DiT model (must have accessible attention layers).
            sample_input: A representative input (dict or tensor) that can be
                passed to ``model(**sample_input)`` or ``model(sample_input)``.
            attention_hook_fn: Optional callable that, given the model, returns
                a list of ``(layer_idx, attention_module)`` pairs.  If None,
                a heuristic search for ``".attn"`` submodules is used.

        Returns:
            Dict mapping layer_idx to entropy score (nats).
        """
        attention_outputs: Dict[int, torch.Tensor] = {}
        hooks = []

        # Find attention modules
        if attention_hook_fn is not None:
            attn_modules = attention_hook_fn(model)
        else:
            attn_modules = self._find_attention_modules(model)

        # Register hooks
        for layer_idx, module in attn_modules:
            def _make_hook(lidx: int):
                def hook(mod, inp, out):
                    # out is typically (attn_output, attn_weights) or just attn_output
                    if isinstance(out, tuple) and len(out) >= 2:
                        weights = out[1]
                    elif isinstance(out, torch.Tensor) and out.dim() >= 3:
                        # Cannot extract weights; skip
                        return
                    else:
                        return
                    if weights is not None:
                        attention_outputs[lidx] = weights.detach()
                return hook
            h = module.register_forward_hook(_make_hook(layer_idx))
            hooks.append(h)

        # Forward pass
        try:
            with torch.no_grad():
                if isinstance(sample_input, dict):
                    model(**sample_input)
                else:
                    model(sample_input)
        finally:
            for h in hooks:
                h.remove()

        # Compute entropy per layer
        for layer_idx, weights in attention_outputs.items():
            self._entropy_scores[layer_idx] = self._compute_entropy(weights)

        # If some layers weren't captured, assign median entropy
        if self._entropy_scores:
            median_entropy = sorted(self._entropy_scores.values())[
                len(self._entropy_scores) // 2
            ]
        else:
            median_entropy = 1.0

        for i in range(self.num_layers):
            if i not in self._entropy_scores:
                self._entropy_scores[i] = median_entropy

        # Assign precision tiers
        self._assign_precision()

        return dict(self._entropy_scores)

    def set_entropy_scores(self, scores: Dict[int, float]) -> None:
        """
        Manually set entropy scores (e.g., from a cached profiling run).

        Args:
            scores: Mapping layer_idx -> entropy (nats).
        """
        self._entropy_scores = dict(scores)
        for i in range(self.num_layers):
            if i not in self._entropy_scores:
                self._entropy_scores[i] = 0.0
        self._assign_precision()

    # ------------------------------------------------------------------
    # Precision assignment
    # ------------------------------------------------------------------

    def assign_layer_precision(
        self,
        entropy_scores: Optional[Dict[int, float]] = None,
    ) -> Dict[int, str]:
        """
        Assign precision tiers based on entropy ranking.

        Args:
            entropy_scores: Optional explicit scores.  If None, uses the
                internally stored scores from ``profile_layer_entropy``.

        Returns:
            Dict mapping layer_idx to one of ``"fp16"``, ``"fp8"``, ``"int4"``.
        """
        if entropy_scores is not None:
            self._entropy_scores = dict(entropy_scores)

        self._assign_precision()
        return dict(self._layer_precision)

    def _assign_precision(self) -> None:
        """Internal: rank layers by entropy and assign tiers."""
        if not self._entropy_scores:
            # Default all to FP16 if no profiling done
            self._layer_precision = {i: "fp16" for i in range(self.num_layers)}
            return

        # Sort layers by entropy (descending)
        sorted_layers = sorted(
            self._entropy_scores.items(), key=lambda x: x[1], reverse=True
        )

        n = len(sorted_layers)
        top_cutoff = n // 3
        mid_cutoff = 2 * n // 3

        self._layer_precision = {}
        for rank, (layer_idx, _entropy) in enumerate(sorted_layers):
            if rank < top_cutoff:
                self._layer_precision[layer_idx] = "fp16"
            elif rank < mid_cutoff:
                self._layer_precision[layer_idx] = "fp8"
            else:
                self._layer_precision[layer_idx] = "int4"

    def get_layer_config(self) -> Dict[int, str]:
        """
        Return the current per-layer precision assignment.

        Returns:
            Dict mapping layer_idx to ``"fp16"`` / ``"fp8"`` / ``"int4"``.
        """
        if not self._layer_precision:
            # Return safe default
            return {i: "fp16" for i in range(self.num_layers)}
        return dict(self._layer_precision)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_entropy(attention_weights: torch.Tensor) -> float:
        """
        Compute the average entropy (nats) of attention distributions.

        Args:
            attention_weights: Shape ``(B, H, Sq, Skv)``.

        Returns:
            Scalar entropy averaged over all heads and positions.
        """
        # Clamp to avoid log(0)
        eps = 1e-10
        w = attention_weights.float().clamp(min=eps)
        entropy = -(w * w.log()).sum(dim=-1)  # (B, H, Sq)
        return entropy.mean().item()

    @staticmethod
    def _find_attention_modules(model: Any) -> List[Tuple[int, Any]]:
        """
        Heuristic: find attention submodules by name patterns.

        Returns list of (layer_index, module) pairs.
        """
        attn_modules = []
        layer_idx = 0
        for name, module in model.named_modules():
            # Common patterns: ".attn", ".self_attn", ".attention"
            name_lower = name.lower()
            if any(pat in name_lower for pat in (".attn", "self_attn", "attention")):
                # Avoid double-counting (e.g., "attention.out_proj")
                # Only grab the top-level attention module
                if not any(sub in name_lower for sub in (".proj", ".linear", ".norm")):
                    attn_modules.append((layer_idx, module))
                    layer_idx += 1
        return attn_modules
