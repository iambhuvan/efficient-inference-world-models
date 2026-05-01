"""
Attention pattern analysis and profiling for WorldServe (Phase 4).

Extracts attention weight matrices from DiT models, computes sparsity and
entropy statistics, classifies heads as spatial vs temporal, and generates
analysis reports. Used to guide SVG2 sparse attention configuration.

Architecture assumptions (Matrix-Game 2.0):
  - ~15,360 spatial tokens/frame
  - 30 DiT blocks, Wan2.1 backbone
  - Rolling KV window of 6 frames
"""

from __future__ import annotations

import json
import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class AttentionProfiler:
    """
    Profiles attention patterns in DiT models to guide sparse attention design.

    Extracts attention weight matrices via forward hooks, computes sparsity
    and entropy statistics, and classifies heads as spatial or temporal.
    """

    def __init__(
        self,
        num_frames: int = 6,
        tokens_per_frame: int = 15360,
    ) -> None:
        """
        Args:
            num_frames: Number of frames in the rolling KV window.
            tokens_per_frame: Spatial tokens per frame (15,360 for MG2).
        """
        self.num_frames = num_frames
        self.tokens_per_frame = tokens_per_frame

    # ------------------------------------------------------------------
    # Attention map extraction
    # ------------------------------------------------------------------

    def extract_attention_maps(
        self,
        model: nn.Module,
        sample_input: Union[torch.Tensor, Dict[str, Any], Tuple[Any, ...], List[Any]],
        layers: Optional[List[int]] = None,
        max_tokens_for_capture: int = 4096,
    ) -> Dict[int, Dict[int, torch.Tensor]]:
        """
        Hook into attention layers, capture attention weight matrices.

        For large sequence lengths (MG2: 92,160 tokens for 6 frames), computing
        and storing full (S, S) attention matrices is infeasible. Instead, we
        capture attention patterns on a subsampled token set, or capture the
        raw Q/K and compute attention lazily on a subsample.

        Args:
            model: The DiT model to profile.
            sample_input: Input to run through the model.
            layers: Specific layer indices to capture (None = all).
            max_tokens_for_capture: Maximum sequence length for full attention
                capture. If S > this, subsample tokens.

        Returns:
            Dict of {layer_idx: {head_idx: attention_weights}}.
            Attention weights are (S', S') tensors where S' <= max_tokens_for_capture.
        """
        attention_maps: Dict[int, Dict[int, torch.Tensor]] = {}
        hooks: List[torch.utils.hooks.RemovableHook] = []

        layer_idx_counter = [0]  # Mutable counter for closure

        for name, module in model.named_modules():
            cls_name = type(module).__name__
            if cls_name not in (
                "Attention", "WanAttention", "FlashSelfAttention",
                "SelfAttention", "MultiheadAttention",
                "SpatialAxialAttention", "TemporalAxialAttention",
            ):
                continue

            current_layer = layer_idx_counter[0]
            layer_idx_counter[0] += 1

            if layers is not None and current_layer not in layers:
                continue

            lidx = current_layer
            max_tok = max_tokens_for_capture

            def _make_hook(layer_id: int, max_tokens: int) -> Any:
                def hook_fn(
                    mod: nn.Module,
                    inputs: Tuple[Any, ...],
                    output: Any,
                ) -> None:
                    # Intercept by looking for stored q, k attributes
                    q = getattr(mod, "_last_q", None)
                    k = getattr(mod, "_last_k", None)

                    if q is None or k is None:
                        # Fallback: try to extract Q, K from inputs
                        if isinstance(inputs, tuple) and len(inputs) >= 2:
                            cq, ck = inputs[0], inputs[1]
                            if isinstance(cq, torch.Tensor) and cq.dim() == 4:
                                q, k = cq, ck
                    if q is None or k is None:
                        return

                    # q, k: (B, H, S, D) or (B, S, H, D)
                    if q.dim() != 4:
                        return

                    # Normalize to (B, H, S, D).
                    # Disambiguate using head_dim (dim 3): it's typically
                    # 64/128, much smaller than seq_len. If dim 1 looks
                    # like seq_len (much larger than dim 3) we have (B, S, H, D).
                    d = q.shape[3]  # head_dim candidate
                    if q.shape[1] > q.shape[2] and q.shape[1] > d * 2:
                        # dim 1 is much larger than head_dim -> likely seq_len
                        # (B, S, H, D) -> (B, H, S, D)
                        q = q.transpose(1, 2)
                        k = k.transpose(1, 2)

                    B, H, S, D = q.shape

                    # Subsample if too large
                    if S > max_tokens:
                        stride = S // max_tokens
                        indices = torch.arange(0, S, stride, device=q.device)[:max_tokens]
                        q_sub = q[:, :, indices]
                        k_sub = k[:, :, indices]
                    else:
                        q_sub = q
                        k_sub = k

                    # Compute attention weights
                    scale = D ** -0.5
                    attn = torch.matmul(q_sub, k_sub.transpose(-2, -1)) * scale
                    attn = F.softmax(attn.float(), dim=-1)

                    # Store mean over batch: {head_idx: (S', S')}
                    attn_mean = attn.mean(dim=0).cpu()  # (H, S', S')
                    head_maps: Dict[int, torch.Tensor] = {}
                    for h in range(H):
                        head_maps[h] = attn_mean[h]

                    attention_maps[layer_id] = head_maps

                return hook_fn

            hook = module.register_forward_hook(_make_hook(lidx, max_tok))
            hooks.append(hook)

        # Forward pass
        model.eval()
        with torch.no_grad():
            if isinstance(sample_input, dict):
                model(**sample_input)
            elif isinstance(sample_input, (tuple, list)):
                model(*sample_input)
            else:
                model(sample_input)

        # Clean up hooks
        for hook in hooks:
            hook.remove()

        logger.info("Extracted attention maps from %d layers.", len(attention_maps))
        return attention_maps

    # ------------------------------------------------------------------
    # Sparsity analysis
    # ------------------------------------------------------------------

    def compute_sparsity(
        self,
        attention_weights: torch.Tensor,
        threshold: float = 0.01,
    ) -> float:
        """
        Compute the fraction of attention weights below the given threshold.

        High sparsity indicates the head is concentrating attention on a few
        tokens, making it a good candidate for sparse attention.

        Args:
            attention_weights: (S, S) attention weight matrix for a single head.
            threshold: Weight threshold below which a connection is considered "sparse".

        Returns:
            Fraction of weights below threshold (0.0 to 1.0).
        """
        total = attention_weights.numel()
        if total == 0:
            return 0.0
        sparse_count = (attention_weights < threshold).sum().item()
        return sparse_count / total

    # ------------------------------------------------------------------
    # Entropy analysis
    # ------------------------------------------------------------------

    def compute_entropy(
        self,
        attention_weights: torch.Tensor,
    ) -> float:
        """
        Compute the average per-query attention entropy for a single head.

        Low entropy means the head concentrates on few tokens (sharp attention).
        High entropy means attention is spread broadly (diffuse attention).

        Args:
            attention_weights: (S, S) attention weight matrix (rows = queries, cols = keys).
                Should be normalized (rows sum to 1).

        Returns:
            Average entropy across all query positions (in nats).
        """
        # Clamp to avoid log(0)
        attn = attention_weights.clamp(min=1e-12)

        # Per-query entropy: -sum(p * log(p))
        entropy_per_query = -(attn * attn.log()).sum(dim=-1)  # (S,)

        return entropy_per_query.mean().item()

    # ------------------------------------------------------------------
    # Spatial vs temporal classification
    # ------------------------------------------------------------------

    def classify_spatial_temporal(
        self,
        attention_weights: torch.Tensor,
        num_frames: Optional[int] = None,
        tokens_per_frame: Optional[int] = None,
        spatial_threshold: float = 0.7,
    ) -> Dict[str, Any]:
        """
        For a single head: determine if attention is mostly within-frame (spatial)
        or across-frame (temporal).

        Args:
            attention_weights: (S, S) attention weight matrix.
            num_frames: Number of frames. Defaults to self.num_frames.
            tokens_per_frame: Tokens per frame. Defaults to self.tokens_per_frame.
            spatial_threshold: Threshold for spatial classification.

        Returns:
            Dict with keys:
              - 'type': 'spatial', 'temporal', or 'mixed'
              - 'spatial_ratio': fraction of attention mass within same frame
              - 'temporal_ratio': fraction of attention mass across frames
              - 'entropy': average attention entropy
        """
        num_frames = num_frames or self.num_frames
        tokens_per_frame = tokens_per_frame or self.tokens_per_frame

        S = attention_weights.shape[0]
        expected_S = num_frames * tokens_per_frame

        # If captured on subsampled tokens, adjust tokens_per_frame
        if S != expected_S and S > 0:
            tokens_per_frame = S // num_frames
            if tokens_per_frame == 0:
                return {
                    "type": "mixed",
                    "spatial_ratio": 0.5,
                    "temporal_ratio": 0.5,
                    "entropy": self.compute_entropy(attention_weights),
                }

        # Build frame assignment for each token
        frame_ids = torch.arange(S) // tokens_per_frame
        frame_ids = frame_ids.clamp(max=num_frames - 1)

        # Spatial mask: same frame
        spatial_mask = frame_ids.unsqueeze(0) == frame_ids.unsqueeze(1)  # (S, S)

        spatial_mass = attention_weights[spatial_mask].sum().item()
        total_mass = attention_weights.sum().item()

        if total_mass < 1e-9:
            spatial_ratio = 0.5
        else:
            spatial_ratio = spatial_mass / total_mass

        temporal_ratio = 1.0 - spatial_ratio

        if spatial_ratio >= spatial_threshold:
            head_type = "spatial"
        elif temporal_ratio >= spatial_threshold:
            head_type = "temporal"
        else:
            head_type = "mixed"

        return {
            "type": head_type,
            "spatial_ratio": spatial_ratio,
            "temporal_ratio": temporal_ratio,
            "entropy": self.compute_entropy(attention_weights),
        }

    # ------------------------------------------------------------------
    # Full analysis pipeline
    # ------------------------------------------------------------------

    def analyze(
        self,
        attention_maps: Dict[int, Dict[int, torch.Tensor]],
        sparsity_threshold: float = 0.01,
    ) -> Dict[str, Any]:
        """
        Run full analysis on extracted attention maps.

        Args:
            attention_maps: Output of extract_attention_maps().
            sparsity_threshold: Threshold for sparsity computation.

        Returns:
            Comprehensive analysis dict with per-layer, per-head statistics.
        """
        results: Dict[str, Any] = {
            "num_layers": len(attention_maps),
            "layers": {},
            "summary": {},
        }

        all_sparsities: List[float] = []
        all_entropies: List[float] = []
        type_counts = {"spatial": 0, "temporal": 0, "mixed": 0}

        for layer_idx in sorted(attention_maps.keys()):
            head_maps = attention_maps[layer_idx]
            layer_result: Dict[str, Any] = {"heads": {}}

            for head_idx in sorted(head_maps.keys()):
                attn_w = head_maps[head_idx]

                sparsity = self.compute_sparsity(attn_w, sparsity_threshold)
                entropy = self.compute_entropy(attn_w)
                classification = self.classify_spatial_temporal(attn_w)

                head_result = {
                    "sparsity": sparsity,
                    "entropy": entropy,
                    "classification": classification,
                }
                layer_result["heads"][head_idx] = head_result

                all_sparsities.append(sparsity)
                all_entropies.append(entropy)
                type_counts[classification["type"]] += 1

            results["layers"][layer_idx] = layer_result

        total_heads = sum(type_counts.values())
        results["summary"] = {
            "total_heads_profiled": total_heads,
            "avg_sparsity": sum(all_sparsities) / len(all_sparsities) if all_sparsities else 0.0,
            "avg_entropy": sum(all_entropies) / len(all_entropies) if all_entropies else 0.0,
            "head_type_distribution": type_counts,
            "spatial_fraction": type_counts["spatial"] / total_heads if total_heads > 0 else 0.0,
            "temporal_fraction": type_counts["temporal"] / total_heads if total_heads > 0 else 0.0,
            "mixed_fraction": type_counts["mixed"] / total_heads if total_heads > 0 else 0.0,
        }

        return results

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(
        self,
        results: Dict[str, Any],
        save_path: str,
    ) -> str:
        """
        Save analysis results as a JSON report with heatmap data.

        Generates:
          - {save_path}/attention_report.json: Full analysis results
          - {save_path}/sparsity_by_layer.json: Per-layer sparsity stats
          - {save_path}/head_classification.json: Per-head type assignments

        Args:
            results: Output of analyze().
            save_path: Directory to save report files.

        Returns:
            Path to the main report JSON file.
        """
        os.makedirs(save_path, exist_ok=True)

        # --- Main report ---
        # Convert any non-serializable values
        report = _make_json_serializable(results)
        main_path = os.path.join(save_path, "attention_report.json")
        with open(main_path, "w") as f:
            json.dump(report, f, indent=2)

        # --- Sparsity by layer ---
        sparsity_data: Dict[str, Any] = {}
        for layer_idx, layer_data in results.get("layers", {}).items():
            layer_sparsities = []
            for head_idx, head_data in layer_data.get("heads", {}).items():
                layer_sparsities.append(head_data["sparsity"])
            sparsity_data[str(layer_idx)] = {
                "mean_sparsity": sum(layer_sparsities) / len(layer_sparsities) if layer_sparsities else 0.0,
                "min_sparsity": min(layer_sparsities) if layer_sparsities else 0.0,
                "max_sparsity": max(layer_sparsities) if layer_sparsities else 0.0,
                "per_head": layer_sparsities,
            }

        sparsity_path = os.path.join(save_path, "sparsity_by_layer.json")
        with open(sparsity_path, "w") as f:
            json.dump(sparsity_data, f, indent=2)

        # --- Head classification ---
        classification_data: Dict[str, Any] = {}
        for layer_idx, layer_data in results.get("layers", {}).items():
            layer_classifications = {}
            for head_idx, head_data in layer_data.get("heads", {}).items():
                cls = head_data.get("classification", {})
                layer_classifications[str(head_idx)] = {
                    "type": cls.get("type", "unknown"),
                    "spatial_ratio": cls.get("spatial_ratio", 0.0),
                    "temporal_ratio": cls.get("temporal_ratio", 0.0),
                }
            classification_data[str(layer_idx)] = layer_classifications

        classification_path = os.path.join(save_path, "head_classification.json")
        with open(classification_path, "w") as f:
            json.dump(classification_data, f, indent=2)

        # --- Heatmap data (sparsity matrix for visualization) ---
        heatmap_data: Dict[str, List[List[float]]] = {}
        for layer_idx, layer_data in results.get("layers", {}).items():
            row: List[float] = []
            for head_idx in sorted(layer_data.get("heads", {}).keys()):
                head_data = layer_data["heads"][head_idx]
                row.append(head_data["sparsity"])
            heatmap_data[str(layer_idx)] = row

        heatmap_path = os.path.join(save_path, "sparsity_heatmap.json")
        with open(heatmap_path, "w") as f:
            json.dump(heatmap_data, f, indent=2)

        logger.info(
            "Attention profiling report saved to %s (4 files).",
            save_path,
        )
        return main_path


def _make_json_serializable(obj: Any) -> Any:
    """Recursively convert non-JSON-serializable types."""
    if isinstance(obj, dict):
        return {str(k): _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(v) for v in obj]
    elif isinstance(obj, (torch.Tensor,)):
        return obj.tolist()
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return str(obj)
        return obj
    elif isinstance(obj, (int, str, bool, type(None))):
        return obj
    else:
        return str(obj)
