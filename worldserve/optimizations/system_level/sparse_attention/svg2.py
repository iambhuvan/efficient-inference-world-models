"""
SVG2 sparse attention integration for WorldServe (Phase 4 — PRIMARY).

Implements head-aware spatial/temporal sparse attention following the SVG2
methodology. Each attention head is classified as 'spatial' (attends within
the same frame) or 'temporal' (attends across frames at the same spatial
position). Sparse masks are applied per-head to skip irrelevant token pairs,
dramatically reducing FLOPs for MG2's 15,360 spatial tokens/frame.

Additionally supports semantic token permutation via k-means clustering, which
reorders tokens so semantically similar tokens are adjacent, making block-sparse
patterns more effective.

Architecture assumptions (Matrix-Game 2.0):
  - ~15,360 spatial tokens per frame
  - 30 DiT blocks (Wan2.1 backbone)
  - Rolling KV window of 6 frames
  - 3 denoising steps
  - Keyboard cross-attention + mouse concatenation
"""

from __future__ import annotations

import logging
import math
import types
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Try to import flash_attn for optimal spatial head attention
try:
    from flash_attn import flash_attn_func as _svg2_fa3_func
    _SVG2_FA3 = True
except ImportError:
    _SVG2_FA3 = False


# ---------------------------------------------------------------------------
# Head classification types
# ---------------------------------------------------------------------------
HEAD_TYPE_SPATIAL = "spatial"
HEAD_TYPE_TEMPORAL = "temporal"
HEAD_TYPE_MIXED = "mixed"  # fallback: no sparse mask applied


@dataclass
class HeadClassification:
    """Per-head classification for a single attention layer."""
    layer_idx: int
    head_types: Dict[int, str]  # {head_idx: 'spatial' | 'temporal' | 'mixed'}
    spatial_ratios: Dict[int, float]  # {head_idx: ratio of spatial attention mass}

    def spatial_heads(self) -> List[int]:
        return [h for h, t in self.head_types.items() if t == HEAD_TYPE_SPATIAL]

    def temporal_heads(self) -> List[int]:
        return [h for h, t in self.head_types.items() if t == HEAD_TYPE_TEMPORAL]


@dataclass
class FLOPSTracker:
    """Tracks FLOPs reduction from sparse attention."""
    dense_flops: int = 0
    sparse_flops: int = 0
    per_layer: Dict[int, Dict[str, int]] = field(default_factory=dict)

    @property
    def reduction_ratio(self) -> float:
        if self.dense_flops == 0:
            return 0.0
        return 1.0 - (self.sparse_flops / self.dense_flops)

    def record(self, layer_idx: int, dense: int, sparse: int) -> None:
        self.dense_flops += dense
        self.sparse_flops += sparse
        self.per_layer[layer_idx] = {"dense": dense, "sparse": sparse}


class SVG2SparseAttention:
    """
    SVG2-style head-aware sparse attention for Matrix-Game 2.0.

    Classifies each attention head as spatial or temporal, then applies
    block-diagonal (spatial) or strided (temporal) sparse masks. Optionally
    applies k-means semantic permutation for better block sparsity.
    """

    # MG2 defaults
    DEFAULT_TOKENS_PER_FRAME = 15360
    DEFAULT_NUM_FRAMES = 6
    DEFAULT_NUM_LAYERS = 30

    def __init__(
        self,
        num_heads: int = 24,
        num_layers: int = 30,
        n_clusters: int = 8,
        spatial_threshold: float = 0.7,
        temporal_threshold: float = 0.7,
        tokens_per_frame: int = 15360,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            num_heads: Number of attention heads per layer.
            num_layers: Number of DiT blocks (30 for MG2).
            n_clusters: Number of k-means clusters for semantic token permutation.
            spatial_threshold: If spatial attention mass ratio > this, classify as spatial.
            temporal_threshold: If temporal attention mass ratio > this, classify as temporal.
            tokens_per_frame: Spatial tokens per frame (15,360 for MG2).
        """
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.n_clusters = n_clusters
        self.spatial_threshold = spatial_threshold
        self.temporal_threshold = temporal_threshold
        self.tokens_per_frame = tokens_per_frame

        self.flops_tracker = FLOPSTracker()
        self._head_classifications: Dict[int, HeadClassification] = {}
        self._patched_modules: List[Tuple[nn.Module, Callable]] = []
        self._mask_cache: Dict[Tuple[str, int, int], torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Head classification
    # ------------------------------------------------------------------

    def classify_heads(
        self,
        model: nn.Module,
        sample_input: Union[torch.Tensor, Dict[str, Any]],
        num_frames: Optional[int] = None,
        tokens_per_frame: Optional[int] = None,
    ) -> Dict[int, HeadClassification]:
        """
        Run model on sample input, extract attention patterns, and classify
        each head as spatial or temporal using the SVG2 methodology.

        Computes the ratio of within-frame (spatial) vs across-frame (temporal)
        attention mass for each head. Heads with dominant spatial mass are
        classified as 'spatial'; dominant temporal mass as 'temporal'; otherwise
        'mixed' (no sparsity applied).

        Args:
            model: The MG2 DiT model.
            sample_input: A sample input tensor or dict to run through the model.
            num_frames: Number of frames in the input (defaults to DEFAULT_NUM_FRAMES).
            tokens_per_frame: Spatial tokens per frame (defaults to self.tokens_per_frame).

        Returns:
            Dict mapping layer_idx -> HeadClassification.
        """
        num_frames = num_frames or self.DEFAULT_NUM_FRAMES
        tokens_per_frame = tokens_per_frame or self.tokens_per_frame

        # Hook into attention layers to capture attention weight matrices
        attention_weights: Dict[int, torch.Tensor] = {}
        hooks: List[torch.utils.hooks.RemovableHook] = []

        layer_idx = 0
        for name, module in model.named_modules():
            cls_name = type(module).__name__
            if cls_name in ("Attention", "WanAttention", "FlashSelfAttention",
                            "SelfAttention", "MultiheadAttention"):
                hook = self._register_attention_hook(module, layer_idx, attention_weights)
                hooks.append(hook)
                layer_idx += 1

        # Forward pass to collect attention patterns
        model.eval()
        with torch.no_grad():
            if isinstance(sample_input, dict):
                model(**sample_input)
            elif isinstance(sample_input, (tuple, list)):
                model(*sample_input)
            else:
                model(sample_input)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Classify each head
        classifications: Dict[int, HeadClassification] = {}
        for lidx, attn_map in attention_weights.items():
            classification = self._classify_layer_heads(
                attn_map, lidx, num_frames, tokens_per_frame
            )
            classifications[lidx] = classification

        self._head_classifications = classifications
        logger.info(
            "Classified %d layers. Spatial heads: %d, Temporal: %d, Mixed: %d",
            len(classifications),
            sum(len(c.spatial_heads()) for c in classifications.values()),
            sum(len(c.temporal_heads()) for c in classifications.values()),
            sum(
                sum(1 for t in c.head_types.values() if t == HEAD_TYPE_MIXED)
                for c in classifications.values()
            ),
        )
        return classifications

    def _register_attention_hook(
        self,
        module: nn.Module,
        layer_idx: int,
        storage: Dict[int, torch.Tensor],
    ) -> torch.utils.hooks.RemovableHook:
        """Register a forward hook that captures attention weights."""

        def hook_fn(
            mod: nn.Module,
            inputs: Tuple[Any, ...],
            output: Any,
        ) -> None:
            # Try to intercept QKV before softmax by temporarily patching
            # scaled_dot_product_attention. This hook stores a post-softmax
            # approximation computed from Q and K.
            #
            # For a proper capture, we look for q, k on the module or inputs.
            # If the module stores q/k as attributes during forward, use those.
            q = getattr(mod, "_last_q", None)
            k = getattr(mod, "_last_k", None)
            # Fallback: try to extract Q, K from hook inputs.
            # Many attention modules pass (q, k, v) as positional args.
            if q is None and isinstance(inputs, tuple) and len(inputs) >= 2:
                candidate_q, candidate_k = inputs[0], inputs[1]
                if isinstance(candidate_q, torch.Tensor) and candidate_q.dim() == 4:
                    q, k = candidate_q, candidate_k
            if q is not None and k is not None:
                # q, k: (B, H, S, D) or (B, S, H, D)
                if q.dim() == 4:
                    if q.shape[1] == self.num_heads:
                        # (B, H, S, D) format
                        scale = q.shape[-1] ** -0.5
                        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
                    else:
                        # (B, S, H, D) format -> transpose to (B, H, S, D)
                        q_t = q.transpose(1, 2)
                        k_t = k.transpose(1, 2)
                        scale = q_t.shape[-1] ** -0.5
                        attn = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale
                    # Softmax to get attention weights
                    attn = F.softmax(attn.float(), dim=-1)
                    # Store mean over batch: (H, S, S)
                    storage[layer_idx] = attn.mean(dim=0).cpu()

        return module.register_forward_hook(hook_fn)

    def _classify_layer_heads(
        self,
        attn_weights: torch.Tensor,
        layer_idx: int,
        num_frames: int,
        tokens_per_frame: int,
    ) -> HeadClassification:
        """
        Classify heads in a single layer based on attention mass distribution.

        Args:
            attn_weights: (H, S, S) attention weights for this layer.
            layer_idx: Index of this layer.
            num_frames: Number of frames.
            tokens_per_frame: Spatial tokens per frame.

        Returns:
            HeadClassification for this layer.
        """
        num_heads_actual = attn_weights.shape[0]
        seq_len = attn_weights.shape[1]
        expected_seq_len = num_frames * tokens_per_frame

        head_types: Dict[int, str] = {}
        spatial_ratios: Dict[int, float] = {}

        for h in range(num_heads_actual):
            attn_h = attn_weights[h]  # (S, S)

            # Build spatial mask: True where query and key are in the same frame
            if seq_len == expected_seq_len:
                # Dynamic fallback: DC-DiT may have compressed tokens_per_frame.
                # Infer from actual sequence length if the stored value doesn't divide evenly.
                eff_tokens_per_frame = self.tokens_per_frame
                if seq_len % eff_tokens_per_frame != 0:
                    eff_tokens_per_frame = max(1, seq_len // max(self.num_layers, 1))
                    logger.debug(
                        "[SVG2] tokens_per_frame mismatch (seq_len=%d, stored=%d); "
                        "inferred eff_tokens_per_frame=%d",
                        seq_len, self.tokens_per_frame, eff_tokens_per_frame,
                    )
                frame_ids = torch.arange(seq_len) // eff_tokens_per_frame
                spatial_mask = frame_ids.unsqueeze(0) == frame_ids.unsqueeze(1)
            else:
                # Fallback: assume tokens are grouped by frame in order
                # Use best-guess tokens_per_frame
                tpf_guess = seq_len // max(num_frames, 1)
                if tpf_guess > 0:
                    frame_ids = torch.arange(seq_len) // tpf_guess
                    spatial_mask = frame_ids.unsqueeze(0) == frame_ids.unsqueeze(1)
                else:
                    # Cannot classify; mark all as mixed
                    head_types[h] = HEAD_TYPE_MIXED
                    spatial_ratios[h] = 0.5
                    continue

            # Compute spatial vs temporal attention mass
            spatial_mass = attn_h[spatial_mask].sum().item()
            total_mass = attn_h.sum().item()

            if total_mass < 1e-9:
                head_types[h] = HEAD_TYPE_MIXED
                spatial_ratios[h] = 0.5
                continue

            ratio = spatial_mass / total_mass
            spatial_ratios[h] = ratio

            if ratio >= self.spatial_threshold:
                head_types[h] = HEAD_TYPE_SPATIAL
            elif (1.0 - ratio) >= self.temporal_threshold:
                head_types[h] = HEAD_TYPE_TEMPORAL
            else:
                head_types[h] = HEAD_TYPE_MIXED

        return HeadClassification(
            layer_idx=layer_idx,
            head_types=head_types,
            spatial_ratios=spatial_ratios,
        )

    # ------------------------------------------------------------------
    # Sparse mask computation
    # ------------------------------------------------------------------

    def compute_sparse_mask(
        self,
        head_type: str,
        seq_len: int,
        num_frames: int,
        tokens_per_frame: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Compute a sparse attention mask for a given head type.

        Args:
            head_type: 'spatial' or 'temporal'.
            seq_len: Total sequence length (num_frames * tokens_per_frame).
            num_frames: Number of frames in the sequence.
            tokens_per_frame: Tokens per frame (inferred from seq_len / num_frames if None).
            device: Target device for the mask tensor.

        Returns:
            Boolean mask of shape (seq_len, seq_len). True = attend, False = mask out.
        """
        cache_key = (head_type, seq_len, num_frames)
        if cache_key in self._mask_cache:
            mask = self._mask_cache[cache_key]
            if device is not None:
                mask = mask.to(device)
            return mask

        tokens_per_frame = tokens_per_frame or (seq_len // num_frames)

        if head_type == HEAD_TYPE_SPATIAL:
            # Block-diagonal: each frame attends only to itself
            mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
            for f in range(num_frames):
                start = f * tokens_per_frame
                end = start + tokens_per_frame
                mask[start:end, start:end] = True

        elif head_type == HEAD_TYPE_TEMPORAL:
            # Temporal: attend only to same spatial position across frames
            mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
            for s in range(tokens_per_frame):
                # Indices of this spatial position across all frames
                indices = [f * tokens_per_frame + s for f in range(num_frames)]
                for i in indices:
                    for j in indices:
                        if i < seq_len and j < seq_len:
                            mask[i, j] = True

        else:
            # Mixed: full attention (no mask)
            mask = torch.ones(seq_len, seq_len, dtype=torch.bool)

        self._mask_cache[cache_key] = mask
        if device is not None:
            mask = mask.to(device)
        return mask

    # ------------------------------------------------------------------
    # Semantic permutation via k-means
    # ------------------------------------------------------------------

    def semantic_permute(
        self,
        tokens: torch.Tensor,
        n_clusters: Optional[int] = None,
        max_iter: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        K-means cluster tokens by semantic similarity and reorder so similar
        tokens are adjacent. This makes block-sparse patterns more effective.

        Args:
            tokens: (B, S, D) token embeddings.
            n_clusters: Number of clusters (defaults to self.n_clusters).
            max_iter: Maximum k-means iterations.

        Returns:
            Tuple of:
              - permuted_tokens: (B, S, D) reordered tokens.
              - inverse_perm: (B, S) index tensor to reconstruct original order.
                Use: original = permuted.gather(1, inverse_perm.unsqueeze(-1).expand_as(permuted))
        """
        n_clusters = n_clusters or self.n_clusters
        B, S, D = tokens.shape
        device = tokens.device

        # Normalize for cosine-like clustering
        tokens_norm = F.normalize(tokens.float(), dim=-1)

        # Simple k-means on each batch element
        permuted_list = []
        inverse_perm_list = []

        for b in range(B):
            x = tokens_norm[b]  # (S, D)

            # Initialize centroids with k-means++ style: evenly spaced indices
            step = max(S // n_clusters, 1)
            centroid_indices = torch.arange(0, min(n_clusters * step, S), step, device=device)
            if len(centroid_indices) < n_clusters:
                # Pad with random indices
                extra = torch.randint(0, S, (n_clusters - len(centroid_indices),), device=device)
                centroid_indices = torch.cat([centroid_indices, extra])
            centroids = x[centroid_indices[:n_clusters]].clone()  # (K, D)

            # K-means iterations
            assignments = torch.zeros(S, dtype=torch.long, device=device)
            for _ in range(max_iter):
                # Assign each token to nearest centroid
                # (S, D) @ (D, K) -> (S, K)
                sims = torch.matmul(x, centroids.t())
                new_assignments = sims.argmax(dim=-1)

                if torch.equal(new_assignments, assignments):
                    break
                assignments = new_assignments

                # Update centroids
                for k in range(n_clusters):
                    members = x[assignments == k]
                    if members.shape[0] > 0:
                        centroids[k] = F.normalize(members.mean(dim=0, keepdim=True), dim=-1).squeeze(0)

            # Build permutation: sort by cluster assignment, then by original index within cluster
            # This groups similar tokens together
            sort_keys = assignments.float() * S + torch.arange(S, dtype=torch.float, device=device)
            perm = sort_keys.argsort()

            # Inverse permutation
            inv_perm = torch.empty_like(perm)
            inv_perm[perm] = torch.arange(S, device=device)

            permuted_list.append(tokens[b][perm])
            inverse_perm_list.append(inv_perm)

        permuted_tokens = torch.stack(permuted_list, dim=0)
        inverse_perm = torch.stack(inverse_perm_list, dim=0)

        return permuted_tokens, inverse_perm

    # ------------------------------------------------------------------
    # Sparse attention forward
    # ------------------------------------------------------------------

    def apply_sparse_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        head_classification: HeadClassification,
        layer_idx: int,
        num_frames: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Apply per-head sparse attention using classification-based masks.

        For each head, applies the appropriate sparse mask (spatial block-diagonal,
        temporal strided, or full for mixed heads). Uses block-sparse matmul where
        possible for efficiency.

        Args:
            q: (B, H, S, D) query tensor.
            k: (B, H, S, D) key tensor.
            v: (B, H, S, D) value tensor.
            head_classification: HeadClassification for this layer.
            layer_idx: Index of the current layer.
            num_frames: Number of frames (defaults to DEFAULT_NUM_FRAMES).

        Returns:
            (B, H, S, D) output tensor.
        """
        num_frames = num_frames or self.DEFAULT_NUM_FRAMES
        B, H, S, D = q.shape
        device = q.device
        tokens_per_frame = S // num_frames if num_frames > 0 else S

        output = torch.zeros_like(q)

        # Group heads by type for batched processing
        spatial_heads = head_classification.spatial_heads()
        temporal_heads = head_classification.temporal_heads()
        mixed_heads = [
            h for h in range(H)
            if h not in spatial_heads and h not in temporal_heads
        ]

        # Dense FLOPs estimate: 2 * B * H * S * S * D (QK^T + AV)
        dense_flops = 2 * B * H * S * S * D

        sparse_flops = 0

        # --- Spatial heads: block-diagonal attention (per-frame, via FA3 if available) ---
        if spatial_heads:
            h_idx = torch.tensor(spatial_heads, device=device)
            q_sp = q[:, h_idx]  # (B, H_sp, S, D)
            k_sp = k[:, h_idx]
            v_sp = v[:, h_idx]

            H_sp = len(spatial_heads)
            # Reshape to batch-of-frames: (B*F, S_tpf, H_sp, D)
            q_sp_f = q_sp.reshape(B, H_sp, num_frames, tokens_per_frame, D)
            k_sp_f = k_sp.reshape(B, H_sp, num_frames, tokens_per_frame, D)
            v_sp_f = v_sp.reshape(B, H_sp, num_frames, tokens_per_frame, D)
            # (B, H_sp, F, tpf, D) → (B*F, H_sp, tpf, D) → (B*F, tpf, H_sp, D) for FA3
            q_sp_f = q_sp_f.permute(0, 2, 1, 3, 4).reshape(B * num_frames, H_sp, tokens_per_frame, D)
            k_sp_f = k_sp_f.permute(0, 2, 1, 3, 4).reshape(B * num_frames, H_sp, tokens_per_frame, D)
            v_sp_f = v_sp_f.permute(0, 2, 1, 3, 4).reshape(B * num_frames, H_sp, tokens_per_frame, D)

            _sp_scale = D ** -0.5
            if _SVG2_FA3:
                # FA3 layout: (batch, seqlen, nheads, headdim)
                q_fa = q_sp_f.transpose(1, 2).contiguous()  # (B*F, tpf, H_sp, D)
                k_fa = k_sp_f.transpose(1, 2).contiguous()
                v_fa = v_sp_f.transpose(1, 2).contiguous()
                # Cast to fp16/bf16 if needed (FA3 requirement)
                _sp_dtype = q_fa.dtype
                if _sp_dtype not in (torch.float16, torch.bfloat16):
                    q_fa = q_fa.to(torch.bfloat16)
                    k_fa = k_fa.to(torch.bfloat16)
                    v_fa = v_fa.to(torch.bfloat16)
                out_sp_f = _svg2_fa3_func(q_fa, k_fa, v_fa, softmax_scale=_sp_scale, causal=False)
                out_sp_f = out_sp_f.to(_sp_dtype).transpose(1, 2)  # (B*F, H_sp, tpf, D)
            else:
                attn = torch.matmul(q_sp_f, k_sp_f.transpose(-2, -1)) * _sp_scale
                attn = F.softmax(attn, dim=-1)
                out_sp_f = torch.matmul(attn, v_sp_f)  # (B*F, H_sp, tpf, D)

            # (B*F, H_sp, tpf, D) → (B, F, H_sp, tpf, D) → (B, H_sp, S, D)
            out_sp = out_sp_f.reshape(B, num_frames, H_sp, tokens_per_frame, D)
            out_sp = out_sp.permute(0, 2, 1, 3, 4).reshape(B, H_sp, S, D)

            for i, h in enumerate(spatial_heads):
                output[:, h] = out_sp[:, i]

            sparse_flops += 2 * B * H_sp * num_frames * tokens_per_frame * tokens_per_frame * D

        # --- Temporal heads: same-position-across-frames attention ---
        if temporal_heads:
            h_idx = torch.tensor(temporal_heads, device=device)
            q_tp = q[:, h_idx]  # (B, H_tp, S, D)
            k_tp = k[:, h_idx]
            v_tp = v[:, h_idx]

            H_tp = len(temporal_heads)
            # Reshape: (B, H_tp, F, tpf, D) -> transpose to (B, H_tp, tpf, F, D)
            q_tp = q_tp.reshape(B, H_tp, num_frames, tokens_per_frame, D).permute(0, 1, 3, 2, 4)
            k_tp = k_tp.reshape(B, H_tp, num_frames, tokens_per_frame, D).permute(0, 1, 3, 2, 4)
            v_tp = v_tp.reshape(B, H_tp, num_frames, tokens_per_frame, D).permute(0, 1, 3, 2, 4)

            # Now each spatial position attends across frames: (B, H_tp, tpf, F, D)
            scale = D ** -0.5
            attn = torch.matmul(q_tp, k_tp.transpose(-2, -1)) * scale  # (B, H_tp, tpf, F, F)
            attn = F.softmax(attn, dim=-1)
            out_tp = torch.matmul(attn, v_tp)  # (B, H_tp, tpf, F, D)

            # Permute back: (B, H_tp, tpf, F, D) -> (B, H_tp, F, tpf, D) -> (B, H_tp, S, D)
            out_tp = out_tp.permute(0, 1, 3, 2, 4).reshape(B, H_tp, S, D)

            for i, h in enumerate(temporal_heads):
                output[:, h] = out_tp[:, i]

            # Temporal FLOPs: each spatial position attends across F frames
            sparse_flops += 2 * B * H_tp * tokens_per_frame * num_frames * num_frames * D

        # --- Mixed heads: full dense attention ---
        if mixed_heads:
            h_idx = torch.tensor(mixed_heads, device=device)
            q_mx = q[:, h_idx]
            k_mx = k[:, h_idx]
            v_mx = v[:, h_idx]

            H_mx = len(mixed_heads)
            scale = D ** -0.5
            attn = torch.matmul(q_mx, k_mx.transpose(-2, -1)) * scale
            attn = F.softmax(attn, dim=-1)
            out_mx = torch.matmul(attn, v_mx)

            for i, h in enumerate(mixed_heads):
                output[:, h] = out_mx[:, i]

            sparse_flops += 2 * B * H_mx * S * S * D

        # Track FLOPs
        self.flops_tracker.record(layer_idx, dense_flops, sparse_flops)

        return output

    # ------------------------------------------------------------------
    # Model wrapping
    # ------------------------------------------------------------------

    def wrap_model(
        self,
        model: nn.Module,
        head_classification: Optional[Dict[int, HeadClassification]] = None,
        num_frames: Optional[int] = None,
    ) -> nn.Module:
        """
        Replace self-attention forward in each DiT block with sparse attention.
        Leaves cross-attention (keyboard actions) untouched.

        Args:
            model: The MG2 DiT model.
            head_classification: Per-layer head classifications. If None,
                uses a default all-spatial classification (useful for testing).
            num_frames: Number of frames (defaults to DEFAULT_NUM_FRAMES).

        Returns:
            The model with patched attention (modified in-place).
        """
        num_frames = num_frames or self.DEFAULT_NUM_FRAMES
        self._patched_modules.clear()

        layer_idx = 0
        for name, module in model.named_modules():
            cls_name = type(module).__name__

            # Only patch self-attention modules, skip cross-attention
            # Cross-attention modules typically have "cross" in the name or
            # have separate q projection from kv projection source
            is_self_attn = cls_name in (
                "Attention", "WanAttention", "FlashSelfAttention",
                "SelfAttention",
            ) and "cross" not in name.lower()

            if not is_self_attn:
                continue

            # Get or create classification for this layer
            if head_classification is not None and layer_idx in head_classification:
                layer_class = head_classification[layer_idx]
            else:
                # Default: assume all spatial (most conservative sparse pattern)
                layer_class = HeadClassification(
                    layer_idx=layer_idx,
                    head_types={h: HEAD_TYPE_SPATIAL for h in range(self.num_heads)},
                    spatial_ratios={h: 1.0 for h in range(self.num_heads)},
                )

            original_forward = module.forward
            self._patched_modules.append((module, original_forward))

            # Create the patched forward
            sparse_attn = self
            lclass = layer_class
            lidx = layer_idx
            nf = num_frames

            def _make_sparse_forward(
                orig_fwd: Callable,
                sa: SVG2SparseAttention,
                lc: HeadClassification,
                li: int,
                nframes: int,
            ) -> Callable:
                """Factory to capture variables in closure properly."""

                def _sparse_forward(self_module: Any, *args: Any, **kwargs: Any) -> Any:
                    # Intercept scaled_dot_product_attention calls
                    original_sdpa = F.scaled_dot_product_attention

                    def _sparse_sdpa(
                        query: torch.Tensor,
                        key: torch.Tensor,
                        value: torch.Tensor,
                        attn_mask: Optional[torch.Tensor] = None,
                        dropout_p: float = 0.0,
                        is_causal: bool = False,
                        scale: Optional[float] = None,
                        **kw: Any,
                    ) -> torch.Tensor:
                        # query shape: (B, H, S, D)
                        return sa.apply_sparse_attention(
                            query, key, value, lc, li, nframes
                        )

                    F.scaled_dot_product_attention = _sparse_sdpa  # type: ignore[assignment]
                    try:
                        result = orig_fwd(*args, **kwargs)
                    finally:
                        F.scaled_dot_product_attention = original_sdpa  # type: ignore[assignment]
                    return result

                return _sparse_forward

            patched_forward = _make_sparse_forward(original_forward, sparse_attn, lclass, lidx, nf)
            module.forward = types.MethodType(patched_forward, module)

            layer_idx += 1

        logger.info(
            "SVG2: wrapped %d self-attention layers with sparse attention",
            layer_idx,
        )
        return model

    def restore_all(self) -> int:
        """Restore all patched modules to original forward methods."""
        count = 0
        for module, original_forward in self._patched_modules:
            module.forward = original_forward
            count += 1
        self._patched_modules.clear()
        self._mask_cache.clear()
        logger.info("SVG2: restored %d attention modules to original forward.", count)
        return count

    def get_flops_report(self) -> Dict[str, Any]:
        """Return FLOPs reduction statistics."""
        return {
            "total_dense_flops": self.flops_tracker.dense_flops,
            "total_sparse_flops": self.flops_tracker.sparse_flops,
            "reduction_ratio": self.flops_tracker.reduction_ratio,
            "per_layer": self.flops_tracker.per_layer,
        }
