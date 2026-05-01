"""
Spatial block eviction for KV cache in Matrix-Game 2.0.

ADVANCED optimization (MG2 only).

Matrix-Game 2.0 generates ~15,360 spatial tokens per frame at 256x256 resolution,
with a rolling KV window of 6 frames.  Even with temporal merging and quantization,
the KV cache can be large.

This module groups spatial tokens into blocks (e.g., 4x4 patches = 16 tokens per
block) and evicts entire blocks from old frames based on their attention scores.
Blocks that consistently receive low attention are safe to drop.  "Attention sink"
blocks (e.g., borders, persistent objects) are always retained.

This is MG2-specific because Open-Oasis has only 144 spatial tokens per frame
(12x12), which is too small to benefit from spatial eviction.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


class SpatialBlockEvictor:
    """
    Evict low-attention spatial blocks from old KV cache frames to reduce
    memory footprint for Matrix-Game 2.0.
    """

    def __init__(
        self,
        block_size: int = 4,
        eviction_ratio: float = 0.3,
        min_age_frames: int = 10,
    ) -> None:
        """
        Args:
            block_size: Side length of spatial patch groups (block_size x block_size
                tokens per block).  For MG2 at ~15,360 tokens (roughly 124x124
                spatial grid), block_size=4 gives ~31x31 = 961 blocks.
            eviction_ratio: Fraction of blocks to evict in eligible (old) frames.
                Must be in (0, 1).
            min_age_frames: Only evict blocks from frames older than this many
                frames.  Protects recent context.
        """
        if not 0.0 < eviction_ratio < 1.0:
            raise ValueError(
                f"eviction_ratio must be in (0, 1), got {eviction_ratio}"
            )
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {block_size}")

        self.block_size = block_size
        self.eviction_ratio = eviction_ratio
        self.min_age_frames = min_age_frames

        # Running attention scores per block (for attention-sink detection)
        self._block_score_ema: Dict[int, torch.Tensor] = {}  # frame_idx -> (num_blocks,)
        self._ema_alpha: float = 0.3

    # ------------------------------------------------------------------
    # Spatial grouping
    # ------------------------------------------------------------------

    def group_into_blocks(
        self,
        kv_cache: torch.Tensor,
        H: int,
        W: int,
    ) -> Tuple[torch.Tensor, int, int]:
        """
        Reshape a flat sequence of spatial tokens into spatial blocks.

        Args:
            kv_cache: Tensor of shape ``(num_tokens, head_dim)`` or
                ``(num_heads, num_tokens, head_dim)`` for a single frame.
            H: Spatial height (in tokens).
            W: Spatial width (in tokens).

        Returns:
            A tuple ``(blocked, num_blocks_h, num_blocks_w)`` where:
              - *blocked* has shape ``(..., num_blocks_h, num_blocks_w,
                block_size, block_size, head_dim)``.
              - num_blocks_h, num_blocks_w are the block grid dimensions.
        """
        bs = self.block_size

        # Crop to multiple of block_size
        H_crop = (H // bs) * bs
        W_crop = (W // bs) * bs
        num_blocks_h = H_crop // bs
        num_blocks_w = W_crop // bs

        if kv_cache.dim() == 2:
            # (N, D) -> (H, W, D)
            spatial = kv_cache[: H_crop * W_crop].reshape(H_crop, W_crop, -1)
            # -> (nbh, bs, nbw, bs, D) -> (nbh, nbw, bs, bs, D)
            blocked = spatial.reshape(num_blocks_h, bs, num_blocks_w, bs, -1)
            blocked = blocked.permute(0, 2, 1, 3, 4)
        elif kv_cache.dim() == 3:
            # (num_heads, N, D)
            num_heads = kv_cache.shape[0]
            D = kv_cache.shape[2]
            spatial = kv_cache[:, : H_crop * W_crop, :].reshape(
                num_heads, H_crop, W_crop, D
            )
            blocked = spatial.reshape(num_heads, num_blocks_h, bs, num_blocks_w, bs, D)
            blocked = blocked.permute(0, 1, 3, 2, 4, 5)
            # (num_heads, nbh, nbw, bs, bs, D)
        else:
            raise ValueError(f"Expected 2D or 3D kv_cache, got {kv_cache.dim()}D")

        return blocked, num_blocks_h, num_blocks_w

    def compute_block_scores(
        self,
        attention_weights: torch.Tensor,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """
        Compute average attention score per spatial block.

        Args:
            attention_weights: Shape ``(batch, num_heads, seq_q, seq_kv)``
                where seq_kv includes spatial tokens for one frame.
            H: Spatial height of the frame (in tokens).
            W: Spatial width.

        Returns:
            Block scores of shape ``(num_blocks_h * num_blocks_w,)``.
        """
        bs = self.block_size
        H_crop = (H // bs) * bs
        W_crop = (W // bs) * bs
        num_blocks_h = H_crop // bs
        num_blocks_w = W_crop // bs

        # Mean attention received per KV position (average over batch, heads, queries)
        per_token_score = attention_weights.float().mean(dim=(0, 1, 2))  # (seq_kv,)

        # Take only the spatial tokens for this frame
        spatial_scores = per_token_score[: H_crop * W_crop]
        spatial_scores = spatial_scores.reshape(H_crop, W_crop)

        # Average within each block
        # Reshape into blocks
        block_scores = spatial_scores.reshape(
            num_blocks_h, bs, num_blocks_w, bs
        ).permute(0, 2, 1, 3)  # (nbh, nbw, bs, bs)
        block_scores = block_scores.reshape(num_blocks_h * num_blocks_w, bs * bs)
        block_scores = block_scores.mean(dim=1)  # (num_blocks,)

        return block_scores

    def evict(
        self,
        kv_cache: torch.Tensor,
        frame_ages: torch.Tensor,
        block_scores: torch.Tensor,
        H: int,
        W: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evict low-scoring spatial blocks from old frames.

        Args:
            kv_cache: Full KV tensor for one component (K or V) of one layer,
                shape ``(num_frames, num_heads, num_tokens, head_dim)``
                or ``(num_frames, num_tokens, head_dim)``.
            frame_ages: Per-frame ages, shape ``(num_frames,)``.
            block_scores: Per-block attention scores for each frame,
                shape ``(num_frames, num_blocks)`` where num_blocks corresponds
                to the spatial grid.
            H: Spatial height in tokens.
            W: Spatial width in tokens.

        Returns:
            Tuple ``(evicted_cache, eviction_mask)`` where:
              - *evicted_cache*: same shape as input with evicted tokens zeroed.
              - *eviction_mask*: bool tensor, True where tokens were evicted.
        """
        bs = self.block_size
        H_crop = (H // bs) * bs
        W_crop = (W // bs) * bs
        num_blocks_h = H_crop // bs
        num_blocks_w = W_crop // bs
        num_blocks = num_blocks_h * num_blocks_w
        num_tokens_cropped = H_crop * W_crop

        num_frames = kv_cache.shape[0]
        result = kv_cache.clone()
        device = kv_cache.device

        # Build eviction mask (per-token)
        if kv_cache.dim() == 4:
            num_tokens = kv_cache.shape[2]
        else:
            num_tokens = kv_cache.shape[1]

        eviction_mask = torch.zeros(num_frames, num_tokens, dtype=torch.bool, device=device)

        for f in range(num_frames):
            age_val = frame_ages[f]
            age = age_val.item() if isinstance(age_val, torch.Tensor) else int(age_val)
            if age < self.min_age_frames:
                continue

            scores = block_scores[f] if block_scores.dim() == 2 else block_scores
            if scores.shape[0] != num_blocks:
                continue

            # Use scores directly (caller has already updated EMA via
            # update_frame_scores with key (layer_idx, frame_idx)).
            # Do NOT maintain a separate EMA here with key `f` — that would
            # use a different key namespace and lose all prior history.
            ema_scores = scores

            # Identify attention sinks: top 10% by EMA are protected
            sink_threshold = torch.quantile(ema_scores, 0.9)
            is_sink = ema_scores >= sink_threshold

            # Among non-sink blocks, evict bottom eviction_ratio
            evictable_mask = ~is_sink
            evictable_indices = evictable_mask.nonzero(as_tuple=True)[0]

            if evictable_indices.numel() == 0:
                continue

            evictable_scores = scores[evictable_indices]
            num_to_evict = max(1, int(evictable_indices.numel() * self.eviction_ratio))

            # Find the lowest-scoring blocks
            _, lowest_indices = evictable_scores.topk(
                min(num_to_evict, evictable_scores.numel()), largest=False
            )
            blocks_to_evict = evictable_indices[lowest_indices]

            # Map block indices back to token indices
            for block_idx in blocks_to_evict:
                block_idx = block_idx.item()
                bh = block_idx // num_blocks_w
                bw = block_idx % num_blocks_w

                for dh in range(bs):
                    for dw in range(bs):
                        row = bh * bs + dh
                        col = bw * bs + dw
                        token_idx = row * W_crop + col
                        if token_idx < num_tokens:
                            eviction_mask[f, token_idx] = True

            # Zero out evicted tokens
            if kv_cache.dim() == 4:
                mask_expanded = eviction_mask[f].unsqueeze(0).unsqueeze(-1)
                mask_expanded = mask_expanded.expand_as(result[f])
                result[f] = result[f].masked_fill(mask_expanded, 0.0)
            else:
                mask_expanded = eviction_mask[f].unsqueeze(-1).expand_as(result[f])
                result[f] = result[f].masked_fill(mask_expanded, 0.0)

        return result, eviction_mask

    # ------------------------------------------------------------------
    # Per-frame accumulated scores
    # ------------------------------------------------------------------

    def update_frame_scores(
        self,
        layer_idx: int,
        frame_idx: int,
        block_scores: torch.Tensor,
    ) -> None:
        """Update EMA block scores for a specific frame."""
        key = (layer_idx, frame_idx)
        if key in self._block_score_ema:
            self._block_score_ema[key] = (
                self._ema_alpha * block_scores
                + (1 - self._ema_alpha) * self._block_score_ema[key].to(block_scores.device)
            )
        else:
            self._block_score_ema[key] = block_scores.clone()

    def get_accumulated_scores(
        self,
        layer_idx: int,
        num_frames: int,
        num_blocks: int,
    ) -> torch.Tensor:
        """
        Return per-frame accumulated block scores.

        Returns:
            (num_frames, num_blocks) tensor of EMA scores. Frames without
            history get uniform scores (no eviction bias).
        """
        device = None
        for k, v in self._block_score_ema.items():
            device = v.device
            break

        scores = torch.ones(num_frames, num_blocks, device=device) * 0.5
        for f in range(num_frames):
            key = (layer_idx, f)
            if key in self._block_score_ema:
                s = self._block_score_ema[key]
                if s.shape[0] == num_blocks:
                    scores[f] = s.to(device)
        return scores

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_eviction_stats(
        self,
        eviction_mask: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Compute eviction statistics from a mask.

        Args:
            eviction_mask: Boolean mask from ``evict()``.

        Returns:
            Dict with per-frame and overall eviction rates, memory savings.
        """
        num_frames = eviction_mask.shape[0]
        per_frame_rates = []
        for f in range(num_frames):
            total = eviction_mask[f].numel()
            evicted = eviction_mask[f].sum().item()
            per_frame_rates.append(evicted / max(total, 1))

        overall = eviction_mask.sum().item() / max(eviction_mask.numel(), 1)

        return {
            "overall_eviction_rate": overall,
            "per_frame_rates": per_frame_rates,
            "memory_savings_ratio": overall,
        }
