"""
QVG (Quantization via Grouping) semantic smoothing for KV cache compression.

ADVANCED optimization.

Idea: raw KV vectors are hard to quantize to very low bit-widths because their
dynamic range is large and entries are heterogeneous.  QVG first clusters KV
tokens by latent similarity (k-means), then stores:
  1. Centroids at full precision (small: n_clusters * head_dim).
  2. Per-token cluster assignments (log2(n_clusters) bits each).
  3. Residuals (token - centroid) quantized to 2-bit — much easier because
     residuals have small, centered dynamic range.

This achieves very high compression (approaching 2-bit effective) with better
quality than naive 2-bit quantization, because the centroids capture the bulk
of the semantic content.

References:
  - QVG / Vector Quantization approaches for KV compression
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class QVGCompressed:
    """Container for a QVG-compressed KV tensor."""
    centroids: torch.Tensor          # (n_clusters, head_dim), full precision
    assignments: torch.Tensor        # (num_tokens,), int32 or int16
    quantized_residuals: torch.Tensor  # packed 2-bit residuals (uint8)
    residual_scales: torch.Tensor    # per-group or per-cluster scale
    residual_zero_points: torch.Tensor
    original_shape: Tuple[int, ...]
    group_size: int


class QVGSmoothingQuantizer:
    """
    Semantic smoothing quantizer: cluster KV tokens, quantize residuals
    to 2-bit, store centroids at full precision.
    """

    def __init__(
        self,
        n_clusters: int = 16,
        residual_bits: int = 2,
        kmeans_max_iters: int = 10,
    ) -> None:
        """
        Args:
            n_clusters: Number of k-means clusters.  16 is a good default
                (4 bits for assignment).
            residual_bits: Bit-width for residual quantization (default 2).
            kmeans_max_iters: Max iterations for k-means (keep low for speed).
        """
        if n_clusters < 2:
            raise ValueError(f"n_clusters must be >= 2, got {n_clusters}")
        if residual_bits not in (1, 2, 3, 4):
            raise ValueError(f"residual_bits must be in [1,4], got {residual_bits}")

        self.n_clusters = n_clusters
        self.residual_bits = residual_bits
        self.kmeans_max_iters = kmeans_max_iters
        self._num_levels = 2 ** residual_bits  # e.g., 4 for 2-bit

    # ------------------------------------------------------------------
    # Compress
    # ------------------------------------------------------------------

    def semantic_smooth(
        self,
        kv_cache_tensor: torch.Tensor,
        group_size: int = 64,
    ) -> QVGCompressed:
        """
        Compress a KV cache tensor using k-means + residual quantization.

        Args:
            kv_cache_tensor: Tensor of shape ``(num_tokens, head_dim)`` or
                ``(num_heads, num_tokens, head_dim)``.
            group_size: Group size for residual quantization.

        Returns:
            QVGCompressed container with all data needed for reconstruction.
        """
        original_shape = kv_cache_tensor.shape

        # Flatten to 2D: (N, D)
        if kv_cache_tensor.dim() == 3:
            num_heads, num_tokens, head_dim = kv_cache_tensor.shape
            flat = kv_cache_tensor.reshape(-1, head_dim)
        elif kv_cache_tensor.dim() == 2:
            flat = kv_cache_tensor
        else:
            raise ValueError(
                f"Expected 2D or 3D tensor, got {kv_cache_tensor.dim()}D"
            )

        N, D = flat.shape

        # ---- K-means clustering ----
        centroids, assignments = self._kmeans(flat, self.n_clusters, self.kmeans_max_iters)

        # ---- Compute residuals ----
        assigned_centroids = centroids[assignments]  # (N, D)
        residuals = flat - assigned_centroids  # (N, D)

        # ---- Quantize residuals ----
        packed, scales, zero_points = self._quantize_residuals(
            residuals, group_size=group_size
        )

        return QVGCompressed(
            centroids=centroids,
            assignments=assignments,
            quantized_residuals=packed,
            residual_scales=scales,
            residual_zero_points=zero_points,
            original_shape=original_shape,
            group_size=group_size,
        )

    # ------------------------------------------------------------------
    # Reconstruct
    # ------------------------------------------------------------------

    def reconstruct(
        self,
        compressed: QVGCompressed,
        target_dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        """
        Reconstruct the KV tensor from QVG-compressed representation.

        Args:
            compressed: Output of ``semantic_smooth``.
            target_dtype: Desired output dtype.

        Returns:
            Reconstructed tensor with the same shape as the original input.
        """
        centroids = compressed.centroids
        assignments = compressed.assignments
        original_shape = compressed.original_shape

        # Dequantize residuals
        if len(original_shape) == 3:
            numel = original_shape[0] * original_shape[1] * original_shape[2]
            flat_shape = (original_shape[0] * original_shape[1], original_shape[2])
        else:
            numel = original_shape[0] * original_shape[1]
            flat_shape = original_shape

        residuals = self._dequantize_residuals(
            compressed.quantized_residuals,
            compressed.residual_scales,
            compressed.residual_zero_points,
            compressed.group_size,
            original_numel=numel,
            flat_shape=flat_shape,
            target_dtype=target_dtype,
        )

        # Reconstruct: centroid + residual
        assigned_centroids = centroids[assignments].to(target_dtype)
        reconstructed = assigned_centroids + residuals

        return reconstructed.reshape(original_shape)

    # ------------------------------------------------------------------
    # Compression ratio
    # ------------------------------------------------------------------

    def compute_compression_ratio(
        self,
        original_shape: Tuple[int, ...],
        compressed: Optional[QVGCompressed] = None,
    ) -> Dict[str, float]:
        """
        Report actual compression vs FP16 baseline.

        Args:
            original_shape: Shape of the original tensor.
            compressed: Optional QVGCompressed object for exact sizing.

        Returns:
            Dict with ``original_bytes``, ``compressed_bytes``,
            ``compression_ratio``, ``effective_bits_per_element``.
        """
        numel = math.prod(original_shape)
        original_bytes = numel * 2  # FP16

        if compressed is not None:
            # Centroids: n_clusters * D * 2 bytes
            centroid_bytes = compressed.centroids.numel() * 2
            # Assignments: N * ceil(log2(n_clusters)) / 8
            assignment_bits = math.ceil(math.log2(max(self.n_clusters, 2)))
            N = compressed.assignments.numel()
            assignment_bytes = math.ceil(N * assignment_bits / 8)
            # Residuals: packed at residual_bits per element
            residual_bytes = compressed.quantized_residuals.numel()
            # Scales/zp
            scale_bytes = (compressed.residual_scales.numel() +
                           compressed.residual_zero_points.numel()) * 4
            compressed_bytes = centroid_bytes + assignment_bytes + residual_bytes + scale_bytes
        else:
            # Estimate
            D = original_shape[-1]
            N = numel // D
            centroid_bytes = self.n_clusters * D * 2
            assignment_bits = math.ceil(math.log2(max(self.n_clusters, 2)))
            assignment_bytes = math.ceil(N * assignment_bits / 8)
            # Residual: N*D elements at residual_bits
            residual_bytes = math.ceil(N * D * self.residual_bits / 8)
            n_groups = math.ceil(N * D / 64)
            scale_bytes = n_groups * 8
            compressed_bytes = centroid_bytes + assignment_bytes + residual_bytes + scale_bytes

        ratio = original_bytes / max(compressed_bytes, 1)
        effective_bpe = compressed_bytes * 8 / max(numel, 1)

        return {
            "original_bytes": original_bytes,
            "compressed_bytes": compressed_bytes,
            "compression_ratio": ratio,
            "effective_bits_per_element": effective_bpe,
        }

    # ------------------------------------------------------------------
    # K-means
    # ------------------------------------------------------------------

    def _kmeans(
        self,
        data: torch.Tensor,
        n_clusters: int,
        max_iters: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simple k-means clustering using torch operations.

        Args:
            data: (N, D) float tensor.
            n_clusters: Number of clusters.
            max_iters: Maximum iterations.

        Returns:
            (centroids, assignments) where centroids is (K, D) and
            assignments is (N,) int64.
        """
        N, D = data.shape
        device = data.device
        dtype = data.dtype

        # Clamp n_clusters to number of data points
        K = min(n_clusters, N)

        # Initialise centroids: k-means++ lite (random pick, then farthest)
        indices = [torch.randint(N, (1,), device=device).item()]
        for _ in range(1, K):
            centroids_so_far = data[indices]  # (k, D)
            dists = torch.cdist(data, centroids_so_far)  # (N, k)
            min_dists = dists.min(dim=1).values  # (N,)
            # Pick the farthest point
            idx = min_dists.argmax().item()
            indices.append(idx)

        centroids = data[indices].clone()  # (K, D)

        assignments = torch.zeros(N, dtype=torch.long, device=device)

        for _ in range(max_iters):
            # Assign to nearest centroid
            dists = torch.cdist(data, centroids)  # (N, K)
            new_assignments = dists.argmin(dim=1)  # (N,)

            # Check convergence
            if torch.equal(new_assignments, assignments):
                assignments = new_assignments
                break
            assignments = new_assignments

            # Update centroids
            for k in range(K):
                mask = assignments == k
                if mask.any():
                    centroids[k] = data[mask].mean(dim=0)

        return centroids, assignments

    # ------------------------------------------------------------------
    # Residual quantization (N-bit, group-wise)
    # ------------------------------------------------------------------

    def _quantize_residuals(
        self,
        residuals: torch.Tensor,
        group_size: int = 64,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize residuals to self.residual_bits using group-wise
        symmetric/asymmetric quantization, packed into uint8.

        Args:
            residuals: (N, D) float tensor.
            group_size: Elements per quantization group.

        Returns:
            (packed_uint8, scales, zero_points).
        """
        flat = residuals.reshape(-1).float()
        numel = flat.numel()

        # Pad
        pad_len = (group_size - numel % group_size) % group_size
        if pad_len > 0:
            flat = F.pad(flat, (0, pad_len), value=0.0)

        groups = flat.reshape(-1, group_size)
        num_groups = groups.shape[0]

        max_val = self._num_levels - 1  # e.g., 3 for 2-bit

        g_min = groups.min(dim=1).values
        g_max = groups.max(dim=1).values
        scale = (g_max - g_min).clamp(min=1e-12) / max_val
        zero_point = g_min

        quantized = ((groups - zero_point.unsqueeze(1)) / scale.unsqueeze(1))
        quantized = quantized.round().clamp(0, max_val).to(torch.uint8)

        # Pack into uint8: for 2-bit, 4 values per byte
        packed = self._pack_nbits(quantized.reshape(-1), self.residual_bits)

        return packed, scale, zero_point

    def _dequantize_residuals(
        self,
        packed: torch.Tensor,
        scales: torch.Tensor,
        zero_points: torch.Tensor,
        group_size: int,
        original_numel: int,
        flat_shape: Tuple[int, ...],
        target_dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        """Dequantize packed residuals back to float."""
        # Unpack
        unpacked = self._unpack_nbits(packed, self.residual_bits)

        num_groups = scales.numel()
        total_padded = num_groups * group_size
        unpacked = unpacked[:total_padded]

        groups = unpacked.reshape(num_groups, group_size).to(target_dtype)
        max_val = self._num_levels - 1

        dequantized = (groups * scales.unsqueeze(1).to(target_dtype)
                       + zero_points.unsqueeze(1).to(target_dtype))

        flat = dequantized.reshape(-1)[:original_numel]
        return flat.reshape(flat_shape)

    # ------------------------------------------------------------------
    # Bit packing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pack_nbits(data: torch.Tensor, bits: int) -> torch.Tensor:
        """
        Pack a uint8 tensor where each element uses only `bits` bits into
        densely-packed uint8 (multiple values per byte).
        """
        vals_per_byte = 8 // bits
        numel = data.numel()
        # Pad to multiple of vals_per_byte
        pad_len = (vals_per_byte - numel % vals_per_byte) % vals_per_byte
        if pad_len > 0:
            data = F.pad(data.float(), (0, pad_len), value=0.0).to(torch.uint8)

        data = data.reshape(-1, vals_per_byte)
        packed = torch.zeros(data.shape[0], dtype=torch.uint8, device=data.device)
        for i in range(vals_per_byte):
            packed |= data[:, i] << (bits * (vals_per_byte - 1 - i))
        return packed

    @staticmethod
    def _unpack_nbits(packed: torch.Tensor, bits: int) -> torch.Tensor:
        """Unpack densely-packed uint8 to individual values."""
        vals_per_byte = 8 // bits
        mask = (1 << bits) - 1

        unpacked_parts = []
        for i in range(vals_per_byte):
            shift = bits * (vals_per_byte - 1 - i)
            unpacked_parts.append((packed >> shift) & mask)

        # Interleave
        stacked = torch.stack(unpacked_parts, dim=-1)  # (N, vals_per_byte)
        return stacked.reshape(-1).to(torch.uint8)
