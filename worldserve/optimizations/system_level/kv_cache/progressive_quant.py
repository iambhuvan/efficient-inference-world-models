"""
Progressive KV cache quantization: FP16 -> FP8 -> INT4.

BASELINE / MUST-DO optimization.

As KV entries age (measured in frames since they were generated), they are
progressively quantized to lower precision:
  1. Recent frames: keep at FP16 (full precision attention).
  2. Medium-age frames: quantize to FP8 (torch.float8_e4m3fn).
  3. Old frames: quantize to INT4 (group-wise with scale + zero-point).

Key insight from DiffKV: Keys are more sensitive to quantization than Values
because they directly control the attention routing.  When ``key_value_asymmetry``
is enabled, Values are quantized more aggressively (earlier thresholds).

Split-attention: each precision group runs its own attention kernel and the
softmax outputs are renormalized via log-sum-exp combination.

References:
  - DiffKV (2024) — asymmetric K/V quantization
  - KVQuant (2024) — per-channel, group-wise INT4 with rotation
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers: check FP8 availability
# ---------------------------------------------------------------------------
_FP8_DTYPE: Optional[torch.dtype] = None
try:
    _FP8_DTYPE = torch.float8_e4m3fn
except AttributeError:
    # Older PyTorch without FP8 — fall back to BF16 as proxy
    _FP8_DTYPE = None

_DEVICE_IS_CUDA = torch.cuda.is_available()


@dataclass
class _PrecisionSlot:
    """Metadata for a single frame's KV entry."""
    frame_idx: int
    age: int  # frames since insertion
    k_precision: str  # "fp16", "fp8", "int4"
    v_precision: str


@dataclass
class _QuantizedTensor:
    """Container for a quantized tensor + metadata needed to dequantize."""
    data: torch.Tensor  # quantized payload (fp8 tensor or packed int4 uint8)
    scale: torch.Tensor
    zero_point: Optional[torch.Tensor] = None
    group_size: int = 128
    precision: str = "fp16"  # one of "fp16", "fp8", "int4"
    original_shape: Optional[Tuple[int, ...]] = None


class ProgressiveKVQuantizer:
    """
    Age-based progressive quantization of KV cache entries with optional
    key/value asymmetry and split-attention renormalization.
    """

    def __init__(
        self,
        fp8_age_threshold: int = 30,
        int4_age_threshold: int = 90,
        key_value_asymmetry: bool = True,
    ) -> None:
        """
        Args:
            fp8_age_threshold: Frame-age at which entries drop from FP16 to FP8.
            int4_age_threshold: Frame-age at which entries drop from FP8 to INT4.
            key_value_asymmetry: If True, Values are quantized earlier (70% of
                the Key thresholds), following the DiffKV insight.
        """
        self.key_value_asymmetry = key_value_asymmetry

        # Key thresholds
        self.k_fp8_age = fp8_age_threshold
        self.k_int4_age = int4_age_threshold

        # Value thresholds (more aggressive if asymmetric)
        if key_value_asymmetry:
            self.v_fp8_age = int(fp8_age_threshold * 0.7)
            self.v_int4_age = int(int4_age_threshold * 0.7)
        else:
            self.v_fp8_age = fp8_age_threshold
            self.v_int4_age = int4_age_threshold

        # Internal storage: layer_idx -> list[_QuantizedTensor] for K and V
        self._cache_k: Dict[int, List[_QuantizedTensor]] = {}
        self._cache_v: Dict[int, List[_QuantizedTensor]] = {}
        self._ages: Dict[int, List[int]] = {}  # layer_idx -> list of ages

        # Tracking
        self._total_fp16_bytes: int = 0
        self._total_actual_bytes: int = 0

    # ------------------------------------------------------------------
    # Quantize / dequantize primitives
    # ------------------------------------------------------------------

    @staticmethod
    def quantize_to_fp8(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize a FP16/BF16 tensor to FP8 (float8_e4m3fn) with per-tensor scale.

        Args:
            tensor: Input tensor in FP16 or BF16.

        Returns:
            (quantized_fp8, scale) where scale is a scalar tensor.
        """
        # Compute per-tensor absmax scale
        amax = tensor.abs().amax().clamp(min=1e-12)
        # float8_e4m3fn max representable value is 448.0
        fp8_max = 448.0
        scale = amax / fp8_max

        scaled = tensor / scale

        if _FP8_DTYPE is not None:
            quantized = scaled.to(_FP8_DTYPE)
        else:
            # Fallback: clamp to FP8 range and keep as BF16/FP16
            quantized = scaled.clamp(-fp8_max, fp8_max).to(torch.bfloat16)

        return quantized, scale

    @staticmethod
    def quantize_to_int4(
        tensor: torch.Tensor,
        group_size: int = 128,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Group-wise INT4 quantization with scale + zero-point per group.

        The output is packed into uint8 (two INT4 values per byte).

        Args:
            tensor: Input tensor (any float dtype).
            group_size: Number of elements per quantization group.

        Returns:
            (packed_uint8, scales, zero_points) where:
              - packed_uint8 has half the elements (2 int4 per byte)
              - scales: ``(num_groups,)``
              - zero_points: ``(num_groups,)``
        """
        original_shape = tensor.shape
        flat = tensor.reshape(-1).float()
        numel = flat.numel()

        # Pad to multiple of group_size
        pad_len = (group_size - numel % group_size) % group_size
        if pad_len > 0:
            flat = F.pad(flat, (0, pad_len), value=0.0)

        groups = flat.reshape(-1, group_size)
        num_groups = groups.shape[0]

        # Per-group min/max
        g_min = groups.min(dim=1).values  # (num_groups,)
        g_max = groups.max(dim=1).values

        # Scale and zero-point for unsigned 4-bit [0, 15]
        scale = (g_max - g_min).clamp(min=1e-12) / 15.0
        zero_point = g_min

        # Quantize to [0, 15]
        quantized = ((groups - zero_point.unsqueeze(1)) / scale.unsqueeze(1))
        quantized = quantized.round().clamp(0, 15).to(torch.uint8)

        # Pack two int4 values into one uint8
        quantized_flat = quantized.reshape(-1)
        # Ensure even length
        if quantized_flat.numel() % 2 != 0:
            quantized_flat = F.pad(quantized_flat, (0, 1), value=0)
        high = quantized_flat[0::2] << 4
        low = quantized_flat[1::2]
        packed = (high | low).to(torch.uint8)

        return packed, scale, zero_point

    @staticmethod
    def dequantize_fp8(
        quantized: torch.Tensor,
        scale: torch.Tensor,
        target_dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        """
        Dequantize FP8 tensor back to FP16/BF16.

        Args:
            quantized: FP8 (or fallback BF16) tensor.
            scale: Per-tensor scale from quantize_to_fp8.
            target_dtype: Desired output dtype.

        Returns:
            Dequantized tensor in *target_dtype*.
        """
        return quantized.to(target_dtype) * scale

    @staticmethod
    def dequantize_int4(
        packed: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        group_size: int,
        original_numel: int,
        original_shape: Tuple[int, ...],
        target_dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        """
        Dequantize packed INT4 tensor back to float.

        Args:
            packed: uint8 tensor from quantize_to_int4.
            scale: Per-group scales.
            zero_point: Per-group zero points.
            group_size: Group size used during quantization.
            original_numel: Number of elements before padding.
            original_shape: Shape to reshape the output to.
            target_dtype: Desired output dtype.

        Returns:
            Dequantized tensor in *target_dtype* with *original_shape*.
        """
        # Unpack uint8 -> two int4 values
        high = (packed >> 4).to(torch.int32)
        low = (packed & 0x0F).to(torch.int32)

        # Interleave back
        unpacked = torch.stack([high, low], dim=-1).reshape(-1)

        # Trim to padded group length
        num_groups = scale.numel()
        total_padded = num_groups * group_size
        unpacked = unpacked[:total_padded]

        # Reshape into groups and dequantize
        groups = unpacked.reshape(num_groups, group_size).to(target_dtype)
        dequantized = groups * scale.unsqueeze(1).to(target_dtype) + zero_point.unsqueeze(1).to(target_dtype)

        # Flatten and trim to original numel
        flat = dequantized.reshape(-1)[:original_numel]
        return flat.reshape(original_shape)

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def update_cache(
        self,
        kv_cache: Dict[int, Dict[str, torch.Tensor]],
        new_kv: Dict[int, Dict[str, torch.Tensor]],
        frame_idx: int,
    ) -> Dict[int, Dict[str, List[_QuantizedTensor]]]:
        """
        Add new KV entries at FP16 and progressively quantize older entries.

        Args:
            kv_cache: Current managed cache (layer_idx -> {"key": ..., "value": ...}).
                On first call, pass empty dicts.
            new_kv: New KV entries to append (same structure, single frame each).
            frame_idx: Current frame index (used for age computation).

        Returns:
            Updated internal cache with mixed-precision entries.
        """
        for layer_idx, kv in new_kv.items():
            if layer_idx not in self._cache_k:
                self._cache_k[layer_idx] = []
                self._cache_v[layer_idx] = []
                self._ages[layer_idx] = []

            # Add new entry at FP16
            k_tensor = kv["key"]
            v_tensor = kv["value"]

            self._cache_k[layer_idx].append(_QuantizedTensor(
                data=k_tensor.clone(),
                scale=torch.tensor(1.0, device=k_tensor.device),
                precision="fp16",
                original_shape=k_tensor.shape,
            ))
            self._cache_v[layer_idx].append(_QuantizedTensor(
                data=v_tensor.clone(),
                scale=torch.tensor(1.0, device=v_tensor.device),
                precision="fp16",
                original_shape=v_tensor.shape,
            ))
            self._ages[layer_idx].append(0)

            # Age all existing entries and quantize as needed
            for i in range(len(self._ages[layer_idx]) - 1):
                self._ages[layer_idx][i] += 1
                age = self._ages[layer_idx][i]

                # --- Key quantization ---
                k_entry = self._cache_k[layer_idx][i]
                if age >= self.k_int4_age and k_entry.precision != "int4":
                    self._promote_to_int4(k_entry)
                elif age >= self.k_fp8_age and k_entry.precision == "fp16":
                    self._promote_to_fp8(k_entry)

                # --- Value quantization ---
                v_entry = self._cache_v[layer_idx][i]
                if age >= self.v_int4_age and v_entry.precision != "int4":
                    self._promote_to_int4(v_entry)
                elif age >= self.v_fp8_age and v_entry.precision == "fp16":
                    self._promote_to_fp8(v_entry)

        return {"keys": self._cache_k, "values": self._cache_v}

    def _promote_to_fp8(self, entry: _QuantizedTensor) -> None:
        """Quantize an FP16 entry in-place to FP8."""
        raw = entry.data
        quantized, scale = self.quantize_to_fp8(raw)
        entry.data = quantized
        entry.scale = scale
        entry.precision = "fp8"

    def _promote_to_int4(self, entry: _QuantizedTensor) -> None:
        """Quantize an entry (FP16 or FP8) in-place to INT4."""
        # First dequantize if currently FP8
        if entry.precision == "fp8":
            raw = self.dequantize_fp8(entry.data, entry.scale)
        else:
            raw = entry.data

        original_shape = raw.shape
        original_numel = raw.numel()
        packed, scale, zp = self.quantize_to_int4(raw)
        entry.data = packed
        entry.scale = scale
        entry.zero_point = zp
        entry.precision = "int4"
        entry.original_shape = original_shape
        entry.group_size = 128

    # ------------------------------------------------------------------
    # Mixed-precision retrieval
    # ------------------------------------------------------------------

    def get_quantized_cache(
        self,
        target_bits: int = 4,
        target_dtype: torch.dtype = torch.float16,
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        Return a dequantized copy of the internal cache forced to *target_bits* precision.

        Used by speculative decoding to obtain an INT4/FP8-quality copy of the
        KV cache for draft forwards.  Each entry is quantized then immediately
        dequantized so the tensor dtype is still float16/bfloat16 but the
        information content is limited to target_bits per element.

        Args:
            target_bits: 4 (INT4) or 8 (FP8) per element.
            target_dtype: Output tensor dtype (default float16).

        Returns:
            Dict mapping layer_idx -> {"key": Tensor, "value": Tensor} where
            tensors have shape (num_frames, heads, tokens, head_dim) and dtype
            *target_dtype*.  Layers not present in internal cache are skipped.
        """
        result: Dict[int, Dict[str, torch.Tensor]] = {}

        all_layers = set(self._cache_k.keys()) | set(self._cache_v.keys())
        for layer_idx in all_layers:
            k_slots = self._cache_k.get(layer_idx, [])
            v_slots = self._cache_v.get(layer_idx, [])
            if not k_slots and not v_slots:
                continue

            k_frames: List[torch.Tensor] = []
            v_frames: List[torch.Tensor] = []

            for qt in k_slots:
                deq = self._dequantize_to_target(qt, target_bits, target_dtype)
                k_frames.append(deq)
            for qt in v_slots:
                deq = self._dequantize_to_target(qt, target_bits, target_dtype)
                v_frames.append(deq)

            if k_frames:
                result[layer_idx] = {
                    "key":   torch.stack(k_frames, dim=0),
                    "value": torch.stack(v_frames, dim=0) if v_frames else torch.stack(k_frames, dim=0),
                }

        return result

    def _dequantize_to_target(
        self,
        qt: "_QuantizedTensor",
        target_bits: int,
        target_dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Re-quantize a _QuantizedTensor entry to target_bits then dequantize.

        If the stored precision is already at or below target_bits, just
        dequantize to target_dtype without re-quantizing.
        """
        # First dequantize whatever precision is stored
        if qt.precision == "fp16":
            deq = qt.data.to(target_dtype)
        elif qt.precision == "fp8":
            deq = self.dequantize_fp8(qt.data, qt.scale, target_dtype)
        elif qt.precision == "int4":
            original_numel = int(qt.original_shape[0] * qt.scale.numel() * 128) if qt.original_shape else qt.data.numel() * 2
            if qt.original_shape is not None:
                numel = 1
                for d in qt.original_shape:
                    numel *= d
                deq = self.dequantize_int4(
                    qt.data, qt.scale, qt.zero_point, qt.group_size,
                    numel, qt.original_shape, target_dtype,
                )
            else:
                deq = qt.data.to(target_dtype)
        else:
            deq = qt.data.to(target_dtype)

        # If target requires re-quantization (simulating limited precision)
        if target_bits == 4 and qt.precision not in ("int4",):
            packed, scale, zp = self.quantize_to_int4(deq)
            numel = deq.numel()
            deq = self.dequantize_int4(
                packed, scale, zp, 128, numel, deq.shape, target_dtype
            )
        elif target_bits == 8 and qt.precision == "fp16":
            q, scale = self.quantize_to_fp8(deq)
            deq = self.dequantize_fp8(q, scale, target_dtype)

        return deq

    def get_mixed_precision_cache(
        self,
        layer_idx: int,
    ) -> Dict[str, Dict[str, Union[torch.Tensor, List[_QuantizedTensor]]]]:
        """
        Return KV cache entries grouped by precision level for split-attention.

        Args:
            layer_idx: DiT layer index.

        Returns:
            Dict with keys ``"fp16_recent"``, ``"fp8_medium"``, ``"int4_old"``,
            each containing ``{"keys": [...], "values": [...]}``.
        """
        groups: Dict[str, Dict[str, List[_QuantizedTensor]]] = {
            "fp16_recent": {"keys": [], "values": []},
            "fp8_medium": {"keys": [], "values": []},
            "int4_old": {"keys": [], "values": []},
        }

        k_entries = self._cache_k.get(layer_idx, [])
        v_entries = self._cache_v.get(layer_idx, [])

        for k_entry, v_entry in zip(k_entries, v_entries):
            # Use the coarser precision of K/V to determine the group
            max_prec = self._coarser_precision(k_entry.precision, v_entry.precision)
            if max_prec == "fp16":
                group_key = "fp16_recent"
            elif max_prec == "fp8":
                group_key = "fp8_medium"
            else:
                group_key = "int4_old"

            groups[group_key]["keys"].append(k_entry)
            groups[group_key]["values"].append(v_entry)

        return groups

    @staticmethod
    def _coarser_precision(p1: str, p2: str) -> str:
        rank = {"fp16": 0, "fp8": 1, "int4": 2}
        return p1 if rank.get(p1, 0) >= rank.get(p2, 0) else p2

    # ------------------------------------------------------------------
    # Split attention
    # ------------------------------------------------------------------

    def split_attention(
        self,
        query: torch.Tensor,
        kv_groups: Dict[str, Dict[str, List[_QuantizedTensor]]],
        head_dim: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Run attention separately over each precision group and renormalize.

        Uses the log-sum-exp trick to correctly combine softmax outputs
        from separate attention computations.

        Args:
            query: Query tensor, shape ``(batch, num_heads, seq_q, head_dim)``.
            kv_groups: Output of ``get_mixed_precision_cache``.
            head_dim: Head dimension (inferred from query if None).

        Returns:
            Combined attention output, same shape as a standard attention result
            ``(batch, num_heads, seq_q, head_dim)``.
        """
        if head_dim is None:
            head_dim = query.shape[-1]
        scale = 1.0 / math.sqrt(head_dim)

        # Collect per-group (output, logsumexp)
        group_outputs: List[torch.Tensor] = []
        group_lse: List[torch.Tensor] = []

        for group_name in ("fp16_recent", "fp8_medium", "int4_old"):
            k_entries = kv_groups[group_name]["keys"]
            v_entries = kv_groups[group_name]["values"]
            if not k_entries:
                continue

            # Dequantize, compute attention, then FREE the dequantized tensors
            # immediately. This ensures only one group's FP16 tensors are
            # materialized at a time, avoiding OOM from holding all groups.
            keys = self._dequantize_entries(k_entries, target_dtype=query.dtype)
            values = self._dequantize_entries(v_entries, target_dtype=query.dtype)

            k_cat = torch.cat(keys, dim=-2)
            v_cat = torch.cat(values, dim=-2)
            del keys, values  # free individual dequantized list

            if k_cat.dim() == 3 and query.dim() == 4:
                k_cat = k_cat.unsqueeze(0).expand(query.shape[0], -1, -1, -1)
                v_cat = v_cat.unsqueeze(0).expand(query.shape[0], -1, -1, -1)

            attn_logits = torch.matmul(query, k_cat.transpose(-2, -1)) * scale
            del k_cat  # free keys immediately after computing logits

            lse = torch.logsumexp(attn_logits, dim=-1, keepdim=True)
            attn_weights = torch.softmax(attn_logits, dim=-1)
            del attn_logits  # free logits after softmax
            out = torch.matmul(attn_weights, v_cat)
            del attn_weights, v_cat  # free after output computed

            group_outputs.append(out)
            group_lse.append(lse)

        if not group_outputs:
            return torch.zeros_like(query)

        if len(group_outputs) == 1:
            return group_outputs[0]

        # Combine using log-sum-exp renormalization
        # For each group g: output_g was computed with local softmax.
        # True output = sum_g [ exp(lse_g - global_lse) * output_g ]
        all_lse = torch.cat(group_lse, dim=-1)  # (B, H, Sq, num_groups)
        global_lse = torch.logsumexp(all_lse, dim=-1, keepdim=True)  # (B, H, Sq, 1)

        combined = torch.zeros_like(query)
        for out, lse in zip(group_outputs, group_lse):
            weight = torch.exp(lse - global_lse)  # (B, H, Sq, 1)
            combined = combined + weight * out

        return combined

    def _dequantize_entries(
        self,
        entries: List[_QuantizedTensor],
        target_dtype: torch.dtype = torch.float16,
    ) -> List[torch.Tensor]:
        """Dequantize a list of _QuantizedTensor back to target_dtype."""
        result = []
        for entry in entries:
            if entry.precision == "fp16":
                result.append(entry.data.to(target_dtype))
            elif entry.precision == "fp8":
                result.append(self.dequantize_fp8(entry.data, entry.scale, target_dtype))
            elif entry.precision == "int4":
                if entry.original_shape is None:
                    raise ValueError("INT4 entry missing original_shape for dequantization.")
                result.append(self.dequantize_int4(
                    entry.data,
                    entry.scale,
                    entry.zero_point,
                    entry.group_size,
                    original_numel=math.prod(entry.original_shape),
                    original_shape=entry.original_shape,
                    target_dtype=target_dtype,
                ))
            else:
                raise ValueError(f"Unknown precision: {entry.precision}")
        return result

    # ------------------------------------------------------------------
    # Memory tracking
    # ------------------------------------------------------------------

    def get_memory_report(self) -> Dict[str, object]:
        """
        Report memory usage: actual bytes vs hypothetical full-FP16 bytes.

        Returns:
            Dict with ``fp16_bytes``, ``actual_bytes``, ``savings_ratio``,
            and per-precision counts.
        """
        fp16_bytes = 0
        actual_bytes = 0
        precision_counts = {"fp16": 0, "fp8": 0, "int4": 0}

        for layer_entries in list(self._cache_k.values()) + list(self._cache_v.values()):
            for entry in layer_entries:
                if entry.original_shape is not None:
                    numel = math.prod(entry.original_shape)
                else:
                    numel = entry.data.numel()

                fp16_bytes += numel * 2  # FP16 = 2 bytes

                if entry.precision == "fp16":
                    actual_bytes += numel * 2
                elif entry.precision == "fp8":
                    actual_bytes += numel * 1  # 1 byte per element
                    actual_bytes += 4  # scale
                elif entry.precision == "int4":
                    actual_bytes += numel // 2  # 0.5 bytes per element (packed)
                    n_groups = (numel + 127) // 128
                    actual_bytes += n_groups * 8  # scale + zp (4 bytes each)

                precision_counts[entry.precision] += 1

        savings = 1.0 - (actual_bytes / max(fp16_bytes, 1))

        return {
            "fp16_bytes": fp16_bytes,
            "actual_bytes": actual_bytes,
            "savings_ratio": savings,
            "precision_counts": precision_counts,
        }
