"""
KV Cache Compression Manager for WorldServe.

Orchestrates all KV cache optimizations:
  - Temporal token merging (baseline)
  - Progressive quantization FP16->FP8->INT4 (baseline)
  - Importance-aware precision (advanced)
  - Layer-aware quantization (advanced)
  - QVG semantic smoothing (advanced)
  - Spatial block eviction (advanced, MG2 only)

The manager exposes a simple API: create_cache -> update -> get_kv_for_attention,
hiding the complexity of mixed-precision storage and multi-technique coordination.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from worldserve.optimizations.system_level.kv_cache.temporal_merge import TemporalTokenMerger
from worldserve.optimizations.system_level.kv_cache.progressive_quant import ProgressiveKVQuantizer


class KVCacheManager:
    """
    End-to-end manager for KV cache compression in video world model serving.

    Supports both Open-Oasis (small spatial, 16 layers) and Matrix-Game 2.0
    (large spatial, 30 layers, rolling window).
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self._wire_step_counter: int = 0  # incremented by post_step; used by wire_to_model hooks
        # current_frame_idx is the *video frame* index (not the denoising step).
        # Set externally by the generation loop via set_current_frame_idx() so that
        # ProgressiveKVQuantizer's age-based thresholds (fp8_age=30, int4_age=90)
        # operate on frames as the proposal intended, not on (frame × ddim_step).
        # Falls back to _wire_step_counter when not set.
        self._current_frame_idx: Optional[int] = None
        self._capture_enabled: bool = True  # set False to bypass F.sdpa monkey-patch (e.g. auto-accept steps)
        """
        Configure which techniques to enable.

        Args:
            config: Dictionary with optional keys:

                - ``temporal_merge`` (dict | bool): Enable temporal merging.
                  Sub-keys: ``similarity_threshold`` (float), ``merge_strategy`` (str).
                - ``progressive_quant`` (dict | bool): Enable progressive quantization.
                  Sub-keys: ``fp8_age_threshold`` (int), ``int4_age_threshold`` (int),
                  ``key_value_asymmetry`` (bool).
                - ``importance_tracker`` (dict | bool): Enable importance-aware precision.
                  Sub-keys: ``alpha`` (float), ``num_layers`` (int).
                - ``layer_aware`` (dict | bool): Enable layer-aware quantization.
                  Sub-keys: ``num_layers`` (int).
                - ``qvg_smoothing`` (dict | bool): Enable QVG smoothing.
                  Sub-keys: ``n_clusters`` (int), ``residual_bits`` (int).
                - ``spatial_eviction`` (dict | bool): Enable spatial block eviction.
                  Sub-keys: ``block_size`` (int), ``eviction_ratio`` (float),
                  ``min_age_frames`` (int), ``H`` (int), ``W`` (int).
                - ``num_layers`` (int): Number of DiT layers (default 30).
                - ``max_frames`` (int): Maximum frames in KV window (default 32).
        """
        self.config = config
        self.num_layers = config.get("num_layers", 30)
        self.max_frames = config.get("max_frames", 32)

        # ---- Instantiate enabled techniques ----
        self.temporal_merger: Optional[TemporalTokenMerger] = None
        self.progressive_quantizer: Optional[ProgressiveKVQuantizer] = None
        self.importance_tracker: Optional[Any] = None  # lazy import
        self.layer_aware_quantizer: Optional[Any] = None
        self.qvg_smoother: Optional[Any] = None
        self.spatial_evictor: Optional[Any] = None

        # Temporal merge (BASELINE)
        tm_cfg = config.get("temporal_merge", False)
        if tm_cfg:
            kwargs = tm_cfg if isinstance(tm_cfg, dict) else {}
            self.temporal_merger = TemporalTokenMerger(
                similarity_threshold=kwargs.get("similarity_threshold", 0.95),
                merge_strategy=kwargs.get("merge_strategy", "weighted_average"),
            )

        # Progressive quant (BASELINE)
        pq_cfg = config.get("progressive_quant", False)
        if pq_cfg:
            kwargs = pq_cfg if isinstance(pq_cfg, dict) else {}
            self.progressive_quantizer = ProgressiveKVQuantizer(
                fp8_age_threshold=kwargs.get("fp8_age_threshold", 30),
                int4_age_threshold=kwargs.get("int4_age_threshold", 90),
                key_value_asymmetry=kwargs.get("key_value_asymmetry", True),
            )

        # Importance tracker (ADVANCED)
        it_cfg = config.get("importance_tracker", False)
        if it_cfg:
            from worldserve.optimizations.system_level.kv_cache.importance_tracker import ImportanceTracker
            kwargs = it_cfg if isinstance(it_cfg, dict) else {}
            self.importance_tracker = ImportanceTracker(
                alpha=kwargs.get("alpha", 0.1),
                num_layers=kwargs.get("num_layers", self.num_layers),
            )

        # Layer-aware quant (ADVANCED)
        la_cfg = config.get("layer_aware", False)
        if la_cfg:
            from worldserve.optimizations.system_level.kv_cache.layer_aware_quant import LayerAwareQuantizer
            kwargs = la_cfg if isinstance(la_cfg, dict) else {}
            self.layer_aware_quantizer = LayerAwareQuantizer(
                num_layers=kwargs.get("num_layers", self.num_layers),
            )

        # QVG smoothing (ADVANCED)
        qvg_cfg = config.get("qvg_smoothing", False)
        if qvg_cfg:
            from worldserve.optimizations.system_level.kv_cache.qvg_smoothing import QVGSmoothingQuantizer
            kwargs = qvg_cfg if isinstance(qvg_cfg, dict) else {}
            self.qvg_smoother = QVGSmoothingQuantizer(
                n_clusters=kwargs.get("n_clusters", 16),
                residual_bits=kwargs.get("residual_bits", 2),
            )

        # Spatial eviction (ADVANCED, MG2 only)
        se_cfg = config.get("spatial_eviction", False)
        if se_cfg:
            from worldserve.optimizations.system_level.kv_cache.spatial_eviction import SpatialBlockEvictor
            kwargs = se_cfg if isinstance(se_cfg, dict) else {}
            self.spatial_evictor = SpatialBlockEvictor(
                block_size=kwargs.get("block_size", 4),
                eviction_ratio=kwargs.get("eviction_ratio", 0.3),
                min_age_frames=kwargs.get("min_age_frames", 10),
            )
            self._spatial_H = kwargs.get("H", 124)
            self._spatial_W = kwargs.get("W", 124)

        # ---- Internal state ----
        # Raw KV cache: layer_idx -> {"key": (frames, heads, tokens, dim),
        #                              "value": (frames, heads, tokens, dim)}
        self._raw_cache: Dict[int, Dict[str, torch.Tensor]] = {}
        self._frame_count: int = 0
        self._frame_ages: Dict[int, List[int]] = {}  # layer_idx -> list of ages

        # Context KV precompute (Diffusion Forcing optimization):
        # Stores KVs for context frames that are fixed across denoising steps.
        # Populated by set_context_kv(); injected into attention by wire_to_model().
        self._context_kv: Dict[int, Dict[str, torch.Tensor]] = {}
        # Number of context frames whose KVs are fixed (set by caller)
        self._num_context_tokens: int = 0
        # Whether context KV reuse is active (armed after step 0 capture)
        self._context_kv_active: bool = False

        # KV injection mode (used by KVQuantSpeculativeDecoder):
        # When armed, _capture_sdpa replaces K/V with stored cache tensors
        # instead of using the live K/V computed by the model.  This is what
        # makes the INT4 draft and FP16 verify passes actually use different
        # precision caches rather than recomputing identical full-precision K/V.
        self._inject_mode: bool = False
        self._inject_precision: str = "fp16"
        # The cache dict to inject from: layer_idx -> {"key": Tensor, "value": Tensor}
        self._inject_cache: Dict[int, Dict[str, torch.Tensor]] = {}

        # Live quantization substitution into attention.
        # When enabled, _capture_sdpa applies a quantize-dequantize roundtrip
        # to the live K/V tensors before calling orig_sdpa, using a precision
        # derived from the current video-frame age. This makes the KV
        # compression honestly affect numerics (and adds the quant work to
        # measured latency) so we are not just an accountant.
        self._enable_quant_substitution: bool = bool(
            config.get("enable_quant_substitution", False)
        )

        # Quantization volume accounting — how MUCH compression actually
        # happened, not just whether it was on. Reset by create_cache().
        self._quant_stats: Dict[str, Any] = {
            "calls_fp16": 0, "calls_fp8": 0, "calls_int4": 0,
            "calls_gated_off": 0,  # routed through orig FP16 because gate said no
            "elements_fp16": 0, "elements_fp8": 0, "elements_int4": 0,
            "bytes_in": 0, "bytes_effective": 0,
            "per_frame": {},  # frame_idx -> {"fp16": n, "fp8": n, "int4": n, "gated_off": n}
        }

        # Per-frame substitution gate. The eval loop sets this to False when
        # the current frame is a "critical moment" (e.g. high action magnitude)
        # so we skip the FP8/INT4 roundtrip for that frame and let the model
        # see clean FP16 K/V. Default True = same behavior as before.
        self._substitution_active_now: bool = True

    # ------------------------------------------------------------------
    # Cache lifecycle
    # ------------------------------------------------------------------

    def create_cache(
        self,
        num_layers: Optional[int] = None,
        max_frames: Optional[int] = None,
    ) -> None:
        """
        Initialize (reset) the managed KV cache.

        Args:
            num_layers: Override number of layers.
            max_frames: Override max frames.
        """
        if num_layers is not None:
            self.num_layers = num_layers
        if max_frames is not None:
            self.max_frames = max_frames

        self._raw_cache.clear()
        self._frame_count = 0
        self._frame_ages.clear()

        if self.importance_tracker is not None:
            self.importance_tracker.reset()

        if self.progressive_quantizer is not None:
            self.progressive_quantizer._cache_k.clear()
            self.progressive_quantizer._cache_v.clear()
            self.progressive_quantizer._ages.clear()

        if hasattr(self, "_qvg_compressed"):
            self._qvg_compressed.clear()
        self._current_frame_idx = None

        # Reset quantization volume accounting for a fresh generation.
        self._quant_stats = {
            "calls_fp16": 0, "calls_fp8": 0, "calls_int4": 0,
            "calls_gated_off": 0,
            "elements_fp16": 0, "elements_fp8": 0, "elements_int4": 0,
            "bytes_in": 0, "bytes_effective": 0,
            "per_frame": {},
        }
        self._substitution_active_now = True

        # Reset accumulating state in submodules so iter-to-iter timing is stable.
        if self.temporal_merger is not None:
            self.temporal_merger._reset_stats()
        if self.spatial_evictor is not None:
            self.spatial_evictor._block_score_ema.clear()
        self._wire_step_counter = 0

    def update(
        self,
        new_kv: Dict[int, Dict[str, torch.Tensor]],
        frame_idx: int,
        attention_weights: Optional[Dict[int, torch.Tensor]] = None,
    ) -> None:
        """
        Process a new frame's KV entries through the optimization pipeline.

        Pipeline order:
          1. Append new KV to raw cache.
          2. Apply temporal merging (if enabled).
          3. If importance tracker enabled + attention_weights provided: update scores.
          4. Apply progressive quantization by age (if enabled).
          5. If spatial eviction enabled: evict old blocks.

        Args:
            new_kv: New KV entries: ``layer_idx -> {"key": Tensor, "value": Tensor}``.
                Each tensor has shape ``(num_heads, num_tokens, head_dim)``
                (single frame).
            frame_idx: Current frame index.
            attention_weights: Optional per-layer attention weights for importance
                tracking: ``layer_idx -> (batch, heads, seq_q, seq_kv)``.
        """
        self._frame_count = frame_idx + 1

        # Step 1: Append to raw cache
        for layer_idx, kv in new_kv.items():
            if layer_idx not in self._raw_cache:
                # Initialise with unsqueeze for frame dim
                self._raw_cache[layer_idx] = {
                    "key": kv["key"].unsqueeze(0),
                    "value": kv["value"].unsqueeze(0),
                }
                self._frame_ages[layer_idx] = [0]
            else:
                self._raw_cache[layer_idx]["key"] = torch.cat(
                    [self._raw_cache[layer_idx]["key"], kv["key"].unsqueeze(0)],
                    dim=0,
                )
                self._raw_cache[layer_idx]["value"] = torch.cat(
                    [self._raw_cache[layer_idx]["value"], kv["value"].unsqueeze(0)],
                    dim=0,
                )
                # Age existing entries
                self._frame_ages[layer_idx] = [
                    a + 1 for a in self._frame_ages[layer_idx]
                ]
                self._frame_ages[layer_idx].append(0)

            # Enforce max_frames (rolling window)
            num_frames = self._raw_cache[layer_idx]["key"].shape[0]
            if num_frames > self.max_frames:
                excess = num_frames - self.max_frames
                self._raw_cache[layer_idx]["key"] = (
                    self._raw_cache[layer_idx]["key"][excess:]
                )
                self._raw_cache[layer_idx]["value"] = (
                    self._raw_cache[layer_idx]["value"][excess:]
                )
                self._frame_ages[layer_idx] = self._frame_ages[layer_idx][excess:]

        # Step 2: Temporal merging
        if self.temporal_merger is not None and frame_idx > 0:
            for layer_idx in self._raw_cache:
                cache = self._raw_cache[layer_idx]
                num_f = cache["key"].shape[0]
                if num_f >= 2:
                    merged_k, _ = self.temporal_merger.merge_tokens(
                        cache["key"], num_f - 1
                    )
                    merged_v, _ = self.temporal_merger.merge_tokens(
                        cache["value"], num_f - 1
                    )
                    self._raw_cache[layer_idx]["key"] = merged_k
                    self._raw_cache[layer_idx]["value"] = merged_v

        # Step 3: Importance tracking
        if self.importance_tracker is not None and attention_weights is not None:
            for layer_idx, attn_w in attention_weights.items():
                self.importance_tracker.update(layer_idx, attn_w)

        # Step 4: Progressive quantization (feeds into internal mixed-precision store)
        if self.progressive_quantizer is not None:
            self.progressive_quantizer.update_cache(
                kv_cache=self._raw_cache,
                new_kv=new_kv,
                frame_idx=frame_idx,
            )

        # Step 4b: QVG semantic smoothing — k-means + 2-bit residual.
        # Maintains a parallel compressed store keyed by (layer_idx, frame_idx)
        # so get_memory_report() can quantify the achievable compression. Like
        # progressive_quant in the current plumbing, this is an accountant — the
        # active attention path still uses _raw_cache. Substitution into the
        # attention forward is a separate (more invasive) change.
        if self.qvg_smoother is not None:
            if not hasattr(self, "_qvg_compressed"):
                self._qvg_compressed: Dict[int, Dict[int, Dict[str, Any]]] = {}
            for layer_idx, kv in new_kv.items():
                self._qvg_compressed.setdefault(layer_idx, {})
                try:
                    self._qvg_compressed[layer_idx][frame_idx] = {
                        "key":   self.qvg_smoother.semantic_smooth(kv["key"]),
                        "value": self.qvg_smoother.semantic_smooth(kv["value"]),
                    }
                except Exception:
                    # Never let auxiliary compression crash inference
                    pass

        # Step 5: Spatial eviction (MG2 only)
        if self.spatial_evictor is not None and attention_weights is not None:
            for layer_idx in self._raw_cache:
                if layer_idx not in attention_weights:
                    continue

                ages = self._frame_ages.get(layer_idx, [])
                if not ages:
                    continue

                ages_tensor = torch.tensor(ages, dtype=torch.float32,
                                           device=self._raw_cache[layer_idx]["key"].device)
                H, W = self._spatial_H, self._spatial_W

                # Compute block scores for each frame
                attn_w = attention_weights[layer_idx]
                num_blocks_h = (H // self.spatial_evictor.block_size)
                num_blocks_w = (W // self.spatial_evictor.block_size)
                num_blocks = num_blocks_h * num_blocks_w

                # Update evictor's per-frame EMA scores.  The attention weights
                # from the current query cover ALL frames in the KV cache
                # (not just the current frame), so we update scores for every
                # visible frame, not just the newest one.
                current_scores = self.spatial_evictor.compute_block_scores(attn_w, H, W)
                # current_scores covers all KV tokens; split by frame
                tokens_per_frame = num_blocks * self.spatial_evictor.block_size ** 2
                for f_idx in range(len(ages)):
                    # Extract the block scores attributable to this frame
                    # (approximate: use overall scores — the attention weights
                    # already reflect cross-frame interactions)
                    self.spatial_evictor.update_frame_scores(
                        layer_idx, f_idx, current_scores
                    )

                # Get accumulated per-frame scores for eviction decisions
                block_scores_all = self.spatial_evictor.get_accumulated_scores(
                    layer_idx, len(ages), num_blocks
                )

                # Evict from keys and values
                evicted_k, _ = self.spatial_evictor.evict(
                    self._raw_cache[layer_idx]["key"],
                    ages_tensor,
                    block_scores_all,
                    H, W,
                )
                evicted_v, _ = self.spatial_evictor.evict(
                    self._raw_cache[layer_idx]["value"],
                    ages_tensor,
                    block_scores_all,
                    H, W,
                )
                self._raw_cache[layer_idx]["key"] = evicted_k
                self._raw_cache[layer_idx]["value"] = evicted_v

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_kv_for_attention(
        self,
        layer_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Return KV tensors ready for attention computation.

        If progressive quantization is enabled, returns dequantized tensors
        from the mixed-precision store.  Otherwise, returns raw cache tensors
        reshaped as ``(total_kv_tokens, head_dim)`` per head.

        Args:
            layer_idx: DiT layer index.

        Returns:
            Dict with ``"key"`` and ``"value"`` tensors.  Shape depends on
            the stored format but is always suitable for attention:
            ``(num_heads, total_kv_tokens, head_dim)`` or
            ``(total_kv_tokens, head_dim)`` if single-head.
        """
        if self.progressive_quantizer is not None:
            # Use mixed-precision path with split attention
            groups = self.progressive_quantizer.get_mixed_precision_cache(layer_idx)

            # Dequantize all groups and concatenate
            all_keys = []
            all_values = []
            for group_name in ("fp16_recent", "fp8_medium", "int4_old"):
                k_entries = groups[group_name]["keys"]
                v_entries = groups[group_name]["values"]
                if k_entries:
                    k_deq = self.progressive_quantizer._dequantize_entries(k_entries)
                    v_deq = self.progressive_quantizer._dequantize_entries(v_entries)
                    all_keys.extend(k_deq)
                    all_values.extend(v_deq)

            if all_keys:
                key = torch.cat(all_keys, dim=-2)
                value = torch.cat(all_values, dim=-2)
            else:
                # Empty cache
                key = torch.zeros(0)
                value = torch.zeros(0)

            return {"key": key, "value": value}

        # Raw cache path
        if layer_idx not in self._raw_cache:
            return {"key": torch.zeros(0), "value": torch.zeros(0)}

        cache = self._raw_cache[layer_idx]
        # Flatten frames: (F, H, N, D) -> (H, F*N, D)
        key = cache["key"]
        value = cache["value"]

        if key.dim() == 4:
            F, H, N, D = key.shape
            key = key.permute(1, 0, 2, 3).reshape(H, F * N, D)
            value = value.permute(1, 0, 2, 3).reshape(H, F * N, D)
        elif key.dim() == 3:
            F, N, D = key.shape
            key = key.reshape(F * N, D)
            value = value.reshape(F * N, D)

        return {"key": key, "value": value}

    def get_cache(
        self,
        precision: str = "fp16",
    ) -> "Dict[int, Dict[str, torch.Tensor]]":
        """
        Return a copy of the KV cache at the requested precision tier.

        Used by KVQuantSpeculativeDecoder to obtain draft (INT4) and verify
        (FP16) caches for the same set of stored KV entries.

        Args:
            precision: One of "fp16", "fp8", "int4".
              "fp16" — deep copy of _raw_cache (full precision).
              "fp8"  — _raw_cache entries cast to float8_e4m3fn (then dequant to fp16).
              "int4" — re-quantized via ProgressiveKVQuantizer if available,
                       else falls back to fp8 cast.

        Returns:
            Dict[layer_idx -> {"key": Tensor, "value": Tensor}] in float16.
            Shape per tensor: (num_frames, heads, tokens, head_dim).
        """
        if precision == "fp16":
            return {
                layer_idx: {
                    "key":   kv["key"].clone(),
                    "value": kv["value"].clone(),
                }
                for layer_idx, kv in self._raw_cache.items()
            }

        if precision == "int4":
            if self.progressive_quantizer is not None:
                try:
                    return self.progressive_quantizer.get_quantized_cache(
                        target_bits=4, target_dtype=torch.float16
                    )
                except Exception:
                    pass  # fall through to fp8 approximation
            # Fallback: simulate INT4 by round-trip through quantize_to_int4
            result = {}
            for layer_idx, kv in self._raw_cache.items():
                k, v = kv["key"].float(), kv["value"].float()
                try:
                    from worldserve.optimizations.system_level.kv_cache.progressive_quant import ProgressiveKVQuantizer as _PQ
                    pk, sk, zpk = _PQ.quantize_to_int4(k)
                    pv, sv, zpv = _PQ.quantize_to_int4(v)
                    k_deq = _PQ.dequantize_int4(pk, sk, zpk, 128, k.numel(), k.shape, torch.float16)
                    v_deq = _PQ.dequantize_int4(pv, sv, zpv, 128, v.numel(), v.shape, torch.float16)
                    result[layer_idx] = {"key": k_deq, "value": v_deq}
                except Exception:
                    result[layer_idx] = {"key": kv["key"].clone(), "value": kv["value"].clone()}
            return result

        # precision == "fp8"
        result = {}
        try:
            fp8_dtype = torch.float8_e4m3fn
        except AttributeError:
            fp8_dtype = None
        for layer_idx, kv in self._raw_cache.items():
            k, v = kv["key"], kv["value"]
            if fp8_dtype is not None:
                # Cast to fp8 then back to fp16 (simulates fp8 quality)
                k_fp8 = k.to(fp8_dtype)
                v_fp8 = v.to(fp8_dtype)
                result[layer_idx] = {
                    "key":   k_fp8.to(torch.float16),
                    "value": v_fp8.to(torch.float16),
                }
            else:
                result[layer_idx] = {"key": k.clone(), "value": v.clone()}
        return result

    def snapshot(self) -> "Dict[int, Dict[str, torch.Tensor]]":
        """
        Deep-copy the current raw KV cache state.

        Call before a batched speculative draft forward (TreeActionBrancher).
        The returned snapshot can be passed to restore() to undo any KV writes
        that occurred during the speculative forward.

        Returns:
            Deep copy of _raw_cache.
        """
        return {
            layer_idx: {
                "key":   kv["key"].clone(),
                "value": kv["value"].clone(),
            }
            for layer_idx, kv in self._raw_cache.items()
        }

    def restore(
        self,
        snapshot: "Dict[int, Dict[str, torch.Tensor]]",
    ) -> None:
        """
        Restore the KV cache from a previously taken snapshot.

        Call after a batched speculative draft on cache miss (TreeActionBrancher)
        to discard KV writes from the K-1 non-selected branches before running
        the actual forward pass with the correct action.

        Args:
            snapshot: Dict returned by snapshot().
        """
        self._raw_cache = snapshot

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_memory_report(self) -> Dict[str, Any]:
        """
        Report total memory usage and per-technique savings.

        Returns:
            Dict with:
              - ``total_fp16_bytes``: hypothetical full-FP16 size
              - ``total_actual_bytes``: actual bytes used
              - ``savings_ratio``: overall compression ratio
              - ``per_technique``: breakdown by technique
        """
        report: Dict[str, Any] = {
            "total_fp16_bytes": 0,
            "total_actual_bytes": 0,
            "savings_ratio": 0.0,
            "per_technique": {},
        }

        # Raw cache size (FP16 baseline)
        fp16_bytes = 0
        for layer_idx, kv in self._raw_cache.items():
            for component in ("key", "value"):
                t = kv[component]
                fp16_bytes += t.numel() * 2  # FP16 = 2 bytes
        report["total_fp16_bytes"] = fp16_bytes

        # Compute actual bytes compositionally (multiplicative).
        # Temporal merge reduces token count, then quantization reduces
        # per-token byte cost on the remaining tokens.
        actual_bytes = float(fp16_bytes)

        # Temporal merge savings: reduces token count
        if self.temporal_merger is not None:
            tm_stats = self.temporal_merger.get_stats()
            report["per_technique"]["temporal_merge"] = tm_stats
            merge_ratio = tm_stats.get("memory_savings_ratio", 0.0)
            actual_bytes *= (1.0 - merge_ratio)

        # Progressive quant savings: reduces per-token byte cost on remaining tokens
        if self.progressive_quantizer is not None:
            pq_report = self.progressive_quantizer.get_memory_report()
            report["per_technique"]["progressive_quant"] = pq_report
            if pq_report.get("fp16_bytes", 0) > 0:
                quant_ratio = pq_report.get("savings_ratio", 0.0)
                actual_bytes *= (1.0 - quant_ratio)

        # QVG smoothing savings: centroids (full precision) + 4-bit assignment
        # + 2-bit residuals. Effective per-element bits = log2(n_clusters)/D + residual_bits.
        if self.qvg_smoother is not None and hasattr(self, "_qvg_compressed"):
            qvg_orig = 0
            qvg_compressed = 0
            for layer_idx, frames in self._qvg_compressed.items():
                for f_idx, kv in frames.items():
                    for comp in ("key", "value"):
                        c = kv[comp]
                        # Original size: full FP16 of original_shape
                        orig_numel = 1
                        for d in c.original_shape:
                            orig_numel *= d
                        qvg_orig += orig_numel * 2  # fp16
                        # Compressed: centroids fp16 + assignments 4-bit + residuals 2-bit
                        qvg_compressed += (
                            c.centroids.numel() * 2  # centroid full precision
                            + c.assignments.numel() * 0.5  # 4-bit/token (ceil)
                            + c.quantized_residuals.numel()  # already packed bytes
                            + c.residual_scales.numel() * 4  # fp32 scales
                            + c.residual_zero_points.numel() * 4
                        )
            qvg_ratio = (
                1.0 - qvg_compressed / max(qvg_orig, 1)
            ) if qvg_orig > 0 else 0.0
            report["per_technique"]["qvg_smoothing"] = {
                "qvg_orig_bytes": qvg_orig,
                "qvg_compressed_bytes": int(qvg_compressed),
                "savings_ratio": qvg_ratio,
            }
            # When QVG is active alongside progressive_quant we don't double-multiply;
            # report it as a separate alternative compression path.

        actual_bytes = max(int(actual_bytes), 0)
        report["total_actual_bytes"] = actual_bytes
        report["savings_ratio"] = (
            1.0 - actual_bytes / max(fp16_bytes, 1)
        )

        return report

    # OptimizationStack hook interface -----------------------------------

    def pre_step(self, step_idx: int, total_steps: int, latents: Any) -> bool:
        """Called before each denoising step. Returns False (no step skipping)."""
        return False

    def post_step(self, step_idx: int, total_steps: int, latents: Any) -> None:
        """Advance the step counter used by wire_to_model hooks."""
        self._wire_step_counter = step_idx + 1

    def get_quantization_stats(self) -> Dict[str, Any]:
        """
        Report HOW MUCH quantization actually happened during generation —
        not just whether the substitution path was on. Useful for understanding
        the trade-off: we want to know how aggressive the compression was so
        we can correlate it with quality metrics.

        Returns a dict containing:
          calls_{tier}            — number of SDPA hooks routed to this tier
          calls_{tier}_pct        — share of SDPA calls at this tier
          elements_{tier}         — total K/V elements that crossed this tier
          bytes_in_total_mb       — FP16 input volume that hit the substitution path
          bytes_effective_total_mb — same volume after the chosen quantization
          compression_ratio       — bytes_effective / bytes_in (lower = more compression)
          memory_savings_pct      — (1 - compression_ratio) * 100
          quantized_call_pct      — share of calls that were NOT FP16 (i.e. did real work)
          per_frame               — frame-indexed call-tier breakdown
        """
        s = self._quant_stats
        total_calls = s["calls_fp16"] + s["calls_fp8"] + s["calls_int4"]
        if total_calls == 0:
            return {**s, "_note": "no calls observed"}

        pct = lambda n: round(100.0 * n / total_calls, 2)
        compr = (s["bytes_effective"] / s["bytes_in"]) if s["bytes_in"] > 0 else 1.0
        return {
            "calls_total": total_calls,
            "calls_fp16": s["calls_fp16"], "calls_fp16_pct": pct(s["calls_fp16"]),
            "calls_fp8":  s["calls_fp8"],  "calls_fp8_pct":  pct(s["calls_fp8"]),
            "calls_int4": s["calls_int4"], "calls_int4_pct": pct(s["calls_int4"]),
            "elements_fp16": s["elements_fp16"],
            "elements_fp8":  s["elements_fp8"],
            "elements_int4": s["elements_int4"],
            "bytes_in_total_mb":        round(s["bytes_in"] / 1e6, 3),
            "bytes_effective_total_mb": round(s["bytes_effective"] / 1e6, 3),
            "compression_ratio":        round(compr, 4),
            "memory_savings_pct":       round((1.0 - compr) * 100.0, 2),
            "quantized_call_pct":       round(
                100.0 * (s["calls_fp8"] + s["calls_int4"]) / total_calls, 2
            ),
            "per_frame": s["per_frame"],
        }

    def set_current_frame_idx(self, frame_idx: int) -> None:
        """
        Set the current *video frame* index. Should be called from the outer
        autoregressive frame loop in oasis.generate(). When set, the
        wire_to_model hooks pass this as `frame_idx` to update() instead of
        the denoising-step counter, so age thresholds in ProgressiveKVQuantizer
        operate on frames (fp8_age=30 frames, int4_age=90 frames) rather than
        on per-step KV captures.
        """
        self._current_frame_idx = int(frame_idx)

    def set_substitution_active(self, active: bool) -> None:
        """
        Per-frame gate for the quantization substitution path. When False,
        the SDPA hook bypasses the FP8/INT4 roundtrip even though the
        manager has substitution generally enabled. Used to suppress
        quantization on critical frames (e.g. high action magnitude /
        scene transitions) so that the trajectory does not bifurcate at
        moments where it locks in.
        """
        self._substitution_active_now = bool(active)

    # ------------------------------------------------------------------
    # Quantize-dequantize roundtrip helpers (used by attention substitution)
    # ------------------------------------------------------------------

    def _precision_for_frame(self, frame_idx: Optional[int]) -> str:
        """
        Pick a precision tier ('fp16' | 'fp8' | 'int4') for the given frame age,
        using the progressive quantizer's thresholds (default 30 / 90 — override
        from config to e.g. 3 / 10 for short Oasis runs).
        """
        if self.progressive_quantizer is None or frame_idx is None:
            return "fp16"
        # We don't track per-tensor age here — we use the running frame_idx as
        # a proxy. In a 16-frame run with fp8_age=3, frames 0-2 stay FP16,
        # frames 3-9 attend through an FP8 roundtrip, frames 10+ through INT4.
        if frame_idx >= self.progressive_quantizer.k_int4_age:
            return "int4"
        if frame_idx >= self.progressive_quantizer.k_fp8_age:
            return "fp8"
        return "fp16"

    def _quant_roundtrip(self, t: torch.Tensor, precision: str) -> torch.Tensor:
        """
        Quantize then dequantize *t* at the requested precision.
        Returns a tensor of the same shape and dtype as the input but with
        information content limited to the precision tier.

        Cheap implementations (PyTorch-only, no custom kernels) — adequate for
        a numerics/quality benchmark; latency overhead is real.
        """
        if precision == "fp16" or self.progressive_quantizer is None:
            return t

        original_dtype = t.dtype
        original_shape = t.shape

        if precision == "fp8":
            q, scale = self.progressive_quantizer.quantize_to_fp8(t)
            return self.progressive_quantizer.dequantize_fp8(q, scale, target_dtype=original_dtype)

        if precision == "int4":
            packed, scale, zp = self.progressive_quantizer.quantize_to_int4(t, group_size=128)
            return self.progressive_quantizer.dequantize_int4(
                packed, scale, zp, group_size=128,
                original_numel=t.numel(),
                original_shape=original_shape,
                target_dtype=original_dtype,
            )

        return t

    def disable_capture(self) -> None:
        """Bypass F.sdpa monkey-patching in _kv_forward for the next forward call(s).

        Use before auto-accept steps in speculative decoding: the KV tensors
        are not needed, so skipping capture removes the Python-level
        patch/unpatch overhead on all 16 attention modules.
        Re-enable with enable_capture() after the forward completes.
        """
        self._capture_enabled = False

    def enable_capture(self) -> None:
        """Re-enable KV capture after a disable_capture() call."""
        self._capture_enabled = True

    def get_stats(self) -> Dict[str, Any]:
        return self.get_memory_report()

    # Context KV precompute (Diffusion Forcing) ------------------------------------

    def set_context_kv(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """
        Store precomputed KV for context frames (Diffusion Forcing optimization).

        Context frames at t=0 have fixed K, V tensors across all denoising steps.
        After calling this for all layers, arm_context_reuse() enables injection.

        Args:
            layer_idx: DiT layer index.
            key:   (H, S_ctx, D) or (S_ctx, D) key tensor for context tokens.
            value: (H, S_ctx, D) or (S_ctx, D) value tensor for context tokens.
        """
        self._context_kv[layer_idx] = {
            "key": key.detach().clone(),
            "value": value.detach().clone(),
        }
        if key.dim() == 3:
            S_ctx = key.shape[1]
        else:
            S_ctx = key.shape[0]
        self._num_context_tokens = max(self._num_context_tokens, S_ctx)

    def arm_context_reuse(self) -> None:
        """
        Arm context KV reuse after step 0 has populated _context_kv.

        After this is called, wire_to_model hooks will prepend context KVs
        to each attention call, giving the model access to context frames
        without recomputation.
        """
        if self._context_kv:
            self._context_kv_active = True
            import logging
            logger = logging.getLogger(__name__)
            logger.info(
                "KVCacheManager: context KV reuse armed (%d layers, %d ctx tokens)",
                len(self._context_kv),
                self._num_context_tokens,
            )

    def disarm_context_reuse(self) -> None:
        """Disable context KV injection (call between generations)."""
        self._context_kv_active = False
        self._context_kv.clear()
        self._num_context_tokens = 0

    def has_context_kv(self, layer_idx: int) -> bool:
        """Return True if context KV is available for this layer."""
        return self._context_kv_active and layer_idx in self._context_kv

    # KV injection (speculative decoding) --------------------------------

    def inject_into_model(
        self,
        model: Any,
        precision: str = "int4",
    ) -> None:
        """
        Arm KV injection mode for the next forward pass.

        When armed, the ``_capture_sdpa`` hook inside ``wire_to_model`` will
        replace the live K and V tensors with the pre-computed cache stored in
        ``_inject_cache`` instead of passing through the model-computed K/V.
        This is what makes the INT4 draft forward actually cheaper: the model's
        attention layers read from the quantized (lower-precision) KV store
        rather than recomputing full-precision projections and then discarding
        the quantized data.

        Call ``disable_inject()`` after the forward pass completes (use a
        try/finally block to guarantee cleanup).

        Args:
            model: Unused — kept for API symmetry with inject pattern; hooks
                are already wired via ``wire_to_model``.
            precision: ``"int4"`` or ``"fp16"`` — which stored cache to inject.
                Determines which precision snapshot populates ``_inject_cache``.
        """
        self._inject_precision = precision
        self._inject_cache = self.get_cache(precision=precision)
        self._inject_mode = True

    def disable_inject(self) -> None:
        """
        Disarm KV injection mode after a speculative draft/verify forward.

        Safe to call even if injection was never armed.
        """
        self._inject_mode = False
        self._inject_cache = {}

    # Wire attention hooks -----------------------------------------------

    def wire_to_model(self, model: Any) -> int:
        """
        Wire K/V capture hooks into all attention modules in *model*.

        Patches each SpatialAxialAttention / TemporalAxialAttention (and
        generic Attention/WanAttention) module's forward to intercept
        F.scaled_dot_product_attention and store the K and V tensors in
        this cache manager after each forward pass.

        The frame_idx passed to update() is the current denoising step
        index, maintained by post_step().  This lets progressive
        quantization age K/V entries across diffusion timesteps.

        Works correctly when torch.compile is *disabled* (e.g. the
        KV-only ablation benchmark run).  In compiled mode Dynamo may
        inline submodule forwards, so capture is best-effort.

        Returns:
            Number of attention modules wired.
        """
        import types as _types
        import torch.nn.functional as F
        import torch.nn as nn

        _ATTN_CLASS_NAMES = frozenset({
            "SpatialAxialAttention",
            "TemporalAxialAttention",
            "Attention",
            "WanAttention",
            "FlashSelfAttention",
        })

        attn_modules = [
            (idx, name, mod)
            for idx, (name, mod) in enumerate(
                (n, m) for n, m in model.named_modules()
                if type(m).__name__ in _ATTN_CLASS_NAMES
            )
        ]

        if not attn_modules:
            return 0

        manager = self  # close over manager, not self (avoids confusion in nested fn)

        def _make_kv_forward(orig_fwd: Any, l_idx: int) -> Any:
            """Return a replacement forward that captures K/V then calls orig_fwd."""

            @torch.compiler.disable  # prevent Dynamo from inlining through the F.sdpa monkey-patch
            def _kv_forward(*args: Any, **kwargs: Any) -> Any:
                # Fast path: capture disabled (e.g. auto-accept steps in speculative decoding).
                # Skip the F.sdpa monkey-patch entirely — removes per-module patch/unpatch
                # overhead on all attention layers for steps that don't need KV capture.
                if not manager._capture_enabled:
                    return orig_fwd(*args, **kwargs)

                captured: Dict[str, torch.Tensor] = {}
                orig_sdpa = F.scaled_dot_product_attention

                def _capture_sdpa(
                    query: torch.Tensor,
                    key: torch.Tensor,
                    value: torch.Tensor,
                    attn_mask: Optional[torch.Tensor] = None,
                    dropout_p: float = 0.0,
                    is_causal: bool = False,
                    scale: Optional[float] = None,
                    **extra_kw: Any,
                ) -> torch.Tensor:
                    # --- Context KV injection (Diffusion Forcing) ---
                    # If context KVs are armed for this layer, prepend them
                    # to the current K, V so the model attends to both
                    # context frames and the current denoising frames.
                    if manager.has_context_kv(l_idx):
                        ctx = manager._context_kv[l_idx]
                        ctx_k = ctx["key"].to(device=key.device, dtype=key.dtype)
                        ctx_v = ctx["value"].to(device=value.device, dtype=value.dtype)
                        if key.dim() == 4:
                            # (B, H, S, D) layout
                            # ctx_k: (H, S_ctx, D) → (1, H, S_ctx, D) → broadcast
                            ctx_k = ctx_k.unsqueeze(0).expand(key.shape[0], -1, -1, -1)
                            ctx_v = ctx_v.unsqueeze(0).expand(value.shape[0], -1, -1, -1)
                            key = torch.cat([ctx_k, key], dim=2)
                            value = torch.cat([ctx_v, value], dim=2)
                        elif key.dim() == 3:
                            # (H, S, D) layout
                            key = torch.cat([ctx_k, key], dim=1)
                            value = torch.cat([ctx_v, value], dim=1)

                    # --- KV injection (speculative decoding: INT4 draft / FP16 verify) ---
                    # When injection mode is armed (by inject_into_model()), replace the
                    # live K/V tensors with the stored precision-reduced cache so that the
                    # attention computation uses the pre-quantized data rather than the
                    # model's freshly-projected full-precision K/V.  Only the query (Q)
                    # is kept live so the attention output reflects the current input.
                    if manager._inject_mode and l_idx in manager._inject_cache:
                        stored = manager._inject_cache[l_idx]
                        stored_k = stored["key"].to(device=key.device, dtype=key.dtype)
                        stored_v = stored["value"].to(device=value.device, dtype=value.dtype)
                        # Match the layout of the live key/value tensor:
                        # (B, H, S, D) — batch-first 4-D (e.g. after unsqueeze)
                        # (H, S, D)    — head-first 3-D
                        # (F, H, S, D) — frame-first 4-D (raw cache shape)
                        if key.dim() == 4 and stored_k.dim() == 4:
                            # Both 4-D: stored has shape (F, H, S, D); live has (B, H, S, D)
                            # Flatten frame dim to match batch or just use stored as-is
                            F_s, H_s, S_s, D_s = stored_k.shape
                            stored_k = stored_k.permute(1, 0, 2, 3).reshape(H_s, F_s * S_s, D_s).unsqueeze(0).expand(key.shape[0], -1, -1, -1)
                            stored_v = stored_v.permute(1, 0, 2, 3).reshape(H_s, F_s * S_s, D_s).unsqueeze(0).expand(value.shape[0], -1, -1, -1)
                        elif key.dim() == 4 and stored_k.dim() == 3:
                            # stored (H, S, D) → (1, H, S, D) → broadcast to batch
                            stored_k = stored_k.unsqueeze(0).expand(key.shape[0], -1, -1, -1)
                            stored_v = stored_v.unsqueeze(0).expand(value.shape[0], -1, -1, -1)
                        elif key.dim() == 3 and stored_k.dim() == 4:
                            # stored (F, H, S, D) → flatten → (H, F*S, D)
                            F_s, H_s, S_s, D_s = stored_k.shape
                            stored_k = stored_k.permute(1, 0, 2, 3).reshape(H_s, F_s * S_s, D_s)
                            stored_v = stored_v.permute(1, 0, 2, 3).reshape(H_s, F_s * S_s, D_s)
                        # If shapes still mismatch (e.g. empty cache on first step),
                        # fall through to the live K/V to avoid a crash.
                        if stored_k.shape[-2] > 0:
                            key = stored_k
                            value = stored_v

                    # --- KV capture (for future context) ---
                    if "k" not in captured:
                        if key.dim() == 4:
                            captured["k"] = key[0].detach()
                            captured["v"] = value[0].detach()
                        else:
                            captured["k"] = key.detach()
                            captured["v"] = value.detach()

                    # --- Live KV substitution (age-based + per-frame gate) ---
                    # Apply a quantize-dequantize roundtrip to key/value so the
                    # downstream attention sees the precision tier dictated by
                    # the current video-frame age, IF the per-frame gate is on.
                    # The gate (set by set_substitution_active) lets the eval
                    # loop suppress quantization on critical frames (e.g. high
                    # action magnitude) while leaving it on for stable frames.
                    if manager._enable_quant_substitution:
                        gate_on = manager._substitution_active_now
                        prec = (
                            manager._precision_for_frame(manager._current_frame_idx)
                            if gate_on else "fp16"
                        )
                        # ---- accountant: how much quantization actually fires ----
                        stats = manager._quant_stats
                        n_elem = key.numel() + value.numel()
                        bytes_in = n_elem * 2  # FP16 → 2 bytes/elem
                        stats["bytes_in"] += bytes_in
                        stats[f"calls_{prec}"] += 1
                        stats[f"elements_{prec}"] += n_elem
                        if not gate_on:
                            stats["calls_gated_off"] += 1
                        if prec == "fp16":
                            stats["bytes_effective"] += bytes_in
                        elif prec == "fp8":
                            stats["bytes_effective"] += n_elem * 1   # 1 B/elem
                        elif prec == "int4":
                            stats["bytes_effective"] += n_elem // 2  # 0.5 B/elem
                        f_idx = manager._current_frame_idx
                        if f_idx is not None:
                            pf = stats["per_frame"].setdefault(
                                int(f_idx), {"fp16": 0, "fp8": 0, "int4": 0, "gated_off": 0}
                            )
                            pf[prec] += 1
                            if not gate_on:
                                pf["gated_off"] += 1
                        # ---- apply the roundtrip ----
                        if prec != "fp16":
                            key = manager._quant_roundtrip(key, prec)
                            value = manager._quant_roundtrip(value, prec)

                    return orig_sdpa(
                        query, key, value,
                        attn_mask=attn_mask,
                        dropout_p=dropout_p,
                        is_causal=is_causal,
                        scale=scale,
                        **extra_kw,
                    )

                F.scaled_dot_product_attention = _capture_sdpa  # type: ignore[assignment]
                try:
                    result = orig_fwd(*args, **kwargs)
                finally:
                    F.scaled_dot_product_attention = orig_sdpa  # type: ignore[assignment]

                if "k" in captured:
                    try:
                        # Prefer the explicit frame index when the generation
                        # loop has set it (one capture per frame is the right
                        # granularity for age-based quantization). Fall back to
                        # the denoising-step counter when not set.
                        f_idx = (
                            manager._current_frame_idx
                            if manager._current_frame_idx is not None
                            else manager._wire_step_counter
                        )
                        manager.update(
                            {l_idx: {"key": captured["k"], "value": captured["v"]}},
                            frame_idx=f_idx,
                        )
                    except Exception:
                        pass  # Never let cache errors crash inference

                return result

            return _kv_forward

        for l_idx, name, mod in attn_modules:
            orig = mod.forward
            mod.forward = _make_kv_forward(orig, l_idx)  # type: ignore[assignment]

        return len(attn_modules)

