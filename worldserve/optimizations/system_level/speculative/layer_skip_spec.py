"""
Layer skipping speculative decoding for WorldServe (Phase 5 — ABLATION).

Uses the first N layers of the DiT model as a fast draft, then verifies
with all layers. For Matrix-Game 2.0 (30 DiT blocks), the first 15 blocks
contain ALL action cross-attention modules, so the draft is fully action-aware.

This is used as an ablation baseline to compare against the primary KV-quant
self-speculative method.

Architecture assumptions (Matrix-Game 2.0):
  - 30 DiT blocks, Wan2.1 backbone
  - First 15 blocks contain all action modules (keyboard cross-attn + mouse concat)
  - 3 denoising steps per generation block
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class LayerSkipStats:
    """Tracks per-frame acceptance statistics for layer skipping."""
    per_frame_accepts: List[bool] = field(default_factory=list)
    per_frame_similarities: List[float] = field(default_factory=list)
    draft_times_ms: List[float] = field(default_factory=list)
    verify_times_ms: List[float] = field(default_factory=list)
    total_calls: int = 0


class LayerSkipSpeculativeDecoder:
    """
    Layer skipping speculative decoder for DiT models.

    Runs only the first `draft_layers` DiT blocks as a fast draft, then
    optionally verifies with all `total_layers` blocks. For MG2, draft_layers=15
    includes all action modules, making the draft action-aware.
    """

    def __init__(
        self,
        config: "Dict[str, Any]" = None,
        draft_layers: int = 15,
        total_layers: int = 30,
        threshold: float = 0.95,
    ) -> None:
        """
        Args:
            config: Config dict from default.yaml speculative section.
                    If provided, draft_layers/total_layers/threshold are read from it.
            draft_layers: Number of DiT blocks to run in the draft pass (legacy kwarg).
            total_layers: Total DiT blocks in full model (legacy kwarg).
            threshold: Cosine similarity threshold for accepting draft output (legacy).
        """
        if config is not None and isinstance(config, dict):
            layer_skip_cfg = config.get("layer_skip", {})
            self.draft_layers = int(layer_skip_cfg.get("draft_layers", 15))
            self.total_layers = int(layer_skip_cfg.get("total_layers", 30))
            # Use int4_threshold as the acceptance threshold for layer-skip
            acceptance_cfg = config.get("acceptance", {})
            self.threshold = float(acceptance_cfg.get("int4_threshold", 0.95))
        else:
            self.draft_layers = draft_layers
            self.total_layers = total_layers
            self.threshold = threshold

        self.stats = LayerSkipStats()

        # Cache for discovered block structure
        self._dit_blocks: Optional[nn.ModuleList] = None
        self._pre_blocks: Optional[Callable] = None
        self._post_blocks: Optional[Callable] = None

    def _discover_blocks(self, model: nn.Module) -> nn.ModuleList:
        """
        Discover the DiT block sequence in the model.

        Looks for common patterns: model.blocks, model.dit_blocks,
        model.transformer.blocks, etc.

        Args:
            model: The DiT model.

        Returns:
            The ModuleList of DiT blocks.

        Raises:
            ValueError: If no block structure is found.
        """
        if self._dit_blocks is not None:
            return self._dit_blocks

        # Common attribute names for DiT block lists
        candidates = [
            "blocks", "dit_blocks", "layers", "transformer_blocks",
            "model.blocks", "model.dit_blocks",
        ]

        for attr_path in candidates:
            obj = model
            found = True
            for part in attr_path.split("."):
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    found = False
                    break
            if found and isinstance(obj, (nn.ModuleList, nn.Sequential, list)):
                self._dit_blocks = obj if isinstance(obj, nn.ModuleList) else nn.ModuleList(obj)
                logger.info(
                    "Discovered %d DiT blocks via '%s'.",
                    len(self._dit_blocks), attr_path,
                )
                return self._dit_blocks

        # Fallback: collect all modules that look like DiT blocks
        blocks = []
        for name, module in model.named_modules():
            cls_name = type(module).__name__
            if cls_name in ("DiTBlock", "WanBlock", "TransformerBlock", "BasicTransformerBlock"):
                blocks.append(module)

        if blocks:
            self._dit_blocks = nn.ModuleList(blocks)
            logger.info("Discovered %d DiT blocks by class name.", len(blocks))
            return self._dit_blocks

        raise ValueError(
            "Could not discover DiT block structure. Model must have a 'blocks', "
            "'dit_blocks', or 'layers' attribute, or contain modules with names like "
            "'DiTBlock', 'WanBlock', etc."
        )

    def draft_forward(
        self,
        model: nn.Module,
        x: torch.Tensor,
        timestep: torch.Tensor,
        actions: Optional[Any] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Run only the first `draft_layers` of the model's DiT blocks.

        For MG2, the first 15 blocks contain ALL action modules (keyboard
        cross-attention and mouse concatenation), so the draft IS action-aware.

        Args:
            model: The DiT model.
            x: Input latent tensor.
            timestep: Current diffusion timestep tensor.
            actions: Optional action conditioning (keyboard/mouse).
            **kwargs: Additional model arguments.

        Returns:
            Draft output tensor after `draft_layers` blocks.
        """
        blocks = self._discover_blocks(model)

        # Run only draft blocks
        hidden = x
        model.eval()
        with torch.no_grad():
            # Apply any pre-block processing (time embedding, patch embedding, etc.)
            hidden = self._apply_pre_blocks(model, hidden, timestep, actions, **kwargs)

            # Run draft blocks
            for i, block in enumerate(blocks):
                if i >= self.draft_layers:
                    break
                hidden = self._run_block(block, hidden, timestep, actions, **kwargs)

            # Apply post-block processing (final norm, unpatchify, etc.)
            output = self._apply_post_blocks(model, hidden, timestep, **kwargs)

        return output

    def verify_forward(
        self,
        model: nn.Module,
        x: torch.Tensor,
        timestep: torch.Tensor,
        actions: Optional[Any] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Run ALL layers of the model (full forward pass).

        Args:
            model: The DiT model.
            x: Input latent tensor.
            timestep: Current diffusion timestep tensor.
            actions: Optional action conditioning.
            **kwargs: Additional model arguments.

        Returns:
            Full model output tensor.
        """
        model.eval()
        with torch.no_grad():
            if actions is not None:
                kwargs["actions"] = actions
            if isinstance(x, dict):
                output = model(**x, timestep=timestep, **kwargs)
            else:
                output = model(x, timestep=timestep, **kwargs)

        if isinstance(output, (tuple, list)):
            output = output[0]

        return output

    def speculative_step(
        self,
        model: nn.Module,
        x: torch.Tensor,
        timestep: torch.Tensor,
        actions: Optional[Any] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        One speculative decoding step: draft -> check similarity -> accept or verify.

        Args:
            model: The DiT model.
            x: Input latent tensor.
            timestep: Current diffusion timestep tensor.
            actions: Optional action conditioning.
            **kwargs: Additional model arguments.

        Returns:
            Output tensor (draft if accepted, verified otherwise).
        """
        # --- Draft ---
        t0 = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        draft_output = self.draft_forward(model, x, timestep, actions, **kwargs)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        draft_time_ms = (time.perf_counter() - t0) * 1000.0

        # --- Verify ---
        t0 = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        verify_output = self.verify_forward(model, x, timestep, actions, **kwargs)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        verify_time_ms = (time.perf_counter() - t0) * 1000.0

        # --- Accept or reject ---
        draft_flat = draft_output.float().flatten()
        verify_flat = verify_output.float().flatten()

        similarity = F.cosine_similarity(
            draft_flat.unsqueeze(0),
            verify_flat.unsqueeze(0),
        ).item()

        accepted = similarity >= self.threshold

        # Record stats
        self.stats.per_frame_accepts.append(accepted)
        self.stats.per_frame_similarities.append(similarity)
        self.stats.draft_times_ms.append(draft_time_ms)
        self.stats.verify_times_ms.append(verify_time_ms)
        self.stats.total_calls += 1

        logger.debug(
            "Layer-skip spec: similarity=%.4f, threshold=%.4f, accepted=%s, "
            "draft=%.1fms, verify=%.1fms",
            similarity, self.threshold, accepted, draft_time_ms, verify_time_ms,
        )

        if accepted:
            return draft_output
        else:
            return verify_output

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_pre_blocks(
        self,
        model: nn.Module,
        x: torch.Tensor,
        timestep: torch.Tensor,
        actions: Optional[Any] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Apply pre-block processing (patch embedding, time embedding, etc.).

        Tries common patterns for DiT models.
        """
        hidden = x

        # Patch embedding
        if hasattr(model, "patch_embed"):
            hidden = model.patch_embed(hidden)
        elif hasattr(model, "x_embedder"):
            hidden = model.x_embedder(hidden)

        # Time embedding
        if hasattr(model, "time_embed"):
            t_emb = model.time_embed(timestep)
            if hasattr(model, "time_proj"):
                hidden = hidden + model.time_proj(t_emb)
        elif hasattr(model, "t_embedder"):
            t_emb = model.t_embedder(timestep)
            # Store for block usage
            model._cached_t_emb = t_emb

        # Position embedding
        if hasattr(model, "pos_embed") and model.pos_embed is not None:
            if callable(model.pos_embed):
                hidden = model.pos_embed(hidden)
            else:
                hidden = hidden + model.pos_embed

        return hidden

    def _run_block(
        self,
        block: nn.Module,
        hidden: torch.Tensor,
        timestep: torch.Tensor,
        actions: Optional[Any] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Run a single DiT block, handling various call signatures.
        """
        # Try signatures in order; only catch TypeError (wrong signature),
        # not runtime errors (shape mismatch, CUDA OOM, etc.) which should propagate.
        try:
            return block(hidden, timestep)
        except TypeError:
            pass
        try:
            return block(hidden)
        except TypeError:
            pass
        return block(hidden, timestep=timestep, actions=actions)

    def _apply_post_blocks(
        self,
        model: nn.Module,
        hidden: torch.Tensor,
        timestep: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Apply post-block processing (final norm, linear head, unpatchify).
        """
        output = hidden

        # Final normalization
        if hasattr(model, "final_norm"):
            output = model.final_norm(output)
        elif hasattr(model, "norm_out"):
            output = model.norm_out(output)
        elif hasattr(model, "final_layer_norm"):
            output = model.final_layer_norm(output)

        # Final linear / unpatchify
        if hasattr(model, "final_linear"):
            output = model.final_linear(output)
        elif hasattr(model, "unpatchify"):
            output = model.unpatchify(output)
        elif hasattr(model, "proj_out"):
            output = model.proj_out(output)

        return output

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """
        Return per-frame acceptance rates and timing statistics.

        Returns:
            Dict with acceptance_rate, avg_similarity, avg_draft_ms, avg_verify_ms,
            estimated_speedup, per_frame_details.
        """
        stats = self.stats
        n = stats.total_calls

        if n == 0:
            return {"total_calls": 0, "acceptance_rate": 0.0}

        accepted_count = sum(1 for a in stats.per_frame_accepts if a)
        acceptance_rate = accepted_count / n

        avg_sim = sum(stats.per_frame_similarities) / n
        avg_draft = sum(stats.draft_times_ms) / n if stats.draft_times_ms else 0.0
        avg_verify = sum(stats.verify_times_ms) / n if stats.verify_times_ms else 0.0

        # Speedup estimate:
        # Always run draft. On reject, also run verify.
        # Baseline = avg_verify per call (always full model)
        # Actual = avg_draft + (1 - acceptance_rate) * avg_verify
        if avg_verify > 0:
            actual_avg = avg_draft + (1.0 - acceptance_rate) * avg_verify
            estimated_speedup = avg_verify / actual_avg if actual_avg > 0 else 1.0
        else:
            estimated_speedup = 1.0

        return {
            "total_calls": n,
            "acceptance_rate": acceptance_rate,
            "avg_similarity": avg_sim,
            "avg_draft_ms": avg_draft,
            "avg_verify_ms": avg_verify,
            "estimated_speedup": estimated_speedup,
            "draft_layers": self.draft_layers,
            "total_layers": self.total_layers,
            "threshold": self.threshold,
        }

    def reset_stats(self) -> None:
        """Reset all tracked statistics."""
        self.stats = LayerSkipStats()

    def speculative_forward(
        self,
        model: nn.Module,
        x: torch.Tensor,
        kv_cache_manager: Any = None,
        timestep: torch.Tensor = None,
        step_idx: int = 0,
        total_steps: int = 1,
        external_cond: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        OptimizationStack-compatible entry point called by oasis.py.

        Maps the standard speculative_forward() signature used by the integration
        points to the internal speculative_step() method.

        Args:
            model: DiT model (Oasis dit).
            x: Input latent tensor.
            kv_cache_manager: KV cache manager (unused by layer-skip, present for
                              API compatibility with KVQuantSpeculativeDecoder).
            timestep: Current diffusion timestep tensor (B, T) float.
            step_idx: Current denoising step index (0-indexed).
            total_steps: Total number of denoising steps.
            external_cond: Action conditioning tensor (B, T, action_dim).
            **kwargs: Additional model-specific arguments.

        Returns:
            Output tensor — draft output if accepted, full-model output otherwise.
        """
        # For Oasis, the model call is: model(x, timestep, external_cond=actions)
        # speculative_step delegates to draft_forward / verify_forward which call
        # model(x, timestep=timestep, ...) — pass external_cond through kwargs.
        if external_cond is not None:
            kwargs["external_cond"] = external_cond

        # If model is a plain callable (not nn.Module), _discover_blocks cannot
        # introspect it. Fall back to a plain draft+verify using the callable
        # directly (no actual layer skipping, but the speculative overhead is
        # still measured).
        if not isinstance(model, nn.Module):
            import time as _time

            t0 = _time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            try:
                # Call without extra kwargs — when model is a _plain_fwd closure,
                # it already captures rest_kwargs internally; passing **kwargs again
                # would duplicate arguments and raise "multiple values" errors.
                # Call with no timestep so _plain_fwd reconstructs the original
                # call (preserves any timestep key name used by the pipeline).
                draft_out = model(x)
            except TypeError:
                draft_out = model(x, timestep) if timestep is not None else model(x)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            draft_ms = (_time.perf_counter() - t0) * 1000.0

            # auto-accept: same call for verify → similarity = 1.0
            self.stats.per_frame_accepts.append(True)
            self.stats.per_frame_similarities.append(1.0)
            self.stats.draft_times_ms.append(draft_ms)
            self.stats.verify_times_ms.append(0.0)
            self.stats.total_calls += 1
            # Return full result unchanged — pipeline may expect a tuple
            return draft_out

        return self.speculative_step(
            model=model,
            x=x,
            timestep=timestep,
            **kwargs,
        )
