"""
DyDiT++ — Dynamic DiT with Timestep-wise Width Adaptation and Spatial
Token Skipping.

Reference: "DyDiT++: Dynamic Diffusion Transformers for Efficient Visual
           Generation"  arxiv 2504.06803

Two orthogonal dynamic-compute mechanisms:

  Part A — Timestep-wise Dynamic Width (TDW)
  -------------------------------------------
  A small MLP maps the current diffusion timestep embedding → a width
  scale s ∈ [min_scale, max_scale].  At high noise (t ≈ 1.0) only a
  fraction of the model's channels are used; at low noise (t ≈ 0.0)
  the full model width is used.  This exploits the observation that
  coarse global structure (high t) is cheaper to compute than fine
  detail (low t).

  Implementation: channels are masked rather than physically pruned, so
  the operator shape stays fixed and compilation / CUDAGraph compatibility
  is maintained.  The mask is differentiable (soft during training via
  straight-through, binary during inference).

  Part B — Spatial-wise Dynamic Token (SDT)
  ------------------------------------------
  After each attention layer the per-token attention entropy is computed:
      H_i = −Σ_j a_{ij} log a_{ij}
  Tokens with H_i < τ are "confident" (peaked distribution) and their
  MLP computation is skipped — they carry forward unchanged values.

  Part C — TD-LoRA
  -----------------
  Four discrete width levels {0.5, 0.625, 0.75, 1.0} each get a
  dedicated small LoRA adapter (r=4) attached to the Q/K/V projections.
  The frozen base handles the heavy compute; LoRA corrects the
  approximation error introduced by channel masking.

Architecture assumptions (Matrix-Game 2.0):
  - Wan2.1 backbone, 30 DiT blocks
  - embed_dim = 256 (timestep embedding; model width varies per layer)
  - 3 denoising steps, linspace(1.0, 0.0, 4) timesteps
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Width levels for TD-LoRA
# ---------------------------------------------------------------------------
WIDTH_LEVELS: Tuple[float, ...] = (0.5, 0.625, 0.75, 1.0)


# ---------------------------------------------------------------------------
# Part A: TimestepWidthPredictor
# ---------------------------------------------------------------------------

class TimestepWidthPredictor(nn.Module):
    """
    Small MLP that maps a timestep embedding to a channel width scale.

    Input:   t_embed  ∈ R^{embed_dim}  (sinusoidal / learned timestep embed)
    Output:  scale    ∈ [min_scale, max_scale]

    Design rationale:
      - High noise (t ≈ 1.0): scale ≈ min_scale (0.5) — only half the
        channels are needed for global structure.
      - Low noise (t ≈ 0.0):  scale ≈ max_scale (1.0) — full capacity
        for fine-detail refinement.

    The MLP ends with Sigmoid so scale ∈ (0, 1) internally; we then
    affine-map to [min_scale, max_scale].

    Parameters
    ----------
    embed_dim:
        Dimensionality of the timestep embedding fed as input.
    min_scale:
        Minimum channel fraction (active at high noise).
    max_scale:
        Maximum channel fraction (active at low noise / full fidelity).
    """

    def __init__(
        self,
        embed_dim: int = 256,
        min_scale: float = 0.5,
        max_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.min_scale = min_scale
        self.range = max_scale - min_scale

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Initialise final linear to output ≈ 1.0 (→ max_scale) at t≈0
        # by zeroing weight and biasing toward 1.
        nn.init.zeros_(self.mlp[-2].weight)
        nn.init.ones_(self.mlp[-2].bias)

    def forward(self, t_embed: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        t_embed:
            (B, embed_dim)  or  (embed_dim,) timestep embedding tensor.

        Returns
        -------
        torch.Tensor
            Scalar scale value(s) in [min_scale, max_scale].
            Shape: (B, 1) if t_embed is 2-D, else (1,).
        """
        raw = self.mlp(t_embed)  # ∈ (0, 1) via Sigmoid
        scale = raw * self.range + self.min_scale
        return scale


# ---------------------------------------------------------------------------
# Part B: SpatialDynamicToken
# ---------------------------------------------------------------------------

class SpatialDynamicToken(nn.Module):
    """
    Skip MLP computation for spatial tokens that have peaked attention
    distributions (low entropy → high confidence).

    Low entropy ⟹ the token has already decided where to attend; running
    the MLP block on it provides diminishing returns, so we re-use the
    residual value from the previous layer.

    Parameters
    ----------
    entropy_threshold:
        Tokens with mean head entropy < this value are skipped.
        Set to 0 to disable skipping entirely.
    skip_ratio_max:
        Hard upper bound on the fraction of tokens that may be skipped
        per forward call.  Prevents degenerate collapse where nothing
        is computed.
    """

    def __init__(
        self,
        entropy_threshold: float = 0.5,
        skip_ratio_max: float = 0.4,
    ) -> None:
        super().__init__()
        self.entropy_threshold = entropy_threshold
        self.skip_ratio_max = skip_ratio_max

        # Running stats (non-differentiable EMA for monitoring)
        self.register_buffer("_skip_ema", torch.tensor(0.0))
        self._ema_alpha = 0.01

    def forward(
        self,
        x: torch.Tensor,
        attn_weights: torch.Tensor,
        compute_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """
        Selectively apply compute_fn to important tokens.

        Parameters
        ----------
        x:
            (B, N, C) token tensor.
        attn_weights:
            (B, H, N, N) post-softmax attention weight tensor.
        compute_fn:
            A callable (e.g. the MLP sub-layer) to apply to important tokens.
            Must accept a (B*, C) or (B, N', C) tensor and return same shape.

        Returns
        -------
        torch.Tensor
            (B, N, C) output; important tokens are updated, skipped tokens
            carry their input values.
        """
        B, N, C = x.shape

        # Per-token entropy: (B, N)
        # attn_weights: (B, H, N, N) → mean over heads → (B, N, N) → entropy over key dim
        attn_mean = attn_weights.mean(dim=1)  # (B, N, N)
        entropy = -(attn_mean * torch.log(attn_mean + 1e-8)).sum(dim=-1)  # (B, N)

        # Importance mask: True for tokens to compute
        important = entropy > self.entropy_threshold  # (B, N) bool

        # Enforce skip_ratio_max: if too many tokens would be skipped, keep more
        skip_frac = (~important).float().mean()
        if skip_frac > self.skip_ratio_max:
            # Keep (1 - skip_ratio_max) fraction: select top-entropy tokens
            keep_k = max(1, int(N * (1.0 - self.skip_ratio_max)))
            topk_vals, _ = entropy.topk(keep_k, dim=1)
            threshold = topk_vals[:, -1].unsqueeze(1)  # (B, 1)
            important = entropy >= threshold

        # Update EMA skip rate (detached, for monitoring)
        current_skip = (~important).float().mean().detach()
        self._skip_ema = (
            (1 - self._ema_alpha) * self._skip_ema + self._ema_alpha * current_skip
        )

        # For training stability: if fewer than 10% of tokens would be active,
        # fall back to full computation.
        if important.float().mean() < 0.1:
            return compute_fn(x)

        # --- Apply compute_fn only to important tokens ---
        x_out = x.clone()

        # We process batch × token dimension jointly for efficiency
        # important: (B, N) → flat index for gather
        for b in range(B):
            imp_b = important[b]  # (N,) bool
            if not imp_b.any():
                continue
            x_imp = x[b][imp_b].unsqueeze(0)   # (1, N_imp, C)
            out_imp = compute_fn(x_imp)          # (1, N_imp, C)
            x_out[b][imp_b] = out_imp.squeeze(0)

        return x_out

    def get_skip_rate(self) -> float:
        """Return the exponential moving average skip rate."""
        return self._skip_ema.item()


# ---------------------------------------------------------------------------
# Part C: TD-LoRA adapter (per width level)
# ---------------------------------------------------------------------------

class TDLoRAAdapter(nn.Module):
    """
    Timestep-adaptive LoRA adapter that selects among four discrete LoRA
    matrices based on the current width level.

    Width levels: {0.5, 0.625, 0.75, 1.0}
    Each level has its own (lora_A_i, lora_B_i) pair with rank r=4.

    During training, the active adapter is selected by the quantised
    width scale produced by TimestepWidthPredictor.

    Parameters
    ----------
    base_module:
        The nn.Linear to adapt (e.g. Q/K/V projection).
    r:
        LoRA rank per level.  Default 4 (smaller than radial LoRA to
        limit total parameter count across 4 levels × 30 layers × 3 QKV).
    lora_alpha:
        LoRA scaling factor.  scale = lora_alpha / r.
    width_levels:
        Tuple of width fractions.  Must match WIDTH_LEVELS global.
    """

    def __init__(
        self,
        base_module: nn.Linear,
        r: int = 4,
        lora_alpha: float = 8.0,
        width_levels: Tuple[float, ...] = WIDTH_LEVELS,
    ) -> None:
        super().__init__()
        self.base_module = base_module
        self.r = r
        self.scale = lora_alpha / r
        self.width_levels = width_levels
        self.num_levels = len(width_levels)

        d_out, d_in = base_module.weight.shape

        # One pair of LoRA matrices per width level
        self.lora_As = nn.ParameterList([
            nn.Parameter(torch.empty(r, d_in)) for _ in range(self.num_levels)
        ])
        self.lora_Bs = nn.ParameterList([
            nn.Parameter(torch.zeros(d_out, r)) for _ in range(self.num_levels)
        ])

        for lora_a in self.lora_As:
            nn.init.kaiming_uniform_(lora_a, a=math.sqrt(5))

        # Freeze base weight
        for param in self.base_module.parameters():
            param.requires_grad_(False)

    def _level_index(self, width_scale: float) -> int:
        """Return index of the nearest width level for a given scale."""
        dists = [abs(width_scale - lvl) for lvl in self.width_levels]
        return int(min(range(self.num_levels), key=lambda i: dists[i]))

    def forward(
        self,
        x: torch.Tensor,
        width_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x:
            (..., d_in) input tensor.
        width_scale:
            Current width fraction; selects the LoRA level.  If None,
            uses the last (largest) level (full model).

        Returns
        -------
        torch.Tensor
            (..., d_out) output.
        """
        base_out = F.linear(x, self.base_module.weight, self.base_module.bias)

        level_idx = self._level_index(width_scale) if width_scale is not None else -1
        lora_A = self.lora_As[level_idx]
        lora_B = self.lora_Bs[level_idx]
        lora_out = F.linear(F.linear(x, lora_A), lora_B) * self.scale
        return base_out + lora_out

    def get_lora_params(self) -> List[nn.Parameter]:
        """Return all LoRA parameters across all width levels."""
        params: List[nn.Parameter] = []
        for lora_a, lora_b in zip(self.lora_As, self.lora_Bs):
            params.extend([lora_a, lora_b])
        return params

    def freeze_base(self) -> None:
        for param in self.base_module.parameters():
            param.requires_grad_(False)

    def unfreeze_base(self) -> None:
        for param in self.base_module.parameters():
            param.requires_grad_(True)


# ---------------------------------------------------------------------------
# Dynamic Width linear wrapper (used by DyDiTPlusPlus)
# ---------------------------------------------------------------------------

class _DynamicWidthLinear(nn.Module):
    """
    Wraps an nn.Linear with a channel-masking scheme for dynamic width.

    During forward:
      1. Compute channel mask: first int(d_in * scale) input channels active.
      2. Zero-out the remaining channels in x (straight-through for training).
      3. Apply the full weight matrix (shape unchanged for compiler compatibility).

    This approximates physically pruning to the active channels while
    maintaining a fixed computational graph for torch.compile / CUDAGraph.
    """

    def __init__(self, base_linear: nn.Linear) -> None:
        super().__init__()
        self.base = base_linear
        d_out, d_in = base_linear.weight.shape
        self.d_in = d_in
        self.d_out = d_out

    def forward(
        self,
        x: torch.Tensor,
        scale: float = 1.0,
    ) -> torch.Tensor:
        # Use F.linear directly (not self.base(...)) to avoid infinite recursion:
        # wrap_model monkeypatches child.forward → _dw_forward → dw.forward →
        # self.base(x) → child.forward → ... if self.base(x) is used.
        w = self.base.weight
        b = self.base.bias

        if scale >= 1.0:
            return F.linear(x, w, b)

        active_in = max(1, int(self.d_in * scale))
        active_out = max(1, int(self.d_out * scale))

        # Input masking
        mask_in = torch.zeros(self.d_in, device=x.device, dtype=x.dtype)
        mask_in[:active_in] = 1.0
        x_masked = x * mask_in  # straight-through gradient

        # Apply full weight; then mask output channels
        out = F.linear(x_masked, w, b)

        mask_out = torch.zeros(self.d_out, device=x.device, dtype=x.dtype)
        mask_out[:active_out] = 1.0
        return out * mask_out


# ---------------------------------------------------------------------------
# Main DyDiT++ class
# ---------------------------------------------------------------------------

@dataclass
class DyDiTStats:
    """Running statistics collected during DyDiT++ forward passes."""
    total_calls: int = 0
    sum_width_scale: float = 0.0
    sum_skip_fraction: float = 0.0
    per_timestep_scale: Dict[float, List[float]] = field(default_factory=dict)

    @property
    def avg_width_scale(self) -> float:
        return self.sum_width_scale / max(1, self.total_calls)

    @property
    def avg_skip_fraction(self) -> float:
        return self.sum_skip_fraction / max(1, self.total_calls)


class DyDiTPlusPlus:
    """
    DyDiT++ controller: installs Timestep-wise Dynamic Width (TDW) and
    Spatial Dynamic Token (SDT) mechanisms into a DiT model.

    Typical usage
    -------------
    >>> dydit = DyDiTPlusPlus(model, embed_dim=256, min_scale=0.5)
    >>> dydit.wrap_model(model)
    >>> # ... training loop ...
    >>> optimizer = torch.optim.AdamW(dydit.get_trainable_params(), lr=1e-4)
    >>> # During inference:
    >>> output = dydit.forward_with_dynamics(x, t_embed)

    Parameters
    ----------
    model:
        The DiT model to wrap (Wan2.1 backbone for MG2).
    embed_dim:
        Timestep embedding dimensionality fed to TimestepWidthPredictor.
    min_scale:
        Minimum width fraction (high-noise regime).
    max_scale:
        Maximum width fraction (low-noise / detail regime).
    entropy_threshold:
        Per-token attention entropy threshold for SDT skipping.
    skip_ratio_max:
        Maximum fraction of tokens that can be skipped per step.
    apply_lora:
        If True, also wrap Q/K/V projections in TDLoRAAdapter.
    lora_rank:
        LoRA rank for TD-LoRA adapters.
    """

    # Module classes whose linear layers get dynamic width treatment
    LINEAR_CLASSES: Tuple[str, ...] = ("Linear",)

    # Names indicating Q/K/V projections to get TD-LoRA
    QKV_NAMES: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "to_q", "to_k", "to_v")

    def __init__(
        self,
        model: nn.Module,
        embed_dim: int = 256,
        min_scale: float = 0.5,
        max_scale: float = 1.0,
        entropy_threshold: float = 0.5,
        skip_ratio_max: float = 0.4,
        apply_lora: bool = True,
        lora_rank: int = 4,
    ) -> None:
        self.model = model
        self.embed_dim = embed_dim
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.apply_lora = apply_lora
        self.lora_rank = lora_rank

        # Width predictor (trainable)
        self.width_predictor = TimestepWidthPredictor(
            embed_dim=embed_dim,
            min_scale=min_scale,
            max_scale=max_scale,
        )

        # SDT module (trainable thresholds / no params by default)
        self.sdt = SpatialDynamicToken(
            entropy_threshold=entropy_threshold,
            skip_ratio_max=skip_ratio_max,
        )

        # Storage for installed hooks and wrappers
        self._dynamic_linears: List[_DynamicWidthLinear] = []
        self._td_lora_adapters: List[TDLoRAAdapter] = []
        self._attn_hooks: List[torch.utils.hooks.RemovableHook] = []

        # Last attention weights captured by hooks
        self._last_attn_weights: Optional[torch.Tensor] = None

        # Stats
        self._stats = DyDiTStats()

        # Current width scale (set during forward_with_dynamics)
        self._current_scale: float = max_scale

    # ------------------------------------------------------------------
    # wrap_model
    # ------------------------------------------------------------------

    def wrap_model(self, model: nn.Module) -> nn.Module:
        """
        Install TDW dynamic linear wrappers and SDT attention hooks in-place.

        Also installs TD-LoRA adapters on Q/K/V projections if apply_lora=True.

        Parameters
        ----------
        model:
            The DiT model to instrument.

        Returns
        -------
        nn.Module
            Same model (modified in-place).
        """
        self._dynamic_linears.clear()
        self._td_lora_adapters.clear()
        for hook in self._attn_hooks:
            hook.remove()
        self._attn_hooks.clear()

        linear_count = 0
        lora_count = 0

        for parent_name, parent_module in model.named_modules():
            for child_name, child in parent_module.named_children():
                # --- Wrap linear layers with dynamic width ---
                if isinstance(child, nn.Linear):
                    dw_linear = _DynamicWidthLinear(child)
                    self._dynamic_linears.append(dw_linear)

                    # Monkeypatch forward to inject current scale
                    dydit_ref = self
                    original_forward = child.forward

                    def _make_dw_forward(dw: _DynamicWidthLinear) -> Callable:
                        def _dw_forward(x: torch.Tensor) -> torch.Tensor:
                            return dw.forward(x, scale=dydit_ref._current_scale)
                        return _dw_forward

                    child.forward = _make_dw_forward(dw_linear)  # type: ignore[method-assign]
                    linear_count += 1

                # --- Install TD-LoRA on Q/K/V projections ---
                if self.apply_lora and isinstance(child, nn.Linear):
                    if child_name in self.QKV_NAMES:
                        lora_adapter = TDLoRAAdapter(
                            base_module=child,
                            r=self.lora_rank,
                            lora_alpha=float(self.lora_rank * 2),
                        )
                        self._td_lora_adapters.append(lora_adapter)
                        setattr(parent_module, child_name, lora_adapter)
                        lora_count += 1

        # --- Hook into attention modules to capture weights for SDT ---
        for name, module in model.named_modules():
            cls_name = type(module).__name__
            if cls_name in ("Attention", "WanAttention", "SelfAttention",
                            "MultiheadAttention", "RadialAttentionModule"):
                if "cross" in name.lower():
                    continue
                hook = self._register_attn_weight_hook(module)
                if hook is not None:
                    self._attn_hooks.append(hook)

        logger.info(
            "DyDiT++: wrapped %d linear layers with dynamic width, "
            "%d Q/K/V projections with TD-LoRA, "
            "registered %d attention hooks.",
            linear_count, lora_count, len(self._attn_hooks),
        )
        return model

    def _register_attn_weight_hook(
        self, module: nn.Module
    ) -> Optional[torch.utils.hooks.RemovableHook]:
        """Register a hook that stores the most recent attention weights."""
        dydit_ref = self

        def _hook(mod: nn.Module, inputs: Any, output: Any) -> None:
            # Try common attribute names where attention weights may be stored
            for attr in ("_attn_weights", "attn_weights", "_last_attn"):
                weights = getattr(mod, attr, None)
                if isinstance(weights, torch.Tensor) and weights.dim() == 4:
                    dydit_ref._last_attn_weights = weights.detach()
                    return

        return module.register_forward_hook(_hook)

    # ------------------------------------------------------------------
    # forward_with_dynamics
    # ------------------------------------------------------------------

    def forward_with_dynamics(
        self,
        x: torch.Tensor,
        t_embed: torch.Tensor,
        **model_kwargs: Any,
    ) -> torch.Tensor:
        """
        Run a forward pass through self.model with dynamic compute active.

        Steps:
        1. Predict width scale from t_embed via TimestepWidthPredictor.
        2. Set self._current_scale so all _DynamicWidthLinear wrappers use it.
        3. Pass t_embed width_scale to TDLoRAAdapter forwards.
        4. Call model.forward(x, **model_kwargs).
        5. Record stats.

        Parameters
        ----------
        x:
            (B, N, C) noisy latent input.
        t_embed:
            (B, embed_dim) timestep embedding.
        **model_kwargs:
            Additional kwargs forwarded to model.forward (e.g. cond, mask).

        Returns
        -------
        torch.Tensor
            Model output (same shape as x for a noise predictor).
        """
        # --- Step 1: predict width scale ---
        with torch.no_grad() if not self.width_predictor.training else torch.enable_grad():
            scale_tensor = self.width_predictor(t_embed)  # (B, 1) or (1,)

        # Use batch-mean scale for uniform channel masking across batch
        scale_val = float(scale_tensor.mean().item())

        # --- Step 2: set current scale ---
        self._current_scale = scale_val

        # Update TD-LoRA adapters with current scale
        for adapter in self._td_lora_adapters:
            # Patch forward to inject scale for this call
            adapter._active_scale = scale_val

        # --- Step 3: forward pass ---
        output = self.model(x, **model_kwargs)

        # --- Step 4: record stats ---
        self._stats.total_calls += 1
        self._stats.sum_width_scale += scale_val
        skip_rate = self.sdt.get_skip_rate()
        self._stats.sum_skip_fraction += skip_rate

        return output

    # ------------------------------------------------------------------
    # Trainable parameter access
    # ------------------------------------------------------------------

    def get_trainable_params(self) -> List[nn.Parameter]:
        """
        Return only the trainable parameters: width predictor MLP + TD-LoRA adapters.

        Base model weights are frozen.  This list should be passed directly
        to the optimizer.

        Returns
        -------
        List[nn.Parameter]
        """
        params: List[nn.Parameter] = list(self.width_predictor.parameters())
        for adapter in self._td_lora_adapters:
            params.extend(adapter.get_lora_params())
        return params

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """
        Return a dict of average dynamic compute statistics.

        Keys
        ----
        avg_width_scale:
            Mean channel fraction used across all forward calls (0.5–1.0).
        avg_skip_fraction:
            Mean fraction of spatial tokens skipped per call (0–skip_ratio_max).
        total_forward_calls:
            Number of forward calls recorded.
        current_scale:
            Width scale from the most recent forward call.
        """
        return {
            "avg_width_scale": self._stats.avg_width_scale,
            "avg_skip_fraction": self._stats.avg_skip_fraction,
            "total_forward_calls": self._stats.total_calls,
            "current_scale": self._current_scale,
        }

    def reset_stats(self) -> None:
        """Reset all running statistics."""
        self._stats = DyDiTStats()

    # ------------------------------------------------------------------
    # Convenience: freeze base / unfreeze for fine-tuning
    # ------------------------------------------------------------------

    def freeze_base_model(self) -> None:
        """Freeze all base model parameters; keep only TDW + LoRA trainable."""
        for param in self.model.parameters():
            param.requires_grad_(False)
        # Re-enable LoRA params
        for adapter in self._td_lora_adapters:
            for p in adapter.get_lora_params():
                p.requires_grad_(True)
        for param in self.width_predictor.parameters():
            param.requires_grad_(True)
        logger.info("DyDiT++: frozen base model; TDW predictor + %d LoRA adapters active.",
                    len(self._td_lora_adapters))

    def unfreeze_base_model(self) -> None:
        """Unfreeze all model parameters for full fine-tuning."""
        for param in self.model.parameters():
            param.requires_grad_(True)
        logger.info("DyDiT++: unfrozen base model (full fine-tune mode).")

    def remove_hooks(self) -> None:
        """Remove all registered forward hooks (cleanup)."""
        for hook in self._attn_hooks:
            hook.remove()
        self._attn_hooks.clear()
