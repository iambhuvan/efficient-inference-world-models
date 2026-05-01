"""
Radial Attention for Matrix-Game 2.0 (Wan2.1 backbone).

Reference: "Radial Attention: O(n log n) Sparse Attention with Energy Decay"
           arxiv 2506.19852

Core idea — Spatiotemporal Energy Decay:
  3D RoPE encodes (i, j, t) positions.  Due to the cosine structure of RoPE,
  the attention energy between two tokens decays monotonically with their
  spatiotemporal distance.  Radial Attention exploits this by shrinking the
  spatial attention window as temporal distance grows:

      spatial_window(temporal_dist) = max(min_window,
                                         base_window >> temporal_dist)

  temporal_dist = 0 → full base_window
  temporal_dist = 1 → base_window / 2
  temporal_dist = 2 → base_window / 4
  ...

  The geometric series Σ_k base_window / 2^k = 2 * base_window, so the
  total non-zeros per query row is O(base_window) regardless of num_frames,
  giving overall mask density O(N * base_window) = O(N log N) when
  base_window ∝ log N.

Architecture assumptions (Matrix-Game 2.0):
  - 15 360 spatial tokens per frame  (44 × 80 × 4.36... ≈ 15 360)
  - 6 rolling KV frames
  - 30 DiT blocks (Wan2.1)
  - 3 denoising steps
  - 3D RoPE (i, j, t axes)
  - Cross-attention used for keyboard conditioning — must NOT be replaced
"""

from __future__ import annotations

import logging
import math
import types
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: spatial grid dimensions from tokens_per_frame
# ---------------------------------------------------------------------------

def _infer_hw(tokens_per_frame: int) -> Tuple[int, int]:
    """Infer (H, W) spatial grid from total token count.

    For MG2: tokens_per_frame = 15360 → H=44, W=349... that does not factor
    cleanly.  We use the largest divisor of tokens_per_frame that is ≤ √N.
    Callers may override by passing explicit grid_h / grid_w to the mask
    builder.
    """
    n = tokens_per_frame
    h = int(math.isqrt(n))
    while h > 1 and n % h != 0:
        h -= 1
    return h, n // h


# ---------------------------------------------------------------------------
# 1. RadialAttentionMask
# ---------------------------------------------------------------------------

class RadialAttentionMask:
    """
    Build a boolean sparse attention mask embodying the radial decay rule.

    For token (t, r, c) attending to (t', r', c'):
      - temporal_dist = |t - t'|
      - window = base_spatial_window >> temporal_dist   (right-shift = halve)
      - window = max(window, min_spatial_window)
      - attend iff |r - r'| <= window // 2 AND |c - c'| <= window // 2

    Parameters
    ----------
    tokens_per_frame:
        Number of spatial tokens per frame.  For MG2 this is 15 360.
    num_frames:
        Number of frames in the rolling KV window (typically 6 for MG2).
    base_spatial_window:
        Full window size at temporal_dist = 0.  Default 64 gives roughly
        64² = 4 096 attended pairs per query at the closest frame.
    min_spatial_window:
        Floor window size (applied once right-shifting would go below it).
    grid_h, grid_w:
        Explicit spatial grid dimensions.  If None, inferred from
        tokens_per_frame via _infer_hw().
    """

    def __init__(
        self,
        tokens_per_frame: int = 15360,
        num_frames: int = 6,
        base_spatial_window: int = 64,
        min_spatial_window: int = 8,
        grid_h: Optional[int] = None,
        grid_w: Optional[int] = None,
    ) -> None:
        self.tokens_per_frame = tokens_per_frame
        self.num_frames = num_frames
        self.base_spatial_window = base_spatial_window
        self.min_spatial_window = min_spatial_window

        if grid_h is not None and grid_w is not None:
            self.grid_h = grid_h
            self.grid_w = grid_w
        else:
            self.grid_h, self.grid_w = _infer_hw(tokens_per_frame)

        # Validate
        if self.grid_h * self.grid_w != tokens_per_frame:
            logger.warning(
                "RadialAttentionMask: grid_h=%d * grid_w=%d = %d != tokens_per_frame=%d. "
                "Mask will be built on the inferred grid; remainder tokens get min window.",
                self.grid_h, self.grid_w, self.grid_h * self.grid_w, tokens_per_frame,
            )

        self._mask_cache: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Mask construction — @torch._dynamo.disable because we use Python
    # for-loops over frame indices and range() which dynamo cannot trace.
    # ------------------------------------------------------------------

    @torch._dynamo.disable  # type: ignore[attr-defined]
    def build_mask(self) -> torch.Tensor:
        """
        Build and return a (T*S, T*S) boolean attention mask.

        True  → token pair is attended (within radial window)
        False → token pair is masked out

        The mask is cached after the first call.  Call clear_cache() to
        force a rebuild.

        Returns
        -------
        torch.Tensor
            Shape (num_frames * tokens_per_frame, num_frames * tokens_per_frame),
            dtype=torch.bool.  Stored on CPU; move to device before use.

        Complexity
        ----------
        Non-zeros per row ≤ Σ_{d=0}^{T-1} w(d)²
          where w(d) = max(min_window, base >> d).
        For base=64, T=6: 64² + 32² + 16² + 8² + 8² + 8² = 4096+1024+256+192 ≈ 5 568
        vs dense 15360² / 15360 = 15360 → ~64× sparsity per row.
        """
        if self._mask_cache is not None:
            return self._mask_cache

        T = self.num_frames
        S = self.tokens_per_frame
        H, W = self.grid_h, self.grid_w
        total = T * S

        # Build coordinate arrays for spatial (row, col) of each token
        # token index within a frame: pos = r * W + c
        rows = torch.arange(H).unsqueeze(1).expand(H, W).reshape(-1)  # (S,)
        cols = torch.arange(W).unsqueeze(0).expand(H, W).reshape(-1)  # (S,)

        # We build the mask in chunks to avoid OOM on large grids
        mask = torch.zeros(total, total, dtype=torch.bool)

        for t in range(T):
            for t_prime in range(T):
                temporal_dist = abs(t - t_prime)
                window = max(
                    self.min_spatial_window,
                    self.base_spatial_window >> temporal_dist,  # right-shift = halve
                )
                half_win = window // 2

                t_offset = t * S
                tp_offset = t_prime * S

                # For each spatial position in frame t, find all positions in
                # frame t' within the spatial window.
                # rows/cols are (S,) tensors; build pairwise diffs.
                # To avoid huge (S, S) intermediate on MG2 (15360 × 15360 = 236 M):
                # process in row-strips of the query frame.
                strip_size = min(256, S)
                for s_start in range(0, S, strip_size):
                    s_end = min(s_start + strip_size, S)
                    # Query positions in this strip
                    q_rows = rows[s_start:s_end]  # (strip,)
                    q_cols = cols[s_start:s_end]  # (strip,)

                    # Key positions: all S tokens in frame t'
                    # diff_r: (strip, S)
                    diff_r = (q_rows.unsqueeze(1) - rows.unsqueeze(0)).abs()
                    diff_c = (q_cols.unsqueeze(1) - cols.unsqueeze(0)).abs()

                    in_window = (diff_r <= half_win) & (diff_c <= half_win)  # (strip, S)

                    mask[
                        t_offset + s_start : t_offset + s_end,
                        tp_offset : tp_offset + S,
                    ] = in_window

        self._mask_cache = mask
        logger.info(
            "RadialAttentionMask: built (%d × %d) mask, "
            "density=%.4f%% (non-zeros=%d)",
            total, total,
            100.0 * mask.float().mean().item(),
            mask.sum().item(),
        )
        return mask

    def clear_cache(self) -> None:
        """Invalidate cached mask (e.g. after changing window parameters)."""
        self._mask_cache = None

    def sparsity_ratio(self) -> float:
        """Return fraction of token pairs that are *masked out* (0 → dense, 1 → empty)."""
        mask = self.build_mask()
        return 1.0 - mask.float().mean().item()

    def nonzeros_per_row(self) -> float:
        """Average number of attended keys per query token."""
        mask = self.build_mask()
        return mask.float().sum(dim=1).mean().item()


# ---------------------------------------------------------------------------
# 2. RadialAttentionModule
# ---------------------------------------------------------------------------

class RadialAttentionModule(nn.Module):
    """
    Drop-in replacement for a self-attention module that uses a precomputed
    radial sparse mask.

    Parameters
    ----------
    embed_dim:
        Token embedding dimension (model width).
    num_heads:
        Number of attention heads.
    radial_mask:
        Precomputed RadialAttentionMask instance.
    dropout:
        Attention dropout probability (applied during training only).
    bias:
        Whether QKV projection layers include a bias term.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        radial_mask: RadialAttentionMask,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.radial_mask = radial_mask

        assert embed_dim % num_heads == 0, (
            f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}"
        )

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Cached device-specific mask (lazily promoted)
        self._device_mask: Optional[torch.Tensor] = None
        self._mask_device: Optional[torch.device] = None

    def _get_mask(self, device: torch.device) -> torch.Tensor:
        """Return mask on the correct device, lazily moving from CPU."""
        if self._device_mask is None or self._mask_device != device:
            cpu_mask = self.radial_mask.build_mask()  # (T*S, T*S) bool
            # Convert to additive float mask: 0 where attended, -inf where masked
            float_mask = torch.zeros_like(cpu_mask, dtype=torch.float32)
            float_mask[~cpu_mask] = float("-inf")
            self._device_mask = float_mask.to(device)
            self._mask_device = device
        return self._device_mask

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x:
            (B, N, C) input token sequence.
        mask:
            Optional additional additive attention mask (e.g. causal).
            Summed with the radial mask if provided.
        key_padding_mask:
            (B, N) bool mask where True indicates padding tokens to ignore.

        Returns
        -------
        torch.Tensor
            (B, N, C) output.
        """
        B, N, C = x.shape
        device = x.device

        Q = self.q_proj(x)  # (B, N, C)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape to (B, H, N, head_dim) for SDPA
        def _split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        Q = _split_heads(Q)  # (B, H, N, D)
        K = _split_heads(K)
        V = _split_heads(V)

        # Build combined mask
        radial_mask_float = self._get_mask(device)  # (T*S, T*S)
        # Slice to actual sequence length (in case of padding)
        if radial_mask_float.shape[0] > N:
            radial_mask_float = radial_mask_float[:N, :N]
        attn_mask = radial_mask_float.unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)

        if mask is not None:
            attn_mask = attn_mask + mask

        if key_padding_mask is not None:
            # key_padding_mask: (B, N) True = ignore
            # Convert to (B, 1, 1, N) additive mask
            kp = key_padding_mask.float().masked_fill(key_padding_mask, float("-inf"))
            kp = kp.masked_fill(~key_padding_mask, 0.0)
            attn_mask = attn_mask + kp.unsqueeze(1).unsqueeze(2)

        dropout_p = self.dropout if self.training else 0.0

        # torch.nn.functional.scaled_dot_product_attention handles the
        # -inf masking natively and dispatches to FlashAttention when available.
        out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
        )  # (B, H, N, D)

        # Merge heads
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)
        return out


# ---------------------------------------------------------------------------
# 3. RadialAttentionReplacer
# ---------------------------------------------------------------------------

class RadialAttentionReplacer:
    """
    Scans a DiT model and replaces self-attention modules with
    RadialAttentionModule, preserving cross-attention untouched.

    Usage
    -----
    >>> replacer = RadialAttentionReplacer(
    ...     tokens_per_frame=15360, num_frames=6,
    ...     base_spatial_window=64, min_window=8,
    ... )
    >>> model = replacer.wrap_model(model)

    Parameters
    ----------
    tokens_per_frame:
        Spatial tokens per frame for MG2 (15 360).
    num_frames:
        Number of rolling-KV frames (6 for MG2).
    base_spatial_window:
        Radial attention base window.  Larger = denser but more accurate.
    min_window:
        Minimum window size — floor for long-range temporal pairs.
    grid_h, grid_w:
        Optional explicit spatial grid; inferred if None.
    """

    # Module class names considered self-attention (cross-attn has "cross" in name)
    SELF_ATTN_CLASSES: Tuple[str, ...] = (
        "Attention",
        "WanAttention",
        "CausalWanSelfAttention",    # Matrix-Game 2.0 (Wan2.1 backbone)
        "FlashSelfAttention",
        "SelfAttention",
        "MultiheadAttention",
    )

    def __init__(
        self,
        tokens_per_frame: int = 15360,
        num_frames: int = 6,
        base_spatial_window: int = 64,
        min_window: int = 8,
        grid_h: Optional[int] = None,
        grid_w: Optional[int] = None,
    ) -> None:
        self.tokens_per_frame = tokens_per_frame
        self.num_frames = num_frames
        self.base_spatial_window = base_spatial_window
        self.min_window = min_window
        self.grid_h = grid_h
        self.grid_w = grid_w

        # Build shared mask once (CPU)
        self.radial_mask = RadialAttentionMask(
            tokens_per_frame=tokens_per_frame,
            num_frames=num_frames,
            base_spatial_window=base_spatial_window,
            min_spatial_window=min_window,
            grid_h=grid_h,
            grid_w=grid_w,
        )

        self._replaced_modules: List[Tuple[nn.Module, str, nn.Module]] = []
        # (parent_module, child_name, original_child)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def wrap_model(self, model: nn.Module) -> nn.Module:
        """
        Replace self-attention modules in-place and return the model.

        Cross-attention modules (names containing "cross") are skipped.
        For each replaced module, the original is stored in
        self._replaced_modules for potential restoration.

        Parameters
        ----------
        model:
            DiT model to patch.

        Returns
        -------
        nn.Module
            Same model with self-attention modules replaced.
        """
        self._replaced_modules.clear()

        # Debug: print all attention modules found and probe result
        attn_modules = self.find_attention_modules(model)
        print(f"[RA DEBUG] SELF_ATTN_CLASSES: {self.SELF_ATTN_CLASSES}")
        print(f"[RA DEBUG] find_attention_modules found {len(attn_modules)} modules:")
        for _par, _cn, _mod, _is_self in attn_modules[:5]:
            _ed, _nh = self._probe_dimensions(_mod)
            # Print non-dunder, non-callable attributes of the first found module
            _attrs = {k: type(v).__name__ for k, v in vars(_mod).items()
                      if not k.startswith('_') and not callable(v)}
            print(f"  {type(_par).__name__}.{_cn}: {type(_mod).__name__} "
                  f"is_self={_is_self} embed_dim={_ed} num_heads={_nh}")
            if _ed is None:
                print(f"    attrs: {list(_attrs.keys())[:20]}")
                # Also check nn.Linear children
                lin_children = [(n, m) for n, m in _mod.named_modules() if isinstance(m, nn.Linear)]
                print(f"    nn.Linear children: {[(n, tuple(m.weight.shape)) for n, m in lin_children[:5]]}")

        for parent, child_name, module, is_self_attn in attn_modules:
            if not is_self_attn:
                continue

            embed_dim, num_heads = self._probe_dimensions(module)
            if embed_dim is None:
                logger.warning(
                    "RadialAttentionReplacer: could not infer embed_dim for "
                    "%s.%s — skipping.", type(parent).__name__, child_name
                )
                continue

            radial_module = RadialAttentionModule(
                embed_dim=embed_dim,
                num_heads=num_heads,
                radial_mask=self.radial_mask,
                dropout=getattr(module, "dropout", 0.0),
                bias=True,
            )
            # Copy existing weights where names/shapes match
            self._copy_weights(module, radial_module)

            self._replaced_modules.append((parent, child_name, module))
            setattr(parent, child_name, radial_module)

        logger.info(
            "RadialAttentionReplacer: replaced %d self-attention modules.",
            len(self._replaced_modules),
        )
        return model

    def restore_model(self, model: nn.Module) -> nn.Module:
        """Restore all replaced modules to their originals."""
        for parent, child_name, original in self._replaced_modules:
            setattr(parent, child_name, original)
        logger.info(
            "RadialAttentionReplacer: restored %d modules.",
            len(self._replaced_modules),
        )
        self._replaced_modules.clear()
        return model

    def find_attention_modules(
        self, model: nn.Module
    ) -> List[Tuple[nn.Module, str, nn.Module, bool]]:
        """
        Scan model for attention modules.

        Returns
        -------
        List of (parent_module, child_attr_name, module, is_self_attn).
        is_self_attn is False for modules whose qualified name contains
        'cross' (cross-attention for keyboard conditioning).
        """
        results: List[Tuple[nn.Module, str, nn.Module, bool]] = []

        for parent_name, parent_module in model.named_modules():
            for child_name, child_module in parent_module.named_children():
                cls_name = type(child_module).__name__
                # Match by explicit whitelist OR by class name containing
                # "attention" or "selfattn" (case-insensitive).
                # Note: "attn" != "attention" — "attention" has 'e' before 'n'.
                # CausalWanSelfAttention is now in SELF_ATTN_CLASSES above.
                cls_lower = cls_name.lower()
                is_attn_cls = (
                    cls_name in self.SELF_ATTN_CLASSES
                    or "selfattention" in cls_lower
                    or "self_attention" in cls_lower
                    or "selfattn" in cls_lower
                )
                if not is_attn_cls:
                    continue
                # Full qualified name
                full_name = f"{parent_name}.{child_name}" if parent_name else child_name
                is_self_attn = "cross" not in full_name.lower()
                results.append((parent_module, child_name, child_module, is_self_attn))

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _probe_dimensions(module: nn.Module) -> Tuple[Optional[int], int]:
        """Infer (embed_dim, num_heads) from common attribute names.

        Supports standard Transformer naming (embed_dim, q_proj, out_proj)
        and Wan2.1/CausalWanSelfAttention naming (dim, q/k/v/o).
        """
        embed_dim: Optional[int] = None
        num_heads: int = 1

        # Attribute-based dim probe (add 'dim' for Wan2.1)
        for attr in ("embed_dim", "hidden_size", "d_model", "width", "dim"):
            v = getattr(module, attr, None)
            if isinstance(v, int):
                embed_dim = v
                break

        # Fallback: infer from Q projection weight shapes
        # Wan2.1 uses module.q (not module.q_proj)
        for proj_attr in ("q_proj", "to_q", "query", "q"):
            proj = getattr(module, proj_attr, None)
            if isinstance(proj, nn.Linear):
                embed_dim = embed_dim or proj.out_features
                break
        if embed_dim is None:
            # Output projection fallback — Wan2.1 uses module.o
            for proj_attr in ("out_proj", "proj", "o"):
                proj = getattr(module, proj_attr, None)
                if isinstance(proj, nn.Linear):
                    embed_dim = proj.in_features
                    break

        for attr in ("num_heads", "nhead", "n_heads", "heads"):
            v = getattr(module, attr, None)
            if isinstance(v, int):
                num_heads = v
                break

        return embed_dim, num_heads

    @staticmethod
    def _copy_weights(src: nn.Module, dst: RadialAttentionModule) -> None:
        """Best-effort weight copy from original attention module."""
        mapping: Dict[str, str] = {
            # common Q/K/V projection names → radial module names
            "q_proj": "q_proj",
            "k_proj": "k_proj",
            "v_proj": "v_proj",
            "to_q": "q_proj",
            "to_k": "k_proj",
            "to_v": "v_proj",
            "out_proj": "out_proj",
            "proj_out": "out_proj",
            "to_out.0": "out_proj",
            # Wan2.1 / CausalWanSelfAttention names (q/k/v/o)
            "q": "q_proj",
            "k": "k_proj",
            "v": "v_proj",
            "o": "out_proj",
        }
        for src_name, dst_name in mapping.items():
            # Handle dotted names
            src_mod = src
            for part in src_name.split("."):
                src_mod = getattr(src_mod, part, None)
                if src_mod is None:
                    break
            if src_mod is None:
                continue
            dst_mod = getattr(dst, dst_name, None)
            if dst_mod is None:
                continue
            if (isinstance(src_mod, nn.Linear) and isinstance(dst_mod, nn.Linear)
                    and src_mod.weight.shape == dst_mod.weight.shape):
                dst_mod.weight.data.copy_(src_mod.weight.data)
                if src_mod.bias is not None and dst_mod.bias is not None:
                    dst_mod.bias.data.copy_(src_mod.bias.data)


# ---------------------------------------------------------------------------
# 4. RadialLoRAAdapter
# ---------------------------------------------------------------------------

class RadialLoRAAdapter(nn.Module):
    """
    LoRA adapter (Hu et al., 2022) for a single linear projection in
    RadialAttentionModule.

    Implements:
        W_eff = W_base + (B @ A) * scale

    where A ∈ R^{r × d_in}, B ∈ R^{d_out × r}, scale = lora_alpha / r.

    Designed to wrap the Q, K, or V projection (nn.Linear) of a
    RadialAttentionModule.  The base weight is frozen; only A and B are
    trained.

    Parameters
    ----------
    base_module:
        The nn.Linear to adapt.
    r:
        LoRA rank.  Default 8 as validated on Wan2.1-14B.
    lora_alpha:
        LoRA scaling factor.  Default 16.  scale = lora_alpha / r = 2.0.
    """

    def __init__(
        self,
        base_module: nn.Linear,
        r: int = 8,
        lora_alpha: float = 16.0,
    ) -> None:
        super().__init__()
        self.base_module = base_module
        self.r = r
        self.lora_alpha = lora_alpha
        self.scale = lora_alpha / r

        d_out, d_in = base_module.weight.shape

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.empty(r, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, r))

        # Kaiming uniform for A (standard LoRA initialisation)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B initialised to zero so delta = 0 at training start

        # Freeze base weights by default
        self.enable_lora_only()

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x:
            (..., d_in) input.

        Returns
        -------
        torch.Tensor
            (..., d_out) = base(x) + lora_delta(x)
        """
        # Base output (no grad through base weight when frozen)
        base_out = F.linear(x, self.base_module.weight, self.base_module.bias)
        # LoRA delta: x @ A^T → (B, N, r), then @ B^T → (B, N, d_out)
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale
        return base_out + lora_out

    def enable_lora_only(self) -> None:
        """Freeze base module weights; only LoRA parameters are trainable."""
        for param in self.base_module.parameters():
            param.requires_grad_(False)
        self.lora_A.requires_grad_(True)
        self.lora_B.requires_grad_(True)

    def enable_full_finetune(self) -> None:
        """Unfreeze everything (base + LoRA) for full fine-tuning."""
        for param in self.base_module.parameters():
            param.requires_grad_(True)
        self.lora_A.requires_grad_(True)
        self.lora_B.requires_grad_(True)

    def get_lora_params(self) -> List[torch.Tensor]:
        """Return the two LoRA parameter tensors for optimiser construction."""
        return [self.lora_A, self.lora_B]

    def merge_weights(self) -> nn.Linear:
        """
        Merge LoRA delta into the base weight and return a plain nn.Linear.

        Useful for inference deployment: eliminates the extra matmul.
        """
        d_out, d_in = self.base_module.weight.shape
        merged = nn.Linear(d_in, d_out, bias=self.base_module.bias is not None)
        merged.weight.data = (
            self.base_module.weight.data + (self.lora_B @ self.lora_A) * self.scale
        )
        if self.base_module.bias is not None:
            merged.bias.data.copy_(self.base_module.bias.data)
        return merged

    @staticmethod
    def apply_to_radial_module(
        radial_module: RadialAttentionModule,
        r: int = 8,
        lora_alpha: float = 16.0,
        target_projections: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj"),
    ) -> RadialAttentionModule:
        """
        Wrap Q, K, V (and optionally out_proj) of a RadialAttentionModule
        with LoRA adapters.

        Parameters
        ----------
        radial_module:
            The module to adapt.
        r:
            LoRA rank.
        lora_alpha:
            LoRA alpha.
        target_projections:
            Which projections to wrap.  Default: q, k, v.

        Returns
        -------
        RadialAttentionModule
            Same module with projections replaced by LoRA-wrapped versions
            (modified in-place).
        """
        for proj_name in target_projections:
            original_proj = getattr(radial_module, proj_name, None)
            if not isinstance(original_proj, nn.Linear):
                continue
            lora_adapter = RadialLoRAAdapter(
                base_module=original_proj,
                r=r,
                lora_alpha=lora_alpha,
            )
            setattr(radial_module, proj_name, lora_adapter)
        return radial_module
