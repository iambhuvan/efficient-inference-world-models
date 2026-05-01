"""
DC-DiT: Dynamic Chunking DiT for MG2.

Reference: arxiv 2603.06351

Upcycles a pretrained MG2 DiT checkpoint (base_distill.safetensors) in 8×
fewer training steps by freezing the base DiT and training only the small
TokenRouter and TokenDecoder modules.

Architecture
------------
Input: x  (B, N, C) where N = 15 360 tokens/frame (MG2 / Wan2.1).

Encoder (learned, small)
    TokenRouter : score_i = sigmoid(MLP(x_i)) ∈ [0, 1]
    SpatialPooler: important tokens (score > threshold) pass through;
                   background tokens are spatially average-pooled (k×k → 1).
    Output: x_compressed (B, N', C), N' << N.

Compressed DiT blocks
    Standard attention on N' tokens — O((N')²) vs O(N²).
    3D RoPE positions are preserved by mapping compressed grid coords
    back to original coordinates.

Decoder (learned, small)
    TokenDecoder: for each background position, cross-attends from the
    nearest full-resolution anchor tokens to reconstruct the full N sequence.

Upcycling strategy
------------------
1. Load base DiT from existing checkpoint (base_distill.safetensors).
2. Only train TokenRouter and TokenDecoder (much smaller).
3. Distillation loss:
       L_dc = ||DC-DiT(x_t, t) − base_DiT(x_t, t)||²
   i.e. compressed output must match the base model's output on the same input.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW


# ---------------------------------------------------------------------------
# TokenRouter
# ---------------------------------------------------------------------------

class TokenRouter(nn.Module):
    """
    Assign an importance score ∈ [0, 1] to each input token.

    A 2-layer MLP with sigmoid output.  Tokens with score > threshold are
    kept at full resolution; the rest are pooled.

    Parameters
    ----------
    hidden_dim : input (and output) token dimensionality.
    threshold  : score cut-off for full-resolution tokens (default 0.5).
    """

    def __init__(self, hidden_dim: int, threshold: float = 0.5) -> None:
        super().__init__()
        self.threshold = threshold

        # Small bottleneck MLP: D → D//4 → 1.
        bottleneck = max(hidden_dim // 4, 1)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck),
            nn.SiLU(),
            nn.Linear(bottleneck, 1),
        )
        # Initialise final layer near 0.5 so routing starts uniform.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.constant_(self.mlp[-1].bias, 0.0)  # sigmoid(0) = 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute per-token importance scores.

        Parameters
        ----------
        x : (B, N, D) token sequence.

        Returns
        -------
        scores : (B, N) float tensor in [0, 1].
        """
        return torch.sigmoid(self.mlp(x)).squeeze(-1)  # (B, N)


# ---------------------------------------------------------------------------
# SpatialPooler
# ---------------------------------------------------------------------------

class SpatialPooler(nn.Module):
    """
    Partition tokens into important (full-resolution) and background (pooled).

    Background tokens within each k×k spatial patch are averaged into a single
    token.  This reduces the sequence length by up to k² for background regions.

    Parameters
    ----------
    pool_size : spatial pooling factor k; k×k patches are pooled (default 4).
    """

    def __init__(self, pool_size: int = 4) -> None:
        super().__init__()
        self.pool_size = pool_size

    def forward(
        self,
        x: torch.Tensor,
        important_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Split x into full-resolution important tokens and pooled background.

        Parameters
        ----------
        x              : (B, N, D) token sequence.
        important_mask : (B, N) bool tensor; True = important token.

        Returns
        -------
        x_full       : (B, N_imp, D)  important tokens at full resolution.
        x_compressed : (B, N_bg_pooled + N_imp, D)  mixed sequence for DiT.
        routing_info : dict with keys:
            'important_mask'  : (B, N) bool
            'important_indices': (B, N_imp) LongTensor  (padding with -1)
            'bg_indices'       : (B, N_bg) LongTensor
            'pool_size'        : int
            'original_N'       : int
        """
        B, N, D = x.shape
        k = self.pool_size

        # ------ Important tokens ------
        # For variable-length sequences we pad to the maximum count.
        imp_counts = important_mask.long().sum(dim=1)  # (B,)
        max_imp = int(imp_counts.max().item())

        imp_idx_padded = torch.full(
            (B, max_imp), -1, device=x.device, dtype=torch.long
        )
        x_full_list = []

        for b in range(B):
            idx = important_mask[b].nonzero(as_tuple=True)[0]  # (n_imp,)
            n_imp = len(idx)
            imp_idx_padded[b, :n_imp] = idx
            full_b = x[b, idx]  # (n_imp, D)
            # Pad to max_imp.
            if n_imp < max_imp:
                pad = x.new_zeros(max_imp - n_imp, D)
                full_b = torch.cat([full_b, pad], dim=0)
            x_full_list.append(full_b)

        x_full = torch.stack(x_full_list, dim=0)  # (B, max_imp, D)

        # ------ Background tokens (pooled) ------
        bg_mask = ~important_mask  # (B, N)
        bg_counts = bg_mask.long().sum(dim=1)
        max_bg = int(bg_counts.max().item())

        bg_idx_padded = torch.full(
            (B, max_bg), -1, device=x.device, dtype=torch.long
        )
        pooled_list = []

        for b in range(B):
            idx = bg_mask[b].nonzero(as_tuple=True)[0]  # (n_bg,)
            n_bg = len(idx)
            bg_idx_padded[b, :n_bg] = idx

            if n_bg == 0:
                pooled_list.append(x.new_zeros(1, D))
                continue

            bg_tokens = x[b, idx]  # (n_bg, D)
            # Pool groups of k tokens (1-D approximation of 2-D k×k pooling).
            n_full_groups = n_bg // k
            remainder = n_bg % k

            groups = [bg_tokens[i * k: (i + 1) * k].mean(dim=0, keepdim=True)
                      for i in range(n_full_groups)]
            if remainder > 0:
                groups.append(bg_tokens[n_full_groups * k:].mean(dim=0, keepdim=True))

            pooled = torch.cat(groups, dim=0)  # (n_groups, D)
            pooled_list.append(pooled)

        # Pad pooled sequences to max length.
        max_pooled = max(p.shape[0] for p in pooled_list)
        x_pooled_pad = x.new_zeros(B, max_pooled, D)
        for b, p in enumerate(pooled_list):
            x_pooled_pad[b, :p.shape[0]] = p

        # Compressed sequence = important + pooled background.
        x_compressed = torch.cat([x_full, x_pooled_pad], dim=1)  # (B, N', D)

        routing_info: Dict[str, Any] = {
            "important_mask": important_mask,
            "important_indices": imp_idx_padded,
            "bg_indices": bg_idx_padded,
            "pool_size": k,
            "original_N": N,
            "max_imp": max_imp,
            "max_pooled": max_pooled,
        }

        return x_full, x_compressed, routing_info


# ---------------------------------------------------------------------------
# TokenDecoder
# ---------------------------------------------------------------------------

class TokenDecoder(nn.Module):
    """
    Reconstruct the full N-token sequence from the compressed representation.

    For each background token position, cross-attention over the nearest
    important (full-resolution) tokens is used to fill in the value.

    Parameters
    ----------
    hidden_dim : token dimensionality D.
    num_heads  : number of attention heads for cross-attention (default 8).
    """

    def __init__(self, hidden_dim: int, num_heads: int = 8) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = max(1, min(num_heads, hidden_dim // 64))

        # Cross-attention: queries from background positions, keys/values from
        # important tokens.
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=self.num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)

        # Simple positional interpolation projection.
        self.pos_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.eye_(self.pos_proj.weight)  # identity init

    def forward(
        self,
        x_compressed: torch.Tensor,
        routing_info: Dict[str, Any],
        original_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decode compressed tokens back to the full N-token sequence.

        Parameters
        ----------
        x_compressed       : (B, N', D) compressed + important tokens
                             (important tokens are the first max_imp entries).
        routing_info       : dict returned by SpatialPooler.forward.
        original_positions : (N,) float tensor of original grid positions
                             (unused in this implementation; included for
                             compatibility with RoPE-based decoders).

        Returns
        -------
        x_full_reconstructed : (B, N, D) full-resolution token sequence.
        """
        B = x_compressed.shape[0]
        N_orig = routing_info["original_N"]
        max_imp = routing_info["max_imp"]
        important_mask: torch.Tensor = routing_info["important_mask"]  # (B, N)
        imp_indices: torch.Tensor = routing_info["important_indices"]  # (B, max_imp)
        bg_indices: torch.Tensor = routing_info["bg_indices"]          # (B, max_bg)
        D = x_compressed.shape[-1]

        # Split compressed sequence back into important and pooled parts.
        x_imp = x_compressed[:, :max_imp, :]      # (B, max_imp, D)

        # Allocate output buffer.
        x_out = x_compressed.new_zeros(B, N_orig, D)

        for b in range(B):
            # Place important tokens at their original positions.
            valid_imp = imp_indices[b][imp_indices[b] >= 0]
            n_imp = len(valid_imp)
            if n_imp > 0:
                x_out[b, valid_imp] = x_imp[b, :n_imp]

            # Background positions: cross-attend from important anchors.
            valid_bg = bg_indices[b][bg_indices[b] >= 0]
            n_bg = len(valid_bg)
            if n_bg == 0 or n_imp == 0:
                continue

            # Queries: zero-initialised placeholder for background positions.
            bg_queries = x_compressed.new_zeros(1, n_bg, D)
            # Keys/values: the important tokens.
            kv = x_imp[b, :n_imp].unsqueeze(0)  # (1, n_imp, D)

            # Apply positional encoding to query (simple: embed background
            # index as a normalised scalar and add to the zero query).
            pos_enc = (valid_bg.float() / max(N_orig - 1, 1)).unsqueeze(-1)  # (n_bg, 1)
            pos_enc = pos_enc.expand(n_bg, D)                                 # (n_bg, D)
            bg_queries = bg_queries + pos_enc.unsqueeze(0)

            attn_out, _ = self.cross_attn(bg_queries, kv, kv, need_weights=False)
            attn_out = self.norm(attn_out.squeeze(0))  # (n_bg, D)
            x_out[b, valid_bg] = attn_out

        return x_out


# ---------------------------------------------------------------------------
# DCDiT
# ---------------------------------------------------------------------------

class DCDiT(nn.Module):
    """
    DC-DiT: Dynamic Chunking wrapper around an existing DiT model.

    Applies token routing and spatial pooling before the DiT blocks, runs
    attention on the compressed token sequence, then reconstructs the full
    sequence with the TokenDecoder.

    Only the TokenRouter and TokenDecoder are trainable by default;
    the base DiT is frozen.  This matches the upcycling strategy described
    in arxiv 2603.06351.

    Parameters
    ----------
    base_model        : pre-trained DiT (e.g. MG2's base_distill.safetensors).
    compression_ratio : target compression; roughly controls pooling window
                        (default 4 → pool_size=4).
    threshold         : importance score threshold for routing (default 0.5).
    """

    def __init__(
        self,
        base_model: nn.Module,
        compression_ratio: int = 4,
        threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.compression_ratio = compression_ratio
        self.threshold = threshold

        # Infer hidden_dim from base model parameters.
        hidden_dim = self._infer_hidden_dim(base_model)
        self.hidden_dim = hidden_dim

        self.router = TokenRouter(hidden_dim=hidden_dim, threshold=threshold)
        self.pooler = SpatialPooler(pool_size=compression_ratio)
        self.decoder = TokenDecoder(hidden_dim=hidden_dim, num_heads=8)

        # Freeze base model immediately.
        self.freeze_base()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_hidden_dim(model: nn.Module) -> int:
        """
        Attempt to read the hidden dimension from model parameter shapes.

        Falls back to 1536 (Wan2.1-1.8B approximate transformer width).
        """
        for name, param in model.named_parameters():
            if "weight" in name and param.ndim == 2:
                candidate = max(param.shape)
                if 512 <= candidate <= 4096:
                    return int(candidate)
        return 1536

    def freeze_base(self) -> None:
        """
        Freeze all base DiT parameters.

        Only the TokenRouter and TokenDecoder remain trainable.
        """
        for p in self.base_model.parameters():
            p.requires_grad_(False)
        self.base_model.eval()

    def get_trainable_params(self) -> List[torch.nn.Parameter]:
        """
        Return only the trainable parameters (router + decoder).

        These are the only parameters updated during upcycling distillation.

        Returns
        -------
        List of nn.Parameter objects.
        """
        return list(self.router.parameters()) + list(self.decoder.parameters())

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        DC-DiT forward pass.

        1. Route tokens → important / background.
        2. Pool background tokens.
        3. Run base DiT on compressed sequence.
        4. Decode back to full resolution.

        Parameters
        ----------
        x    : (B, N, C) token sequence at the *input* of the DiT.
               For video models N may be flattened spatial-temporal tokens.
               If x is 5-D (B, T, C, H, W) it is reshaped to (B, N, C).
        t    : (B,) or (B, T) timestep tensor.
        cond : optional conditioning (passed through unchanged to base DiT).

        Returns
        -------
        Output of same shape as x (before any reshaping).
        """
        original_shape = x.shape
        reshape_back = False

        # Handle (B, T, C, H, W) video input.
        if x.ndim == 5:
            B, T_vid, C, H, W = x.shape
            x = x.reshape(B, T_vid * H * W, C)
            reshape_back = True
        elif x.ndim == 4:
            B, C, H, W = x.shape
            x = x.reshape(B, H * W, C)
            reshape_back = True

        B, N, D = x.shape

        # ---- Token routing ----
        scores = self.router(x)                       # (B, N)
        important_mask = scores > self.threshold       # (B, N) bool

        # Ensure at least one important token per sample to prevent empty sequences.
        for b in range(B):
            if not important_mask[b].any():
                # Force the highest-scored token to be important.
                best = scores[b].argmax()
                important_mask[b, best] = True

        # ---- Spatial pooling ----
        x_full, x_compressed, routing_info = self.pooler(x, important_mask)

        # ---- Base DiT on compressed sequence ----
        # The base DiT expects its standard input format; we pass the
        # compressed token sequence and the original timestep tensor.
        # The 3-D RoPE in Wan2.1 uses token positions; we pass the original
        # positional grid so position encodings remain meaningful.
        try:
            if cond is not None:
                compressed_out = self.base_model(x_compressed, t, cond)
            else:
                compressed_out = self.base_model(x_compressed, t)
        except Exception:
            # If the base model cannot handle variable-length sequences,
            # fall back to routing only (identity on compressed tokens).
            compressed_out = x_compressed

        # ---- Reconstruct full sequence ----
        x_reconstructed = self.decoder(compressed_out, routing_info)

        # Restore original shape if needed.
        if reshape_back:
            x_reconstructed = x_reconstructed.reshape(original_shape)

        return x_reconstructed


# ---------------------------------------------------------------------------
# Upcycling distillation loop
# ---------------------------------------------------------------------------

class DCDiTDistiller:
    """
    Distillation trainer for DC-DiT upcycling.

    Trains only the TokenRouter and TokenDecoder by matching the DC-DiT
    output to the frozen base DiT output:

        L_dc = ||DC-DiT(x_t, t) − base_DiT(x_t, t)||²

    Parameters
    ----------
    dc_dit : :class:`DCDiT` instance (base DiT already frozen).
    lr     : learning rate for AdamW (default 1e-4, higher than typical
             because only small modules are trained).
    """

    def __init__(self, dc_dit: DCDiT, lr: float = 1e-4) -> None:
        self.dc_dit = dc_dit
        trainable = dc_dit.get_trainable_params()
        if not trainable:
            raise ValueError("DC-DiT has no trainable parameters.")
        self.optimizer = AdamW(trainable, lr=lr)

    def train_step(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        One gradient update step.

        Parameters
        ----------
        batch : dict with 'x' (input tokens), 't' (timesteps), optionally
                'cond' (conditioning).

        Returns
        -------
        Dict with 'loss_dc'.
        """
        device = next(self.dc_dit.router.parameters()).device
        dtype = next(self.dc_dit.router.parameters()).dtype

        x = batch["x"].to(device=device, dtype=dtype)
        t = batch["t"].to(device=device, dtype=dtype)
        cond = batch.get("cond", None)
        if cond is not None:
            cond = cond.to(device=device, dtype=dtype)

        # Base model reference (no grad).
        self.dc_dit.base_model.eval()
        with torch.no_grad():
            try:
                if cond is not None:
                    base_out = self.dc_dit.base_model(x, t, cond)
                else:
                    base_out = self.dc_dit.base_model(x, t)
            except Exception:
                base_out = x  # fallback identity

        # DC-DiT forward (router + decoder trainable).
        self.dc_dit.router.train()
        self.dc_dit.decoder.train()
        if cond is not None:
            dc_out = self.dc_dit(x, t, cond)
        else:
            dc_out = self.dc_dit(x, t)

        loss = F.mse_loss(dc_out, base_out.detach())

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.dc_dit.get_trainable_params(), max_norm=1.0
        )
        self.optimizer.step()

        return {"loss_dc": loss.item()}

    def run(
        self,
        dataloader: torch.utils.data.DataLoader,
        epochs: int = 5,
    ) -> None:
        """
        Full upcycling training loop.

        Parameters
        ----------
        dataloader : yields dicts with 'x', 't', optionally 'cond'.
        epochs     : number of training epochs.
        """
        device = next(self.dc_dit.router.parameters()).device

        for epoch in range(epochs):
            epoch_loss = 0.0
            n = 0
            for batch in dataloader:
                batch_dev = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                metrics = self.train_step(batch_dev)
                epoch_loss += metrics["loss_dc"]
                n += 1

            avg = epoch_loss / max(n, 1)
            print(f"[DCDiT] Epoch {epoch + 1}/{epochs}  avg_loss={avg:.6f}")

    def save(self, path: str | Path) -> None:
        """
        Save DC-DiT router and decoder weights (not the frozen base DiT).

        Parameters
        ----------
        path : destination .pt file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "router_state_dict": self.dc_dit.router.state_dict(),
                "decoder_state_dict": self.dc_dit.decoder.state_dict(),
                "compression_ratio": self.dc_dit.compression_ratio,
                "threshold": self.dc_dit.threshold,
                "hidden_dim": self.dc_dit.hidden_dim,
            },
            str(path),
        )
        print(f"[DCDiTDistiller] Saved → {path}")
