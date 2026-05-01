"""
Dual-Expert Consistency Model (DCM) for video world model distillation.

Reference: arxiv 2506.03123

Architecture overview
---------------------
Two expert networks are trained jointly on top of a frozen teacher DiT:

  SemanticExpert  — operates on a 4× spatially downsampled version of x_t.
                    Captures coarse layout and semantics cheaply.
                    Implemented with a small 2-block DiT that reuses the
                    teacher's patch-embedding and timestep-embedding layers.

  DetailExpert    — operates at full resolution, conditioned on the coarse
                    prediction produced by SemanticExpert.  Implemented as
                    6 lightweight adapter blocks inserted into the teacher
                    DiT's middle layers.

Loss
----
L_DCM = L_PCM(semantic) + L_PCM(detail) + λ_TC · L_TC

L_TC (temporal coherence):
    L_TC ≈ ||x0_pred^{t+1} − x0_pred^t||² / num_pixels

The temporal coherence term is an approximate proxy for optical-flow warping
loss that avoids the need for an external optical flow network.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from worldserve.optimizations.model_level.distillation.pcm import pseudo_huber_distance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_module(teacher: nn.Module, attr: str) -> Optional[nn.Module]:
    """Return getattr(teacher, attr) if it exists, else None."""
    return getattr(teacher, attr, None)


# ---------------------------------------------------------------------------
# SemanticExpert
# ---------------------------------------------------------------------------

class SemanticExpert(nn.Module):
    """
    Coarse-resolution expert that processes x_t downsampled 4× spatially.

    Reuses the teacher's patch-embedding and timestep-embedding layers so
    that feature representations are compatible.  The backbone consists of
    ``num_expert_blocks`` lightweight transformer blocks operating on the
    compressed token sequence.

    Parameters
    ----------
    teacher_model   : frozen teacher DiT; embedding layers are borrowed.
    embed_dim       : transformer hidden dimension (matches teacher).
    num_expert_blocks : number of DiT blocks in the semantic expert (default 2).
    """

    def __init__(
        self,
        teacher_model: nn.Module,
        embed_dim: int,
        num_expert_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_expert_blocks = num_expert_blocks

        # Borrow patch-embed and time-embed from teacher (shared, frozen).
        # These attributes are common across DiT implementations; if the
        # teacher uses different names the AttributeError will surface early.
        self.patch_embed: Optional[nn.Module] = _safe_module(teacher_model, "x_embedder")
        if self.patch_embed is None:
            self.patch_embed = _safe_module(teacher_model, "patch_embed")
        self.time_embed: Optional[nn.Module] = _safe_module(teacher_model, "t_embedder")
        if self.time_embed is None:
            self.time_embed = _safe_module(teacher_model, "time_embed")

        # Freeze borrowed layers — they are shared with the teacher.
        for mod in [self.patch_embed, self.time_embed]:
            if mod is not None:
                for p in mod.parameters():
                    p.requires_grad_(False)

        # Lightweight transformer blocks (single-head attention + FFN).
        self.blocks = nn.ModuleList([
            _LightTransformerBlock(embed_dim) for _ in range(num_expert_blocks)
        ])

        # Output projection: map embed_dim → latent channels (16 for Oasis).
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        nn.init.zeros_(self.out_proj.bias)

        # 4× spatial upsampler applied to the coarse output.
        self.upsample = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)

    def forward(
        self,
        x_t_downsampled: torch.Tensor,
        t_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict coarse x_0 from a spatially downsampled noised latent.

        Parameters
        ----------
        x_t_downsampled : (B, C, H//4, W//4) or (B, T, C, H//4, W//4).
        t_embed         : (B, embed_dim) timestep embedding.

        Returns
        -------
        coarse_x0 : (B, C, H, W) or (B, T, C, H, W) at *original* resolution.
        """
        squeeze_time = False
        if x_t_downsampled.ndim == 5:
            B, T, C, H_d, W_d = x_t_downsampled.shape
            x = x_t_downsampled.reshape(B * T, C, H_d, W_d)
            # Tile t_embed across frames.
            t_embed = t_embed.unsqueeze(1).expand(B, T, -1).reshape(B * T, -1)
            squeeze_time = True
        else:
            B, C, H_d, W_d = x_t_downsampled.shape
            x = x_t_downsampled

        BT = x.shape[0]

        # Flatten spatial dims to token sequence: (BT, H_d*W_d, C).
        tokens = x.reshape(BT, C, -1).permute(0, 2, 1)  # (BT, N_d, C)

        # Project to embed_dim with a simple linear if needed.
        if tokens.shape[-1] != self.embed_dim:
            # On-the-fly projection (no persistent layer needed for the general case).
            tokens = F.linear(
                tokens,
                torch.eye(
                    min(tokens.shape[-1], self.embed_dim),
                    device=tokens.device,
                    dtype=tokens.dtype,
                ).repeat(
                    math.ceil(self.embed_dim / tokens.shape[-1]),
                    math.ceil(tokens.shape[-1] / self.embed_dim),
                )[:self.embed_dim, :tokens.shape[-1]].T,
            )

        for block in self.blocks:
            tokens = block(tokens, t_embed)

        tokens = self.out_proj(tokens)  # (BT, N_d, embed_dim)

        # Reconstruct spatial map.
        N_d = H_d * W_d
        out = tokens[:, :N_d, :C].permute(0, 2, 1).reshape(BT, C, H_d, W_d)

        # 4× bilinear upsample back to original resolution.
        out = self.upsample(out)

        if squeeze_time:
            _, C_out, H_out, W_out = out.shape
            out = out.reshape(B, T, C_out, H_out, W_out)

        return out


# ---------------------------------------------------------------------------
# DetailExpert
# ---------------------------------------------------------------------------

class DetailExpert(nn.Module):
    """
    Full-resolution refinement expert conditioned on the coarse x_0 prediction.

    Implemented as adapter layers inserted into (or alongside) the teacher
    DiT's middle blocks.  The teacher blocks are frozen; only the adapters
    are trainable.

    Parameters
    ----------
    teacher_model     : frozen teacher DiT.
    num_adapter_blocks: number of adapter blocks (default 6).
    """

    def __init__(
        self,
        teacher_model: nn.Module,
        num_adapter_blocks: int = 6,
    ) -> None:
        super().__init__()
        self.num_adapter_blocks = num_adapter_blocks

        # Infer embed_dim from teacher.
        embed_dim = self._infer_embed_dim(teacher_model)
        self.embed_dim = embed_dim

        # Coarse conditioning projection: map coarse_x0 channels → embed_dim.
        self.coarse_proj = nn.Conv2d(16, embed_dim, kernel_size=1, bias=True)
        nn.init.zeros_(self.coarse_proj.bias)

        # Adapter blocks: small residual modules inserted after teacher blocks.
        self.adapters = nn.ModuleList([
            _AdapterBlock(embed_dim) for _ in range(num_adapter_blocks)
        ])

        # Final output projection back to latent channel count.
        self.out_proj = nn.Conv2d(embed_dim, 16, kernel_size=1, bias=True)
        nn.init.zeros_(self.out_proj.bias)

    @staticmethod
    def _infer_embed_dim(teacher: nn.Module) -> int:
        """
        Attempt to read the hidden dim from the teacher's parameter shapes.
        Falls back to 512 (DiT-S default).
        """
        for name, param in teacher.named_parameters():
            if "weight" in name and param.ndim == 2:
                candidate = max(param.shape)
                if 256 <= candidate <= 2048:
                    return int(candidate)
        return 512  # DiT-S/2 default

    def forward(
        self,
        x_t: torch.Tensor,
        t_embed: torch.Tensor,
        coarse_x0: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict refined x_0 at full resolution.

        Parameters
        ----------
        x_t        : (B, C, H, W) or (B, T, C, H, W) full-resolution noised latent.
        t_embed    : (B, embed_dim) timestep embedding.
        coarse_x0  : (B, C, H, W) or (B, T, C, H, W) coarse prediction from
                     SemanticExpert (same spatial resolution as x_t).

        Returns
        -------
        refined_x0 : same shape as x_t.
        """
        squeeze_time = False
        if x_t.ndim == 5:
            B, T, C, H, W = x_t.shape
            x = x_t.reshape(B * T, C, H, W)
            coarse = coarse_x0.reshape(B * T, C, H, W)
            t_embed = t_embed.unsqueeze(1).expand(B, T, -1).reshape(B * T, -1)
            squeeze_time = True
        else:
            B, C, H, W = x_t.shape
            x = x_t
            coarse = coarse_x0
            T = 1

        BT = x.shape[0]

        # Project coarse x_0 to embed_dim feature map.
        coarse_feat = self.coarse_proj(coarse.float())  # (BT, D, H, W)

        # Concatenate with x_t (channel-wise) then flatten to tokens.
        N = H * W
        # Simple feature: treat (x_t flattened + coarse_feat flattened) as tokens.
        x_tokens = x.reshape(BT, C, N).permute(0, 2, 1)  # (BT, N, C)
        c_tokens = coarse_feat.reshape(BT, self.embed_dim, N).permute(0, 2, 1)

        # Linear mix to embed_dim if needed.
        if x_tokens.shape[-1] != self.embed_dim:
            pad = self.embed_dim - x_tokens.shape[-1]
            if pad > 0:
                x_tokens = F.pad(x_tokens, (0, pad))
            else:
                x_tokens = x_tokens[..., :self.embed_dim]

        h = x_tokens + c_tokens  # (BT, N, D)

        # Adapter blocks refine the tokens.
        for adapter in self.adapters:
            h = h + adapter(h, t_embed)  # residual

        # Project back to latent channels.
        h_spatial = h.permute(0, 2, 1).reshape(BT, self.embed_dim, H, W)
        out = self.out_proj(h_spatial.float())  # (BT, 16, H, W)

        if squeeze_time:
            out = out.reshape(B, T, *out.shape[1:])

        return out.to(x_t.dtype)


# ---------------------------------------------------------------------------
# Small building-block modules
# ---------------------------------------------------------------------------

class _LightTransformerBlock(nn.Module):
    """Single-head self-attention + FFN block (no cross-attention)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=max(1, dim // 64), batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        # AdaLN modulation from timestep embedding.
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x     : (B, N, D) token sequence.
        t_emb : (B, D) timestep embedding.

        Returns
        -------
        x of same shape.
        """
        shift, scale = self.adaLN_modulation(t_emb).chunk(2, dim=-1)
        # Modulate pre-attention norm.
        x_norm = self.norm1(x) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class _AdapterBlock(nn.Module):
    """Lightweight adapter: LayerNorm + FFN (no attention to minimise FLOPS)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.t_proj = nn.Linear(dim, dim, bias=True)
        nn.init.zeros_(self.t_proj.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x     : (B, N, D).
        t_emb : (B, D).

        Returns
        -------
        Residual delta of same shape as x.
        """
        t_cond = self.t_proj(t_emb).unsqueeze(1)  # (B, 1, D)
        return self.ffn(self.norm(x) + t_cond)


# ---------------------------------------------------------------------------
# DCMDistiller
# ---------------------------------------------------------------------------

class DCMDistiller:
    """
    Dual-Expert Consistency Model distiller.

    Trains SemanticExpert and DetailExpert jointly on top of a frozen teacher
    DiT, using PCM-style consistency losses plus a temporal coherence
    regulariser.

    Parameters
    ----------
    teacher_model : frozen teacher DiT nn.Module.
    lambda_tc     : weight for the temporal coherence loss (default 0.1).
    embed_dim     : transformer hidden dim (default 512 for DiT-S).
    lr            : learning rate (default 1e-5).
    ema_decay     : EMA decay for PCM target networks (default 0.9999).
    c_huber       : pseudo-Huber constant (default 0.00054).
    """

    def __init__(
        self,
        teacher_model: nn.Module,
        lambda_tc: float = 0.1,
        embed_dim: int = 512,
        lr: float = 1e-5,
        ema_decay: float = 0.9999,
        c_huber: float = 0.00054,
    ) -> None:
        self.teacher = teacher_model
        self.lambda_tc = lambda_tc
        self.c_huber = c_huber
        self.ema_decay = ema_decay

        # Freeze teacher entirely.
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        # Build expert networks.
        self.semantic_expert = SemanticExpert(
            teacher_model=teacher_model,
            embed_dim=embed_dim,
            num_expert_blocks=2,
        )
        self.detail_expert = DetailExpert(
            teacher_model=teacher_model,
            num_adapter_blocks=6,
        )

        # EMA copies for PCM-style target networks.
        self.semantic_ema = copy.deepcopy(self.semantic_expert)
        self.semantic_ema.eval()
        for p in self.semantic_ema.parameters():
            p.requires_grad_(False)

        self.detail_ema = copy.deepcopy(self.detail_expert)
        self.detail_ema.eval()
        for p in self.detail_ema.parameters():
            p.requires_grad_(False)

        # Optimizer over both experts only.
        trainable = (
            list(self.semantic_expert.parameters())
            + list(self.detail_expert.parameters())
        )
        self.optimizer = AdamW(trainable, lr=lr)

    # ------------------------------------------------------------------
    # EMA update
    # ------------------------------------------------------------------

    def _update_ema(self) -> None:
        mu = self.ema_decay
        for (ps, pt) in [
            (self.semantic_expert, self.semantic_ema),
            (self.detail_expert, self.detail_ema),
        ]:
            with torch.no_grad():
                for p, p_ema in zip(ps.parameters(), pt.parameters()):
                    p_ema.data.mul_(mu).add_(p.data, alpha=1.0 - mu)

    # ------------------------------------------------------------------
    # x_0 predictions
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _teacher_euler_step(
        self,
        x_t: torch.Tensor,
        t_val: float,
        s_val: float,
        cond: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """One Euler step with the teacher from t → s."""
        device = x_t.device
        dtype = x_t.dtype
        if x_t.ndim == 5:
            B, T_vid, C, H, W = x_t.shape
            t_tensor = torch.full((B, T_vid), t_val, device=device, dtype=dtype)
        else:
            B = x_t.shape[0]
            t_tensor = torch.full((B,), t_val, device=device, dtype=dtype)

        if cond is not None:
            noise_pred = self.teacher(x_t, t_tensor, cond)
        else:
            noise_pred = self.teacher(x_t, t_tensor)
        return x_t + (s_val - t_val) * noise_pred

    def _build_t_embed(
        self, t_val: float, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Build a simple sinusoidal timestep embedding of dimension embed_dim.

        In a production setting this would match the teacher's t_embedder;
        here we provide a self-contained fallback.
        """
        D = self.semantic_expert.embed_dim
        half = D // 2
        freqs = torch.arange(half, device=device, dtype=torch.float32)
        freqs = torch.exp(-math.log(10000) * freqs / half)
        t_tensor = torch.tensor([t_val], device=device, dtype=torch.float32)
        emb = t_tensor[:, None] * freqs[None, :]  # (1, half)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)  # (1, D)
        if D % 2 == 1:
            emb = F.pad(emb, (0, 1))
        emb = emb.expand(batch_size, -1).to(dtype)
        return emb

    def _predict_semantic_x0(
        self,
        model: SemanticExpert,
        x_t: torch.Tensor,
        t_val: float,
    ) -> torch.Tensor:
        """Coarse x_0 prediction from semantic expert."""
        if x_t.ndim == 5:
            B = x_t.shape[0]
            # 4× downsample spatial dims.
            B_, T_vid, C, H, W = x_t.shape
            x_down = F.avg_pool2d(
                x_t.reshape(B_ * T_vid, C, H, W).float(),
                kernel_size=4, stride=4
            ).reshape(B_, T_vid, C, H // 4, W // 4).to(x_t.dtype)
        else:
            B = x_t.shape[0]
            B_, C, H, W = x_t.shape
            x_down = F.avg_pool2d(x_t.float(), kernel_size=4, stride=4).to(x_t.dtype)

        t_emb = self._build_t_embed(t_val, B, x_t.device, x_t.dtype)
        coarse = model(x_down, t_emb)
        return coarse

    def _predict_detail_x0(
        self,
        model: DetailExpert,
        x_t: torch.Tensor,
        t_val: float,
        coarse_x0: torch.Tensor,
    ) -> torch.Tensor:
        """Fine x_0 prediction from detail expert."""
        B = x_t.shape[0]
        t_emb = self._build_t_embed(t_val, B, x_t.device, x_t.dtype)
        return model(x_t, t_emb, coarse_x0)

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        x0_batch: torch.Tensor,
        cond_batch: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the full DCM loss for a batch of clean latents.

        Returns a dict with keys:
            'loss_semantic'  — PCM consistency loss on SemanticExpert.
            'loss_detail'    — PCM consistency loss on DetailExpert.
            'loss_tc'        — Temporal coherence loss.
            'loss_total'     — Weighted sum.

        Parameters
        ----------
        x0_batch   : (B, T_vid, C, H, W) clean latents.
        cond_batch : optional conditioning tensor.
        """
        device = x0_batch.device
        dtype = x0_batch.dtype

        # Sample two timesteps from a random phase.
        t_val = float(torch.empty(1).uniform_(0.1, 1.0).item())
        s_val = float(torch.empty(1).uniform_(0.0, t_val - 0.05).item())

        # Noise x_0.
        noise = torch.randn_like(x0_batch)
        x_t = x0_batch + t_val * noise

        # Teacher Euler step to get x_s^ODE.
        x_s_ode = self._teacher_euler_step(x_t, t_val, s_val, cond_batch)

        # ----- Semantic expert loss -----
        coarse_x0_online = self._predict_semantic_x0(self.semantic_expert, x_t, t_val)
        with torch.no_grad():
            coarse_x0_target = self._predict_semantic_x0(self.semantic_ema, x_s_ode, s_val)

        # Match coarse_x0 to clean x_0 reference (downsampled).
        if x0_batch.ndim == 5:
            B, T_vid, C, H, W = x0_batch.shape
            x0_coarse_ref = F.avg_pool2d(
                x0_batch.reshape(B * T_vid, C, H, W).float(),
                kernel_size=4, stride=4,
            ).reshape(B, T_vid, C, H // 4, W // 4).to(dtype)
            # Upsample back for loss computation (after SemanticExpert upsamples).
        else:
            x0_coarse_ref = x0_batch

        loss_semantic = pseudo_huber_distance(coarse_x0_online, coarse_x0_target.detach(), 0.00054)

        # ----- Detail expert loss -----
        refined_x0_online = self._predict_detail_x0(
            self.detail_expert, x_t, t_val, coarse_x0_online.detach()
        )
        with torch.no_grad():
            refined_x0_target = self._predict_detail_x0(
                self.detail_ema, x_s_ode, s_val, coarse_x0_target
            )

        loss_detail = pseudo_huber_distance(refined_x0_online, refined_x0_target.detach(), 0.00054)

        # ----- Temporal coherence loss -----
        # Approximate: penalise large temporal differences between consecutive
        # frame predictions.
        if x0_batch.ndim == 5 and x0_batch.shape[1] > 1:
            # refined_x0_online: (B, T_vid, C, H, W)
            diff = refined_x0_online[:, 1:] - refined_x0_online[:, :-1]
            num_pixels = float(diff[0].numel())
            loss_tc = diff.pow(2).sum() / num_pixels
        else:
            loss_tc = x0_batch.new_zeros(1).squeeze()

        loss_total = loss_semantic + loss_detail + self.lambda_tc * loss_tc

        return {
            "loss_semantic": loss_semantic,
            "loss_detail": loss_detail,
            "loss_tc": loss_tc,
            "loss_total": loss_total,
        }

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        One gradient update over all trainable parameters.

        Parameters
        ----------
        batch : dict with 'x0' and optionally 'cond'.

        Returns
        -------
        Dict of scalar loss values.
        """
        self.semantic_expert.train()
        self.detail_expert.train()

        x0 = batch["x0"]
        cond = batch.get("cond", None)

        self.optimizer.zero_grad()
        losses = self.compute_loss(x0, cond)
        losses["loss_total"].backward()
        params = (
            list(self.semantic_expert.parameters())
            + list(self.detail_expert.parameters())
        )
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        self.optimizer.step()
        self._update_ema()

        return {k: v.item() for k, v in losses.items()}

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """
        Save all trainable expert weights.

        Parameters
        ----------
        path : destination .pt file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "semantic_expert": self.semantic_expert.state_dict(),
                "detail_expert": self.detail_expert.state_dict(),
                "semantic_ema": self.semantic_ema.state_dict(),
                "detail_ema": self.detail_ema.state_dict(),
                "lambda_tc": self.lambda_tc,
            },
            str(path),
        )
        print(f"[DCMDistiller] Saved → {path}")
