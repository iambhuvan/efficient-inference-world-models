"""
Adaptive Projected Guidance (APG) for WorldServe / Oasis.

Reference: arxiv 2410.02416  "Eliminating Oversaturation and Artifacts of
High Guidance Scales in Diffusion Models" (Sadat et al., 2024).

Standard CFG computes:
    eps_guided = eps_uncond + w * (eps_cond - eps_uncond)

APG decomposes the guidance vector g = eps_cond - eps_uncond into a component
parallel to eps_uncond (responsible for oversaturation) and a component
orthogonal to eps_uncond (responsible for fine structure).  It then:

  1. Down-weights the parallel component by alpha_parallel ∈ [0, 1].
  2. Accumulates a momentum term over the orthogonal component across steps
     and adds it back to improve temporal coherence.

Final formula:
    g_parallel = (g · eps_uncond_hat) * eps_uncond_hat
    g_ortho    = g - g_parallel
    m_t        = eta * m_{t-1} + (1 - eta) * g_ortho
    eps_guided = eps_uncond + w * (g_ortho + alpha_parallel * g_parallel)
                            + beta * m_t
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor


class APGGuidance:
    """
    Adaptive Projected Guidance.

    Parameters
    ----------
    guidance_scale : float
        CFG guidance weight ``w``.
    alpha_parallel : float
        Scaling factor for the parallel (oversaturation) component.
        0.0 = remove entirely, 1.0 = standard CFG.  Default 0.5.
    momentum_eta : float
        EMA decay for the orthogonal momentum buffer.  Default 0.9.
    momentum_beta : float
        Coefficient applied to the momentum term when adding to the
        guided prediction.  Default 0.1.
    """

    def __init__(
        self,
        guidance_scale: float = 7.5,
        alpha_parallel: float = 0.5,
        momentum_eta: float = 0.9,
        momentum_beta: float = 0.1,
    ) -> None:
        self.guidance_scale = guidance_scale
        self.alpha_parallel = alpha_parallel
        self.momentum_eta = momentum_eta
        self.momentum_beta = momentum_beta

        # Momentum buffer — initialised lazily on first apply() call.
        self._momentum: Optional[Tensor] = None

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear the momentum buffer.  Call between generations."""
        self._momentum = None

    # ------------------------------------------------------------------
    # Core guidance computation
    # ------------------------------------------------------------------

    def apply(self, eps_cond: Tensor, eps_uncond: Tensor) -> Tensor:
        """
        Compute the APG-modified guided prediction.

        Parameters
        ----------
        eps_cond : Tensor
            Conditional model prediction.  Arbitrary shape, e.g.
            (B, T, C, H, W) or (B, C, H, W).
        eps_uncond : Tensor
            Unconditional model prediction, same shape as *eps_cond*.

        Returns
        -------
        Tensor
            Guided prediction, same shape as inputs.
        """
        w = self.guidance_scale
        g = eps_cond - eps_uncond  # guidance vector, shape same as eps_cond

        # Flatten all dims except batch for dot-product computation.
        # We work in float32 to avoid precision loss in the normalisation.
        orig_dtype = eps_uncond.dtype
        eps_u = eps_uncond.float()
        g_f = g.float()

        batch_size = eps_u.shape[0]
        eps_u_flat = eps_u.reshape(batch_size, -1)   # (B, D)
        g_flat = g_f.reshape(batch_size, -1)          # (B, D)

        # Unit vector along eps_uncond for each batch element.
        eps_u_norm = eps_u_flat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        eps_u_hat = eps_u_flat / eps_u_norm            # (B, D)

        # Parallel component: scalar projection then broadcast.
        dot = (g_flat * eps_u_hat).sum(dim=-1, keepdim=True)  # (B, 1)
        g_par_flat = dot * eps_u_hat                           # (B, D)

        # Orthogonal component.
        g_ort_flat = g_flat - g_par_flat                       # (B, D)

        # Reshape back to original spatial shape.
        spatial_shape = eps_uncond.shape[1:]
        g_par = g_par_flat.reshape(batch_size, *spatial_shape).to(orig_dtype)
        g_ort = g_ort_flat.reshape(batch_size, *spatial_shape).to(orig_dtype)

        # Update momentum buffer (EMA over ortho component).
        if self._momentum is None:
            self._momentum = torch.zeros_like(g_ort)
        self._momentum = (
            self.momentum_eta * self._momentum
            + (1.0 - self.momentum_eta) * g_ort
        )

        # Assemble guided prediction.
        eps_guided = (
            eps_uncond
            + w * (g_ort + self.alpha_parallel * g_par)
            + self.momentum_beta * self._momentum
        )

        return eps_guided


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class APGWrapper(nn.Module):
    """
    Thin wrapper that runs both conditional and unconditional forward passes
    through *model*, then applies APG to produce the final guided prediction.

    Works with any model whose signature is:
        model(x_t, t, cond) -> Tensor

    Parameters
    ----------
    model : nn.Module
        The underlying diffusion model (e.g. Oasis DiT).
    guidance_scale : float
        CFG guidance weight passed to APGGuidance.
    **apg_kwargs
        Additional keyword arguments forwarded to APGGuidance (alpha_parallel,
        momentum_eta, momentum_beta).
    """

    def __init__(
        self,
        model: nn.Module,
        guidance_scale: float = 7.5,
        **apg_kwargs: Any,
    ) -> None:
        super().__init__()
        self.model = model
        self.apg = APGGuidance(guidance_scale=guidance_scale, **apg_kwargs)

    def reset(self) -> None:
        """Clear APG momentum.  Delegate to inner APGGuidance."""
        self.apg.reset()

    def forward(
        self,
        x_t: Tensor,
        t: Tensor,
        cond: Any,
        uncond: Optional[Any] = None,
    ) -> Tensor:
        """
        Run the model and apply APG.

        Parameters
        ----------
        x_t : Tensor
            Noisy latent input.
        t : Tensor
            Timestep tensor (B, T) for Oasis, or (B,) for image models.
        cond : any
            Conditional context (action tokens, text embeddings, …).
        uncond : any, optional
            Unconditional context.  If None, a zero-tensor of the same shape
            as the conditional output is used as the uncond prediction (i.e.,
            effectively the null/zero unconditional).  Pass an explicit uncond
            object whenever possible for best results.

        Returns
        -------
        Tensor
            APG-guided noise/velocity prediction, same shape as x_t.
        """
        eps_cond = self.model(x_t, t, cond)

        if uncond is not None:
            eps_uncond = self.model(x_t, t, uncond)
        else:
            # Fallback: treat unconditional as all-zeros (null prediction).
            eps_uncond = torch.zeros_like(eps_cond)

        return self.apg.apply(eps_cond, eps_uncond)
