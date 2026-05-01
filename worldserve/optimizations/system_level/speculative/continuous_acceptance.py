"""
CSpD Continuous Acceptance Criterion for speculative decoding with diffusion models.

Generalizes the standard speculative decoding acceptance/rejection sampling from
discrete token distributions to continuous denoising distributions.

Instead of comparing point estimates via cosine similarity, CSpD models both the
draft and verify outputs as isotropic Gaussian distributions:
  p_draft(x_{t-1} | x_t)  ~ N(x̂_draft,  σ_t² I)
  p_verify(x_{t-1} | x_t) ~ N(x̂_verify, σ_t² I)

Acceptance probability:
  a = min(1, exp(-KL(p_draft || p_verify) / temperature))
  KL = ||x̂_draft - x̂_verify||² / (2 σ_t²)

On rejection, x̂_verify is returned (already computed — zero wasted work).

Reference:
  CSpD (Nov 2024) — https://hf.co/papers/2411.11925
  "Continuous Speculative Decoding for Autoregressive Image Generation"
"""

from __future__ import annotations

import math
import random
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def cspd_accept(
    x_draft: torch.Tensor,
    x_verify: torch.Tensor,
    sigma_t: float,
    threshold: float = 0.90,
    temperature: float = 1.0,
    deterministic: bool = False,
) -> Tuple[bool, float]:
    """
    CSpD acceptance criterion for continuous denoising distributions.

    Computes KL divergence between draft and verify isotropic Gaussians,
    then applies acceptance sampling with the given threshold.

    Args:
        x_draft:      Predicted x_{t-1} from draft model.  Any shape.
        x_verify:     Predicted x_{t-1} from verify model. Same shape.
        sigma_t:      Noise level at current timestep t.  Scalar.
                      For DDIM: sigma_t = sqrt(1 - alpha_bar_t).
                      For flow-matching: sigma_t ≈ t (the flow time).
        threshold:    Minimum acceptance probability.  Samples u ~ Uniform(0,1)
                      and accepts if u < min(1, accept_prob / threshold).
        temperature:  Temperature scaling of KL divergence (higher = more lenient).
        deterministic: If True, accept iff accept_prob >= threshold (no sampling).

    Returns:
        Tuple of (accepted: bool, accept_prob: float).
        On acceptance: use x_draft.
        On rejection:  use x_verify (already computed).
    """
    with torch.no_grad():
        mse = F.mse_loss(x_draft.float(), x_verify.float(), reduction="mean").item()

    sigma_eff = max(float(sigma_t), 1e-6)
    kl_div = mse / (2.0 * sigma_eff ** 2 + 1e-12)

    # Acceptance probability: exp(-KL/temperature)
    accept_prob = math.exp(-kl_div / max(temperature, 1e-6))

    if deterministic:
        accepted = accept_prob >= threshold
    else:
        u = random.random()
        accepted = u < min(1.0, accept_prob / max(threshold, 1e-12))

    return accepted, accept_prob


def cosine_accept(
    x_draft: torch.Tensor,
    x_verify: torch.Tensor,
    threshold: float = 0.90,
) -> Tuple[bool, float]:
    """
    Simple cosine-similarity acceptance criterion (baseline).

    Args:
        x_draft:   Draft output tensor.
        x_verify:  Verify output tensor.
        threshold: Accept if cos_sim >= threshold.

    Returns:
        Tuple of (accepted: bool, similarity: float).
    """
    with torch.no_grad():
        d = x_draft.float().flatten()
        v = x_verify.float().flatten()
        sim = F.cosine_similarity(d.unsqueeze(0), v.unsqueeze(0)).item()
    return sim >= threshold, sim


def adaptive_sigma_from_ddim_step(
    t_int: int,
    alpha_bar: torch.Tensor,
) -> float:
    """
    Compute sigma_t from an integer DDIM timestep and the alpha_bar schedule.

    Args:
        t_int:      Integer timestep in [0, 999] (Oasis DDPM scale).
        alpha_bar:  Cumulative product alpha schedule, shape (1000,).

    Returns:
        sigma_t = sqrt(1 - alpha_bar[t]).
    """
    ab = float(alpha_bar[t_int].item())
    return math.sqrt(max(1.0 - ab, 0.0))


def adaptive_sigma_from_flow_t(t: float) -> float:
    """
    Compute sigma_t from a flow-matching timestep t ∈ [0, 1].

    For flow-matching (MG2 / Wan2.1):
      sigma_t ≈ t (noise level decreases from 1 to 0 as t goes 1 → 0).

    Args:
        t: Flow-matching timestep (0 = clean, 1 = pure noise).

    Returns:
        sigma_t = t.
    """
    return max(float(t), 1e-6)
