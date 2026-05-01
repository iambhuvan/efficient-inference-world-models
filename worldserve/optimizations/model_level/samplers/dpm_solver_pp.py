"""
DPM-Solver++ sampler adapted for the Oasis / WorldServe denoising loop.

The Oasis denoising loop integrates the ODE:

    dx/dt = v_theta(x_t, t)

using an Euler method:

    x_{t+h} = x_t + h * v_theta(x_t, t)    h = t_end - t_start < 0

This module replaces that Euler step with a multi-step Adams-Bashforth
predictor that re-uses previous network evaluations to achieve higher-order
accuracy without extra forward passes.

References
----------
- DPM-Solver++: arxiv 2211.01095  (Lu et al., 2022)
- SA-Solver:    arxiv 2309.05019  (Xue et al., 2023)
"""

from __future__ import annotations

from collections import deque
from typing import Callable, List, Optional

import torch
from torch import Tensor


class DPMSolverPPSampler:
    """
    Multi-step DPM-Solver++ sampler for the Oasis Euler-ODE denoising loop.

    Supports three orders:
      - order=1 : plain Euler (baseline, identical to original Oasis loop)
      - order=2 : 2nd-order Adams-Bashforth (AB2)
      - order=3 : 3rd-order Adams-Bashforth (AB3)

    The AB2 formula is:
        x_{t+h} = x_t + h * [3/2 * D0 - 1/2 * D0_prev]

    The AB3 formula is:
        x_{t+h} = x_t + h * [23/12 * D0 - 4/3 * D0_prev + 5/12 * D0_pp]

    where D0, D0_prev, D0_pp are the current and two previous noise predictions.

    For the first k-1 steps (where k = order) the sampler automatically falls
    back to Euler so there is no cold-start issue.

    Parameters
    ----------
    order : int
        Solver order.  1 = Euler, 2 = AB2, 3 = AB3.
    """

    def __init__(self, order: int = 2) -> None:
        if order not in (1, 2, 3):
            raise ValueError(f"order must be 1, 2, or 3; got {order}")
        self.order = order
        # Ring buffer of past noise predictions, oldest first.
        # Capacity = order - 1 (we need that many past values for AB).
        self._history: deque[Tensor] = deque(maxlen=max(1, order - 1))

    # ------------------------------------------------------------------
    # OptimizationStack interface
    # ------------------------------------------------------------------

    def pre_step(self, step_idx: int, total_steps: int, latents: Tensor) -> bool:
        """Never skip; DPM-Solver++ always needs the network evaluation."""
        return False

    def post_step(self, step_idx: int, total_steps: int, latents: Tensor) -> None:
        """No-op post-step hook (history is updated inside ``step``)."""
        return

    def reset(self) -> None:
        """Clear stored noise-prediction history.  Call between generations."""
        self._history.clear()

    # ------------------------------------------------------------------
    # Core ODE step
    # ------------------------------------------------------------------

    def step(
        self,
        latents: Tensor,
        noise_pred: Tensor,
        t_start: Tensor,
        t_end: Tensor,
    ) -> Tensor:
        """
        Perform one ODE step, updating *latents* from ``t_start`` to ``t_end``.

        Parameters
        ----------
        latents : Tensor
            Current latent state, shape (B, T, C, H, W).
        noise_pred : Tensor
            Network output v_theta(x_t, t), same shape as *latents*.
        t_start : Tensor
            Scalar (or broadcastable) timestep at the start of the step.
        t_end : Tensor
            Scalar (or broadcastable) timestep at the end of the step.

        Returns
        -------
        Tensor
            Updated latents after the ODE step.
        """
        h = t_end - t_start  # negative scalar; h < 0

        history = list(self._history)  # oldest ... newest
        n_history = len(history)

        if self.order == 1 or n_history == 0:
            # Euler (order-1 or cold-start fallback)
            new_latents = latents + h * noise_pred

        elif self.order == 2 or n_history == 1:
            # AB2: use current D0 and most-recent D0_prev
            d0_prev = history[-1]
            new_latents = latents + h * (1.5 * noise_pred - 0.5 * d0_prev)

        else:
            # AB3: use current D0 and two previous predictions
            # history[-1] = D0_prev (one step ago)
            # history[-2] = D0_pp  (two steps ago)
            d0_prev = history[-1]
            d0_pp = history[-2]
            new_latents = latents + h * (
                (23.0 / 12.0) * noise_pred
                - (4.0 / 3.0) * d0_prev
                + (5.0 / 12.0) * d0_pp
            )

        # Store current noise prediction for the next step.
        # .clone() is critical when torch.compile(mode="reduce-overhead") uses
        # CUDA Graphs — without it the graph's output memory is overwritten on the
        # next replay, corrupting d0_prev/d0_pp in subsequent AB steps.
        self._history.append(noise_pred.detach().clone())

        return new_latents

    # ------------------------------------------------------------------
    # Convenience: full denoising loop
    # ------------------------------------------------------------------

    def wrap_oasis_loop(
        self,
        dit_fn: Callable[[Tensor, Tensor], Tensor],
        latents: Tensor,
        timesteps: Tensor,
    ) -> Tensor:
        """
        Run the complete Oasis denoising loop using DPM-Solver++.

        Replaces the inner Euler loop in ``oasis.py`` lines 125-149.

        Parameters
        ----------
        dit_fn : callable
            Function ``(x: Tensor, t: Tensor) -> Tensor`` — the DiT forward
            pass.  ``t`` is expected to be shape (B, T).
        latents : Tensor
            Initial noisy latents, shape (B, T, C, H, W).
        timesteps : Tensor
            1-D tensor of length ``num_steps + 1``, linearly spaced from 1.0
            to 0.0 (as produced by ``torch.linspace(1.0, 0.0, num_steps+1)``).

        Returns
        -------
        Tensor
            Clean latents after the full denoising loop.
        """
        self.reset()
        num_steps = len(timesteps) - 1

        for i in range(num_steps):
            t_start = timesteps[i]
            t_end = timesteps[i + 1]
            # DiT expects (B, T) timestep tensor
            t_batch = t_start.expand(latents.shape[0], latents.shape[1])
            noise_pred = dit_fn(latents, t_batch)
            latents = self.step(latents, noise_pred, t_start, t_end)

        return latents


# ---------------------------------------------------------------------------
# Convenience aliases
# ---------------------------------------------------------------------------

def euler_sampler() -> DPMSolverPPSampler:
    """Return a plain Euler sampler (order=1)."""
    return DPMSolverPPSampler(order=1)


def dpm_solver_pp_2() -> DPMSolverPPSampler:
    """Return a 2nd-order DPM-Solver++ (Adams-Bashforth 2) sampler."""
    return DPMSolverPPSampler(order=2)


def dpm_solver_pp_3() -> DPMSolverPPSampler:
    """Return a 3rd-order DPM-Solver++ (Adams-Bashforth 3) sampler."""
    return DPMSolverPPSampler(order=3)
