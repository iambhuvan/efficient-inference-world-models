"""
ProReflow: Progressive Reflow + Aligned Velocity Prediction for MG2.

Reference: arxiv 2503.04824 — "ProReflow"

Theory
------
Matrix-Game 2.0 uses flow matching (Wan2.1 backbone):

    x_t = t · x_1 + (1 − t) · x_0          (linear OT interpolation)
    v_target(x_t, t) = x_1 − x_0            (straight-line velocity)

The pretrained model already approximates straight paths with 3 distilled
steps.  ProReflow pushes them even straighter through two complementary
objectives:

1. Progressive Reflow
   Refine one local interval [t_k, t_{k+1}] at a time.  For the current
   interval, generate a (x_{t_k}, x_{t_{k+1}}) pair from the current model,
   then train to predict the exact straight-line velocity.

   L_prog = E[ ||v_θ(x_{t_k}, t_k) − (x_{t_{k+1}} − x_{t_k})|| ² ]

   where x_{t_{k+1}} is obtained by a one-step ODE using the *current*
   model (not the teacher), yielding a self-consistency target.

2. Aligned Velocity Prediction
   Match the velocity *direction*, not just magnitude:

   L_align = E[ 1 − cos(v_θ, v_target) ]
           = E[ 1 − (v_θ / ||v_θ||) · (v_target / ||v_target||) ]

   Combined loss:
   L_total = L_flow_match + λ_align · L_align
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# ProReflowDistiller
# ---------------------------------------------------------------------------

class ProReflowDistiller:
    """
    Progressive Reflow + Aligned-v distiller for MG2 (Wan2.1 flow matching).

    Parameters
    ----------
    model                : the MG2 flow-matching DiT (trained or partially
                           distilled); weights are updated in-place.
    lambda_align         : weight for the cosine alignment loss (default 0.5).
    num_reflow_intervals : number of local time intervals for progressive
                           reflow (default 3 — matches MG2's 3 distilled steps).
    lr                   : AdamW learning rate (default 1e-5).
    """

    def __init__(
        self,
        model: nn.Module,
        lambda_align: float = 0.5,
        num_reflow_intervals: int = 3,
        lr: float = 1e-5,
    ) -> None:
        self.model = model
        self.lambda_align = lambda_align
        self.num_reflow_intervals = num_reflow_intervals

        # Build interval boundaries: [0, 1] split into equal sub-intervals.
        self._interval_boundaries = self._build_intervals(num_reflow_intervals)

        self.optimizer = AdamW(self.model.parameters(), lr=lr)

    # ------------------------------------------------------------------
    # Interval helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_intervals(n: int) -> List[Tuple[float, float]]:
        """
        Divide [0, 1] into n equal sub-intervals.

        Returns list of (t_start, t_end) tuples, e.g. for n=3:
            [(0.0, 0.333), (0.333, 0.667), (0.667, 1.0)]
        """
        step = 1.0 / n
        return [(k * step, (k + 1) * step) for k in range(n)]

    def _get_interval(self, interval_idx: int) -> Tuple[float, float]:
        """Return (t_start, t_end) for a given interval index."""
        if not (0 <= interval_idx < self.num_reflow_intervals):
            raise ValueError(
                f"interval_idx must be in [0, {self.num_reflow_intervals - 1}]"
            )
        return self._interval_boundaries[interval_idx]

    # ------------------------------------------------------------------
    # Velocity prediction
    # ------------------------------------------------------------------

    def _predict_velocity(
        self,
        x: torch.Tensor,
        t_val: float,
        cond: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Query the model for the velocity v_θ(x, t).

        Parameters
        ----------
        x     : (B, ...) latent tensor.
        t_val : scalar timestep in [0, 1].
        cond  : optional conditioning.

        Returns
        -------
        Velocity tensor of same shape as x.
        """
        device = x.device
        dtype = x.dtype
        B = x.shape[0]
        t_tensor = torch.full((B,), t_val, device=device, dtype=dtype)

        if cond is not None:
            return self.model(x, t_tensor, cond)
        return self.model(x, t_tensor)

    # ------------------------------------------------------------------
    # Reflow pair generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_reflow_pairs(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        num_steps: int = 10,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, float]]:
        """
        Generate (x_{t_k}, x_{t_{k+1}}, t_k) pairs by running a multi-step
        Euler integration of the *current* model from x_0 ~ N(0, I).

        These pairs are used as self-consistency targets for progressive reflow.

        Parameters
        ----------
        model     : flow-matching model (used for ODE integration).
        x0        : (B, ...) starting noise sample x_0 ~ N(0, I).
        num_steps : number of Euler integration steps.

        Returns
        -------
        List of (x_t, x_{t+dt}, t) triples — one per integration step,
        covering the full [0, 1] trajectory.
        """
        B = x0.shape[0]
        device = x0.device
        dtype = x0.dtype

        timesteps = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
        pairs: List[Tuple[torch.Tensor, torch.Tensor, float]] = []

        x = x0.clone()
        for i in range(num_steps):
            t_k = float(timesteps[i].item())
            t_k1 = float(timesteps[i + 1].item())
            dt = t_k1 - t_k

            t_tensor = torch.full((B,), t_k, device=device, dtype=dtype)
            v = model(x, t_tensor)            # velocity at x_{t_k}
            x_next = x + dt * v              # Euler step

            pairs.append((x.clone(), x_next.clone(), t_k))
            x = x_next

        return pairs

    # ------------------------------------------------------------------
    # Aligned velocity loss
    # ------------------------------------------------------------------

    def compute_aligned_v_loss(
        self,
        v_pred: torch.Tensor,
        v_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the cosine-alignment loss between predicted and target velocities.

        L_align = mean(1 − cos(v_pred, v_target))
                = mean(1 − (v_pred · v_target) / (||v_pred|| · ||v_target||))

        Small ε prevents division by zero.

        Parameters
        ----------
        v_pred   : predicted velocity (B, ...).
        v_target : target velocity (B, ...).

        Returns
        -------
        Scalar loss tensor.
        """
        # Flatten spatial/channel dims for cosine similarity.
        flat_pred = v_pred.reshape(v_pred.shape[0], -1).float()
        flat_tgt = v_target.reshape(v_target.shape[0], -1).float()

        cosine_sim = F.cosine_similarity(flat_pred, flat_tgt, dim=-1, eps=1e-8)
        return (1.0 - cosine_sim).mean()

    # ------------------------------------------------------------------
    # Progressive reflow loss
    # ------------------------------------------------------------------

    def compute_progressive_loss(
        self,
        model: nn.Module,
        interval_idx: int,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute the progressive reflow loss for a single interval.

        Procedure
        ---------
        1. Sample x_0 ~ N(0, I) (or use batch noise).
        2. Euler-integrate the model to obtain x_{t_k}.
        3. Compute v_θ(x_{t_k}, t_k).
        4. Target = straight-line velocity = (x_{t_{k+1}} − x_{t_k}) / Δt.
        5. L_prog = ||v_θ − v_target||².

        Parameters
        ----------
        model        : current model.
        interval_idx : which local interval to train on.
        batch        : dict with 'x_clean' (B, ...) clean samples used as
                       reference x_1, and optionally 'cond'.

        Returns
        -------
        Scalar loss tensor.
        """
        t_start, t_end = self._get_interval(interval_idx)
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        x_clean = batch["x_clean"].to(device=device, dtype=dtype)
        cond = batch.get("cond", None)
        B = x_clean.shape[0]

        # Sample noise x_0.
        x_noise = torch.randn_like(x_clean)

        # Interpolate to t_start: x_{t_start} = t_start · x_clean + (1−t_start) · x_noise
        x_tk = t_start * x_clean + (1.0 - t_start) * x_noise

        # One Euler step from t_start → t_end using the current model.
        with torch.no_grad():
            t_tensor = torch.full((B,), t_start, device=device, dtype=dtype)
            if cond is not None:
                v_tk = model(x_tk, t_tensor, cond)
            else:
                v_tk = model(x_tk, t_tensor)
            dt = t_end - t_start
            x_tk1 = x_tk + dt * v_tk

        # Target: straight-line velocity = (x_{t_{k+1}} − x_{t_k}) / Δt
        v_straight = (x_tk1 - x_tk) / max(dt, 1e-8)

        # Online velocity prediction at t_start.
        t_tensor = torch.full((B,), t_start, device=device, dtype=dtype)
        if cond is not None:
            v_pred = model(x_tk, t_tensor, cond)
        else:
            v_pred = model(x_tk, t_tensor)

        return F.mse_loss(v_pred, v_straight.detach())

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        One gradient update step combining flow-matching, progressive reflow
        and cosine alignment losses.

        Parameters
        ----------
        batch : dict with keys:
            'x_clean'  — (B, ...) clean data samples (x_1 in flow matching).
            'x_noise'  — (B, ...) noise samples (x_0 ~ N(0, I)).  If absent,
                         fresh noise is sampled.
            'cond'     — optional conditioning tensor.

        Returns
        -------
        Dict of scalar loss values.
        """
        self.model.train()
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype

        x_clean = batch["x_clean"].to(device=device, dtype=dtype)
        cond = batch.get("cond", None)
        B = x_clean.shape[0]

        if "x_noise" in batch:
            x_noise = batch["x_noise"].to(device=device, dtype=dtype)
        else:
            x_noise = torch.randn_like(x_clean)

        # Sample a random timestep.
        t_val = float(torch.empty(1).uniform_(0.0, 1.0).item())

        # Flow-matching interpolation: x_t = t · x_1 + (1−t) · x_0
        x_t = t_val * x_clean + (1.0 - t_val) * x_noise

        # Ground-truth straight velocity.
        v_target = x_clean - x_noise  # x_1 − x_0

        # Model prediction.
        t_tensor = torch.full((B,), t_val, device=device, dtype=dtype)
        if cond is not None:
            v_pred = self.model(x_t, t_tensor, cond)
        else:
            v_pred = self.model(x_t, t_tensor)

        # Flow-matching MSE loss.
        loss_fm = F.mse_loss(v_pred, v_target)

        # Cosine alignment loss.
        loss_align = self.compute_aligned_v_loss(v_pred, v_target)

        # Progressive reflow on a randomly selected interval.
        interval_idx = int(torch.randint(0, self.num_reflow_intervals, (1,)).item())
        loss_prog = self.compute_progressive_loss(self.model, interval_idx, batch)

        loss_total = loss_fm + self.lambda_align * loss_align + loss_prog

        self.optimizer.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {
            "loss_fm": loss_fm.item(),
            "loss_align": loss_align.item(),
            "loss_prog": loss_prog.item(),
            "loss_total": loss_total.item(),
            "interval_idx": interval_idx,
            "t_val": t_val,
        }

    # ------------------------------------------------------------------
    # Full refinement loop
    # ------------------------------------------------------------------

    def run_full_refinement(
        self,
        dataloader: DataLoader,
        epochs: int = 5,
    ) -> None:
        """
        Run the full ProReflow training loop.

        Iterates over all epochs and all data batches, updating the model
        with combined flow-matching, progressive-reflow and alignment losses.

        Parameters
        ----------
        dataloader : yields dicts with at least 'x_clean', optionally 'cond'.
        epochs     : number of training epochs (default 5).
        """
        device = next(self.model.parameters()).device

        for epoch in range(epochs):
            epoch_metrics: Dict[str, float] = {}
            num_batches = 0

            for batch in dataloader:
                # Move to device.
                batch_dev = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                metrics = self.train_step(batch_dev)
                for k, v in metrics.items():
                    epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v
                num_batches += 1

            # Log epoch averages.
            avg = {k: v / max(num_batches, 1) for k, v in epoch_metrics.items()}
            print(
                f"[ProReflow] Epoch {epoch + 1}/{epochs}  "
                + "  ".join(f"{k}={v:.6f}" for k, v in avg.items() if isinstance(v, float))
            )

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save_refined_model(self, path: str | Path) -> None:
        """
        Save the refined model weights.

        Parameters
        ----------
        path : destination .pt file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "lambda_align": self.lambda_align,
                "num_reflow_intervals": self.num_reflow_intervals,
            },
            str(path),
        )
        print(f"[ProReflowDistiller] Saved refined model → {path}")
