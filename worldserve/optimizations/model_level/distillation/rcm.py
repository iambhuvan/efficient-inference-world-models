"""
rCM: Score-Regularised Continuous-time Consistency Model for MG2.

Reference: arxiv 2510.08431 — "Score-Regularized Consistency Models"

Validated on Wan2.1 (the base of MG2), achieving 15-50× speedup.

Theory
------
sCM (continuous-time consistency) loss:

    L_sCM = E_{t, s<t} [ λ(t) · ||f_θ(x_t, t) − sg[f_θ⁻(x_s^ODE, s)]||² ]

where:
    x_s^ODE = x_t + (s − t) · v_θ(x_t, t)   (one-step ODE solve to s)
    f_θ      = consistency function (student, online)
    f_θ⁻     = EMA of f_θ (target network, stop-gradient)
    λ(t)     = SNR-based weighting = 1 / (t² + σ_min²)

rCM adds score regularisation to prevent mode-covering collapse:

    L_rCM = L_sCM + λ_score · L_score
    L_score = E_t [ ||v_θ(x_t, t) − v_teacher(x_t, t)||² ]

JVP computation
---------------
For the continuous-time consistency gradient a Jacobian-vector product (JVP)
is needed:

    d/dt [f_θ(x_t + t·v, t)]  requires J·v

Use torch.func.jvp (functorch API):

    _, jvp_val = torch.func.jvp(lambda x: model(x, t), (x,), (v,))

This is implemented in ``compute_jvp_loss``.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW


# ---------------------------------------------------------------------------
# rCMDistiller
# ---------------------------------------------------------------------------

class rCMDistiller:
    """
    Score-regularised continuous-time consistency model distiller for MG2.

    MG2 is a Wan2.1-based flow-matching DiT.  The consistency function
    ``f_θ(x_t, t)`` maps any point along the ODE trajectory to x_0
    (the clean sample), enforcing trajectory consistency across all t.

    Parameters
    ----------
    student_model : the MG2 DiT to distil (weights updated in-place).
    teacher_model : frozen teacher DiT (provides velocity field).
    sigma_min     : minimum noise level for numerical stability (default 1e-4).
    lambda_score  : weight for the score regularisation loss (default 0.1).
    ema_decay     : EMA decay for the target network (default 0.9999).
    total_steps   : total training steps (used for λ schedule; default 1000).
    lr            : AdamW learning rate (default 1e-5).
    """

    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        sigma_min: float = 1e-4,
        lambda_score: float = 0.1,
        ema_decay: float = 0.9999,
        total_steps: int = 1000,
        lr: float = 1e-5,
    ) -> None:
        self.student = student_model
        self.teacher = teacher_model
        self.sigma_min = sigma_min
        self.lambda_score = lambda_score
        self.ema_decay = ema_decay
        self.total_steps = total_steps

        # EMA target network.
        self.target = copy.deepcopy(student_model)
        self.target.eval()
        for p in self.target.parameters():
            p.requires_grad_(False)

        # Freeze teacher.
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        self.optimizer = AdamW(self.student.parameters(), lr=lr)
        self._step: int = 0

    # ------------------------------------------------------------------
    # SNR-based loss weighting
    # ------------------------------------------------------------------

    def get_lambda_weight(self, t: float) -> float:
        """
        Compute the SNR-based loss weight λ(t) = 1 / (t² + σ_min²).

        Higher noise levels (large t) are down-weighted to prevent
        trivially easy high-noise samples from dominating.

        Parameters
        ----------
        t : scalar timestep in [0, 1].

        Returns
        -------
        Scalar float weight.
        """
        return 1.0 / (t ** 2 + self.sigma_min ** 2)

    # ------------------------------------------------------------------
    # Velocity / consistency function helpers
    # ------------------------------------------------------------------

    def _velocity(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: float,
        cond: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Query the model for the velocity v(x, t).

        For flow-matching models the forward pass directly returns a velocity
        field.  For DDPM-style noise predictors one would convert ε → v;
        here we call the model as a velocity predictor (MG2 is flow-matching).

        Parameters
        ----------
        model : DiT.
        x     : (B, ...) input latent.
        t     : scalar timestep.
        cond  : optional conditioning.

        Returns
        -------
        Velocity tensor of same shape as x.
        """
        B = x.shape[0]
        device = x.device
        dtype = x.dtype

        # Build per-sample timestep tensor.
        if x.ndim == 5:
            # (B, T_vid, C, H, W) — Oasis-style
            T_vid = x.shape[1]
            t_tensor = torch.full((B, T_vid), t, device=device, dtype=dtype)
        else:
            t_tensor = torch.full((B,), t, device=device, dtype=dtype)

        if cond is not None:
            return model(x, t_tensor, cond)
        return model(x, t_tensor)

    def _consistency_fn(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: float,
        cond: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute the consistency function f_θ(x_t, t) → x_0_pred.

        For flow-matching:
            x_0_pred = x_t − t · v_θ(x_t, t)

        This is the ODE-flow equivalent of DDPM's x_0 prediction.

        Parameters
        ----------
        model : DiT (student or target).
        x     : (B, ...) noised latent at time t.
        t     : scalar timestep in [0, 1].
        cond  : optional conditioning.

        Returns
        -------
        x_0_pred of same shape as x.
        """
        v = self._velocity(model, x, t, cond)
        return x - t * v

    # ------------------------------------------------------------------
    # ODE step (teacher)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _ode_step_to_s(
        self,
        x_t: torch.Tensor,
        t: float,
        s: float,
        cond: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        One-step ODE solve from t → s using the student velocity field:

            x_s^ODE = x_t + (s − t) · v_θ(x_t, t)

        Note: per the sCM formulation the ODE step uses the *student*
        velocity (not teacher), so that the target is self-consistent.

        Parameters
        ----------
        x_t  : (B, ...) latent at time t.
        t, s : source and target timesteps (s < t).
        cond : optional conditioning.

        Returns
        -------
        x_s^ODE  of same shape as x_t.
        """
        v = self._velocity(self.student, x_t, t, cond)
        return x_t + (s - t) * v

    # ------------------------------------------------------------------
    # Core losses
    # ------------------------------------------------------------------

    def compute_consistency_loss(
        self,
        x0: torch.Tensor,
        t: float,
        s: float,
        cond: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute the sCM consistency loss between (x_t, t) and (x_s^ODE, s).

        Procedure
        ---------
        1. Noise x_0 to x_t at level t.
        2. One-step ODE solve with student: x_t → x_s^ODE.
        3. Online:  f_θ(x_t, t)           (student, with gradients).
        4. Target:  sg[f_θ⁻(x_s^ODE, s)]  (EMA, stop-gradient).
        5. Loss:  λ(t) · ||online − target||²

        Parameters
        ----------
        x0   : (B, ...) clean latents.
        t, s : timesteps (t > s ≥ 0).
        cond : optional conditioning.

        Returns
        -------
        Scalar loss tensor.
        """
        noise = torch.randn_like(x0)
        x_t = x0 + t * noise          # noised input

        # ODE step (no grad through this path).
        x_s_ode = self._ode_step_to_s(x_t, t, s, cond)

        # Online consistency function (with gradients).
        x0_online = self._consistency_fn(self.student, x_t, t, cond)

        # EMA target (stop-gradient).
        with torch.no_grad():
            x0_target = self._consistency_fn(self.target, x_s_ode, s, cond)

        lam = self.get_lambda_weight(t)
        return lam * F.mse_loss(x0_online, x0_target.detach())

    def compute_score_loss(
        self,
        x_t: torch.Tensor,
        t: float,
        cond: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Regularise the student velocity field to match the teacher.

        L_score = ||v_θ(x_t, t) − v_teacher(x_t, t)||²

        Prevents the consistency mapping from collapsing to a degenerate
        mode-covering solution.

        Parameters
        ----------
        x_t  : (B, ...) noised latent.
        t    : scalar timestep.
        cond : optional conditioning.

        Returns
        -------
        Scalar loss tensor.
        """
        v_student = self._velocity(self.student, x_t, t, cond)
        with torch.no_grad():
            v_teacher = self._velocity(self.teacher, x_t, t, cond)
        return F.mse_loss(v_student, v_teacher.detach())

    def compute_jvp_loss(
        self,
        x_t: torch.Tensor,
        t: float,
        v_t: torch.Tensor,
        cond: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute the JVP-based consistency gradient loss.

        For continuous-time consistency training, the directional derivative
        of the consistency function along the ODE flow must be zero:

            JVP = d/dt [f_θ(x_t + t·v, t)]
                = J_x f_θ(x_t, t) · v_t  + ∂_t f_θ(x_t, t)

        We minimise ||JVP||² to encourage a flat (consistent) surface.

        Uses ``torch.func.jvp`` (functorch functional API, requires
        PyTorch >= 2.0).  Falls back to finite differences on older releases.

        Parameters
        ----------
        x_t  : (B, ...) input latent.
        t    : scalar timestep.
        v_t  : (B, ...) velocity vector (tangent direction).
        cond : optional conditioning.

        Returns
        -------
        Scalar JVP loss tensor.
        """
        # Build a pure-function wrapper for torch.func.jvp.
        # cond is captured in the closure; only x varies.
        if cond is not None:
            def f_x(x: torch.Tensor) -> torch.Tensor:
                return self._consistency_fn(self.student, x, t, cond)
        else:
            def f_x(x: torch.Tensor) -> torch.Tensor:
                return self._consistency_fn(self.student, x, t, None)

        try:
            # Preferred path: torch.func.jvp (PyTorch >= 2.0).
            _, jvp_val = torch.func.jvp(f_x, (x_t,), (v_t,))
        except AttributeError:
            # Fallback: finite difference approximation.
            eps = 1e-4
            f_plus = f_x(x_t + eps * v_t)
            f_minus = f_x(x_t - eps * v_t)
            jvp_val = (f_plus - f_minus) / (2.0 * eps)

        # Minimise the magnitude of the directional derivative.
        return jvp_val.pow(2).mean()

    # ------------------------------------------------------------------
    # EMA update
    # ------------------------------------------------------------------

    def _update_ema(self) -> None:
        """Update EMA target: θ⁻ ← μ·θ⁻ + (1−μ)·θ."""
        mu = self.ema_decay
        with torch.no_grad():
            for p_s, p_t in zip(self.student.parameters(), self.target.parameters()):
                p_t.data.mul_(mu).add_(p_s.data, alpha=1.0 - mu)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Execute one combined rCM training update.

        Losses combined:
          L_rCM = L_sCM + λ_score · L_score

        The JVP loss is computed optionally (can be expensive) every 10 steps.

        Parameters
        ----------
        batch : dict with:
            'x0'   — (B, ...) clean latents.
            'cond' — optional conditioning tensor.

        Returns
        -------
        Dict of scalar metric values.
        """
        self.student.train()
        device = next(self.student.parameters()).device
        dtype = next(self.student.parameters()).dtype

        x0 = batch["x0"].to(device=device, dtype=dtype)
        cond = batch.get("cond", None)
        if cond is not None:
            cond = cond.to(device=device, dtype=dtype)

        # Sample timesteps: t > s > 0.
        t_val = float(torch.empty(1).uniform_(self.sigma_min + 0.02, 1.0).item())
        s_val = float(
            torch.empty(1).uniform_(self.sigma_min, max(self.sigma_min, t_val - 0.02)).item()
        )

        # Consistency loss.
        loss_scm = self.compute_consistency_loss(x0, t_val, s_val, cond)

        # Score regularisation.
        noise = torch.randn_like(x0)
        x_t = x0 + t_val * noise
        loss_score = self.compute_score_loss(x_t, t_val, cond)

        # JVP loss (every 10 steps to save compute).
        loss_jvp = x0.new_zeros(1).squeeze()
        if self._step % 10 == 0:
            v_t = self._velocity(self.student, x_t, t_val, cond).detach()
            loss_jvp = self.compute_jvp_loss(x_t, t_val, v_t, cond)

        loss_total = loss_scm + self.lambda_score * loss_score + 0.01 * loss_jvp

        self.optimizer.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
        self.optimizer.step()
        self._update_ema()
        self._step += 1

        return {
            "loss_scm": loss_scm.item(),
            "loss_score": loss_score.item(),
            "loss_jvp": loss_jvp.item() if isinstance(loss_jvp, torch.Tensor) else float(loss_jvp),
            "loss_total": loss_total.item(),
            "t_val": t_val,
            "s_val": s_val,
            "lambda_weight": self.get_lambda_weight(t_val),
        }

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """
        Save student and EMA target weights.

        Parameters
        ----------
        path : destination .pt file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "student_state_dict": self.student.state_dict(),
                "target_state_dict": self.target.state_dict(),
                "sigma_min": self.sigma_min,
                "lambda_score": self.lambda_score,
                "ema_decay": self.ema_decay,
                "step": self._step,
            },
            str(path),
        )
        print(f"[rCMDistiller] Saved → {path}")
