"""
Phased Consistency Model (PCM) distillation for the Oasis DiT-S/2 world model.

Reference: arxiv 2405.18407 — "Phased Consistency Model"

Theory
------
Consistency models enforce the mapping

    f_θ(x_t, t) = f_θ(x_s, s)   for all (s, t) on the same ODE trajectory.

PCM phases the trajectory [0, T] into K non-overlapping segments.  Within
segment k the student is required to output the same x_0 prediction from any
noised input in that segment, using a pseudo-Huber distance for robustness.

The EMA "target network" f_θ⁻ provides stable training targets via
stop-gradient.

Oasis-specific notes
--------------------
* Euler ODE:  x_{t_end} = x_{t_start} + (t_end − t_start) · ε_θ(x_t, t)
* x_0 prediction (simplified first-order):
      f_θ(x_t, t) = x_t − t · ε_θ(x_t, t)
* 20 denoising steps, 5 phases of 4 steps each, t ∈ [0, 1].
* Latent shape: (B, T, C=16, H=18, W=32).
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


# ---------------------------------------------------------------------------
# Pseudo-Huber distance
# ---------------------------------------------------------------------------

def pseudo_huber_distance(a: torch.Tensor, b: torch.Tensor, c: float = 0.00054) -> torch.Tensor:
    """
    Compute the pseudo-Huber distance between two tensors, averaged over all
    non-batch dimensions.

    d(a, b) = sqrt(||a − b||² + c²) − c

    Parameters
    ----------
    a, b : tensors of identical shape (B, ...)
    c    : smoothing constant; default 0.00054 is calibrated for Oasis latents.

    Returns
    -------
    Scalar loss tensor.
    """
    diff_sq = (a - b).pow(2).mean()          # mean over all elements
    return (diff_sq + c * c).sqrt() - c


# ---------------------------------------------------------------------------
# PCMDistiller
# ---------------------------------------------------------------------------

class PCMDistiller:
    """
    Phased Consistency Model distiller for the Oasis DiT-S/2 (DDPM with
    sigmoid_beta_schedule, v-prediction, 10-step DDIM at inference, latents
    B×T×16×18×32).

    Training is *progressive*: phase 5 (hardest, high noise) is learned first,
    then phase 4, …, down to phase 1 (low noise, easiest).

    Phase layout (5 phases, 4 Euler steps each)
    -------------------------------------------
    Phase 5  t ∈ [1.0, 0.8]
    Phase 4  t ∈ [0.8, 0.6]
    Phase 3  t ∈ [0.6, 0.4]
    Phase 2  t ∈ [0.4, 0.2]
    Phase 1  t ∈ [0.2, 0.0]

    Parameters
    ----------
    student_model : Oasis DiT nn.Module (f_θ, to be distilled).
    teacher_model : Oasis DiT nn.Module (fixed teacher, eval mode).
    num_phases    : Number of PCM phases (default 5).
    num_student_steps : Denoising steps per phase (default 4).
    ema_decay     : EMA decay for the target network (default 0.9999).
    c_huber       : Pseudo-Huber smoothing constant (default 0.00054).
    lr            : Learning rate for AdamW (default 1e-5).
    """

    # Phase boundaries: (t_start, t_end), ordered from hardest to easiest.
    # PCM-training trajectory (5 phases × 4 substeps = 20 fine teacher steps);
    # the *inference* student runs at 10-step DDIM Oasis canonical.
    # Overridden if num_phases != 5.
    DEFAULT_PHASE_BOUNDARIES: List[Tuple[float, float]] = [
        (1.0, 0.8),  # phase 5 (index 4) — hardest
        (0.8, 0.6),  # phase 4 (index 3)
        (0.6, 0.4),  # phase 3 (index 2)
        (0.4, 0.2),  # phase 2 (index 1)
        (0.2, 0.0),  # phase 1 (index 0) — easiest
    ]

    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        num_phases: int = 5,
        num_student_steps: int = 4,
        ema_decay: float = 0.9999,
        c_huber: float = 0.00054,
        lr: float = 1e-5,
    ) -> None:
        self.student = student_model
        self.teacher = teacher_model
        self.num_phases = num_phases
        self.num_student_steps = num_student_steps
        self.ema_decay = ema_decay
        self.c_huber = c_huber

        # EMA target network — a deep copy of the student, kept in eval mode.
        self.target = copy.deepcopy(student_model)
        self.target.eval()
        for p in self.target.parameters():
            p.requires_grad_(False)

        # Teacher is always frozen.
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        self.optimizer = AdamW(self.student.parameters(), lr=lr)

        # Build phase boundaries dynamically if num_phases != 5.
        if num_phases == 5:
            self._phase_boundaries = self.DEFAULT_PHASE_BOUNDARIES.copy()
        else:
            self._phase_boundaries = self._build_phase_boundaries(num_phases)

    # ------------------------------------------------------------------
    # Phase helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_phase_boundaries(num_phases: int) -> List[Tuple[float, float]]:
        """Build equal-width phase boundaries for arbitrary num_phases."""
        boundaries = []
        step = 1.0 / num_phases
        for k in range(num_phases):
            t_start = 1.0 - k * step
            t_end = 1.0 - (k + 1) * step
            boundaries.append((t_start, t_end))
        return boundaries

    def get_phase_timesteps(self, phase_idx: int) -> Tuple[float, float]:
        """
        Return (t_start, t_end) for the given phase index.

        Phase indices run from 0 (easiest, low noise) to num_phases-1
        (hardest, high noise) when accessed directly.  For progressive
        training, hardest phases (high index) are trained first.

        Parameters
        ----------
        phase_idx : 0-based index into the phase list (0 = easiest).

        Returns
        -------
        (t_start, t_end) float tuple.
        """
        if not (0 <= phase_idx < self.num_phases):
            raise ValueError(
                f"phase_idx must be in [0, {self.num_phases - 1}], got {phase_idx}"
            )
        # Boundaries are stored hardest-first; phase 0 = easiest = last entry.
        return self._phase_boundaries[self.num_phases - 1 - phase_idx]

    # ------------------------------------------------------------------
    # x_0 prediction
    # ------------------------------------------------------------------

    def _predict_x0(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: float,
        cond: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute the x_0 prediction from a noise-prediction model.

        Using the simplified first-order Euler approximation:
            f(x_t, t) = x_t − t · ε_θ(x_t, t)

        Parameters
        ----------
        model : noise-prediction network (Oasis DiT).
        x_t   : noised latent  (B, T_vid, C, H, W).
        t     : scalar timestep in [0, 1].
        cond  : optional action conditioning tensor (B, T_vid, A).

        Returns
        -------
        x_0_pred of same shape as x_t.
        """
        B, T_vid = x_t.shape[:2]
        device = x_t.device
        dtype = x_t.dtype

        # Oasis DiT expects t as (B, T_vid) — one value per frame.
        t_tensor = torch.full(
            (B, T_vid), t, device=device, dtype=dtype
        )

        if cond is not None:
            noise_pred = model(x_t, t_tensor, cond)
        else:
            noise_pred = model(x_t, t_tensor)

        # First-order Euler x_0 prediction.
        return x_t - t * noise_pred

    # ------------------------------------------------------------------
    # ODE step for generating x_s from x_t (teacher)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _euler_step(
        self,
        x_t: torch.Tensor,
        t: float,
        s: float,
        cond: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        One Euler ODE step with the teacher model:
            x_s^ODE = x_t + (s − t) · ε_teacher(x_t, t)

        Parameters
        ----------
        x_t  : noised latent at time t.
        t, s : source and target timesteps (s < t).
        cond : optional conditioning.

        Returns
        -------
        x_s^ODE  — latent at time s along the teacher ODE trajectory.
        """
        B, T_vid = x_t.shape[:2]
        device = x_t.device
        dtype = x_t.dtype

        t_tensor = torch.full((B, T_vid), t, device=device, dtype=dtype)

        if cond is not None:
            noise_pred = self.teacher(x_t, t_tensor, cond)
        else:
            noise_pred = self.teacher(x_t, t_tensor)

        return x_t + (s - t) * noise_pred

    # ------------------------------------------------------------------
    # Core loss
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        x0: torch.Tensor,
        cond: Optional[torch.Tensor],
        current_phase: int,
    ) -> torch.Tensor:
        """
        Compute the PCM consistency loss for a single phase.

        Procedure
        ---------
        1. Sample two timesteps t > s within the current phase.
        2. Noise x_0 to x_t with Gaussian noise at level t.
        3. Euler-step with teacher from x_t → x_s^ODE.
        4. Student predicts x_0 from x_t (online) and x_s^ODE (target, EMA).
        5. Loss = pseudo_huber(f_θ(x_t, t),  sg[f_θ⁻(x_s^ODE, s)]).

        Parameters
        ----------
        x0          : clean latents  (B, T_vid, C, H, W).
        cond        : action conditioning or None.
        current_phase : 0-based phase index (0 = easiest).

        Returns
        -------
        Scalar loss tensor.
        """
        t_start, t_end = self.get_phase_timesteps(current_phase)
        device = x0.device
        dtype = x0.dtype

        # Sample two timesteps: t ∈ (t_end, t_start], s ∈ [t_end, t)
        # Use uniform sampling within the phase interval.
        t_val = float(
            torch.empty(1).uniform_(t_end + 1e-5, t_start).item()
        )
        s_val = float(
            torch.empty(1).uniform_(t_end, t_val - 1e-5).item()
        )

        # Add noise: x_t = x_0 + t · ε,   ε ~ N(0, I)
        noise = torch.randn_like(x0)
        x_t = x0 + t_val * noise

        # Teacher Euler step: x_s^ODE  (no grad)
        x_s_ode = self._euler_step(x_t, t_val, s_val, cond)

        # Online student: f_θ(x_t, t)
        x0_pred_online = self._predict_x0(self.student, x_t, t_val, cond)

        # EMA target: sg[ f_θ⁻(x_s^ODE, s) ]
        with torch.no_grad():
            x0_pred_target = self._predict_x0(self.target, x_s_ode, s_val, cond)

        return pseudo_huber_distance(x0_pred_online, x0_pred_target, self.c_huber)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        current_phase: int,
    ) -> Dict[str, float]:
        """
        Execute one gradient update step.

        Parameters
        ----------
        batch : dict with key 'x0' (clean latents) and optionally 'cond'.
        current_phase : 0-based phase index.

        Returns
        -------
        Dict with training metrics: 'loss', 'phase', 't', 's'.
        """
        self.student.train()

        x0 = batch["x0"]
        cond = batch.get("cond", None)

        self.optimizer.zero_grad()
        loss = self.compute_loss(x0, cond, current_phase)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update EMA target network.
        self._update_ema()

        return {
            "loss": loss.item(),
            "phase": current_phase,
        }

    # ------------------------------------------------------------------
    # EMA update
    # ------------------------------------------------------------------

    def _update_ema(self) -> None:
        """
        Perform one EMA update:
            θ⁻ ← μ · θ⁻ + (1 − μ) · θ
        """
        mu = self.ema_decay
        with torch.no_grad():
            for p_student, p_target in zip(
                self.student.parameters(), self.target.parameters()
            ):
                p_target.data.mul_(mu).add_(p_student.data, alpha=1.0 - mu)

    # ------------------------------------------------------------------
    # Progressive training loop
    # ------------------------------------------------------------------

    def train_progressive(
        self,
        dataloader: DataLoader,
        num_phases: int = 5,
        epochs_per_phase: int = 10,
    ) -> None:
        """
        Train progressively from the hardest phase to the easiest.

        Phase ordering: phase (num_phases-1) → phase 0.

        Parameters
        ----------
        dataloader      : yields dicts with 'x0' and optionally 'cond'.
        num_phases      : total number of phases to train.
        epochs_per_phase: number of epochs for each phase.
        """
        # Train from hardest to easiest.
        phase_order = list(range(num_phases - 1, -1, -1))

        for phase_idx in phase_order:
            t_start, t_end = self.get_phase_timesteps(phase_idx)
            print(
                f"[PCM] Training phase {phase_idx}  "
                f"t ∈ [{t_end:.2f}, {t_start:.2f}]  "
                f"for {epochs_per_phase} epochs …"
            )

            for epoch in range(epochs_per_phase):
                epoch_loss = 0.0
                num_batches = 0

                for batch in dataloader:
                    # Move tensors to the student model's device.
                    device = next(self.student.parameters()).device
                    batch = {
                        k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    }
                    metrics = self.train_step(batch, phase_idx)
                    epoch_loss += metrics["loss"]
                    num_batches += 1

                avg_loss = epoch_loss / max(num_batches, 1)
                print(
                    f"  [PCM] Phase {phase_idx}  Epoch {epoch + 1}/{epochs_per_phase}  "
                    f"avg_loss={avg_loss:.6f}"
                )

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save_distilled_model(self, path: str | Path) -> None:
        """
        Save the distilled student weights.

        Only the student (not the teacher or EMA target) is saved.  The saved
        file is a plain safetensors / torch state dict usable in inference.

        Parameters
        ----------
        path : destination file path (e.g. 'pcm_student.pt').
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "student_state_dict": self.student.state_dict(),
            "ema_target_state_dict": self.target.state_dict(),
            "num_phases": self.num_phases,
            "num_student_steps": self.num_student_steps,
            "ema_decay": self.ema_decay,
            "c_huber": self.c_huber,
        }
        torch.save(state, str(path))
        print(f"[PCMDistiller] Saved distilled model → {path}")
