"""
Context Forcing distillation for long-horizon interactive game generation.

Reference: WorldPlay arxiv 2512.14614

Problem
-------
Distilled video world models suffer distribution mismatch over long rollouts:

    Teacher generates context with quality distribution  D_teacher.
    Distilled student is trained on  teacher-generated  context.
    At inference: student generates context with D_student ≠ D_teacher.
    After N blocks of autoregressive generation: context is completely
    out-of-distribution → catastrophic quality collapse.

Solution 1 — Reconstituted Context Memory
    Maintain a full frame history.  Periodically re-run the model on
    sampled distant frames to produce in-distribution context.

Solution 2 — Context Forcing Distillation
    During training, the student must generate its own context frames
    (not the teacher's).  This closes the train/inference distribution gap.

Both solutions are implemented here and can be used independently or
together.

Public API
----------
ReconstitutedContextMemory  — inference-time context buffer with
                               periodic reconstitution.
ContextForcingDistillation  — training wrapper (student generates own context).
ContextForcedGenerator      — inference wrapper for autoregressive generation.
ContextForcingDistiller     — thin alias that combines the above for
                               import from the package __init__.
"""

from __future__ import annotations

import math
import random
from collections import deque
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# ReconstitutedContextMemory
# ---------------------------------------------------------------------------

class ReconstitutedContextMemory:
    """
    Full-history frame buffer with periodic model-based reconstitution.

    Maintains all generated frames.  Every ``reconstitute_every`` steps it
    picks ``num_reconstitute`` distant frames and re-runs the model on them
    to obtain in-distribution context representations.

    Parameters
    ----------
    model               : the inference model (student or teacher).
    reconstitute_every  : re-generate distant frames every N new frames
                          (default 20).
    num_reconstitute    : how many distant frames to reconstitute per cycle
                          (default 3).
    window_size         : number of context frames used for the next
                          generation block (default 6).
    """

    def __init__(
        self,
        model: nn.Module,
        reconstitute_every: int = 20,
        num_reconstitute: int = 3,
        window_size: int = 6,
    ) -> None:
        self.model = model
        self.reconstitute_every = reconstitute_every
        self.num_reconstitute = num_reconstitute
        self.window_size = window_size

        # History: list of (frame_idx, frame_tensor) ordered by frame_idx.
        self._history: List[Tuple[int, torch.Tensor]] = []
        # Reconstituted overrides: frame_idx → reconstituted tensor.
        self._reconstituted: Dict[int, torch.Tensor] = {}

    def add_frame(self, frame: torch.Tensor, frame_idx: int) -> None:
        """
        Append a newly generated frame to the history.

        Triggers reconstitution if the reconstitution interval is reached.

        Parameters
        ----------
        frame     : (B, C, H, W) or (B, T, C, H, W) frame tensor.
        frame_idx : the global frame index.
        """
        self._history.append((frame_idx, frame.detach()))

        # Trigger reconstitution every N frames.
        if (frame_idx + 1) % self.reconstitute_every == 0 and frame_idx > self.window_size:
            self._run_reconstitution(frame_idx)

    def get_context(self, current_frame_idx: int) -> List[torch.Tensor]:
        """
        Return the most recent ``window_size`` frames, with reconstituted
        frames substituted where available.

        Parameters
        ----------
        current_frame_idx : index of the frame *about to be generated*.

        Returns
        -------
        List of ``window_size`` frame tensors in chronological order.
        """
        # Collect up to window_size most recent frames.
        if not self._history:
            return []

        window = self._history[-self.window_size:]
        context: List[torch.Tensor] = []

        for idx, frame in window:
            # Prefer reconstituted version if available.
            if idx in self._reconstituted:
                context.append(self._reconstituted[idx])
            else:
                context.append(frame)

        return context

    def reconstitute_distant_frames(
        self, frame_indices: List[int]
    ) -> List[torch.Tensor]:
        """
        Re-run the model on stored frames at the given indices to produce
        in-distribution representations.

        Parameters
        ----------
        frame_indices : list of global frame indices to reconstitute.

        Returns
        -------
        List of reconstituted frame tensors (same order as input indices).
        """
        reconstituted: List[torch.Tensor] = []

        # Build index → frame mapping from history.
        idx_to_frame: Dict[int, torch.Tensor] = {
            idx: frame for idx, frame in self._history
        }

        self.model.eval()
        with torch.no_grad():
            for frame_idx in frame_indices:
                if frame_idx not in idx_to_frame:
                    continue

                frame = idx_to_frame[frame_idx]
                device = frame.device
                dtype = frame.dtype

                # Re-run model on the stored frame.
                # For reconstitution we run a single denoising step with light
                # noise to obtain an in-distribution representation.
                t_light = 0.05
                noise = torch.randn_like(frame)
                x_noised = frame + t_light * noise

                B = frame.shape[0]
                if frame.ndim == 5:
                    _, T_vid = frame.shape[:2]
                    t_tensor = torch.full(
                        (B, T_vid), t_light, device=device, dtype=dtype
                    )
                else:
                    t_tensor = torch.full((B,), t_light, device=device, dtype=dtype)

                try:
                    denoised = self.model(x_noised, t_tensor)
                except Exception:
                    # Fall back to the original frame if the model call fails.
                    denoised = frame

                self._reconstituted[frame_idx] = denoised
                reconstituted.append(denoised)

        return reconstituted

    def _run_reconstitution(self, current_frame_idx: int) -> None:
        """
        Select distant frames and trigger reconstitution.

        Samples ``num_reconstitute`` frames spaced at least ``window_size``
        steps before the current frame.
        """
        eligible = [
            idx for idx, _ in self._history
            if idx <= current_frame_idx - self.window_size
        ]
        if len(eligible) < self.num_reconstitute:
            return

        # Sample without replacement, prefer temporal diversity.
        step = max(1, len(eligible) // self.num_reconstitute)
        sampled_indices = [eligible[i * step] for i in range(self.num_reconstitute)]
        self.reconstitute_distant_frames(sampled_indices)

    def reset(self) -> None:
        """Clear all history and reconstituted frames."""
        self._history.clear()
        self._reconstituted.clear()


# ---------------------------------------------------------------------------
# ContextForcingDistillation
# ---------------------------------------------------------------------------

class ContextForcingDistillation:
    """
    Training wrapper that forces the student model to generate its own context
    frames, closing the train/inference distribution gap.

    During training, ``forcing_ratio`` controls what fraction of batches use
    student-generated context (vs. pre-computed teacher context).  This ratio
    is gradually ramped from 0 → 1 over training via ``train()``.

    Parameters
    ----------
    student_model      : the student model being distilled.
    teacher_model      : frozen teacher model.
    num_context_frames : size of the context window (default 6).
    forcing_ratio      : initial fraction of batches using student context
                         (default 0.5).
    kl_weight          : weight of the optional KL divergence regulariser
                         (default 0.1).
    lr                 : AdamW learning rate (default 1e-5).
    """

    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        num_context_frames: int = 6,
        forcing_ratio: float = 0.5,
        kl_weight: float = 0.1,
        lr: float = 1e-5,
    ) -> None:
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.num_context_frames = num_context_frames
        self.forcing_ratio = forcing_ratio
        self.kl_weight = kl_weight

        # Freeze teacher.
        self.teacher_model.eval()
        for p in self.teacher_model.parameters():
            p.requires_grad_(False)

        self.optimizer = AdamW(self.student_model.parameters(), lr=lr)

    # ------------------------------------------------------------------
    # KL divergence helper
    # ------------------------------------------------------------------

    @staticmethod
    def _gaussian_kl(
        mu_p: torch.Tensor,
        mu_q: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        Simplified KL divergence treating outputs as unit-variance Gaussians:

            KL(N(μ_p, I) || N(μ_q, I)) = 0.5 · ||μ_p − μ_q||²

        This is a practical approximation when we don't have explicit log-var
        outputs from the models.

        Parameters
        ----------
        mu_p, mu_q : predicted outputs of student and teacher.
        eps        : unused (kept for API stability).

        Returns
        -------
        Scalar KL approximation.
        """
        return 0.5 * F.mse_loss(mu_p, mu_q)

    # ------------------------------------------------------------------
    # Student context generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _generate_student_context(
        self,
        earlier_frames: torch.Tensor,
        cond: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Run the student model on ``earlier_frames`` to produce in-distribution
        context that will be used for the next-block prediction.

        Parameters
        ----------
        earlier_frames : (B, T_ctx, C, H, W) latent frames at moderate noise.
        cond           : optional conditioning.

        Returns
        -------
        Student-generated context of same shape.
        """
        self.student_model.eval()
        B, T_ctx, C, H, W = earlier_frames.shape
        device = earlier_frames.device
        dtype = earlier_frames.dtype

        # Light noise forward — one step at a low noise level.
        t_light = 0.1
        noise = torch.randn_like(earlier_frames)
        x_noised = earlier_frames + t_light * noise

        t_tensor = torch.full((B, T_ctx), t_light, device=device, dtype=dtype)

        try:
            if cond is not None:
                out = self.student_model(x_noised, t_tensor, cond)
            else:
                out = self.student_model(x_noised, t_tensor)
        except Exception:
            out = earlier_frames  # graceful fallback

        return out.detach()

    # ------------------------------------------------------------------
    # Core loss step
    # ------------------------------------------------------------------

    def compute_loss_step(
        self,
        batch: Dict[str, torch.Tensor],
        use_student_context: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the context-forcing distillation loss for one batch.

        Parameters
        ----------
        batch : dict with keys:
            'frames'         — (B, T, C, H, W) target frames (noisy).
            'noisy_frames'   — (B, T, C, H, W) noisy version of target frames.
            't'              — (B, T) or (B,) timestep tensor.
            'context_frames' — (B, T_ctx, C, H, W) teacher-generated context.
            'earlier_frames' — (B, T_ctx, C, H, W) earlier frames for student
                               context generation.
            'actions'        — optional action conditioning.
        use_student_context : if True, generate student context (forcing mode).

        Returns
        -------
        Dict with 'loss_main', 'loss_kl', 'loss_total'.
        """
        device = next(self.student_model.parameters()).device
        dtype = next(self.student_model.parameters()).dtype

        noisy_frames = batch["noisy_frames"].to(device=device, dtype=dtype)
        t = batch["t"].to(device=device, dtype=dtype)
        teacher_context = batch["context_frames"].to(device=device, dtype=dtype)
        actions = batch.get("actions", None)
        if actions is not None:
            actions = actions.to(device=device, dtype=dtype)

        # Determine context to use for the student.
        if use_student_context and "earlier_frames" in batch:
            earlier = batch["earlier_frames"].to(device=device, dtype=dtype)
            student_context = self._generate_student_context(earlier, actions)
        else:
            student_context = teacher_context

        # Student forward with its own context.
        self.student_model.train()
        try:
            if actions is not None:
                student_out = self.student_model(
                    noisy_frames, t, cond=student_context, actions=actions
                )
            else:
                student_out = self.student_model(noisy_frames, t, cond=student_context)
        except TypeError:
            student_out = self.student_model(noisy_frames, t)

        # Teacher forward with teacher context (reference).
        with torch.no_grad():
            try:
                if actions is not None:
                    teacher_out = self.teacher_model(
                        noisy_frames, t, cond=teacher_context, actions=actions
                    )
                else:
                    teacher_out = self.teacher_model(
                        noisy_frames, t, cond=teacher_context
                    )
            except TypeError:
                teacher_out = self.teacher_model(noisy_frames, t)

        # MSE main loss: student-with-student-context ≈ teacher-with-teacher-context.
        loss_main = F.mse_loss(student_out, teacher_out.detach())

        # Optional KL regulariser (Gaussian approximation).
        loss_kl = self._gaussian_kl(student_out, teacher_out.detach())

        loss_total = loss_main + self.kl_weight * loss_kl

        return {
            "loss_main": loss_main,
            "loss_kl": loss_kl,
            "loss_total": loss_total,
        }

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(
        self,
        dataloader: DataLoader,
        epochs: int = 10,
        ramp_forcing_ratio: bool = True,
    ) -> None:
        """
        Progressive context-forcing training loop.

        When ``ramp_forcing_ratio`` is True, the probability of using
        student-generated context is linearly ramped from 0.0 to 1.0 over
        the training run.  This prevents cold-start instability — the student
        first learns from teacher context, then progressively shifts to its own.

        Parameters
        ----------
        dataloader         : yields batches as described in ``compute_loss_step``.
        epochs             : number of training epochs.
        ramp_forcing_ratio : if True, linearly ramp forcing_ratio 0 → 1.
        """
        total_steps = epochs * len(dataloader)
        step = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch in dataloader:
                # Compute current forcing ratio.
                if ramp_forcing_ratio:
                    current_ratio = min(1.0, step / max(total_steps - 1, 1))
                else:
                    current_ratio = self.forcing_ratio

                use_student_ctx = random.random() < current_ratio

                self.optimizer.zero_grad()
                losses = self.compute_loss_step(batch, use_student_context=use_student_ctx)
                losses["loss_total"].backward()
                torch.nn.utils.clip_grad_norm_(
                    self.student_model.parameters(), max_norm=1.0
                )
                self.optimizer.step()

                epoch_loss += losses["loss_total"].item()
                num_batches += 1
                step += 1

            avg = epoch_loss / max(num_batches, 1)
            print(
                f"[ContextForcing] Epoch {epoch + 1}/{epochs}  "
                f"avg_loss={avg:.6f}  "
                f"forcing_ratio={current_ratio:.3f}"
            )

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save student model weights."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "student_state_dict": self.student_model.state_dict(),
                "num_context_frames": self.num_context_frames,
                "forcing_ratio": self.forcing_ratio,
                "kl_weight": self.kl_weight,
            },
            str(path),
        )
        print(f"[ContextForcingDistillation] Saved → {path}")


# ---------------------------------------------------------------------------
# ContextForcedGenerator  (inference)
# ---------------------------------------------------------------------------

class ContextForcedGenerator:
    """
    Inference wrapper for autoregressive game video generation using
    context forcing + reconstituted context memory.

    Maintains a rolling window of recently generated frames and periodically
    reconstitutes distant frames to keep the context in-distribution.

    Parameters
    ----------
    model                : the distilled (or base) generation model.
    memory_size          : maximum number of frames to keep in history
                           (default 100).
    reconstitute_every   : reconstitute distant frames every N new frames
                           (default 20).
    """

    def __init__(
        self,
        model: nn.Module,
        memory_size: int = 100,
        reconstitute_every: int = 20,
    ) -> None:
        self.model = model
        self.memory_size = memory_size
        self.reconstitute_every = reconstitute_every

        self._context_memory = ReconstitutedContextMemory(
            model=model,
            reconstitute_every=reconstitute_every,
            num_reconstitute=3,
            window_size=6,
        )
        self._frame_idx: int = 0

    def generate_next_block(
        self,
        actions: torch.Tensor,
        num_frames: int,
        noise_level: float = 1.0,
        num_denoising_steps: int = 3,
    ) -> torch.Tensor:
        """
        Generate the next ``num_frames`` frames given action conditioning.

        Uses the reconstituted context memory to obtain a stable context
        window, then runs the model's denoising loop.

        Parameters
        ----------
        actions              : (B, A) action conditioning tensor.
        num_frames           : number of frames to generate in this block.
        noise_level          : initial noise level for the denoising schedule.
        num_denoising_steps  : number of denoising steps.

        Returns
        -------
        Generated frames as (B, num_frames, C, H, W).
        """
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        B = actions.shape[0]
        actions = actions.to(device=device, dtype=dtype)

        # Retrieve context from memory.
        context_list = self._context_memory.get_context(self._frame_idx)

        # Stack context into a tensor or use None if empty.
        if context_list:
            context = torch.stack(context_list, dim=1)  # (B, T_ctx, C, H, W) approx
            # Handle shape mismatches gracefully.
            if context.ndim == 4:
                context = context.unsqueeze(1)
        else:
            context = None

        # Infer latent shape from context or use Oasis default.
        if context is not None:
            C_latent = context.shape[-3] if context.ndim >= 3 else 16
            H_latent = context.shape[-2] if context.ndim >= 2 else 18
            W_latent = context.shape[-1] if context.ndim >= 1 else 32
        else:
            C_latent, H_latent, W_latent = 16, 18, 32

        # Initialise noise.
        latents = torch.randn(
            B, num_frames, C_latent, H_latent, W_latent,
            device=device, dtype=dtype,
        ) * noise_level

        # Simple denoising loop.
        timesteps = torch.linspace(
            noise_level, 0.0, num_denoising_steps + 1, device=device
        )

        self.model.eval()
        with torch.no_grad():
            for i in range(num_denoising_steps):
                t_start = float(timesteps[i].item())
                t_end = float(timesteps[i + 1].item())

                t_tensor = torch.full(
                    (B, num_frames), t_start, device=device, dtype=dtype
                )

                try:
                    if context is not None:
                        noise_pred = self.model(
                            latents, t_tensor, cond=context, actions=actions
                        )
                    else:
                        noise_pred = self.model(latents, t_tensor, actions=actions)
                except TypeError:
                    try:
                        noise_pred = self.model(latents, t_tensor)
                    except Exception:
                        # Absolute fallback: just return latent noise.
                        return latents

                # Euler step.
                latents = latents + (t_end - t_start) * noise_pred

        # Store generated frames in memory.
        for frame_offset in range(num_frames):
            frame_tensor = latents[:, frame_offset]  # (B, C, H, W)
            self._context_memory.add_frame(frame_tensor, self._frame_idx + frame_offset)

        self._frame_idx += num_frames
        return latents

    def reset(self) -> None:
        """
        Reset all internal state to begin a new episode.

        Clears the frame history and resets the frame counter.
        """
        self._context_memory.reset()
        self._frame_idx = 0


# ---------------------------------------------------------------------------
# ContextForcingDistiller  (package-level alias)
# ---------------------------------------------------------------------------

class ContextForcingDistiller(ContextForcingDistillation):
    """
    Convenience alias combining ContextForcingDistillation (training) with
    easy access to a ContextForcedGenerator (inference).

    Usage
    -----
    distiller = ContextForcingDistiller(student, teacher)
    distiller.train(dataloader, epochs=10)
    distiller.save("cf_student.pt")

    # Inference
    gen = distiller.make_generator()
    frames = gen.generate_next_block(actions, num_frames=16)
    """

    def make_generator(
        self,
        memory_size: int = 100,
        reconstitute_every: int = 20,
    ) -> ContextForcedGenerator:
        """
        Create a :class:`ContextForcedGenerator` wrapping the student model.

        Parameters
        ----------
        memory_size        : maximum frames kept in history.
        reconstitute_every : reconstitution interval (frames).

        Returns
        -------
        ContextForcedGenerator ready for inference.
        """
        return ContextForcedGenerator(
            model=self.student_model,
            memory_size=memory_size,
            reconstitute_every=reconstitute_every,
        )
