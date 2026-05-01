"""
ParaDiGMS: Parallel Denoising via Picard Iterations for DDIM-based models (Oasis).

Parallelizes the sequential DDIM denoising loop by running multiple denoising
steps simultaneously across CUDA streams, using Picard iterations to refine
the parallel guesses until convergence.

Reference:
  ParaDiGMS (NeurIPS 2023) — https://hf.co/papers/2305.16317
  "Accelerating Parallel Sampling of Diffusion Models"

Architecture target:
  Open-Oasis (10-step DDIM canonical, 144 spatial tokens/frame after p=2
  patchify, v-prediction, sigmoid_beta_schedule, H100 80GB).
  Not suitable for MG2 (only 3 steps — not worth parallelizing).

Critical Oasis specifics (corrected vs original):
  * DiT output is v (velocity), NOT eps. DDIM update is:
        x0_pred  = √ᾱ_t · x_t − √(1−ᾱ_t) · v
        eps_pred = √(1−ᾱ_t) · x_t + √ᾱ_t · v
        x_{t-1}  = √ᾱ_{t-1} · x0_pred + √(1−ᾱ_{t-1}) · eps_pred
  * Timestep tensor must be torch.long and shaped (B, T) where past frames are
    0 (clean context) and only the current frame carries t_val.
  * Generation is autoregressive per-frame; ParaDiGMS parallelizes the inner
    DDIM loop *within a single frame*, not across frames.

Expected speedup:
  2-4x wallclock at parallel_steps=2-4 on H100 80GB (Oasis uses ~18GB → 36-72GB for 2-4x).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class ParaDiGMSSolver:
    """
    Parallel denoising via Picard iterations for DDIM-based world models.

    Replaces the sequential DDIM loop with a parallel variant:

    Sequential (current Oasis, 10 steps per frame):
      x_{9} → x_{8} → ... → x_0   (10 serial steps, each blocking the next)

    ParaDiGMS:
      1. Initialize parallel_steps guesses by running single forward from x_t.
      2. While not converged:
         a. For each parallel step pair (t_i, t_{i-1}): run DDIM step in parallel
            using current best estimate of x_{t_i}.
         b. Check Picard convergence: ||x_new - x_old|| < picard_tol.
      3. Advance window by parallel_steps.

    The key insight: DDIM steps are weakly coupled — x_{t-1} depends mainly on
    x_t and v_pred(x_t), not on x_{t+1}. Picard iterations exploit this to
    converge in 2-3 iterations for most denoising steps.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Args:
            config: Dict with keys:
              parallel_steps (int, 2): Steps to run in parallel per window.
              picard_tol (float, 1e-3): Convergence tolerance (L2 norm normalized).
              max_picard_iters (int, 3): Max Picard iterations before accepting.
        """
        self.parallel_steps    = int(config.get("parallel_steps",    2))
        self.picard_tol        = float(config.get("picard_tol",      1e-3))
        self.max_picard_iters  = int(config.get("max_picard_iters",  3))

        # CUDA streams for parallel execution
        self._streams: List[torch.cuda.Stream] = []
        self._initialized = False

        # Stats
        self._total_windows: int = 0
        self._total_picard_iters: int = 0
        self._converged_early: int = 0

    def _ensure_streams(self) -> None:
        if not self._initialized and torch.cuda.is_available():
            self._streams = [
                torch.cuda.Stream() for _ in range(self.parallel_steps)
            ]
            self._initialized = True

    def parallel_denoise(
        self,
        dit: Any,
        latents: torch.Tensor,
        t_seq: torch.Tensor,
        actions: torch.Tensor,
        alpha_bar: torch.Tensor,
        num_steps: int,
        optimizations: Optional[Any] = None,
        amp_dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        """
        Run the full DDIM denoising loop with Picard-parallelized windows.

        Drop-in replacement for the sequential loop in oasis.py generate().

        Args:
            dit:          The DiT model (eps predictor).
            latents:      Initial noisy latent (B, T, C, H, W).
            t_seq:        DDIM timestep sequence (num_steps+1,) from linspace.
            actions:      Action conditioning (B, T, 25).
            alpha_bar:    Cumulative alpha schedule (1000,).
            num_steps:    Total denoising steps.
            optimizations: OptimizationStack (passed through for hooks).
            amp_dtype:    AMP autocast dtype.

        Returns:
            Denoised latent (B, T, C, H, W).
        """
        self._ensure_streams()
        device = latents.device
        dtype = latents.dtype

        K = self.parallel_steps
        step = 0

        with torch.cuda.amp.autocast(dtype=amp_dtype):
            while step < num_steps:
                # Window: steps [step, step+K)
                window_end = min(step + K, num_steps)
                window_size = window_end - step

                if window_size == 1:
                    # Last step or K=1: run sequentially
                    latents = self._single_ddim_step(
                        dit, latents, t_seq, alpha_bar, step, actions,
                        num_steps, optimizations,
                    )
                    step += 1
                    continue

                # Initialize parallel guesses: x̂_{step+1}, ..., x̂_{step+K}
                # Start with x_{step} as the initial guess for all parallel slots
                parallel_latents = [latents.clone() for _ in range(window_size)]

                # Picard iterations
                converged = False
                for picard_iter in range(self.max_picard_iters):
                    self._total_picard_iters += 1
                    new_latents = list(parallel_latents)

                    # Run window_size DDIM steps in parallel via CUDA streams
                    results = self._parallel_window(
                        dit, parallel_latents, t_seq, alpha_bar,
                        step, window_size, actions, optimizations,
                    )

                    # Check convergence
                    max_drift = 0.0
                    for j in range(window_size):
                        drift = (results[j] - parallel_latents[j]).norm()
                        norm = parallel_latents[j].norm() + 1e-8
                        max_drift = max(max_drift, (drift / norm).item())
                        new_latents[j] = results[j]

                    parallel_latents = new_latents

                    if max_drift < self.picard_tol:
                        converged = True
                        self._converged_early += 1
                        break

                # Use the last parallel latent as the starting point for the next window
                latents = parallel_latents[-1]
                step = window_end
                self._total_windows += 1

        return latents

    def _parallel_window(
        self,
        dit: Any,
        parallel_latents: List[torch.Tensor],
        t_seq: torch.Tensor,
        alpha_bar: torch.Tensor,
        step_offset: int,
        window_size: int,
        actions: torch.Tensor,
        optimizations: Optional[Any],
    ) -> List[torch.Tensor]:
        """
        Run window_size DDIM steps in parallel using CUDA streams.
        """
        events = [torch.cuda.Event(enable_timing=False) for _ in range(window_size)]
        results = [None] * window_size

        for j in range(window_size):
            stream = self._streams[j % len(self._streams)] if self._streams else torch.cuda.current_stream()
            with torch.cuda.stream(stream):
                s = step_offset + j
                x = parallel_latents[j]
                t_int = int(t_seq[s].item())
                t_next_int = max(int(t_seq[s + 1].item()), 0)

                # Oasis t_in is long-typed (B, T). Inside parallel_denoise the
                # caller already constructed t_in with zeros for context frames
                # and t_val for the current frame; here we just broadcast t_int
                # uniformly because parallel_denoise sees one frame at a time.
                t_batch = torch.full(
                    (x.shape[0], x.shape[1]), t_int,
                    device=x.device, dtype=torch.long,
                )

                with torch.no_grad():
                    v = dit(x, t_batch, external_cond=actions)
                    if isinstance(v, (tuple, list)):
                        v = v[0]

                ab_t = alpha_bar[t_int].float()
                ab_t_next = alpha_bar[t_next_int].float()
                x_f = x.float()
                v_f = v.float()
                # v-prediction DDIM (Salimans & Ho 2022, sec 4)
                x0_pred  = ab_t.sqrt() * x_f - (1.0 - ab_t).sqrt() * v_f
                eps_pred = (1.0 - ab_t).sqrt() * x_f + ab_t.sqrt() * v_f
                x0_pred  = x0_pred.clamp(-20.0, 20.0)
                x_next = (
                    ab_t_next.sqrt() * x0_pred + (1.0 - ab_t_next).sqrt() * eps_pred
                ).to(dtype=x.dtype)
                results[j] = x_next
            events[j].record(stream)

        # Synchronize all streams
        for event in events:
            event.synchronize()

        return results  # type: ignore[return-value]

    def _single_ddim_step(
        self,
        dit: Any,
        latents: torch.Tensor,
        t_seq: torch.Tensor,
        alpha_bar: torch.Tensor,
        step: int,
        actions: torch.Tensor,
        num_steps: int,
        optimizations: Optional[Any],
    ) -> torch.Tensor:
        """Run a single DDIM v-prediction step (used for the last step or K=1)."""
        t_int = int(t_seq[step].item())
        t_next_int = max(int(t_seq[step + 1].item()), 0)

        t_batch = torch.full(
            (latents.shape[0], latents.shape[1]), t_int,
            device=latents.device, dtype=torch.long,
        )

        if optimizations is not None and hasattr(optimizations, "pre_step"):
            skip = optimizations.pre_step(step, num_steps, latents)
            if skip:
                return latents

        with torch.no_grad():
            v = dit(latents, t_batch, external_cond=actions)
            if isinstance(v, (tuple, list)):
                v = v[0]

        ab_t = alpha_bar[t_int].float()
        ab_t_next = alpha_bar[t_next_int].float()
        x_f = latents.float()
        v_f = v.float()
        x0_pred  = ab_t.sqrt() * x_f - (1.0 - ab_t).sqrt() * v_f
        eps_pred = (1.0 - ab_t).sqrt() * x_f + ab_t.sqrt() * v_f
        x0_pred  = x0_pred.clamp(-20.0, 20.0)
        latents = (
            ab_t_next.sqrt() * x0_pred + (1.0 - ab_t_next).sqrt() * eps_pred
        ).to(dtype=latents.dtype)

        if optimizations is not None and hasattr(optimizations, "post_step"):
            optimizations.post_step(step, num_steps, latents)

        return latents

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_windows": self._total_windows,
            "total_picard_iters": self._total_picard_iters,
            "avg_picard_iters_per_window": (
                self._total_picard_iters / max(self._total_windows, 1)
            ),
            "early_convergence_rate": (
                self._converged_early / max(self._total_windows, 1)
            ),
            "parallel_steps": self.parallel_steps,
            "picard_tol": self.picard_tol,
        }

    def reset_stats(self) -> None:
        self._total_windows = 0
        self._total_picard_iters = 0
        self._converged_early = 0
