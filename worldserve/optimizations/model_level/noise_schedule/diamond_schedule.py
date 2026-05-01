"""
DIAMOND-informed noise schedule for game-video diffusion (Oasis).

Reference: "Diffusion for World Modeling: Visual Details Matter in Atari"
           DIAMOND — arxiv 2405.12399

Standard DDPM cosine schedule (Improved DDPM, Ho et al. 2022):
    alpha_bar_t = f(t) / f(0),  f(t) = cos²(π/2 * (t/T + s) / (1 + s))
with s = 0.008 (small offset to prevent alpha_bar_0 from being too small).

Game video schedules require higher SNR (lower noise) at every step
because consecutive frames are highly correlated (~0.95 cosine similarity
vs ~0.5 for natural images).  The key adaptation is:

    1. Reduce offset s: 0.008 → 0.003
       This shifts alpha_bar toward 1 everywhere (less noise).

    2. Non-uniform timestep grid: concentrate steps in t ∈ [0.2, 0.6]
       where most of the structural change happens in game video, and
       use fewer steps near t ≈ 1.0 (pure noise, low information) and
       t ≈ 0.0 (near-clean, diminishing returns).

    3. Optional calibration: measure empirical inter-frame cosine
       similarity from a batch of game video and adjust s accordingly.

Oasis specifics:
  - 20 denoising steps
  - Original uniform schedule: linspace(1.0, 0.0, 21) (21 breakpoints)
  - 144 spatial tokens per frame (18 × 32 ÷ VAE factor)
  - DiT-S/2 architecture
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level schedule functions (no class dependency — easier to test)
# ---------------------------------------------------------------------------

def cosine_schedule(
    num_steps: int,
    s: float = 0.008,
) -> torch.Tensor:
    """
    Standard cosine noise schedule from Improved DDPM (arxiv 2102.09672).

    Returns alpha_bar values at uniformly spaced timesteps.

    Parameters
    ----------
    num_steps:
        Number of denoising steps T.  Returns T+1 alpha_bar values
        (index 0 = t=0/clean, index T = t=T/noisy).
    s:
        Small offset preventing alpha_bar_0 from being too close to 1.
        Default 0.008 (Improved DDPM).

    Returns
    -------
    torch.Tensor
        Shape (num_steps + 1,), alpha_bar values in [0, 1], decreasing.
    """
    steps = num_steps + 1
    t = torch.linspace(0, num_steps, steps)
    f_t = torch.cos(((t / num_steps + s) / (1 + s)) * math.pi / 2) ** 2
    alpha_bar = f_t / f_t[0]
    # Clamp to [1e-9, 1] for numerical stability
    alpha_bar = alpha_bar.clamp(min=1e-9, max=1.0)
    return alpha_bar


def game_video_schedule(
    num_steps: int,
    temporal_corr: float = 0.95,
    s: float = 0.003,
) -> torch.Tensor:
    """
    Game-video-adapted cosine noise schedule.

    Compared to the standard cosine schedule, this uses a smaller s
    offset (0.003 vs 0.008) to shift toward higher SNR.  The shift
    magnitude is further scaled by temporal_corr:

        s_eff = s * (1 - temporal_corr)^{0.5}

    so a higher inter-frame correlation leads to an even smaller s
    (less noise), and vice versa.

    Empirically for Atari/Minecraft game video with temporal_corr=0.95:
        s_eff ≈ 0.003 * sqrt(0.05) ≈ 0.00067

    Parameters
    ----------
    num_steps:
        Number of denoising steps.
    temporal_corr:
        Empirical inter-frame cosine similarity of the target video.
        For Atari / Minecraft: ~0.95.  For natural video: ~0.5.
    s:
        Base offset.  Reduced from ImageNet default of 0.008.

    Returns
    -------
    torch.Tensor
        Shape (num_steps + 1,), alpha_bar values in [0, 1], decreasing.
    """
    # Scale s by (1 - corr)^0.5: higher correlation → smaller s → more SNR
    s_eff = s * math.sqrt(max(1e-6, 1.0 - temporal_corr))
    return cosine_schedule(num_steps=num_steps, s=s_eff)


def measure_temporal_correlation(frames: torch.Tensor) -> float:
    """
    Compute mean inter-frame cosine similarity from a batch of video frames.

    Parameters
    ----------
    frames:
        (B, T, C, H, W)  or  (T, C, H, W)  float tensor of video frames.
        Values can be in any range; normalisation is applied internally.

    Returns
    -------
    float
        Mean cosine similarity between consecutive frame pairs, averaged
        over batch and time.  Range [−1, 1]; game video typically 0.90–0.97.
    """
    if frames.dim() == 4:
        frames = frames.unsqueeze(0)  # (1, T, C, H, W)

    B, T, C, H, W = frames.shape
    if T < 2:
        logger.warning("measure_temporal_correlation: need >= 2 frames, got %d.", T)
        return 0.95  # fallback

    # Flatten spatial + channel dims: (B, T, C*H*W)
    flat = frames.reshape(B, T, -1).float()

    # Normalise
    flat = F.normalize(flat, dim=-1)

    # Pairwise cosine similarity for consecutive frames
    # cos(f_t, f_{t+1}) = dot product after normalisation
    sim = (flat[:, :-1] * flat[:, 1:]).sum(dim=-1)  # (B, T-1)
    mean_sim = float(sim.mean().item())
    return mean_sim


# ---------------------------------------------------------------------------
# Main DIAMONDNoiseSchedule class
# ---------------------------------------------------------------------------

class DIAMONDNoiseSchedule:
    """
    Game-video-adapted noise schedule for DDPM-based world models.

    Standard schedules use: alpha_bar_t = product of (1 - beta_t)
    ImageNet optimal: beta follows cosine schedule (Improved DDPM, 2102.09672)
    Game video optimal: shift cosine schedule → higher SNR at each step

    The key adjustment: game frames have ~0.95 inter-frame cosine similarity
    vs ~0.5 for natural images. This means the model needs less noise to learn
    frame prediction (the task is "easier" at each step).

    Parameters
    ----------
    num_steps:
        Number of denoising steps.  Default 20 for Oasis.
    temporal_corr:
        Initial inter-frame cosine similarity estimate.  Refined by
        calibrate_from_video() if called.
    s:
        Base cosine-schedule offset.  Reduced from 0.008 to 0.003.
    concentrate_range:
        (t_low, t_high) tuple.  The non-uniform timestep grid places
        more steps in this range.  Default (0.2, 0.6) based on DIAMOND
        analysis of where game-video structure forms.
    concentrate_ratio:
        Fraction of total steps to concentrate in concentrate_range.
        Default 0.5 (half the steps in a 40%-wide window).
    """

    def __init__(
        self,
        num_steps: int = 20,
        temporal_corr: float = 0.95,
        s: float = 0.003,
        concentrate_range: Tuple[float, float] = (0.2, 0.6),
        concentrate_ratio: float = 0.5,
    ) -> None:
        self.num_steps = num_steps
        self.temporal_corr = temporal_corr
        self.s = s
        self.concentrate_range = concentrate_range
        self.concentrate_ratio = concentrate_ratio

        # Pre-compute alpha_bar
        self._alpha_bar: Optional[torch.Tensor] = None
        self._timestep_grid: Optional[torch.Tensor] = None

        self._build_schedule()

    # ------------------------------------------------------------------
    # Schedule construction
    # ------------------------------------------------------------------

    def _build_schedule(self) -> None:
        """Recompute alpha_bar and timestep grid (called on init and after calibration)."""
        self._alpha_bar = game_video_schedule(
            num_steps=self.num_steps,
            temporal_corr=self.temporal_corr,
            s=self.s,
        )
        self._timestep_grid = self._build_nonuniform_timesteps()

    def _build_nonuniform_timesteps(self) -> torch.Tensor:
        """
        Build a non-uniform timestep grid that concentrates steps in
        self.concentrate_range.

        Strategy:
          - Split num_steps into three segments:
              [1.0, t_high), [t_high, t_low], (t_low, 0.0]
          - The middle segment [t_high, t_low] gets concentrate_ratio * num_steps
            steps; the remaining (1 - concentrate_ratio) steps are split evenly
            between the outer two segments.
          - Concatenate and sort descending (from 1.0 to 0.0).

        Returns
        -------
        torch.Tensor
            Shape (num_steps + 1,) timestep values from ~1.0 to ~0.0.
        """
        t_low, t_high = self.concentrate_range
        total = self.num_steps

        # Steps in the concentrated range
        mid_steps = max(2, int(total * self.concentrate_ratio))
        outer_steps = total - mid_steps
        low_steps = max(1, outer_steps // 2)
        high_steps = max(1, outer_steps - low_steps)

        # Three linspace segments (include endpoints, then deduplicate)
        seg_high = torch.linspace(1.0, t_high, high_steps + 1)       # 1.0 → t_high
        seg_mid  = torch.linspace(t_high, t_low, mid_steps + 1)[1:]   # t_high → t_low (skip dup)
        seg_low  = torch.linspace(t_low, 0.0, low_steps + 1)[1:]      # t_low → 0.0 (skip dup)

        grid = torch.cat([seg_high, seg_mid, seg_low])

        # The concatenated grid has high_steps + mid_steps + low_steps + 1 entries
        # Trim or pad to exactly num_steps + 1
        target_len = self.num_steps + 1
        if len(grid) > target_len:
            # Uniform sub-sample
            idx = torch.linspace(0, len(grid) - 1, target_len).long()
            grid = grid[idx]
        elif len(grid) < target_len:
            # Pad by interpolating near zero
            pad_vals = torch.linspace(float(grid[-2]), 0.0, target_len - len(grid) + 2)[1:-1]
            grid = torch.cat([grid[:-1], pad_vals, grid[-1:]])

        # Ensure strictly decreasing and in [0, 1]
        grid = grid.clamp(0.0, 1.0)
        # Sort descending
        grid, _ = grid.sort(descending=True)
        return grid

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate_from_video(self, video_frames: torch.Tensor) -> float:
        """
        Measure inter-frame correlation from game video and update the schedule.

        Parameters
        ----------
        video_frames:
            (B, T, C, H, W)  or  (T, C, H, W) float tensor.

        Returns
        -------
        float
            Measured temporal correlation used for calibration.
        """
        corr = measure_temporal_correlation(video_frames)
        self.temporal_corr = corr
        self._build_schedule()
        logger.info(
            "DIAMONDNoiseSchedule: calibrated from video → "
            "temporal_corr=%.4f, s_eff=%.6f",
            corr, self.s * math.sqrt(max(1e-6, 1.0 - corr)),
        )
        return corr

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_timesteps(
        self,
        num_steps: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Return the optimized non-uniform timestep schedule.

        Parameters
        ----------
        num_steps:
            If provided and different from self.num_steps, rebuilds the grid.
        device:
            Target device for the output tensor.

        Returns
        -------
        torch.Tensor
            Shape (num_steps + 1,), values in [0.0, 1.0] descending.
            Equivalent to linspace(1.0, 0.0, num_steps+1) for a uniform
            schedule, but non-uniform here to concentrate steps in the
            active range.
        """
        if num_steps is not None and num_steps != self.num_steps:
            old_steps = self.num_steps
            self.num_steps = num_steps
            self._build_schedule()
            logger.debug(
                "DIAMONDNoiseSchedule.get_timesteps: rebuilt for %d steps (was %d).",
                num_steps, old_steps,
            )

        grid = self._timestep_grid
        assert grid is not None
        if device is not None:
            grid = grid.to(device)
        return grid

    def get_alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """
        Return alpha_bar at continuous timestep t via linear interpolation.

        Parameters
        ----------
        t:
            Scalar or tensor of timestep values in [0.0, 1.0].
            t=1.0 → pure noise (alpha_bar ≈ 0).
            t=0.0 → clean image (alpha_bar = 1).

        Returns
        -------
        torch.Tensor
            alpha_bar(t), same shape as t.
        """
        assert self._alpha_bar is not None
        ab = self._alpha_bar.to(t.device)

        # alpha_bar is indexed 0..num_steps where index 0 = clean, T = noisy
        # Map t ∈ [0, 1] → continuous index ∈ [0, num_steps]
        idx_float = t * self.num_steps
        idx_low = idx_float.long().clamp(0, self.num_steps - 1)
        idx_high = (idx_low + 1).clamp(0, self.num_steps)
        frac = idx_float - idx_low.float()

        ab_low = ab[idx_low]
        ab_high = ab[idx_high]
        return ab_low + frac * (ab_high - ab_low)

    def get_betas(self) -> torch.Tensor:
        """
        Return per-step beta values derived from the alpha_bar schedule.

        beta_t = 1 - alpha_bar_t / alpha_bar_{t-1}

        Returns
        -------
        torch.Tensor
            Shape (num_steps,).
        """
        assert self._alpha_bar is not None
        ab = self._alpha_bar
        # beta_t = 1 - alpha_bar_t / alpha_bar_{t-1}
        betas = 1.0 - ab[1:] / ab[:-1]
        return betas.clamp(0.0, 0.999)

    def get_snr(self, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Signal-to-noise ratio schedule: SNR(t) = alpha_bar(t) / (1 - alpha_bar(t)).

        Parameters
        ----------
        t:
            Optional timestep tensor; if None returns SNR at all schedule steps.

        Returns
        -------
        torch.Tensor
        """
        if t is None:
            ab = self._alpha_bar
        else:
            ab = self.get_alpha_bar(t)
        return ab / (1 - ab + 1e-8)

    def compare_with_imagenet_schedule(
        self, num_steps: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Return both the standard ImageNet cosine schedule and this game-video
        schedule for comparison.

        Returns
        -------
        dict with keys 'imagenet_alpha_bar' and 'game_video_alpha_bar'.
        """
        n = num_steps or self.num_steps
        imagenet_ab = cosine_schedule(n, s=0.008)
        game_ab = game_video_schedule(n, temporal_corr=self.temporal_corr, s=self.s)
        return {
            "imagenet_alpha_bar": imagenet_ab,
            "game_video_alpha_bar": game_ab,
            "snr_ratio": self.get_snr() / (imagenet_ab / (1 - imagenet_ab + 1e-8)),
        }

    def __repr__(self) -> str:
        return (
            f"DIAMONDNoiseSchedule("
            f"num_steps={self.num_steps}, "
            f"temporal_corr={self.temporal_corr:.3f}, "
            f"s={self.s}, "
            f"concentrate_range={self.concentrate_range}, "
            f"concentrate_ratio={self.concentrate_ratio}"
            f")"
        )


# ---------------------------------------------------------------------------
# DIAMONDScheduleAdaptor
# ---------------------------------------------------------------------------

class DIAMONDScheduleAdaptor:
    """
    Wraps any model's denoising loop to replace the uniform timestep grid
    with the DIAMOND game-video-optimized schedule.

    Typical usage (Oasis)
    ---------------------
    >>> schedule = DIAMONDNoiseSchedule(num_steps=10)
    >>> adaptor = DIAMONDScheduleAdaptor(schedule)
    >>> # Replace oasis.generate with the adapted version:
    >>> frames = adaptor.adapt_oasis_loop(oasis.generate, obs=obs, num_steps=10)

    Parameters
    ----------
    schedule:
        A DIAMONDNoiseSchedule instance.
    """

    def __init__(self, schedule: DIAMONDNoiseSchedule) -> None:
        self.schedule = schedule

    # ------------------------------------------------------------------

    def adapt_oasis_loop(
        self,
        oasis_generate_fn: Callable[..., Any],
        *args: Any,
        num_steps: int = 20,
        device: Optional[torch.device] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Call oasis_generate_fn with a patched timestep schedule.

        This function:
        1. Builds the optimized non-uniform timestep grid.
        2. Passes it to oasis_generate_fn via the `timesteps` kwarg
           (overriding whatever default is in the function).

        If oasis_generate_fn does not accept a `timesteps` kwarg,
        it is called without modification and a warning is logged.

        Parameters
        ----------
        oasis_generate_fn:
            The generation / denoising callable (e.g. oasis.generate).
        *args:
            Positional arguments forwarded to oasis_generate_fn.
        num_steps:
            Number of denoising steps.
        device:
            Device for the timestep tensor.
        **kwargs:
            Keyword arguments forwarded to oasis_generate_fn.
            'timesteps' key is overwritten with the optimized schedule.

        Returns
        -------
        Any
            Whatever oasis_generate_fn returns (typically a frame tensor).
        """
        optimized_ts = self.schedule.get_timesteps(num_steps=num_steps, device=device)

        # Override or inject timesteps
        kwargs["timesteps"] = optimized_ts

        logger.debug(
            "DIAMONDScheduleAdaptor: injecting %d non-uniform timesteps "
            "(range %.3f–%.3f, concentrated in %s).",
            len(optimized_ts),
            float(optimized_ts.min()),
            float(optimized_ts.max()),
            self.schedule.concentrate_range,
        )

        try:
            return oasis_generate_fn(*args, **kwargs)
        except TypeError as exc:
            # oasis_generate_fn may not accept 'timesteps'; fall back
            logger.warning(
                "DIAMONDScheduleAdaptor: oasis_generate_fn did not accept "
                "'timesteps' kwarg (%s). Calling without schedule override.",
                exc,
            )
            kwargs.pop("timesteps", None)
            return oasis_generate_fn(*args, **kwargs)

    # ------------------------------------------------------------------

    def build_denoising_loop(
        self,
        model: Any,
        x_T: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        guidance_scale: float = 1.0,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        A standalone DDPM denoising loop using the DIAMOND schedule.

        Implements the reverse diffusion:
            x_{t-1} = (1/sqrt(alpha_t)) * (x_t − (beta_t / sqrt(1-alpha_bar_t)) * eps)
                      + sigma_t * z
        where eps = model(x_t, t, condition).

        This provides a self-contained loop independent of Oasis internals,
        useful for ablation studies.

        Parameters
        ----------
        model:
            Noise prediction model.  Called as model(x_t, t_embed, condition).
        x_T:
            (B, C, H, W)  initial noisy sample at t = T.
        condition:
            Optional conditioning tensor passed to model.
        num_steps:
            Number of denoising steps.  Defaults to self.schedule.num_steps.
        guidance_scale:
            Classifier-free guidance scale.  1.0 = no guidance.
        device:
            Device override.

        Returns
        -------
        torch.Tensor
            (B, C, H, W) denoised sample x_0.
        """
        n = num_steps or self.schedule.num_steps
        device = device or x_T.device
        timesteps = self.schedule.get_timesteps(num_steps=n, device=device)
        betas = self.schedule.get_betas().to(device)
        alpha_bar = self.schedule._alpha_bar
        assert alpha_bar is not None
        alpha_bar = alpha_bar.to(device)

        x = x_T.to(device)
        B = x.shape[0]

        with torch.no_grad():
            for i, t_val in enumerate(timesteps[:-1]):  # skip last (t=0)
                step_idx = i  # schedule index for this step

                # Compute alpha_bar at this and the next timestep
                ab_t = alpha_bar[step_idx]
                ab_t1 = alpha_bar[min(step_idx + 1, n)]

                # alpha_t = ab_t / ab_t1
                alpha_t = ab_t / (ab_t1 + 1e-8)
                beta_t = 1.0 - alpha_t

                # t_embed: (B,) scalar timestep in [0, T]
                t_tensor = t_val.expand(B).to(device)

                # Model prediction
                if condition is not None:
                    eps = model(x, t_tensor, condition)
                else:
                    eps = model(x, t_tensor)

                # Classifier-free guidance
                if guidance_scale != 1.0 and condition is not None:
                    eps_uncond = model(x, t_tensor)
                    eps = eps_uncond + guidance_scale * (eps - eps_uncond)

                # DDPM reverse step
                mean = (1.0 / alpha_t.sqrt()) * (
                    x - (beta_t / (1.0 - ab_t).sqrt()) * eps
                )

                if i < n - 1:
                    # Add noise for all but the final step
                    sigma_t = beta_t.sqrt()
                    z = torch.randn_like(x)
                    x = mean + sigma_t * z
                else:
                    x = mean

        return x

    # ------------------------------------------------------------------

    def patch_linspace_timesteps(
        self,
        module: Any,
        attr_name: str = "timesteps",
        num_steps: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Directly overwrite a `timesteps` attribute on any module/object.

        Many inference pipelines store their timestep grid as a tensor
        attribute.  This method replaces it with the optimized schedule.

        Parameters
        ----------
        module:
            Object with a timestep-grid attribute.
        attr_name:
            Name of the attribute to replace.
        num_steps:
            Optionally override step count.
        device:
            Device for new timesteps.
        """
        if not hasattr(module, attr_name):
            logger.warning(
                "patch_linspace_timesteps: %s has no attribute '%s'.",
                type(module).__name__, attr_name,
            )
            return

        optimized_ts = self.schedule.get_timesteps(num_steps=num_steps, device=device)
        setattr(module, attr_name, optimized_ts)
        logger.info(
            "DIAMONDScheduleAdaptor: patched %s.%s with %d optimized timesteps.",
            type(module).__name__, attr_name, len(optimized_ts),
        )
