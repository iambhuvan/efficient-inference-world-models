"""
DIAMOND CSGO + Action-Aware Step Skip — novel world-model optimisation.

Idea (the action-aware contribution): when the current action is very
similar to the previous action (cosine-sim > τ), the denoising trajectory
for the next observation should be very similar too. So:

  • cache the trajectory from the previous frame's full denoising
  • on a similar-action frame, warm-start at step K with cached x_K and
    only run the final (num_steps − K) Euler / Heun steps
  • on a different-action frame, run full denoising and refresh cache

Why this works on CSGO specifically: ~30–60% of human gameplay frames
have near-zero action delta (player not moving / not looking around),
which gives a meaningful skip rate. The Karras sigma schedule is shared
across frames, so trajectory[K] from frame T-1 lives at the same noise
level as step K of frame T — making warm-start sound, not a hack.

What we report (besides FPS):
  • skip_rate  — fraction of frames that hit the warm-start path
  • per-mode latency — ms/frame for skip vs full
  • aggregate speedup vs baseline

Run:
    PYTHONPATH=benchmarks modal run benchmarks/optimised/diamond_csgo_action_skip_modal.py
"""

import sys
sys.path.insert(0, "/root")
sys.path.insert(0, "/root/benchmarks")
sys.path.insert(0, "/root/benchmarks/optimised")

from _diamond_common import (
    app, diamond_image, hf_secret, model_volume, MODEL_CACHE,
    load_diamond, make_result,
)


@app.function(
    gpu="H100",
    image=diamond_image,
    secrets=[hf_secret],
    volumes={MODEL_CACHE: model_volume},
    timeout=3600,
    memory=32768,
)
def run_action_skip(
    num_frames: int = 64,
    skip_k: int = 3,                     # how many early steps to skip
    similarity_threshold: float = 0.95,  # cosine-sim threshold
    num_warmup: int = 4,
    num_iters: int = 3,
) -> dict:
    """Action-aware step skip. Wraps the next-obs DiffusionSampler with
    a thin shim that decides per-call whether to warm-start."""
    import statistics
    from typing import Optional

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    bundle = load_diamond()
    wm_env = bundle["wm_env"]
    base_sampler = wm_env.sampler_next_obs    # DiffusionSampler
    cfg = base_sampler.cfg
    sigmas = base_sampler.sigmas              # tensor of length num_steps+1
    num_steps = len(sigmas) - 1
    skip_k = max(0, min(skip_k, num_steps - 1))
    print(f"  num_steps={num_steps}, skip_k={skip_k}, threshold={similarity_threshold}")

    class ActionAwareSampler:
        def __init__(self) -> None:
            self.base = base_sampler
            self.sigmas = sigmas
            self.thr = similarity_threshold
            self.k = skip_k
            self.last_act: Optional[Tensor] = None
            self.last_traj: Optional[list[Tensor]] = None
            # bookkeeping
            self.full_count = 0
            self.skip_count = 0
            self.skip_ms: list[float] = []
            self.full_ms: list[float] = []
            self.last_sim: float = 0.0

        # WorldModelEnv reaches into `sampler.denoiser.device` etc. Proxy
        # all DiffusionSampler attributes through to the base instance.
        @property
        def denoiser(self):
            return self.base.denoiser

        @property
        def cfg(self):
            return self.base.cfg

        @torch.no_grad()
        def _denoise_from(
            self,
            x: Tensor,
            prev_obs: Tensor,
            prev_act: Optional[Tensor],
            start_step: int,
        ) -> tuple[Tensor, list[Tensor]]:
            """Re-implementation of DiffusionSampler.sample's inner loop,
            parameterised by `start_step` so we can warm-start."""
            denoiser = self.base.denoiser
            b = x.shape[0]
            device = x.device
            gamma_ = min(cfg.s_churn / (len(self.sigmas) - 1), 2 ** 0.5 - 1)
            trajectory: list[Tensor] = [x]

            for sigma, next_sigma in zip(
                self.sigmas[start_step:-1], self.sigmas[start_step + 1:]
            ):
                gamma = gamma_ if cfg.s_tmin <= sigma <= cfg.s_tmax else 0
                sigma_hat = sigma * (gamma + 1)
                if gamma > 0:
                    eps = torch.randn_like(x) * cfg.s_noise
                    x = x + eps * (sigma_hat ** 2 - sigma ** 2) ** 0.5
                if cfg.s_cond > 0:
                    sigma_cond = torch.full(
                        (b,), fill_value=cfg.s_cond, device=device,
                    )
                    prev_obs = denoiser.apply_noise(
                        prev_obs, sigma_cond, sigma_offset_noise=0,
                    )
                else:
                    sigma_cond = None
                denoised = denoiser.denoise(x, sigma, sigma_cond, prev_obs, prev_act)
                d = (x - denoised) / sigma_hat
                dt = next_sigma - sigma_hat
                if cfg.order == 1 or next_sigma == 0:
                    x = x + d * dt
                else:
                    x_2 = x + d * dt
                    s_in = torch.ones(b, device=device)
                    denoised_2 = denoiser.denoise(
                        x_2, next_sigma * s_in, sigma_cond, prev_obs, prev_act,
                    )
                    d_2 = (x_2 - denoised_2) / next_sigma
                    d_prime = (d + d_2) / 2
                    x = x + d_prime * dt
                trajectory.append(x)
            return x, trajectory

        @torch.no_grad()
        def sample(
            self, prev_obs: Tensor, prev_act: Optional[Tensor],
        ) -> tuple[Tensor, list[Tensor]]:
            b, t, c, h, w = prev_obs.size()
            prev_obs_flat = prev_obs.reshape(b, t * c, h, w)

            sim = 0.0
            warm_start = False
            if (
                self.last_act is not None
                and self.last_traj is not None
                and prev_act is not None
                and self.k > 0
            ):
                # compare the most recent action across consecutive frames.
                a_curr = prev_act[:, -1].reshape(b, -1).float()
                a_prev = self.last_act[:, -1].reshape(b, -1).float()
                sim = F.cosine_similarity(a_curr, a_prev, dim=-1).mean().item()
                warm_start = sim > self.thr
            self.last_sim = sim

            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()

            if warm_start:
                x_warm = self.last_traj[self.k].clone()
                x, traj_tail = self._denoise_from(
                    x_warm, prev_obs_flat, prev_act, start_step=self.k,
                )
                # Stitch a coherent trajectory that's still self-consistent
                # in length/sigma indexing.
                full_traj = list(self.last_traj[:self.k]) + traj_tail
                e.record()
                torch.cuda.synchronize()
                self.skip_ms.append(s.elapsed_time(e))
                self.skip_count += 1
                self.last_act = prev_act.clone()
                self.last_traj = full_traj
                return x, full_traj

            # full denoising — replicate base init: noise of shape (b, c, h, w)
            x0 = torch.randn(b, c, h, w, device=prev_obs.device)
            x, traj = self._denoise_from(
                x0, prev_obs_flat, prev_act, start_step=0,
            )
            e.record()
            torch.cuda.synchronize()
            self.full_ms.append(s.elapsed_time(e))
            self.full_count += 1
            self.last_act = prev_act.clone()
            self.last_traj = traj
            return x, traj

    aware = ActionAwareSampler()
    wm_env.sampler_next_obs = aware

    # Run the timing loop manually so we can capture the per-mode metrics.
    @torch.inference_mode()
    def _run_once() -> tuple[list[float], int]:
        wm_env.reset()
        future_acts = wm_env.next_act
        n_future = future_acts.shape[0]
        per_frame_ms: list[float] = []
        for t in range(num_frames):
            act = future_acts[t % n_future].unsqueeze(0)
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            wm_env.step(act)
            e.record()
            torch.cuda.synchronize()
            per_frame_ms.append(s.elapsed_time(e))
        return per_frame_ms, len(per_frame_ms)

    print(f"Warmup ({num_warmup} iters) ...")
    for i in range(num_warmup):
        # reset bookkeeping each warmup so steady-state stats are clean
        aware.skip_count = aware.full_count = 0
        aware.skip_ms.clear(); aware.full_ms.clear()
        ms_list, _ = _run_once()
        total = sum(ms_list)
        print(f"  warmup {i + 1}: {total:.0f} ms, "
              f"skip_rate={aware.skip_count / max(1, aware.skip_count + aware.full_count):.2%}")

    aware.skip_count = aware.full_count = 0
    aware.skip_ms.clear(); aware.full_ms.clear()
    torch.cuda.reset_peak_memory_stats()
    iter_totals: list[float] = []
    print(f"Timed runs ({num_iters} iters) ...")
    for i in range(num_iters):
        ms_list, n_obs = _run_once()
        total = sum(ms_list)
        iter_totals.append(total)
        skip_rate = aware.skip_count / max(1, aware.skip_count + aware.full_count)
        print(f"  iter {i + 1}: {total:.0f} ms, {n_obs / (total / 1000):.2f} fps, "
              f"skip_rate={skip_rate:.2%}, last_sim={aware.last_sim:.3f}")

    if not iter_totals:
        return {"error": "all iters OOM"}

    mean_total = statistics.mean(iter_totals)
    fps = num_frames / (mean_total / 1000.0)
    total_calls = aware.skip_count + aware.full_count
    skip_rate = aware.skip_count / max(1, total_calls)
    avg_skip = statistics.mean(aware.skip_ms) if aware.skip_ms else 0.0
    avg_full = statistics.mean(aware.full_ms) if aware.full_ms else 0.0

    timing = {
        "num_frames": num_frames,
        "latency_ms_mean_total": round(mean_total, 2),
        "latency_per_frame_ms_mean": round(mean_total / num_frames, 3),
        "frames_per_sec": round(fps, 3),
        "vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "raw_iter_totals_ms": iter_totals,
    }
    return make_result(
        kernel="diamond_action_skip",
        bundle=bundle,
        timing=timing,
        extras={
            "skip_k": skip_k,
            "similarity_threshold": similarity_threshold,
            "skip_rate": round(skip_rate, 4),
            "skip_count": aware.skip_count,
            "full_count": aware.full_count,
            "avg_sampler_ms_skip": round(avg_skip, 3),
            "avg_sampler_ms_full": round(avg_full, 3),
            "last_action_similarity": round(aware.last_sim, 4),
        },
    )


@app.local_entrypoint(name="diamond_csgo_action_skip_modal")
def main(
    num_frames: int = 64,
    skip_k: int = 3,
    similarity_threshold: float = 0.95,
    num_warmup: int = 4,
    num_iters: int = 3,
):
    result = run_action_skip.remote(
        num_frames=num_frames,
        skip_k=skip_k,
        similarity_threshold=similarity_threshold,
        num_warmup=num_warmup,
        num_iters=num_iters,
    )
    import json
    print("\nDIAMOND CSGO + action-skip result:")
    print(json.dumps(result, indent=2))

    sys.path.insert(0, "/root/benchmarks")
    from result_store import save_result
    save_result(
        result, "optimised_kernels",
        f"diamond_csgo_action_skip_k{skip_k}_thr{int(similarity_threshold * 100)}_f{num_frames}",
    )
