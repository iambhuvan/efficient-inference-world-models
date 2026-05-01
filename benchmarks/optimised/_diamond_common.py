"""
Shared DIAMOND-CSGO scaffolding for optimisation scripts.

Centralises:
  - The Modal image (extends image_base + DIAMOND deps)
  - Model loading from HF snapshot (Agent + WorldModelEnv) with the
    rew_end_model stub (training-only head whose torcheval/torchaudio
    chain breaks on cu124).
  - The per-frame timing loop driven by spawn-provided action sequences.
  - Result-dict construction.

Each individual optimisation script imports these helpers, applies its
specific tweak (torch.compile / BF16 autocast / step-count override /
action-aware sampler), and calls `time_world_model(...)`.

Why one shared module: DIAMOND model loading is ~80 lines of boilerplate
(snapshot, Hydra compose, agent build, WorldModelEnv build). Repeating it
in every optimisation script makes diffs noisy and bug-prone.
"""

from __future__ import annotations

import os
import sys
import statistics
import types as _types
from pathlib import Path
from typing import Any, Callable

# Make the benchmarks/ root importable when this module loads inside a
# Modal container (where /root/benchmarks is mounted).
sys.path.insert(0, "/root")
sys.path.insert(0, "/root/benchmarks")

from modal_common import (  # noqa: F401
    app, image_base, _add_common_layers,
    hf_secret, model_volume, MODEL_CACHE,
)

DIAMOND_REPO = "eloialonso/diamond"
DIAMOND_GH = "https://github.com/eloialonso/diamond.git"

# Shared image — same recipe as the baseline. Optimisation scripts that
# need extra deps can extend further before re-applying _add_common_layers.
diamond_image = _add_common_layers(
    image_base
    .pip_install(
        "gymnasium==0.29.1",
        "ale-py==0.9.0",
        "h5py==3.11.0",
        "opencv-python-headless==4.10.0.84",
        "pillow==10.3.0",
        "torcheval==0.0.7",
        "tqdm==4.66.4",
        "hydra-core==1.3",
        "pygame==2.5.2",
        "wandb>=0.18.0,<0.20.0",
    )
    .run_commands(
        f"git clone --depth=1 -b csgo {DIAMOND_GH} /root/diamond "
        "|| echo 'diamond clone failed'"
    )
    .env({
        "PYTHONPATH": "/root/diamond/src:/root/diamond:/root/benchmarks:/root",
        "SDL_VIDEODRIVER": "dummy",
        "DISPLAY": "",
    })
)


def stub_rew_end_model() -> None:
    """Inject a stub for `models.rew_end_model` to bypass torcheval's
    torchaudio cu13 import. Must run BEFORE importing `agent` or `envs`."""
    if "models.rew_end_model" in sys.modules:
        return
    stub = _types.ModuleType("models.rew_end_model")
    class _Stub:  # noqa: D401
        pass
    stub.RewEndModel = _Stub
    stub.RewEndModelConfig = _Stub
    sys.modules["models.rew_end_model"] = stub


def load_diamond(
    *,
    num_steps_denoising: int | None = None,
    num_steps_upsampling: int | None = None,
):
    """Snapshot weights, build Agent + WorldModelEnv. Returns a bundle."""
    import torch
    from huggingface_hub import snapshot_download
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import DictConfig, OmegaConf

    OmegaConf.register_new_resolver("eval", eval, replace=True)

    sys.path.insert(0, "/root/diamond/src")
    sys.path.insert(0, "/root/diamond")
    stub_rew_end_model()

    hf_token = os.environ.get("HF_TOKEN", "").strip() or None
    print(f"Downloading {DIAMOND_REPO} (csgo only) ...")
    local_dir = snapshot_download(
        DIAMOND_REPO, cache_dir=MODEL_CACHE, token=hf_token,
        allow_patterns=["csgo/*", "README.md"],
    )
    print(f"  weights at: {local_dir}")

    path_ckpt = os.path.join(local_dir, "csgo/model/csgo.pt")
    spawn_dir = Path(os.path.join(local_dir, "csgo/spawn"))
    agent_yaml = os.path.join(local_dir, "csgo/config/agent/csgo.yaml")
    env_yaml = os.path.join(local_dir, "csgo/config/env/csgo.yaml")

    with initialize_config_dir(version_base="1.3", config_dir="/root/diamond/config"):
        cfg: DictConfig = compose(config_name="trainer")
    cfg.agent = OmegaConf.load(agent_yaml)
    cfg.env = OmegaConf.load(env_yaml)
    cfg.agent.rew_end_model = None              # bypass training head

    if num_steps_denoising is not None:
        cfg.world_model_env.diffusion_sampler_next_obs.num_steps_denoising = num_steps_denoising
    if num_steps_upsampling is not None:
        if cfg.world_model_env.get("diffusion_sampler_upsampling", None) is not None:
            cfg.world_model_env.diffusion_sampler_upsampling.num_steps_denoising = (
                num_steps_upsampling
            )

    num_actions = int(cfg.env.num_actions)
    print(f"  num_actions={num_actions}, "
          f"num_steps_denoising={cfg.world_model_env.diffusion_sampler_next_obs.num_steps_denoising}")

    from agent import Agent                                # type: ignore
    from envs import WorldModelEnv                         # type: ignore

    device = torch.device("cuda")
    agent = Agent(instantiate(cfg.agent, num_actions=num_actions)).to(device).eval()
    agent.load(path_ckpt)
    n_params = sum(p.numel() for p in agent.parameters())

    sl = cfg.agent.denoiser.inner_model.num_steps_conditioning
    if agent.upsampler is not None:
        sl = max(sl, cfg.agent.upsampler.inner_model.num_steps_conditioning)
    wm_env_cfg = instantiate(cfg.world_model_env, num_batches_to_preload=1)
    wm_env = WorldModelEnv(
        agent.denoiser, agent.upsampler, agent.rew_end_model,
        spawn_dir, 1, sl, wm_env_cfg, return_denoising_trajectory=False,
    )
    return {
        "agent": agent,
        "wm_env": wm_env,
        "num_actions": num_actions,
        "context_length": int(sl),
        "n_params": n_params,
        "device": device,
        "cfg": cfg,
    }


def time_world_model(
    wm_env,
    num_frames: int,
    num_warmup: int,
    num_iters: int,
) -> dict[str, Any]:
    """Run the standard timing loop. Same logic as the baseline."""
    import torch

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
        ms_list, _ = _run_once()
        total = sum(ms_list)
        print(f"  warmup {i + 1}: {total:.0f} ms total, "
              f"{total / len(ms_list):.2f} ms/frame")

    torch.cuda.reset_peak_memory_stats()
    iter_totals: list[float] = []
    all_per_frame: list[list[float]] = []
    print(f"Timed runs ({num_iters} iters) ...")
    for i in range(num_iters):
        try:
            ms_list, n_obs = _run_once()
            total = sum(ms_list)
            iter_totals.append(total)
            all_per_frame.append(ms_list)
            fps = n_obs / (total / 1000.0)
            print(f"  iter {i + 1}: {total:.0f} ms ({n_obs} frames, "
                  f"{fps:.2f} fps, {total / n_obs:.2f} ms/frame)")
        except torch.cuda.OutOfMemoryError as oom:
            print(f"  iter {i + 1}: OOM — {oom}")
            torch.cuda.empty_cache()

    if not iter_totals:
        return {"error": "All iterations OOM"}

    mean_total = statistics.mean(iter_totals)
    fps = num_frames / (mean_total / 1000.0)
    flat = [v for it in all_per_frame for v in it]
    p50 = statistics.median(flat)
    p95 = sorted(flat)[int(0.95 * len(flat)) - 1] if flat else 0.0

    return {
        "num_frames": num_frames,
        "latency_ms_mean_total": round(mean_total, 2),
        "latency_ms_std_total": round(
            statistics.stdev(iter_totals) if len(iter_totals) > 1 else 0.0, 2
        ),
        "latency_per_frame_ms_mean": round(mean_total / num_frames, 3),
        "latency_per_frame_ms_p50": round(p50, 3),
        "latency_per_frame_ms_p95": round(p95, 3),
        "frames_per_sec": round(fps, 3),
        "vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "raw_iter_totals_ms": iter_totals,
    }


def make_result(
    kernel: str,
    bundle: dict,
    timing: dict,
    extras: dict[str, Any] | None = None,
) -> dict:
    """Assemble the standard result dict for save_result."""
    res = {
        "model": DIAMOND_REPO,
        "variant": "csgo",
        "kernel": kernel,
        "n_params_M": round(bundle["n_params"] / 1e6, 2),
        "num_actions": bundle["num_actions"],
        "context_length": bundle["context_length"],
        "gpu": "H100",
        **timing,
    }
    if extras:
        res.update(extras)
    return res
