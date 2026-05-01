"""
Modal H100 benchmark: DIAMOND CSGO world model baseline (no optimizations).

Model    : eloialonso/diamond  (csgo/model/csgo.pt — single 1.5 GB checkpoint)
Pipeline : DIAMOND `Agent` + `WorldModelEnv` (DDIM, action-conditioned)
Task     : Action-conditioned next-frame prediction
Source   : github.com/eloialonso/diamond  (branch=`csgo`, NOT main)
License  : Apache-2.0
Paper    : arxiv.org/abs/2405.12399

Why:
  Cleanest open *truly action-conditioned* world model. Small (~50–100M
  denoiser) → fast iteration on action-aware optimisations (KV reuse on
  repeated actions, action-routed heads, etc.).  CSGO frames are
  visually meaningful, unlike Atari — keeps FVD-style metrics useful.

Loading strategy (verified against upstream src/play.py on the csgo branch):
  1. Clone the `csgo` branch (CSGO support is NOT on main).
  2. snapshot_download(eloialonso/diamond, allow_patterns="csgo/*") — pulls
     csgo/model/csgo.pt + config YAMLs + spawn states.
  3. Use Hydra's `initialize_config_dir` against /root/diamond/config to
     compose the trainer config, then override agent + env from the HF
     snapshot YAMLs (matches play.py's `prepare_play_mode`).
  4. Build `Agent` via `instantiate(cfg.agent, num_actions=...)`, load
     csgo.pt, build `WorldModelEnv`.
  5. Loop wm_env.step(random_action) — that's the per-frame deployed cost
     (predict_next_obs + upsample + rew_end_model).

Run:
    PYTHONPATH=benchmarks modal run benchmarks/baseline/diamond_csgo_baseline_modal.py
"""

import sys
sys.path.insert(0, "/root")
sys.path.insert(0, "/root/benchmarks")

import modal
from modal_common import (
    app, image_base, _add_common_layers,
    hf_secret, model_volume, MODEL_CACHE,
)

DIAMOND_REPO = "eloialonso/diamond"
DIAMOND_GH = "https://github.com/eloialonso/diamond.git"

# DIAMOND deps (verified against csgo-branch requirements.txt). pygame is
# imported transitively by `envs/__init__.py`'s `make_atari_env` even
# though we never start a window; SDL_VIDEODRIVER=dummy keeps it headless.
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
        # Required by src/utils.py at import-time, even though we never log.
        # 0.17.0 (DIAMOND's pin) uses removed np.float_; bump for numpy>=2.
        "wandb>=0.18.0,<0.20.0",
    )
    .run_commands(
        # Clone the csgo branch (main does not have csgo support).
        f"git clone --depth=1 -b csgo {DIAMOND_GH} /root/diamond "
        "|| echo 'diamond clone failed'"
    )
    .env({
        "PYTHONPATH": "/root/diamond/src:/root/diamond:/root/benchmarks:/root",
        "SDL_VIDEODRIVER": "dummy",     # keep pygame headless
        "DISPLAY": "",
    })
)


@app.function(
    gpu="H100",
    image=diamond_image,
    secrets=[hf_secret],
    volumes={MODEL_CACHE: model_volume},
    timeout=3600,
    memory=32768,
)
def run_diamond_csgo_baseline(
    num_frames: int = 64,
    seed: int = 42,
    num_warmup: int = 4,
    num_iters: int = 3,
) -> dict:
    """DIAMOND CSGO baseline: action-conditioned per-frame generation FPS."""
    import json
    import os
    import statistics
    from pathlib import Path

    import torch
    from huggingface_hub import snapshot_download
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import DictConfig, OmegaConf

    hf_token = os.environ.get("HF_TOKEN", "").strip() or None

    # play.py registers an "eval" resolver before composing — required
    # because some YAMLs use ${eval:...} interpolations. `replace=True`
    # makes this idempotent across iterations / re-runs in the same proc.
    OmegaConf.register_new_resolver("eval", eval, replace=True)

    # ---------------------------------------------------------------- weights
    print(f"Downloading {DIAMOND_REPO} (csgo only) ...")
    local_dir = snapshot_download(
        DIAMOND_REPO,
        cache_dir=MODEL_CACHE,
        token=hf_token,
        allow_patterns=["csgo/*", "README.md"],
    )
    print(f"  weights at: {local_dir}")

    path_ckpt = os.path.join(local_dir, "csgo/model/csgo.pt")
    spawn_dir = os.path.join(local_dir, "csgo/spawn")
    agent_yaml = os.path.join(local_dir, "csgo/config/agent/csgo.yaml")
    env_yaml = os.path.join(local_dir, "csgo/config/env/csgo.yaml")
    for p in (path_ckpt, spawn_dir, agent_yaml, env_yaml):
        if not os.path.exists(p):
            return {"error": f"missing {p}", "model": DIAMOND_REPO}

    # ---------------------------------------------------------------- imports
    sys.path.insert(0, "/root/diamond/src")
    sys.path.insert(0, "/root/diamond")

    # `from envs import WorldModelEnv` transitively pulls
    # `models.rew_end_model` → `torcheval` → `torchaudio`, and the latter
    # gets upgraded to a CUDA-13 build during pip resolution that fails on
    # our cu124 image. The reward/end head is training-only — we never use
    # it for FPS benchmarking. Stub the module so the import succeeds, and
    # null out cfg.agent.rew_end_model so Agent never instantiates it.
    import types as _types
    _stub = _types.ModuleType("models.rew_end_model")
    class _StubRewEndModel:  # noqa: D401 - placeholder
        pass
    class _StubRewEndModelConfig:
        pass
    _stub.RewEndModel = _StubRewEndModel
    _stub.RewEndModelConfig = _StubRewEndModelConfig
    sys.modules["models.rew_end_model"] = _stub

    from agent import Agent                                # type: ignore
    from envs import WorldModelEnv                         # type: ignore

    # ---------------------------------------------------------------- config
    print("Composing Hydra trainer config ...")
    with initialize_config_dir(version_base="1.3", config_dir="/root/diamond/config"):
        cfg: DictConfig = compose(config_name="trainer")
    cfg.agent = OmegaConf.load(agent_yaml)
    cfg.env = OmegaConf.load(env_yaml)
    # Skip rew_end_model: training-only head whose import chain breaks on
    # cu124 (see the stub above).
    cfg.agent.rew_end_model = None
    assert cfg.env.train.id == "csgo", f"unexpected env id: {cfg.env.train.id}"
    num_actions = int(cfg.env.num_actions)
    print(f"  cfg.env.num_actions = {num_actions}")

    # ---------------------------------------------------------------- model
    device = torch.device("cuda")
    print(f"Building Agent on {device} ...")
    agent = Agent(instantiate(cfg.agent, num_actions=num_actions)).to(device).eval()
    agent.load(path_ckpt)
    n_params = sum(p.numel() for p in agent.parameters())
    print(f"  Agent loaded: {n_params / 1e6:.2f}M params total "
          f"(denoiser + upsampler + rew_end_model)")

    # WorldModelEnv signature mirrors play.py exactly.
    sl = cfg.agent.denoiser.inner_model.num_steps_conditioning
    if agent.upsampler is not None:
        sl = max(sl, cfg.agent.upsampler.inner_model.num_steps_conditioning)
    wm_env_cfg = instantiate(cfg.world_model_env, num_batches_to_preload=1)
    wm_env = WorldModelEnv(
        agent.denoiser,
        agent.upsampler,
        agent.rew_end_model,
        Path(spawn_dir),                # `make_generator_init` calls .iterdir()
        1,                              # num_envs (CSGO requires =1)
        sl,                             # seq_length
        wm_env_cfg,
        return_denoising_trajectory=False,
    )

    # ---------------------------------------------------------------- timing
    # CSGO actions are mixed discrete+continuous (51-dim per step). Spawn
    # data provides ~200 future actions of shape [200, 51]; slice each
    # step to shape [1, 51] to match act_buffer[:, -1]. Cycle through
    # actions so we exercise variety, but content doesn't change timing.
    @torch.inference_mode()
    def _run_once() -> tuple[list[float], int]:
        wm_env.reset()
        future_acts = wm_env.next_act           # shape [N_future, act_dim]
        n_future = future_acts.shape[0]
        per_frame_ms: list[float] = []
        for t in range(num_frames):
            act = future_acts[t % n_future].unsqueeze(0)   # [1, act_dim]
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
            ms_list, n_observed = _run_once()
            total = sum(ms_list)
            iter_totals.append(total)
            all_per_frame.append(ms_list)
            fps = n_observed / (total / 1000.0)
            print(
                f"  iter {i + 1}: {total:.0f} ms "
                f"({n_observed} frames, {fps:.2f} fps, {total / n_observed:.2f} ms/frame)"
            )
        except torch.cuda.OutOfMemoryError as oom:
            print(f"  iter {i + 1}: OOM — {oom}")
            torch.cuda.empty_cache()

    if not iter_totals:
        return {"error": "All iterations OOM", "model": DIAMOND_REPO}

    mean_total = statistics.mean(iter_totals)
    fps = num_frames / (mean_total / 1000.0)
    flat = [v for it in all_per_frame for v in it]
    p50 = statistics.median(flat)
    p95 = sorted(flat)[int(0.95 * len(flat)) - 1] if flat else 0.0

    results = {
        "model": DIAMOND_REPO,
        "variant": "csgo",
        "kernel": "baseline_diamond_world_model",
        "num_frames": num_frames,
        "num_actions": num_actions,
        "context_length": int(sl),
        "n_params_M": round(n_params / 1e6, 2),
        "latency_ms_mean_total": round(mean_total, 2),
        "latency_ms_std_total": round(
            statistics.stdev(iter_totals) if len(iter_totals) > 1 else 0.0, 2
        ),
        "latency_per_frame_ms_mean": round(mean_total / num_frames, 3),
        "latency_per_frame_ms_p50": round(p50, 3),
        "latency_per_frame_ms_p95": round(p95, 3),
        "frames_per_sec": round(fps, 3),
        "vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "gpu": "H100",
        "raw_iter_totals_ms": iter_totals,
    }
    print("\n" + json.dumps(results, indent=2))
    return results


@app.local_entrypoint(name="diamond_csgo_baseline_modal")
def main(
    num_frames: int = 64,
    num_warmup: int = 4,
    num_iters: int = 3,
):
    result = run_diamond_csgo_baseline.remote(
        num_frames=num_frames,
        num_warmup=num_warmup,
        num_iters=num_iters,
    )

    print("\nDIAMOND CSGO baseline result:")
    import json
    print(json.dumps(result, indent=2))

    sys.path.insert(0, "/root/benchmarks")
    from result_store import save_result
    save_result(
        result, "baseline",
        f"diamond_csgo_baseline_f{num_frames}",
    )
