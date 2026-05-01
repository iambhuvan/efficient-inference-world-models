"""
Helios (BestWishYsh/Helios-*) model backend for WorldServe.

Helios is a 14B autoregressive diffusion video model finetuned from
Wan2.1-T2V-14B-Diffusers. Native support for T2V, I2V, and V2V tasks.

Paper: "Helios — Real Real-Time Long Video Generation Model"
       (Yuan et al., arXiv 2603.04379, March 2026).

Authors explicitly report **19.5 FPS on a single H100 GPU** for the
Helios-Distilled checkpoint at 832×480 × 81 frames, *without* using
KV-cache compression, sparse / linear attention, or quantization — i.e.
the entire WorldServe optimization category is uncharted on this model.

Three checkpoints are supported:

  * ``base``       — BestWishYsh/Helios-Base (HeliosPipeline,
                     full 50-step diffusion, ~2-5 FPS H100)
  * ``mid``        — BestWishYsh/Helios-Mid (HeliosPyramidPipeline,
                     intermediate pyramid sampling)
  * ``distilled``  — BestWishYsh/Helios-Distilled
                     (HeliosPyramidPipeline, DMD-distilled, 19.5 FPS H100)

All three share the Wan2.1-T2V-14B backbone (40 transformer blocks,
hidden=5120, 40 heads × head_dim=128) — kernels developed against any
of them transfer to the others without changes.

Loading uses ``DiffusionPipeline.from_pretrained(..., trust_remote_code=True)``
which auto-resolves the pipeline class from the checkpoint's
``model_index.json``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch


# ── Checkpoint registry ──────────────────────────────────────────────────
# Centralizes metadata so optimization scripts can pick the right variant
# without hardcoding HF repo IDs in every benchmark file.

CheckpointName = Literal["base", "mid", "distilled"]

CHECKPOINTS: dict[str, dict[str, Any]] = {
    "base": {
        "hf_repo": "BestWishYsh/Helios-Base",
        "pipeline_class": "HeliosPipeline",
        "default_num_steps": 50,         # full undistilled diffusion
        "paper_fps_h100": None,          # not the paper's headline variant
        "notes": (
            "Undistilled full-quality variant. Slower per-generation "
            "(~25-40 s/gen) but largest optimization headroom — every "
            "per-step kernel saving compounds 50×."
        ),
    },
    "mid": {
        "hf_repo": "BestWishYsh/Helios-Mid",
        "pipeline_class": "HeliosPyramidPipeline",
        "default_num_steps": None,       # pipeline default
        "paper_fps_h100": None,
        "notes": "Intermediate pyramid pipeline; less downloaded, less tested.",
    },
    "distilled": {
        "hf_repo": "BestWishYsh/Helios-Distilled",
        "pipeline_class": "HeliosPyramidPipeline",
        "default_num_steps": None,       # use pipeline's distilled schedule
        "paper_fps_h100": 19.5,          # the published headline number
        "notes": (
            "DMD-distilled few-step variant. Published 19.5 FPS H100 single-GPU. "
            "Authors explicitly state no KV-cache / sparse-attn / quantization "
            "in baseline — every WorldServe optimization is uncharted here."
        ),
    },
}


# ── Wan2.1-T2V-14B canonical generation config ───────────────────────────
DEFAULT_WIDTH = 832
DEFAULT_HEIGHT = 480
DEFAULT_NUM_FRAMES = 81          # ~5 s at 16 fps native
DEFAULT_GUIDANCE = 5.0
DEFAULT_PROMPT = (
    "A camera slowly pans through a sunlit forest clearing, golden light "
    "filtering through tall trees, soft wind moving the leaves."
)


# ── Loader ───────────────────────────────────────────────────────────────

def resolve_checkpoint(variant_or_dir: str) -> tuple[str, dict[str, Any] | None]:
    """
    Return (checkpoint_dir, metadata) for a CheckpointName or a raw path.

    Use this when you have either a registered variant name (``"base"`` etc.)
    or an already-downloaded local directory path.
    """
    if variant_or_dir in CHECKPOINTS:
        from huggingface_hub import snapshot_download

        meta = CHECKPOINTS[variant_or_dir]
        local_dir = snapshot_download(meta["hf_repo"])
        return local_dir, meta

    # Raw directory path — no metadata available.
    if Path(variant_or_dir).exists():
        return variant_or_dir, None

    raise ValueError(
        f"Helios variant_or_dir '{variant_or_dir}' is not a registered "
        f"variant ({list(CHECKPOINTS)}) and not a local directory."
    )


def load_model(
    variant_or_dir: str = "base",
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    cache_dir: str | None = None,
) -> dict[str, Any]:
    """
    Load any Helios checkpoint by variant name or local directory.

    Parameters
    ----------
    variant_or_dir : ``"base"`` / ``"mid"`` / ``"distilled"`` (registered names)
                     OR a local diffusers-format checkpoint directory path.
    device         : ``"cuda"`` (Helios is single-H100 by design).
    dtype          : bf16 by default (matches Wan2.1 training precision).
    cache_dir      : optional override for huggingface_hub cache.

    Returns
    -------
    dict with keys:
        pipe       — the loaded HeliosPipeline / HeliosPyramidPipeline
        variant    — checkpoint name if known, else ``None``
        meta       — entry from CHECKPOINTS, or ``None`` for raw paths
        device     — torch device
        dtype      — model dtype
        n_params_B — total parameters across pipeline components, in billions
    """
    from diffusers import DiffusionPipeline

    if variant_or_dir in CHECKPOINTS:
        from huggingface_hub import snapshot_download

        meta = CHECKPOINTS[variant_or_dir]
        print(f"[helios] Pulling {meta['hf_repo']} via snapshot_download")
        ckpt_dir = snapshot_download(meta["hf_repo"], cache_dir=cache_dir)
        variant = variant_or_dir
    else:
        ckpt_dir = variant_or_dir
        meta = None
        variant = None

    ckpt_path = Path(ckpt_dir)
    if not (ckpt_path / "model_index.json").exists():
        raise FileNotFoundError(
            f"Helios checkpoint missing model_index.json: {ckpt_path}. "
            f"Run snapshot_download('BestWishYsh/Helios-<Variant>') first."
        )

    print(
        f"[helios] Loading {meta['pipeline_class'] if meta else 'HeliosPipeline'} "
        f"from {ckpt_path} (trust_remote_code=True)"
    )
    pipe = DiffusionPipeline.from_pretrained(
        str(ckpt_path),
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    n_params = sum(
        p.numel()
        for module in pipe.components.values()
        if isinstance(module, torch.nn.Module)
        for p in module.parameters()
    )
    n_params_B = n_params / 1e9
    print(f"[helios] Total params: {n_params_B:.2f}B")

    return {
        "pipe": pipe,
        "variant": variant,
        "meta": meta,
        "device": device,
        "dtype": dtype,
        "n_params_B": n_params_B,
    }


# ── Component accessors (used by optimization scripts) ───────────────────

def get_transformer(model: dict[str, Any]) -> torch.nn.Module:
    """Return the DiT transformer module (target of compile, INT4-wo, LayerSkip)."""
    pipe = model["pipe"]
    if not hasattr(pipe, "transformer"):
        raise AttributeError(
            "HeliosPipeline has no .transformer attribute — "
            "check pipeline structure with `print(pipe.components.keys())`."
        )
    return pipe.transformer


def get_vae(model: dict[str, Any]) -> torch.nn.Module:
    """Return the VAE module (target of separate quant treatment)."""
    pipe = model["pipe"]
    if not hasattr(pipe, "vae"):
        raise AttributeError("HeliosPipeline has no .vae attribute.")
    return pipe.vae


def get_scheduler(model: dict[str, Any]) -> Any:
    """Return the diffusion scheduler (e.g. for step-caching / DPM++ swap)."""
    pipe = model["pipe"]
    if not hasattr(pipe, "scheduler"):
        raise AttributeError("HeliosPipeline has no .scheduler attribute.")
    return pipe.scheduler


def get_attention_modules(model: dict[str, Any]) -> dict[str, torch.nn.Module]:
    """
    Return all attention sub-modules of the transformer, keyed by dotted name.

    Used by the SageAttn2 / FA3 swap and the QVG 2-bit-KV layer to enumerate
    every spot where attention is computed.
    """
    transformer = get_transformer(model)
    out: dict[str, torch.nn.Module] = {}
    for name, sub in transformer.named_modules():
        cls_name = type(sub).__name__.lower()
        if "attention" in cls_name or "attn" in cls_name:
            out[name] = sub
    return out


# ── Generation ───────────────────────────────────────────────────────────

@torch.inference_mode()
def generate(
    model: dict[str, Any],
    prompt: str = DEFAULT_PROMPT,
    num_frames: int = DEFAULT_NUM_FRAMES,
    num_steps: int | None = None,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    guidance_scale: float = DEFAULT_GUIDANCE,
    seed: int = 42,
    image: Any | None = None,
    config: dict | None = None,
    optimizations: Any | None = None,
) -> dict[str, Any]:
    """
    Run a single Helios generation.

    ``num_steps=None`` lets the pipeline use its built-in default schedule.
    For the Distilled checkpoint, this is the configuration that produces
    the paper's reported 19.5 FPS — *do not* override unless you know the
    distilled schedule. For the Base checkpoint, this defaults to 50 via
    the registered metadata.

    ``image`` triggers I2V mode when supported by the pipeline.

    ``config`` and ``optimizations`` are forwarded to support a uniform
    OptimizationStack interface across model wrappers — Helios's own hooks
    operate at the diffusers component level (transformer / vae / scheduler)
    rather than per-step DDIM, so most optimization injection happens via
    the dedicated optimization scripts (compile, INT4-wo, sage-attn, etc.)
    before this function is called.
    """
    pipe = model["pipe"]
    device = model["device"]
    meta = model.get("meta")

    # Resolve effective num_steps from arg → variant default → pipeline default.
    if num_steps is None and meta is not None:
        num_steps = meta.get("default_num_steps")  # may still be None

    gen = torch.Generator(device=device).manual_seed(seed)

    call_kwargs: dict[str, Any] = {
        "prompt": prompt,
        "num_frames": num_frames,
        "width": width,
        "height": height,
        "guidance_scale": guidance_scale,
        "generator": gen,
        "return_dict": True,
    }
    if num_steps is not None:
        call_kwargs["num_inference_steps"] = num_steps
    if image is not None:
        call_kwargs["image"] = image  # I2V mode

    out = pipe(**call_kwargs)

    frames_attr = getattr(out, "frames", None) or getattr(out, "videos", None)
    if frames_attr is None:
        raise RuntimeError(
            f"HeliosPipeline output missing .frames / .videos: type={type(out)}"
        )

    frames = _to_tchw_tensor(frames_attr)

    return {
        "frames": frames,
        "variant": model.get("variant"),
        "num_frames": num_frames,
        "num_steps": num_steps,             # None = pipeline default
        "width": width,
        "height": height,
        "guidance_scale": guidance_scale,
        "seed": seed,
    }


# ── Output normalization ─────────────────────────────────────────────────

def _to_tchw_tensor(frames_obj: Any) -> torch.Tensor:
    """
    Normalize HeliosPipeline output to a (T, C, H, W) float tensor in [0, 1].

    Accepted inputs:
        - torch.Tensor of shape (T, C, H, W) or (B, T, C, H, W) — passes through.
        - torch.Tensor of shape (T, H, W, C) — permutes to (T, C, H, W).
        - list[PIL.Image] — stacks into (T, C, H, W).
        - list[list[PIL.Image]] (batched) — takes batch[0].
    """
    import numpy as np

    if isinstance(frames_obj, torch.Tensor):
        t = frames_obj
        if t.ndim == 5:
            t = t[0]
        if t.ndim == 4 and t.shape[-1] == 3:  # T H W C
            t = t.permute(0, 3, 1, 2)
        return t.float().clamp(0.0, 1.0).cpu()

    if isinstance(frames_obj, list):
        if frames_obj and isinstance(frames_obj[0], list):
            frames_obj = frames_obj[0]
        arrs = [np.asarray(im) for im in frames_obj]  # each (H, W, 3) uint8
        stacked = np.stack(arrs, axis=0)              # (T, H, W, 3)
        t = torch.from_numpy(stacked).permute(0, 3, 1, 2).float() / 255.0
        return t.clamp(0.0, 1.0)

    raise TypeError(f"Unrecognized frames type: {type(frames_obj)}")


# ── Public API ───────────────────────────────────────────────────────────

__all__ = [
    "CHECKPOINTS",
    "DEFAULT_GUIDANCE",
    "DEFAULT_HEIGHT",
    "DEFAULT_NUM_FRAMES",
    "DEFAULT_PROMPT",
    "DEFAULT_WIDTH",
    "generate",
    "get_attention_modules",
    "get_scheduler",
    "get_transformer",
    "get_vae",
    "load_model",
    "resolve_checkpoint",
]
