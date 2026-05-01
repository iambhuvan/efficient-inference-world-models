"""
Base model wrappers for Open-Oasis and Matrix-Game 2.0.

Provides a uniform interface for generation, profiling, attention-weight
extraction, KV-cache inspection, and optimization injection.
"""

from __future__ import annotations

import importlib
import sys
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from worldserve import PROJECT_ROOT
from worldserve.utils.profiler import WorldServeProfiler

# Expected clone locations (relative to project root)
OASIS_REPO_DIR = PROJECT_ROOT / "open-oasis"
MATRIX_GAME_REPO_DIR = PROJECT_ROOT / "Matrix-Game"


# ── Abstract Base ─────────────────────────────────────────────────────────

class BaseWorldModel(ABC):
    """
    Abstract interface for video world model wrappers.

    Subclasses handle model-specific loading, generation logic, and
    architecture details while exposing a common API for profiling
    and optimization.
    """

    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.float16):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype
        self.profiler: Optional[WorldServeProfiler] = None
        self._optimizations: Dict[str, Any] = {}
        self._attention_hooks: List[torch.utils.hooks.RemovableHook] = []
        self._captured_attention_weights: Dict[str, torch.Tensor] = {}

    # ── abstract API ──────────────────────────────────────────────────
    @abstractmethod
    def load_checkpoint(self, **kwargs: Any) -> None:
        """Load model weights and prepare for inference."""
        ...

    @abstractmethod
    def generate_frame(
        self,
        actions: torch.Tensor,
        context_frames: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate the next frame(s) given an action tensor.

        Parameters
        ----------
        actions : action conditioning tensor (model-specific shape).
        context_frames : optional previously-generated frames for context.

        Returns
        -------
        Generated frame(s) as (B, C, H, W) or (B, T, C, H, W).
        """
        ...

    @abstractmethod
    def get_kv_cache(self) -> Optional[Dict[str, torch.Tensor]]:
        """Return current KV-cache contents, or None if not applicable."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear all internal state (KV caches, buffers, etc.)."""
        ...

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata (name, params, architecture details)."""
        ...

    # ── profiling ─────────────────────────────────────────────────────
    def attach_profiler(self, profiler: WorldServeProfiler) -> None:
        """Attach a profiler instance for per-layer timing."""
        self.profiler = profiler

    def profile_forward(
        self,
        actions: torch.Tensor,
        context_frames: Optional[torch.Tensor] = None,
        warmup_steps: int = 3,
        measure_steps: int = 10,
    ) -> Dict[str, Any]:
        """
        Run generation with profiling, including warmup.

        Returns dict with ``profiler_summary`` and ``frames``.
        """
        profiler = self.profiler or WorldServeProfiler()
        self.attach_profiler(profiler)

        # Warmup (not profiled)
        for _ in range(warmup_steps):
            with torch.no_grad():
                _ = self.generate_frame(actions, context_frames)
            self.reset()

        profiler.reset()

        # Measured runs
        all_frames = []
        profiler.snapshot_memory("before_generation")
        for step in range(measure_steps):
            profiler.start_region(f"step_{step}")
            with torch.no_grad():
                frame = self.generate_frame(actions, context_frames)
            profiler.end_region(f"step_{step}")
            all_frames.append(frame)

            # Track KV-cache size each step
            kv = self.get_kv_cache()
            if kv is not None:
                total_bytes = sum(v.nelement() * v.element_size() for v in kv.values())
                profiler.record_kv_cache(
                    num_entries=len(kv),
                    size_mb=total_bytes / (1024 ** 2),
                    label=f"step_{step}",
                )

        profiler.snapshot_memory("after_generation")
        summary = profiler.get_summary(total_frames=measure_steps)

        return {
            "profiler_summary": summary,
            "frames": all_frames,
        }

    # ── optimization injection ────────────────────────────────────────
    def inject_optimization(self, name: str, optimization: Any) -> None:
        """
        Register a named optimization to be applied during generation.

        The optimization object should implement an ``apply(model)`` method
        or be a callable that takes the model wrapper as argument.
        """
        self._optimizations[name] = optimization
        if hasattr(optimization, "apply"):
            optimization.apply(self)
        elif callable(optimization):
            optimization(self)
        else:
            warnings.warn(
                f"Optimization '{name}' is not callable and has no .apply() method. "
                "Stored but not applied."
            )

    def list_optimizations(self) -> List[str]:
        """Return names of all registered optimizations."""
        return list(self._optimizations.keys())

    # ── attention weight hooks ────────────────────────────────────────
    def _register_attention_hooks(self, attention_modules: Dict[str, nn.Module]) -> None:
        """
        Register forward hooks to capture attention weights from named modules.
        """
        self.clear_attention_hooks()

        def _make_hook(name: str):
            def hook_fn(module, input, output):
                # Many attention implementations return (output, weights) or
                # store weights as module.attn_weights
                if isinstance(output, tuple) and len(output) >= 2:
                    self._captured_attention_weights[name] = output[1].detach()
                elif hasattr(module, "attn_weights"):
                    self._captured_attention_weights[name] = module.attn_weights.detach()
            return hook_fn

        for name, module in attention_modules.items():
            handle = module.register_forward_hook(_make_hook(name))
            self._attention_hooks.append(handle)

    def get_captured_attention_weights(self) -> Dict[str, torch.Tensor]:
        """Return attention weights captured during the last forward pass."""
        return dict(self._captured_attention_weights)

    def clear_attention_hooks(self) -> None:
        """Remove all attention hooks."""
        for h in self._attention_hooks:
            h.remove()
        self._attention_hooks.clear()
        self._captured_attention_weights.clear()


# ── Open-Oasis Wrapper ───────────────────────────────────────────────────

class OasisWrapper(BaseWorldModel):
    """
    Wrapper for Open-Oasis 500M (DiT-based Minecraft world model).

    Architecture notes:
    - Factored spatial-temporal attention: 144 spatial tokens/frame, up to 32 frames.
    - Actions: 25-dim one-hot added to AdaLN conditioning.
    - VAE encoder/decoder for frame compression/reconstruction.
    """

    MODEL_NAME = "open-oasis"
    SPATIAL_TOKENS = 144
    MAX_FRAMES = 32
    ACTION_DIM = 25  # Minecraft keyboard/mouse actions

    def __init__(
        self,
        repo_dir: Optional[str | Path] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__(device=device, dtype=dtype)
        self.repo_dir = Path(repo_dir) if repo_dir else OASIS_REPO_DIR
        self.dit = None
        self.vae = None
        self._frame_buffer: List[torch.Tensor] = []

    def _ensure_repo(self) -> None:
        """Verify the Open-Oasis repo is cloned and importable."""
        if not self.repo_dir.exists():
            raise FileNotFoundError(
                f"Open-Oasis repository not found at {self.repo_dir}. "
                f"Clone it with: git clone https://github.com/etched-ai/open-oasis.git {self.repo_dir}"
            )
        repo_str = str(self.repo_dir)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

    def load_checkpoint(
        self,
        dit_checkpoint: Optional[str | Path] = None,
        vae_checkpoint: Optional[str | Path] = None,
        **kwargs: Any,
    ) -> None:
        """
        Load DiT and VAE checkpoints.

        If no checkpoint paths given, attempts default locations within
        the repo or HuggingFace hub.
        """
        self._ensure_repo()

        # Import from the oasis repo
        try:
            # Open-Oasis typically exposes model classes at top level
            oasis_module = importlib.import_module("oasis")
        except ImportError:
            # Try alternative import paths
            try:
                oasis_module = importlib.import_module("model")
            except ImportError:
                raise ImportError(
                    f"Cannot import oasis model code from {self.repo_dir}. "
                    "Ensure the repo is properly set up (pip install -e . or "
                    "check the repo README)."
                )

        # Load DiT
        if hasattr(oasis_module, "DiT"):
            dit_cls = oasis_module.DiT
        elif hasattr(oasis_module, "create_dit"):
            dit_cls = oasis_module.create_dit
        else:
            # Scan sub-modules
            dit_cls = self._find_class_in_repo("DiT", "dit")

        if callable(dit_cls) and not isinstance(dit_cls, type):
            self.dit = dit_cls(**kwargs.get("dit_kwargs", {}))
        else:
            self.dit = dit_cls(**kwargs.get("dit_kwargs", {})) if kwargs.get("dit_kwargs") else dit_cls()

        if dit_checkpoint:
            state = torch.load(dit_checkpoint, map_location="cpu", weights_only=True)
            if "model" in state:
                state = state["model"]
            elif "state_dict" in state:
                state = state["state_dict"]
            self.dit.load_state_dict(state, strict=False)

        self.dit = self.dit.to(device=self.device, dtype=self.dtype).eval()

        # Load VAE (typically a pretrained stable-diffusion VAE)
        try:
            from diffusers import AutoencoderKL  # type: ignore[import-untyped]

            if vae_checkpoint:
                self.vae = AutoencoderKL.from_pretrained(str(vae_checkpoint))
            else:
                # Default: try loading from repo config or diffusers hub
                vae_path = self.repo_dir / "vae"
                if vae_path.exists():
                    self.vae = AutoencoderKL.from_pretrained(str(vae_path))
                else:
                    self.vae = AutoencoderKL.from_pretrained(
                        "stabilityai/sd-vae-ft-mse"
                    )
            self.vae = self.vae.to(device=self.device, dtype=self.dtype).eval()
        except Exception as e:
            warnings.warn(f"Could not load VAE: {e}. Decoding will not be available.")

        # Register attention hooks
        self._setup_attention_hooks()

    def _find_class_in_repo(self, *names: str) -> Any:
        """Try to find a class or factory function in the repo modules."""
        for mod_name in ["model", "dit", "models.dit", "oasis.model"]:
            try:
                mod = importlib.import_module(mod_name)
                for name in names:
                    if hasattr(mod, name):
                        return getattr(mod, name)
            except ImportError:
                continue
        raise ImportError(
            f"Could not find any of {names} in Open-Oasis repo at {self.repo_dir}."
        )

    def _setup_attention_hooks(self) -> None:
        """Register hooks on DiT attention layers."""
        if self.dit is None:
            return
        attn_modules: Dict[str, nn.Module] = {}
        for name, module in self.dit.named_modules():
            # Match common attention class names
            cls_name = type(module).__name__.lower()
            if "attention" in cls_name or "attn" in cls_name:
                attn_modules[name] = module
        if attn_modules:
            self._register_attention_hooks(attn_modules)

    def generate_frame(
        self,
        actions: torch.Tensor,
        context_frames: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate frame(s) conditioned on actions.

        Parameters
        ----------
        actions : (B, A) where A=25, one-hot Minecraft actions.
        context_frames : (B, T, C, H, W) previously generated/observed frames.

        Returns
        -------
        (B, C, H, W) next frame in pixel space (if VAE available) or
        latent space.
        """
        if self.dit is None:
            raise RuntimeError("Model not loaded. Call load_checkpoint() first.")

        B = actions.shape[0]
        actions = actions.to(device=self.device, dtype=self.dtype)

        # Encode context frames to latent if VAE available
        latent_context = None
        if context_frames is not None and self.vae is not None:
            ctx = context_frames.to(device=self.device, dtype=self.dtype)
            if ctx.ndim == 5:
                B_ctx, T_ctx, C, H, W = ctx.shape
                ctx_flat = ctx.reshape(B_ctx * T_ctx, C, H, W)
            else:
                ctx_flat = ctx
            with torch.no_grad():
                latent_context = self.vae.encode(ctx_flat).latent_dist.sample()
                latent_context = latent_context * 0.18215  # SD scaling factor
            if context_frames.ndim == 5:
                latent_context = latent_context.reshape(B_ctx, T_ctx, *latent_context.shape[1:])

        # DiT forward (model-specific — adapt to actual Open-Oasis API)
        with torch.no_grad():
            if self.profiler:
                self.profiler.start_region("dit_forward")

            # The actual call depends on Open-Oasis's API. Common patterns:
            try:
                latent_out = self.dit(
                    context=latent_context,
                    actions=actions,
                )
            except TypeError:
                # Alternative: some models take timestep + noise
                noise = torch.randn(
                    B, 4, 18, 32,  # typical Oasis latent shape
                    device=self.device, dtype=self.dtype,
                )
                timestep = torch.zeros(B, device=self.device, dtype=torch.long)
                latent_out = self.dit(noise, timestep, actions)

            if self.profiler:
                self.profiler.end_region("dit_forward")

        # Decode to pixel space
        if self.vae is not None:
            with torch.no_grad():
                if self.profiler:
                    self.profiler.start_region("vae_decode")
                decoded = self.vae.decode(latent_out / 0.18215).sample
                if self.profiler:
                    self.profiler.end_region("vae_decode")
                return decoded.clamp(0, 1)

        return latent_out

    def get_kv_cache(self) -> Optional[Dict[str, torch.Tensor]]:
        """Return KV-cache if the DiT maintains one (spatial-temporal attention)."""
        if self.dit is None:
            return None
        cache: Dict[str, torch.Tensor] = {}
        for name, module in self.dit.named_modules():
            # Look for cached keys/values
            for attr in ("kv_cache", "cache", "_kv_cache", "key_cache", "value_cache"):
                val = getattr(module, attr, None)
                if val is not None and isinstance(val, torch.Tensor):
                    cache[f"{name}.{attr}"] = val
                elif isinstance(val, (tuple, list)):
                    for i, v in enumerate(val):
                        if isinstance(v, torch.Tensor):
                            cache[f"{name}.{attr}.{i}"] = v
        return cache if cache else None

    def reset(self) -> None:
        """Clear KV caches and frame buffers."""
        self._frame_buffer.clear()
        self._captured_attention_weights.clear()
        if self.dit is not None:
            for module in self.dit.modules():
                for attr in ("kv_cache", "cache", "_kv_cache", "key_cache", "value_cache"):
                    if hasattr(module, attr):
                        val = getattr(module, attr)
                        if isinstance(val, torch.Tensor):
                            setattr(module, attr, None)
                        elif isinstance(val, (list, tuple)):
                            setattr(module, attr, type(val)())

    @property
    def model(self) -> Optional[nn.Module]:
        """Uniform access to the core model module (returns self.dit)."""
        return self.dit

    def get_model_info(self) -> Dict[str, Any]:
        """Return Oasis architecture metadata."""
        num_params = 0
        if self.dit is not None:
            num_params = sum(p.numel() for p in self.dit.parameters())
        return {
            "name": self.MODEL_NAME,
            "num_params": num_params,
            "num_params_M": num_params / 1e6,
            "spatial_tokens_per_frame": self.SPATIAL_TOKENS,
            "max_frames": self.MAX_FRAMES,
            "action_dim": self.ACTION_DIM,
            "dtype": str(self.dtype),
            "device": str(self.device),
        }


# ── Factory ───────────────────────────────────────────────────────────────

def load_model(
    model_name: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    **kwargs: Any,
) -> BaseWorldModel:
    """
    Factory function to create and load a model wrapper.

    Parameters
    ----------
    model_name : one of ``"oasis"`` / ``"open-oasis"``.
    """
    name = model_name.lower().replace("_", "-").replace(" ", "-")

    if name in ("oasis", "open-oasis"):
        return OasisWrapper(device=device, dtype=dtype, **kwargs)
    raise ValueError(f"Unknown model '{model_name}'. Supported: oasis")
