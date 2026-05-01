"""
torch.compile wrapper for WorldServe video world models.

Applies regional compilation to DiT blocks and compiles VAE components,
with automatic graph-break detection and fallback.

Open-Oasis 500M: 16 DiT-S blocks + ViT-L/20 VAE (encoder + decoder).
Matrix-Game 2.0 1.8B: 30 Wan DiT blocks + VAE (decoder already compiled).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CompileOptimizer:
    """
    Manages torch.compile for WorldServe models.

    Uses regional compilation (compile one DiT layer, reuse across all blocks)
    to minimize compile time while maximizing kernel fusion.

    Usage::

        compiler = CompileOptimizer(mode="reduce-overhead")
        model = compiler.compile_oasis(model)
        compiler.warmup(model, dummy_input)
    """

    def __init__(
        self,
        mode: str = "max-autotune",
        backend: str = "inductor",
        fullgraph: bool = True,
        dynamic: bool = False,
    ) -> None:
        """
        Args:
            mode: torch.compile mode. One of "default", "reduce-overhead",
                  "max-autotune", "max-autotune-no-cudagraphs".
            backend: Compilation backend (usually "inductor").
            fullgraph: If True, attempt full-graph compilation. Falls back
                       to partial on graph breaks.
            dynamic: Enable dynamic shapes.
        """
        self.mode = mode
        self.backend = backend
        self.fullgraph = fullgraph
        self.dynamic = dynamic
        self._dynamic_modules: list = []  # modules requiring dynamic shapes (e.g., DyDiT++)
        self._compiled_modules: Dict[str, nn.Module] = {}
        self._graph_break_log: List[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compile_oasis(self, model: nn.Module) -> nn.Module:
        """
        Compile Open-Oasis model components.

        Strategy:
          - VAE encoder: mode="reduce-overhead", fullgraph=True
          - VAE decoder: mode="reduce-overhead", fullgraph=True
          - DiT backbone: regional compilation -- compile one DiT block,
            then reuse the compiled version across all 16 blocks.

        IMPORTANT: call FlashAttention3Replacer.replace_oasis_attention()
        BEFORE this method.  The cuDNN SDPA global flags must be set before
        torch.compile so Dynamo bakes the cuDNN dispatch into the compiled
        graph.  If called after compile() the flags are ignored inside CUDA
        graphs (pytorch issue #134608).

        Args:
            model: The Oasis model (expected to have .vae and .dit or
                   .backbone attributes, or similar structure).

        Returns:
            The model with compiled components (in-place modification).
        """
        # Guard: warn if cuDNN SDPA is not active.  This means
        # replace_oasis_attention() was not called before compile_oasis().
        if (
            torch.cuda.is_available()
            and hasattr(torch.backends.cuda, "cudnn_sdp_enabled")
            and not torch.backends.cuda.cudnn_sdp_enabled()
        ):
            logger.warning(
                "compile_oasis: cuDNN SDPA is NOT enabled. "
                "For best performance on H100, call "
                "FlashAttention3Replacer().replace_oasis_attention(model) "
                "before compile_oasis(). The cuDNN flag must be set before "
                "torch.compile so Dynamo captures it in the CUDA graph."
            )

        # --- Compile VAE ---
        vae = self._find_submodule(model, ["vae", "vae_model", "autoencoder"])
        if vae is not None:
            encoder = self._find_submodule(vae, ["encoder", "enc"])
            decoder = self._find_submodule(vae, ["decoder", "dec"])

            if encoder is not None:
                compiled_enc = self._safe_compile(
                    encoder,
                    name="oasis_vae_encoder",
                    mode="reduce-overhead",
                    fullgraph=True,
                )
                self._replace_submodule(vae, ["encoder", "enc"], compiled_enc)
                logger.info("Compiled Oasis VAE encoder (reduce-overhead, fullgraph).")

            if decoder is not None:
                compiled_dec = self._safe_compile(
                    decoder,
                    name="oasis_vae_decoder",
                    mode="reduce-overhead",
                    fullgraph=True,
                )
                self._replace_submodule(vae, ["decoder", "dec"], compiled_dec)
                logger.info("Compiled Oasis VAE decoder (reduce-overhead, fullgraph).")
        else:
            logger.warning("Could not find VAE submodule in Oasis model.")

        # --- Regional compilation of DiT blocks ---
        dit = self._find_submodule(model, ["dit", "backbone", "transformer", "model"])
        if dit is not None:
            self._regional_compile_blocks(
                dit,
                block_names=["blocks", "layers", "transformer_blocks"],
                model_name="oasis",
                mode="reduce-overhead",
                fullgraph=True,
            )
        else:
            # The model itself might be the DiT
            self._regional_compile_blocks(
                model,
                block_names=["blocks", "layers", "transformer_blocks"],
                model_name="oasis",
                mode="reduce-overhead",
                fullgraph=True,
            )

        return model

    def compile_mg2(self, model: nn.Module) -> nn.Module:
        """
        Compile Matrix-Game 2.0 model components.

        Strategy:
          - VAE decoder: already compiled upstream -- skip or verify.
          - DiT backbone: regional compilation across 30 blocks.
            Uses mode="max-autotune-no-cudagraphs" to avoid conflicts with
            the existing VAE CUDA graph.

        Args:
            model: The MG2 / Wan model.

        Returns:
            The model with compiled components (in-place modification).
        """
        # --- Check / skip VAE (MG2 already compiles its VAE decoder) ---
        vae = self._find_submodule(model, ["vae", "vae_model", "autoencoder"])
        if vae is not None:
            decoder = self._find_submodule(vae, ["decoder", "dec"])
            if decoder is not None:
                # Check if already compiled
                if self._is_compiled(decoder):
                    logger.info("MG2 VAE decoder already compiled; skipping.")
                else:
                    compiled_dec = self._safe_compile(
                        decoder,
                        name="mg2_vae_decoder",
                        mode="max-autotune-no-cudagraphs",
                        fullgraph=False,
                    )
                    self._replace_submodule(vae, ["decoder", "dec"], compiled_dec)
                    logger.info("Compiled MG2 VAE decoder (max-autotune-no-cudagraphs).")

        # --- Regional compilation of DiT blocks ---
        dit = self._find_submodule(model, ["dit", "backbone", "transformer", "model"])
        target = dit if dit is not None else model
        self._regional_compile_blocks(
            target,
            block_names=["blocks", "layers", "transformer_blocks"],
            model_name="mg2",
            mode="max-autotune-no-cudagraphs",
            fullgraph=True,
        )

        return model

    def warmup(
        self,
        model: nn.Module,
        dummy_input: Any,
        num_warmup: int = 1,
    ) -> float:
        """
        Run the compiled model to trigger JIT compilation (the first run
        is slow due to Triton kernel compilation).

        Args:
            model: The compiled model.
            dummy_input: A representative input tensor, tuple, or dict.
            num_warmup: Number of warmup forward passes.

        Returns:
            Total warmup time in seconds.
        """
        model.eval()
        start = time.perf_counter()

        with torch.no_grad():
            for i in range(num_warmup):
                logger.info("Warmup pass %d/%d ...", i + 1, num_warmup)
                if isinstance(dummy_input, dict):
                    _ = model(**dummy_input)
                elif isinstance(dummy_input, (tuple, list)):
                    _ = model(*dummy_input)
                else:
                    _ = model(dummy_input)

                # Sync after each pass to get accurate timing
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        logger.info("Warmup completed in %.2f seconds.", elapsed)
        return elapsed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _safe_compile(
        self,
        module: nn.Module,
        name: str,
        mode: Optional[str] = None,
        fullgraph: Optional[bool] = None,
    ) -> nn.Module:
        """
        Compile a module with fallback on graph breaks.

        If fullgraph=True fails due to graph breaks, retries with
        fullgraph=False and logs the issue.
        """
        compile_mode = mode or self.mode
        compile_fullgraph = fullgraph if fullgraph is not None else self.fullgraph

        try:
            # dynamic=True required when DyDiT++ or DC-DiT are active
            # (variable activation shapes per timestep / per routing decision).
            # register_dynamic_module() sets self.dynamic=True automatically.
            compiled = torch.compile(
                module,
                mode=compile_mode,
                backend=self.backend,
                fullgraph=compile_fullgraph,
                dynamic=self.dynamic,
            )
            self._compiled_modules[name] = compiled
            logger.info(
                "Compiled %s (mode=%s, fullgraph=%s, backend=%s)",
                name, compile_mode, compile_fullgraph, self.backend,
            )
            return compiled

        except Exception as e:
            if compile_fullgraph:
                msg = (
                    f"Graph break detected compiling {name} with fullgraph=True: {e}. "
                    f"Falling back to fullgraph=False."
                )
                logger.warning(msg)
                self._graph_break_log.append(msg)

                # Retry without fullgraph
                compiled = torch.compile(
                    module,
                    mode=compile_mode,
                    backend=self.backend,
                    fullgraph=False,
                    dynamic=self.dynamic,
                )
                self._compiled_modules[name] = compiled
                logger.info(
                    "Compiled %s with fullgraph=False fallback (mode=%s)",
                    name, compile_mode,
                )
                return compiled
            else:
                logger.error("Failed to compile %s: %s", name, e)
                raise

    def _regional_compile_blocks(
        self,
        parent: nn.Module,
        block_names: List[str],
        model_name: str,
        mode: str,
        fullgraph: bool,
    ) -> int:
        """
        Regional compilation: compile the first DiT block, then reuse
        the compiled object for all blocks of the same type.

        This drastically reduces compile time (compile once, run N times)
        because all blocks share the same architecture.

        Args:
            parent: The module containing the block list.
            block_names: Possible attribute names for the block list.
            model_name: "oasis" or "mg2" (for logging).
            mode: torch.compile mode.
            fullgraph: Whether to attempt fullgraph.

        Returns:
            Number of blocks compiled.
        """
        blocks = None
        attr_name = None
        for bn in block_names:
            if hasattr(parent, bn):
                candidate = getattr(parent, bn)
                if isinstance(candidate, (nn.ModuleList, nn.Sequential, list)):
                    blocks = candidate
                    attr_name = bn
                    break

        if blocks is None or len(blocks) == 0:
            logger.warning(
                "%s: could not find DiT blocks under attribute names %s",
                model_name, block_names,
            )
            return 0

        num_blocks = len(blocks)
        logger.info(
            "%s: found %d DiT blocks under '%s'. Applying regional compilation.",
            model_name, num_blocks, attr_name,
        )

        # Compile each block individually.  torch.compile's Inductor code
        # cache ensures that blocks with identical architecture reuse the
        # same compiled Triton kernels after the first block is compiled,
        # so subsequent compiles are fast.  We CANNOT share the compiled
        # forward method across blocks because a bound method carries the
        # originating module's weights — all blocks would run block 0's
        # parameters.
        for i in range(num_blocks):
            compiled_block = self._safe_compile(
                blocks[i],
                name=f"{model_name}_dit_block_{i}",
                mode=mode,
                fullgraph=fullgraph,
            )
            blocks[i] = compiled_block

        logger.info(
            "%s: regionally compiled %d DiT blocks (mode=%s).",
            model_name, num_blocks, mode,
        )
        return num_blocks

    def register_dynamic_module(self, module: nn.Module) -> None:
        """Mark a module as requiring dynamic shapes (e.g., DyDiT++ width scaling).

        DyDiT++ wraps nn.Linear with _DynamicWidthLinear, producing variable
        activation shapes per timestep. torch.compile(dynamic=False) would trigger
        recompilation or graph breaks on every step. Calling this method switches
        the compile mode for that module to dynamic=True so Dynamo traces once for
        all width scales.

        Must be called BEFORE compile_oasis() / compile_mg2().

        Args:
            module: The DyDiT++-wrapped model (or any module with dynamic shapes).
        """
        self._dynamic_modules.append(id(module))
        self.dynamic = True  # upgrade the whole compilation to dynamic mode
        logger.info(
            "[CompileOptimizer] dynamic=True enabled for module %s "
            "(DyDiT++ or DC-DiT dynamic shapes detected)",
            type(module).__name__,
        )

    def has_dynamic_modules(self) -> bool:
        """Return True if any dynamic-shape modules have been registered."""
        return len(self._dynamic_modules) > 0

    @staticmethod
    def _find_submodule(
        model: nn.Module,
        candidates: List[str],
    ) -> Optional[nn.Module]:
        """Find a submodule by trying multiple attribute names."""
        for name in candidates:
            if hasattr(model, name):
                return getattr(model, name)
            # Also check one level deeper via named_modules
            for mod_name, mod in model.named_modules():
                if mod_name == name:
                    return mod
        return None

    @staticmethod
    def _replace_submodule(
        parent: nn.Module,
        candidates: List[str],
        new_module: nn.Module,
    ) -> bool:
        """Replace a submodule in-place by trying multiple attribute names."""
        for name in candidates:
            if hasattr(parent, name):
                setattr(parent, name, new_module)
                return True
        return False

    @staticmethod
    def _is_compiled(module: nn.Module) -> bool:
        """Check if a module is already torch.compiled."""
        # torch.compile wraps the module in an OptimizedModule
        return type(module).__name__ == "OptimizedModule" or hasattr(module, "_orig_mod")

    @property
    def graph_breaks(self) -> List[str]:
        """Return a log of graph breaks encountered during compilation."""
        return list(self._graph_break_log)
