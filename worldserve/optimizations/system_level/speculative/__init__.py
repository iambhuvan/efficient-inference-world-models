"""
Speculative decoding optimizations for WorldServe (Phase 5).

Primary method: KV-quant self-speculative — uses INT4 KV cache as fast draft,
FP16 KV cache for verification. Step-adaptive thresholds relax acceptance
at noisy early steps and tighten at final denoising steps.

Ablation: Layer skipping — runs only first 15/30 DiT blocks as draft.
For MG2, first 15 blocks contain all action modules, so draft is action-aware.

Novel contribution: Tree-based action branching — predicts K likely next
actions, batches draft forward passes in parallel, serves matching frame
instantly when actual input arrives.

Architecture assumptions (Matrix-Game 2.0, or PCM-distilled Oasis):
  - MG2: 30 DiT blocks, 3 denoising steps — use LayerSkipSpeculativeDecoder
  - PCM Oasis: 16 DiT blocks, 4 steps — use KVQuantSpeculativeDecoder
  - rCM MG2: 1 denoising step — speculative decoding not beneficial
"""

from typing import Optional, Dict, Any

__all__ = [
    "KVQuantSpeculativeDecoder",
    "LayerSkipSpeculativeDecoder",
    "TreeActionBrancher",
    "build_speculative_decoder",
    "SpeculativeDecoder",
]


class SpeculativeDecoder:
    """
    Stack-compatible wrapper for speculative decoding.

    Speculative decoding requires the model reference to select the right
    decoder (block count determines kv_quant vs layer_skip). This class
    stores the config so OptimizationStack can hold it, and exposes
    build(model) for modal_app.py to call after model loading.

    Usage in modal_app.py:
        if "speculative" in opt_stack.modules:
            decoder = opt_stack.modules["speculative"].build(model["dit"])
            opt_stack.modules["speculative"] = decoder
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self._decoder = None

    def build(self, model) -> object:
        """Instantiate the concrete decoder against the loaded model."""
        self._decoder = build_speculative_decoder(model, self.config)
        return self._decoder

    # OptimizationStack protocol — delegate to concrete decoder once built
    def pre_step(self, step_idx: int, total_steps: int, latents) -> bool:
        if self._decoder is not None and hasattr(self._decoder, "pre_step"):
            return self._decoder.pre_step(step_idx, total_steps, latents)
        return False

    def post_step(self, step_idx: int, total_steps: int, latents) -> None:
        if self._decoder is not None and hasattr(self._decoder, "post_step"):
            self._decoder.post_step(step_idx, total_steps, latents)

    def get_stats(self) -> Dict[str, Any]:
        if self._decoder is not None and hasattr(self._decoder, "get_stats"):
            return self._decoder.get_stats()
        return {"built": self._decoder is not None}


def build_speculative_decoder(model: "nn.Module", config: dict) -> object:
    """
    Build the appropriate speculative decoder based on model architecture.

    PCM-distilled Oasis has 16 DiT blocks — LayerSkipSpeculativeDecoder's
    15-block draft would use nearly all blocks, providing no speedup.
    Use KVQuantSpeculativeDecoder instead for models with <20 blocks.
    """
    from torch import nn

    # Detect actual block count
    num_blocks = sum(1 for _ in model.modules()
                     if type(_).__name__ in ("DiTBlock", "WanBlock", "TransformerBlock", "BasicTransformerBlock"))
    if num_blocks == 0:
        # Fallback: count named children with "block" in name
        num_blocks = sum(1 for name, _ in model.named_modules() if "block" in name.lower())

    method = config.get("method", "kv_quant")

    if method == "layer_skip" and num_blocks < 20:
        import logging
        logging.getLogger(__name__).warning(
            "[speculative] LayerSkipSpeculativeDecoder requires >=20 DiT blocks "
            "(model has %d). Falling back to kv_quant speculative. "
            "This typically means a PCM-distilled Oasis model is being used -- "
            "use KVQuantSpeculativeDecoder with 4-step Oasis.",
            num_blocks,
        )
        method = "kv_quant"

    if method == "layer_skip":
        from worldserve.optimizations.system_level.speculative.layer_skip_spec import LayerSkipSpeculativeDecoder
        return LayerSkipSpeculativeDecoder(config)
    else:
        from worldserve.optimizations.system_level.speculative.kv_quant_spec import KVQuantSpeculativeDecoder
        return KVQuantSpeculativeDecoder(config)


def get_speculative_optimizers() -> Dict[str, type]:
    """Return a dictionary of available speculative decoding classes."""
    optimizers: Dict[str, type] = {}

    try:
        from worldserve.optimizations.system_level.speculative.kv_quant_spec import KVQuantSpeculativeDecoder
        optimizers["kv_quant_spec"] = KVQuantSpeculativeDecoder
    except ImportError:
        pass

    try:
        from worldserve.optimizations.system_level.speculative.layer_skip_spec import LayerSkipSpeculativeDecoder
        optimizers["layer_skip"] = LayerSkipSpeculativeDecoder
    except ImportError:
        pass

    try:
        from worldserve.optimizations.system_level.speculative.tree_branching import TreeActionBrancher
        optimizers["tree_branching"] = TreeActionBrancher
    except ImportError:
        pass

    return optimizers
