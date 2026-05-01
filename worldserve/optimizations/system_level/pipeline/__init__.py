"""
Pipeline-level optimizations for WorldServe.

These are model-agnostic (or model-adapted) optimizations applied at the
inference pipeline level: FlashAttention-3, torch.compile, FP8 quantization,
CUDA stream pipelining, and fused kernels.
"""

from typing import Optional, Dict, Any

__all__ = [
    "FlashAttention3Replacer",
    "CompileOptimizer",
    "DiTFP8Quantizer",
    "CUDAStreamPipeline",
    "FusedAdaLN",
    "FusedTimeEmbedding",
]


def apply_pipeline_optimizations(
    model: Any,
    model_type: str,
    config: Dict[str, Any],
) -> Any:
    """
    Apply all enabled pipeline optimizations to a model.

    Args:
        model: The model (Oasis or MG2) to optimize.
        model_type: One of "oasis" or "mg2".
        config: The pipeline section of the WorldServe config.

    Returns:
        The optimized model.
    """
    # 1. Flash Attention replacement
    if config.get("flash_attention", {}).get("enabled", False):
        from worldserve.optimizations.system_level.pipeline.flash_attention import FlashAttention3Replacer
        replacer = FlashAttention3Replacer()
        if model_type == "oasis":
            replacer.replace_oasis_attention(model)
        elif model_type == "mg2":
            replacer.replace_mg2_attention(model)

    # 2. FP8 quantization (before compile so compiled graph uses FP8 ops)
    if config.get("fp8", {}).get("enabled", False):
        from worldserve.optimizations.system_level.pipeline.fp8_quantizer import DiTFP8Quantizer
        quantizer = DiTFP8Quantizer()
        quantizer.quantize_torchao(model)

    # 3. torch.compile
    if config.get("compile", {}).get("enabled", False):
        from worldserve.optimizations.system_level.pipeline.compile_optimizer import CompileOptimizer
        compiler = CompileOptimizer(
            mode=config.get("compile", {}).get("mode", "max-autotune"),
            backend=config.get("compile", {}).get("backend", "inductor"),
        )
        if model_type == "oasis":
            model = compiler.compile_oasis(model)
        elif model_type == "mg2":
            model = compiler.compile_mg2(model)

    return model
