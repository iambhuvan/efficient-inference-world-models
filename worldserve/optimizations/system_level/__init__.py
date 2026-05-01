"""
System-level optimizations for WorldServe.

These optimizations change *how the hardware executes* — kernel fusion,
CUDA Graph compilation, quantization, KV cache compression, sparse attention
patterns, and speculative decoding. Model weights and algorithm unchanged.

Components:
  pipeline/         FA3, torch.compile+CUDA Graphs, INT4wo, fused kernels, Triton
  kv_cache/         Temporal merge, progressive quantization, DiffKV asymmetry
  step_caching/     SeaCache, FlowCache, SpeCa
  sparse_attention/ SVG2, NATTEN
  speculative/      Self-speculative decoding with INT4 KV draft
"""
