"""
WorldServe Triton kernels — H100 SXM (sm_90a).

Exports
-------
flash_attn_func          FA3-style 2-stage tiling FlashAttention
sliding_tile_attention   STA window-aligned tiling (arXiv 2502.04507)
tempache_attention       TempCache K-dedup exact attention (arXiv 2602.01801)
sage_attention_h100      SageAttention2 INT8-QK + FP8-PV (arXiv 2411.10958)
"""

from .flash_attention import flash_attn_func
from .sliding_tile_attention import sliding_tile_attention
from .tempache import tempache_attention
from .int4_fp8_attention import sage_attention_h100

__all__ = [
    "flash_attn_func",
    "sliding_tile_attention",
    "tempache_attention",
    "sage_attention_h100",
]
