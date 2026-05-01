#pragma once
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

// Per-block FP8 E4M3 quantization for H100
// scale = amax / 448.0  (448.0 = max representable E4M3 value)

// Quantize FP16 → FP8 E4M3 (block-level scale)
// scale is a single FP32 value for the entire tensor or a per-block array
__global__ void quantize_fp8_kernel(
    const __half*  __restrict__ input,   // [N] FP16
    __nv_fp8_e4m3*              output,  // [N] FP8
    const float*   __restrict__ scale,   // [ceil(N/block_size)] per-block scale
    int N,
    int block_size
);

// Dequantize FP8 E4M3 → FP16
__global__ void dequantize_fp8_kernel(
    const __nv_fp8_e4m3* __restrict__ input,   // [N] FP8
    const float*         __restrict__ scale,   // [ceil(N/block_size)] per-block scale
    __half*                           output,  // [N] FP16
    int N,
    int block_size
);

// Host launchers
void launch_quantize_fp8(
    const __half* input,
    __nv_fp8_e4m3* output,
    const float*   scale,
    int N, int block_size,
    cudaStream_t stream
);

void launch_dequantize_fp8(
    const __nv_fp8_e4m3* input,
    const float*          scale,
    __half*               output,
    int N, int block_size,
    cudaStream_t stream
);
