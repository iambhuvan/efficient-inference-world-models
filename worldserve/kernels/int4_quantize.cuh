#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// Per-group INT4 quantization — one warp per group
// group_size must be a multiple of 32 (warp size)

// Quantize FP16 → packed INT4 (2 nibbles per byte)
// output is uint8_t packed, scales is __half per group
__global__ void quantize_int4_kernel(
    const __half*  __restrict__ input,    // [N] FP16
    uint8_t*                    output,   // [N/2] packed INT4
    __half*                     scales,   // [N/group_size] FP16 scales
    int N,
    int group_size
);

// Dequantize packed INT4 → FP16
__global__ void dequantize_int4_kernel(
    const uint8_t* __restrict__ input,    // [N/2] packed INT4
    const __half*  __restrict__ scales,   // [N/group_size] FP16 scales
    __half*                     output,   // [N] FP16
    int N,
    int group_size
);

// Host launchers
void launch_quantize_int4(
    const __half* input,
    uint8_t*      output,
    __half*       scales,
    int N, int group_size,
    cudaStream_t stream
);

void launch_dequantize_int4(
    const uint8_t* input,
    const __half*  scales,
    __half*        output,
    int N, int group_size,
    cudaStream_t stream
);
