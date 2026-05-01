#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Fused AdaLN-zero kernel declaration
// y = alpha * ((x - mean) / sqrt(var + eps)) * (1 + gamma) + beta
// Template on hidden dim D for compile-time unrolling

template <int D>
__global__ void fused_adaln_kernel(
    const __half* __restrict__ x,       // [B, T, D]
    const __half* __restrict__ gamma,   // [B, D] conditioning scale
    const __half* __restrict__ beta,    // [B, D] conditioning shift
    const __half* __restrict__ alpha,   // [B, D] conditioning gate
    __half*       out,                  // [B, T, D]
    int B,
    int T,
    float eps
);

// Host launcher: selects correct template instantiation
void launch_fused_adaln(
    const __half* x,
    const __half* gamma,
    const __half* beta,
    const __half* alpha,
    __half*       out,
    int B, int T, int D,
    float eps,
    cudaStream_t stream
);
