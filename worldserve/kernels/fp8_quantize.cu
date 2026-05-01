// fp8_quantize.cu  —  Per-block FP8 E4M3 quantization for H100 (sm_90a)
// Uses cuda_fp8.h, __nv_fp8x4_e4m3, float4 vectorized loads (8 FP16/thread).
// scale = amax / 448.0  (FP8 E4M3 max representable value)

#include "fp8_quantize.cuh"
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

// FP8 E4M3 maximum representable value
static constexpr float FP8_E4M3_MAX = 448.0f;

// ─── quantize_fp8_kernel ──────────────────────────────────────────────────────
// Grid: ceil(N / (blockDim.x * 8)) blocks
// Each thread processes 8 FP16 values (float4 = 128-bit load) → 8 FP8 output
// scale[block_id] = amax(block) / 448.0  (precomputed on host or via reduce kernel)

__global__ void quantize_fp8_kernel(
    const __half*  __restrict__ input,
    __nv_fp8_e4m3*              output,
    const float*   __restrict__ scale,
    int N,
    int block_size)
{
    const int tid      = threadIdx.x + blockIdx.x * blockDim.x;
    const int blk_id   = (tid * 8) / block_size;  // which scale block
    const int base_idx = tid * 8;

    if (base_idx >= N) return;

    float s      = scale[blk_id];
    float inv_s  = (s > 0.f) ? __frcp_rn(s) : 1.0f;

    // Load 8 FP16 as float4 (128-bit)
    if (base_idx + 7 < N) {
        float4 raw = *reinterpret_cast<const float4*>(input + base_idx);
        const __half* hv = reinterpret_cast<const __half*>(&raw);

        // Convert pairs using __nv_fp8x4_e4m3 (packs 4 FP8 into 32-bit)
        // First 4 values
        __half scaled0 = __float2half(__half2float(hv[0]) * inv_s);
        __half scaled1 = __float2half(__half2float(hv[1]) * inv_s);
        __half scaled2 = __float2half(__half2float(hv[2]) * inv_s);
        __half scaled3 = __float2half(__half2float(hv[3]) * inv_s);
        __half scaled4 = __float2half(__half2float(hv[4]) * inv_s);
        __half scaled5 = __float2half(__half2float(hv[5]) * inv_s);
        __half scaled6 = __float2half(__half2float(hv[6]) * inv_s);
        __half scaled7 = __float2half(__half2float(hv[7]) * inv_s);

        // Pack using __nv_fp8x4_e4m3(float4) constructor
        __nv_fp8x4_e4m3 pack0(make_float4(
            __half2float(scaled0), __half2float(scaled1),
            __half2float(scaled2), __half2float(scaled3)));
        __nv_fp8x4_e4m3 pack1(make_float4(
            __half2float(scaled4), __half2float(scaled5),
            __half2float(scaled6), __half2float(scaled7)));

        // Write 8 FP8 values (two uint32 stores)
        *reinterpret_cast<uint32_t*>(output + base_idx)     = pack0.__x;
        *reinterpret_cast<uint32_t*>(output + base_idx + 4) = pack1.__x;
    } else {
        // Tail: scalar fallback
        for (int i = base_idx; i < N; i++) {
            float val = __half2float(input[i]) * inv_s;
            output[i] = __nv_fp8_e4m3(val);
        }
    }
}

// ─── dequantize_fp8_kernel ────────────────────────────────────────────────────
// Each thread processes 8 FP8 → 8 FP16

__global__ void dequantize_fp8_kernel(
    const __nv_fp8_e4m3* __restrict__ input,
    const float*         __restrict__ scale,
    __half*                           output,
    int N,
    int block_size)
{
    const int tid      = threadIdx.x + blockIdx.x * blockDim.x;
    const int base_idx = tid * 8;

    if (base_idx >= N) return;

    const int blk_id = (base_idx) / block_size;
    float s = __ldg(&scale[blk_id]);

    if (base_idx + 7 < N) {
        // Load 8 FP8 as two uint32 (packed __nv_fp8x4_e4m3)
        uint32_t pk0 = *reinterpret_cast<const uint32_t*>(input + base_idx);
        uint32_t pk1 = *reinterpret_cast<const uint32_t*>(input + base_idx + 4);

        // Unpack: __nv_fp8x4_e4m3 stores 4 FP8 values
        // Access individual bytes and convert
        const __nv_fp8_e4m3* fp8_0 = reinterpret_cast<const __nv_fp8_e4m3*>(&pk0);
        const __nv_fp8_e4m3* fp8_1 = reinterpret_cast<const __nv_fp8_e4m3*>(&pk1);

        float4 out_vec0, out_vec1;
        __half* oh0 = reinterpret_cast<__half*>(&out_vec0);
        __half* oh1 = reinterpret_cast<__half*>(&out_vec1);

#pragma unroll
        for (int i = 0; i < 4; i++) {
            oh0[i] = __float2half((float)fp8_0[i] * s);
            oh1[i] = __float2half((float)fp8_1[i] * s);
        }

        *reinterpret_cast<float4*>(output + base_idx)     = out_vec0;
        *reinterpret_cast<float4*>(output + base_idx + 4) = out_vec1;
    } else {
        // Tail
        for (int i = base_idx; i < N; i++) {
            output[i] = __float2half((float)input[i] * s);
        }
    }
}

// ─── Host launchers ───────────────────────────────────────────────────────────
void launch_quantize_fp8(
    const __half*  input,
    __nv_fp8_e4m3* output,
    const float*   scale,
    int N, int block_size,
    cudaStream_t stream)
{
    // Each thread handles 8 FP16 values
    int threads_per_block = 256;
    int total_threads = (N + 7) / 8;
    int grid = (total_threads + threads_per_block - 1) / threads_per_block;
    quantize_fp8_kernel<<<grid, threads_per_block, 0, stream>>>(
        input, output, scale, N, block_size);
}

void launch_dequantize_fp8(
    const __nv_fp8_e4m3* input,
    const float*          scale,
    __half*               output,
    int N, int block_size,
    cudaStream_t stream)
{
    int threads_per_block = 256;
    int total_threads = (N + 7) / 8;
    int grid = (total_threads + threads_per_block - 1) / threads_per_block;
    dequantize_fp8_kernel<<<grid, threads_per_block, 0, stream>>>(
        input, scale, output, N, block_size);
}
