// int4_quantize.cu  —  Per-group INT4 quantization/dequantization for H100
// One warp per group; warp_absmax via __shfl_xor_sync; lop3.b32 nibble packing;
// float4 vectorized loads (8 FP16 per thread = 128 bits).

#include "int4_quantize.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// ─── Warp absmax reduction ────────────────────────────────────────────────────
__device__ __forceinline__ float warp_absmax(float v)
{
    v = fabsf(v);
#pragma unroll
    for (int mask = 16; mask >= 1; mask >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, mask));
    return v;
}

// ─── quantize_int4_kernel ─────────────────────────────────────────────────────
// Grid: ceil(N / group_size) blocks of 32 threads (1 warp per group)
// Each thread covers group_size/32 elements.
// Loads via float4 (8 FP16 at once) when possible.

__global__ void quantize_int4_kernel(
    const __half*  __restrict__ input,
    uint8_t*                    output,
    __half*                     scales,
    int N,
    int group_size)
{
    const int group_id = blockIdx.x;
    const int lane     = threadIdx.x;                // 0..31
    const int g_start  = group_id * group_size;

    if (g_start >= N) return;
    const int g_end = min(g_start + group_size, N);
    const int g_len = g_end - g_start;

    // ── Step 1: find per-group absmax ─────────────────────────────────────
    float local_max = 0.f;

    // Each lane handles elements spaced 32 apart; load 8 FP16 per iter via float4
    for (int base = lane * 8; base < g_len; base += 32 * 8) {
        if (base + 7 < g_len) {
            // Full float4 load
            float4 raw = *reinterpret_cast<const float4*>(input + g_start + base);
            const __half* hv = reinterpret_cast<const __half*>(&raw);
#pragma unroll
            for (int i = 0; i < 8; i++)
                local_max = fmaxf(local_max, fabsf(__half2float(hv[i])));
        } else {
            // Partial tail
            for (int i = base; i < g_len; i++)
                local_max = fmaxf(local_max, fabsf(__half2float(input[g_start + i])));
        }
    }

    float group_amax = warp_absmax(local_max);
    float scale      = (group_amax > 0.f) ? group_amax / 7.0f : 1.0f;
    float inv_scale  = 1.0f / scale;

    if (lane == 0)
        scales[group_id] = __float2half(scale);

    // ── Step 2: quantize and pack 2 INT4 per byte ─────────────────────────
    // Each warp lane handles 2 elements per stride, packing them into 1 byte.
    for (int base = lane * 2; base < g_len - 1; base += 64) {
        int abs_idx0 = g_start + base;
        int abs_idx1 = g_start + base + 1;

        float f0 = __half2float(input[abs_idx0]);
        float f1 = __half2float(input[abs_idx1]);

        // Clamp to [-8, 7]
        int q0 = max(-8, min(7, (int)rintf(f0 * inv_scale)));
        int q1 = max(-8, min(7, (int)rintf(f1 * inv_scale)));

        // Pack using lop3.b32: low nibble = q0 & 0xF, high nibble = q1 & 0xF
        uint32_t packed;
        uint32_t lo = (uint32_t)(q0 & 0x0F);
        uint32_t hi = (uint32_t)(q1 & 0x0F) << 4;
        // lop3: (lo | hi) — same as OR, but use inline asm for explicit lop3
        asm volatile(
            "lop3.b32 %0, %1, %2, %3, 0xFE;"
            : "=r"(packed) : "r"(lo), "r"(hi), "r"(0u)
        );
        output[(g_start + base) / 2] = (uint8_t)(packed & 0xFF);
    }
    // Handle odd tail element
    if (g_len & 1) {
        int last_idx = g_start + g_len - 1;
        float f0 = __half2float(input[last_idx]);
        int q0 = max(-8, min(7, (int)rintf(f0 * inv_scale)));
        output[last_idx / 2] = (uint8_t)(q0 & 0x0F);
    }
}

// ─── dequantize_int4_kernel ───────────────────────────────────────────────────
// Grid: ceil(N / group_size) blocks of 32 threads

__global__ void dequantize_int4_kernel(
    const uint8_t* __restrict__ input,
    const __half*  __restrict__ scales,
    __half*                     output,
    int N,
    int group_size)
{
    const int group_id = blockIdx.x;
    const int lane     = threadIdx.x;
    const int g_start  = group_id * group_size;

    if (g_start >= N) return;
    const int g_end = min(g_start + group_size, N);
    const int g_len = g_end - g_start;

    float scale = __half2float(__ldg(&scales[group_id]));

    for (int base = lane * 2; base < g_len - 1; base += 64) {
        int byte_idx = (g_start + base) / 2;
        uint8_t packed = __ldg(&input[byte_idx]);

        // Extract nibbles: low 4 bits = q0, high 4 bits = q1
        // Sign-extend from 4-bit signed: if bit3 set, extend to int8
        uint32_t raw_lo, raw_hi;
        uint32_t pk = (uint32_t)packed;
        asm volatile(
            "lop3.b32 %0, %2, 0x0F, 0, 0xC0;"   // lo = pk & 0x0F
            "lop3.b32 %1, %2, 0xF0, 0, 0xC0;"   // hi = pk & 0xF0
            : "=r"(raw_lo), "=r"(raw_hi) : "r"(pk)
        );
        // Sign-extend 4-bit → int (bit 3 is sign)
        int q0 = (int)(raw_lo & 0x0F); if (q0 & 8) q0 |= ~0xF;
        int q1 = (int)((raw_hi >> 4) & 0x0F); if (q1 & 8) q1 |= ~0xF;

        output[g_start + base]     = __float2half((float)q0 * scale);
        output[g_start + base + 1] = __float2half((float)q1 * scale);
    }
    // Handle odd tail
    if (g_len & 1) {
        int last_idx = g_start + g_len - 1;
        uint8_t packed = __ldg(&input[last_idx / 2]);
        int q0 = (int)(packed & 0x0F); if (q0 & 8) q0 |= ~0xF;
        output[last_idx] = __float2half((float)q0 * scale);
    }
}

// ─── Host launchers ───────────────────────────────────────────────────────────
void launch_quantize_int4(
    const __half* input,
    uint8_t*      output,
    __half*       scales,
    int N, int group_size,
    cudaStream_t stream)
{
    int n_groups = (N + group_size - 1) / group_size;
    quantize_int4_kernel<<<n_groups, 32, 0, stream>>>(
        input, output, scales, N, group_size);
}

void launch_dequantize_int4(
    const uint8_t* input,
    const __half*  scales,
    __half*        output,
    int N, int group_size,
    cudaStream_t stream)
{
    int n_groups = (N + group_size - 1) / group_size;
    dequantize_int4_kernel<<<n_groups, 32, 0, stream>>>(
        input, scales, output, N, group_size);
}
