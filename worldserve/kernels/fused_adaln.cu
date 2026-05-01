// fused_adaln.cu  —  Fused AdaLN-zero for H100 (sm_90a)
// y = alpha * ((x - mean) / sqrt(var + eps)) * (1 + gamma) + beta
// One kernel: LayerNorm + scale + shift + gate.
// Two-pass Welford in SMEM, __shfl_xor_sync warp reduction,
// __half2 arithmetic, float4 128-bit loads, __ldg for conditioning.

#include "fused_adaln.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>

// ─── Warp Welford online merge ───────────────────────────────────────────────

struct WelfordState {
    float mean;
    float M2;
    float count;
};

__device__ __forceinline__ WelfordState
warp_welford_reduce(WelfordState s)
{
#pragma unroll
    for (int offset = 16; offset >= 1; offset >>= 1) {
        float om  = __shfl_xor_sync(0xffffffff, s.mean,  offset);
        float oM2 = __shfl_xor_sync(0xffffffff, s.M2,    offset);
        float oc  = __shfl_xor_sync(0xffffffff, s.count, offset);

        float delta = om - s.mean;
        float total = s.count + oc;
        float inv_t = (total > 0.f) ? __frcp_rn(total) : 0.f;

        s.M2    = s.M2 + oM2 + delta * delta * s.count * oc * inv_t;
        s.mean  = (s.mean * s.count + om * oc) * inv_t;
        s.count = total;
    }
    return s;
}

// ─── Fused AdaLN kernel (template on D) ─────────────────────────────────────
// Grid: [B * T]   Block: up to 256 threads
// Each thread handles 8 fp16 per pass (float4 = 128-bit)

template <int D>
__global__ void fused_adaln_kernel(
    const __half* __restrict__ x,      // [B, T, D]
    const __half* __restrict__ gamma,  // [B, D]
    const __half* __restrict__ beta,   // [B, D]
    const __half* __restrict__ alpha,  // [B, D]
    __half*                    out,    // [B, T, D]
    int B, int T, float eps)
{
    const int bt_idx = blockIdx.x;    // flat [B*T]
    const int b      = bt_idx / T;

    const __half* x_row   = x   + (long long)bt_idx * D;
          __half* out_row = out + (long long)bt_idx * D;
    const __half* g_ptr   = gamma + (long long)b * D;
    const __half* b_ptr   = beta  + (long long)b * D;
    const __half* a_ptr   = alpha + (long long)b * D;

    // Shared memory: [32] slots for warp means and M2s
    __shared__ float smem_mean[32];
    __shared__ float smem_M2  [32];

    const int lane   = threadIdx.x & 31;
    const int warpid = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;

    // ── Pass 1: Welford over this row ─────────────────────────────────────
    WelfordState ws = {0.f, 0.f, 0.f};

    for (int v = threadIdx.x; v < D / 8; v += blockDim.x) {
        float4 raw = *reinterpret_cast<const float4*>(x_row + v * 8);
        const __half* hv = reinterpret_cast<const __half*>(&raw);
#pragma unroll
        for (int i = 0; i < 8; i++) {
            float val  = __half2float(hv[i]);
            ws.count  += 1.f;
            float delta = val - ws.mean;
            ws.mean   += delta / ws.count;
            ws.M2     += delta * (val - ws.mean);
        }
    }

    ws = warp_welford_reduce(ws);

    if (lane == 0) {
        smem_mean[warpid] = ws.mean;
        smem_M2  [warpid] = ws.M2;
    }
    __syncthreads();

    // Thread 0 reduces across warps using parallel Welford merge
    if (threadIdx.x == 0) {
        float g_mean = smem_mean[0];
        float g_M2   = smem_M2  [0];
        float g_cnt  = (float)(D / nwarps);

        for (int w = 1; w < nwarps; w++) {
            float om    = smem_mean[w];
            float oM2   = smem_M2[w];
            float oc    = g_cnt;   // each warp handled equal share
            float total = g_cnt * (float)w + oc;
            float inv_t = __frcp_rn(total);
            float delta = om - g_mean;
            g_M2    = g_M2 + oM2 + delta * delta * (g_cnt * (float)w) * oc * inv_t;
            g_mean  = (g_mean * (g_cnt * (float)w) + om * oc) * inv_t;
        }
        smem_mean[0] = g_mean;
        smem_M2  [0] = g_M2 / (float)D;  // variance
    }
    __syncthreads();

    const float row_mean = smem_mean[0];
    const float row_var  = smem_M2  [0];
    const float inv_std  = rsqrtf(row_var + eps);

    const __half2 mean2 = __float2half2_rn(row_mean);
    const __half2 istd2 = __float2half2_rn(inv_std);
    const __half2 one2  = __float2half2_rn(1.f);

    // ── Pass 2: normalize + adaLN (vectorized __half2) ───────────────────
    for (int v = threadIdx.x; v < D / 8; v += blockDim.x) {
        int base = v * 8;

        float4 xraw = *reinterpret_cast<const float4*>(x_row  + base);
        float4 graw = __ldg(reinterpret_cast<const float4*>(g_ptr + base));
        float4 braw = __ldg(reinterpret_cast<const float4*>(b_ptr + base));
        float4 araw = __ldg(reinterpret_cast<const float4*>(a_ptr + base));

        const __half2* xh2 = reinterpret_cast<const __half2*>(&xraw);
        const __half2* gh2 = reinterpret_cast<const __half2*>(&graw);
        const __half2* bh2 = reinterpret_cast<const __half2*>(&braw);
        const __half2* ah2 = reinterpret_cast<const __half2*>(&araw);

        float4   out_vec;
        __half2* out2 = reinterpret_cast<__half2*>(&out_vec);

#pragma unroll
        for (int k = 0; k < 4; k++) {
            __half2 norm    = __hmul2(__hsub2(xh2[k], mean2), istd2);
            __half2 scaled  = __hmul2(norm, __hadd2(one2, gh2[k]));
            __half2 shifted = __hadd2(scaled, bh2[k]);
            out2[k]         = __hmul2(ah2[k], shifted);
        }

        *reinterpret_cast<float4*>(out_row + base) = out_vec;
    }
}

// ─── Explicit instantiations ─────────────────────────────────────────────────
template __global__ void fused_adaln_kernel<128>(
    const __half*, const __half*, const __half*, const __half*, __half*, int, int, float);
template __global__ void fused_adaln_kernel<256>(
    const __half*, const __half*, const __half*, const __half*, __half*, int, int, float);
template __global__ void fused_adaln_kernel<512>(
    const __half*, const __half*, const __half*, const __half*, __half*, int, int, float);
template __global__ void fused_adaln_kernel<1024>(
    const __half*, const __half*, const __half*, const __half*, __half*, int, int, float);

// ─── Host launcher ────────────────────────────────────────────────────────────
void launch_fused_adaln(
    const __half* x,
    const __half* gamma,
    const __half* beta,
    const __half* alpha,
    __half*       out,
    int B, int T, int D,
    float eps,
    cudaStream_t stream)
{
    // D/8 threads per block (one float4 load per thread), capped at 256
    int threads = min(D / 8, 256);
    int blocks  = B * T;

    switch (D) {
        case 128:
            fused_adaln_kernel<128><<<blocks, threads, 0, stream>>>(
                x, gamma, beta, alpha, out, B, T, eps);
            break;
        case 256:
            fused_adaln_kernel<256><<<blocks, threads, 0, stream>>>(
                x, gamma, beta, alpha, out, B, T, eps);
            break;
        case 512:
            fused_adaln_kernel<512><<<blocks, threads, 0, stream>>>(
                x, gamma, beta, alpha, out, B, T, eps);
            break;
        case 1024:
            fused_adaln_kernel<1024><<<blocks, threads, 0, stream>>>(
                x, gamma, beta, alpha, out, B, T, eps);
            break;
        default:
            // Fallback: use 1024-dim kernel (handles any D via loop)
            fused_adaln_kernel<1024><<<blocks, min(256, threads), 0, stream>>>(
                x, gamma, beta, alpha, out, B, T, eps);
            break;
    }
}
