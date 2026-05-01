"""
load.py — JIT-compiles WorldServe custom CUDA kernels at first use.
Uses torch.utils.cpp_extension.load for incremental builds (cached in /tmp).

Target: H100 SXM, sm_90a, CUDA 12.4+

Active Oasis kernels (after pruning):
    fused_adaln     — fused LayerNorm + scale + shift + gate (used by FusedAdaLN nn.Module)
    int4_quantize   — INT4 weight-only quantization (oasis_int4_modal, oasis_custom_modal)
    fp8_quantize    — FP8 E4M3 V-tensor compression (oasis_custom_modal)
"""

import os
import torch
from torch.utils.cpp_extension import load

_KERNELS = None          # cached module
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Compiler flags ─────────────────────────────────────────────────────────────
_NVCC_FLAGS = [
    "-O3",
    "-arch=sm_90a",
    "-use_fast_math",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "-std=c++17",
    # Enable FP8 (CUDA 12.1+)
    "-DENABLE_FP8",
]

_EXTRA_CFLAGS = ["-O3", "-std=c++17"]


def _src(fname: str) -> str:
    """Absolute path to a kernel source file."""
    return os.path.join(_SRC_DIR, fname)


def load_kernels(verbose: bool = False) -> object:
    """
    Compile and load WorldServe Oasis CUDA kernels.

    Returns a C++ extension module exposing:
        fused_adaln(x, gamma, beta, alpha, eps)        -> Tensor
        quantize_int4(input, group_size)               -> (packed, scales)
        dequantize_int4(packed, scales, group_size, N) -> Tensor
        quantize_fp8(input, scale, block_size)         -> Tensor
        dequantize_fp8(packed, scale, block_size)      -> Tensor
    """
    global _KERNELS
    if _KERNELS is not None:
        return _KERNELS

    cpp_binding = _src("kernels_binding.cpp")
    if not os.path.exists(cpp_binding):
        _write_binding(cpp_binding)

    _KERNELS = load(
        name="worldserve_kernels",
        sources=[
            cpp_binding,
            _src("fused_adaln.cu"),
            _src("int4_quantize.cu"),
            _src("fp8_quantize.cu"),
        ],
        extra_cuda_cflags=_NVCC_FLAGS,
        extra_cflags=_EXTRA_CFLAGS,
        verbose=verbose,
        with_cuda=True,
    )
    return _KERNELS


def _write_binding(path: str) -> None:
    """Write the PyTorch C++ binding bridging Python ↔ CUDA kernels."""
    binding_code = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

// Forward declarations from each .cu
#include "fused_adaln.cuh"
#include "int4_quantize.cuh"
#include "fp8_quantize.cuh"

// ── fused_adaln ───────────────────────────────────────────────────────────────
torch::Tensor py_fused_adaln(
    torch::Tensor x,       // [B, T, D] fp16
    torch::Tensor gamma,   // [B, D]    fp16
    torch::Tensor beta,    // [B, D]    fp16
    torch::Tensor alpha,   // [B, D]    fp16
    float eps)
{
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kHalf);
    int B = x.size(0), T = x.size(1), D = x.size(2);
    auto out = torch::empty_like(x);
    auto stream = at::cuda::getCurrentCUDAStream();
    launch_fused_adaln(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(gamma.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(beta.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(alpha.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        B, T, D, eps, stream);
    return out;
}

// ── int4 quantize ─────────────────────────────────────────────────────────────
std::tuple<torch::Tensor, torch::Tensor> py_quantize_int4(
    torch::Tensor input,   // [N] fp16
    int group_size)
{
    TORCH_CHECK(input.is_cuda() && input.scalar_type() == torch::kHalf);
    int N = input.numel();
    int n_groups = (N + group_size - 1) / group_size;
    auto packed = torch::empty({(N + 1) / 2}, torch::TensorOptions()
                    .dtype(torch::kUInt8).device(input.device()));
    auto scales = torch::empty({n_groups}, torch::TensorOptions()
                    .dtype(torch::kHalf).device(input.device()));
    auto stream = at::cuda::getCurrentCUDAStream();
    launch_quantize_int4(
        reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
        packed.data_ptr<uint8_t>(),
        reinterpret_cast<__half*>(scales.data_ptr<at::Half>()),
        N, group_size, stream);
    return {packed, scales};
}

torch::Tensor py_dequantize_int4(
    torch::Tensor packed,   // [N/2] uint8
    torch::Tensor scales,   // [N/group_size] fp16
    int group_size,
    int N)
{
    TORCH_CHECK(packed.is_cuda());
    auto out = torch::empty({N}, torch::TensorOptions()
                    .dtype(torch::kHalf).device(packed.device()));
    auto stream = at::cuda::getCurrentCUDAStream();
    launch_dequantize_int4(
        packed.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        N, group_size, stream);
    return out;
}

// ── fp8 quantize ──────────────────────────────────────────────────────────────
torch::Tensor py_quantize_fp8(
    torch::Tensor input,      // [N] fp16
    torch::Tensor scale,      // [ceil(N/block_size)] float32
    int block_size)
{
    TORCH_CHECK(input.is_cuda() && input.scalar_type() == torch::kHalf);
    int N = input.numel();
    auto out = torch::empty({N}, torch::TensorOptions()
                    .dtype(torch::kUInt8).device(input.device()));
    auto stream = at::cuda::getCurrentCUDAStream();
    launch_quantize_fp8(
        reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<__nv_fp8_e4m3*>(out.data_ptr<uint8_t>()),
        scale.data_ptr<float>(),
        N, block_size, stream);
    return out;
}

torch::Tensor py_dequantize_fp8(
    torch::Tensor packed,   // [N] uint8 (FP8)
    torch::Tensor scale,    // [ceil(N/block_size)] float32
    int block_size)
{
    TORCH_CHECK(packed.is_cuda());
    int N = packed.numel();
    auto out = torch::empty({N}, torch::TensorOptions()
                    .dtype(torch::kHalf).device(packed.device()));
    auto stream = at::cuda::getCurrentCUDAStream();
    launch_dequantize_fp8(
        reinterpret_cast<const __nv_fp8_e4m3*>(packed.data_ptr<uint8_t>()),
        scale.data_ptr<float>(),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        N, block_size, stream);
    return out;
}

// ── pybind ────────────────────────────────────────────────────────────────────
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_adaln",      &py_fused_adaln,      "Fused AdaLN-zero (fp16)");
    m.def("quantize_int4",    &py_quantize_int4,    "INT4 quantize (fp16 -> packed int4 + scales)");
    m.def("dequantize_int4",  &py_dequantize_int4,  "INT4 dequantize (packed int4 + scales -> fp16)");
    m.def("quantize_fp8",     &py_quantize_fp8,     "FP8 E4M3 quantize (fp16 -> uint8)");
    m.def("dequantize_fp8",   &py_dequantize_fp8,   "FP8 E4M3 dequantize (uint8 -> fp16)");
}
"""
    with open(path, "w") as f:
        f.write(binding_code)
