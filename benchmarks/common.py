"""
Shared Modal app setup, Docker image definition, and volume/secret references
for all WorldServe benchmark scripts.

Import from any benchmark script with:
    from modal_common import app, image, image_cuda_devel, hf_secret,
                              model_volume, MODEL_CACHE

Container layout (set up by `_add_common_layers`):
    /root/open-oasis     — the open-oasis repo (DiT-S/2 + ViT-L VAE source)
    /root/worldserve     — our package (kernels, models, optimizations, utils)
    /root/benchmarks     — this directory (entrypoints + result_store)
"""

import os
import modal

# Project root is one level above this file (benchmarks/common.py → project_root/)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Container image
# ---------------------------------------------------------------------------

# Shared pip packages and source mounts used by both images
_COMMON_PIP = [
    "diffusers>=0.32.0",
    "transformers>=4.44.0",
    "accelerate>=0.30.0",
    "huggingface_hub>=0.24.0",
    "torchao==0.5.0",
    "einops",
    "imageio[ffmpeg]",
    "nvtx",
    "tabulate",
    "sentencepiece",
    "protobuf",
    "packaging",
    "ninja",
    "timm",
    "safetensors",
    "ftfy",  # required by HeliosPyramidPipeline.prompt_clean for T5 prompt encoding
]

def _add_common_layers(img):
    """Mount open-oasis repo + worldserve package + benchmarks dir."""
    return (
        img
        .run_commands(
            "git clone https://github.com/etched-ai/open-oasis.git /root/open-oasis "
            "|| echo 'open-oasis clone failed'"
        )
        .add_local_dir(os.path.join(_ROOT, "worldserve"), "/root/worldserve")
        .add_local_dir(os.path.dirname(os.path.abspath(__file__)), "/root/benchmarks")
    )


# ---------------------------------------------------------------------------
# Standard image (debian_slim) — used by all baselines and non-CUDA-compile
# optimised scripts (torchao, third-party sageattention, etc.)
#
# image_base is the *pre-mount* version: scripts that need additional pip /
# run_commands layers must extend image_base, then call _add_common_layers()
# themselves, because Modal forbids any build step after add_local_*.
# ---------------------------------------------------------------------------
image_base = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1", "torchvision", "torchaudio",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(*_COMMON_PIP)
    .apt_install("git", "build-essential")
    # Note: debian_slim has no nvcc → flash-attn / sageattn 2.x source builds
    # will fail. Use prebuilt wheels only; if those fail, fall through to
    # SDPA fallback. For attention-kernel benchmarks, prefer image_cuda_devel.
    .run_commands(
        "pip install flash-attn==2.7.4.post1 --no-build-isolation "
        "|| echo 'flash-attn prebuilt wheel install failed (need cuda-devel for source build)'"
    )
    .run_commands(
        "pip install 'sageattention>=2.0.0' "
        "|| pip install sageattention "
        "|| echo 'SageAttention install failed'"
    )
)
image = _add_common_layers(image_base)

# ---------------------------------------------------------------------------
# CUDA devel image — used by all WorldServe benchmarks that need JIT-compiled
# CUDA extensions (flash-attn, SageAttention 2.x, custom kernels).
# Built from nvidia/cuda:12.4.1-devel-ubuntu22.04 which ships nvcc + headers.
#
# Critical ordering:
#   1. CUDA_HOME env MUST be set BEFORE any pip install that builds CUDA exts.
#      Modal's .env() applies only at *runtime*, not retroactively to RUN
#      layers built earlier. So we set CUDA_HOME inline via run_commands.
#   2. SageAttention >= 2.x is required for the SM90 INT4-Q/FP8-PV kernels
#      (`_sage_qk_int8_pv_fp8_cuda_sm90`). 1.x lacks them entirely.
#   3. flash-attn 2.7.4.post1 has prebuilt wheels for torch 2.5 + cu12.
#      Falls back to source build with CUDA_HOME if wheel install fails.
# ---------------------------------------------------------------------------
image_cuda_devel_base = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    # CUDA_HOME visible to every subsequent RUN (Modal env() applies image-wide).
    .env({"CUDA_HOME": "/usr/local/cuda", "PATH": "/usr/local/cuda/bin:$PATH"})
    .pip_install(
        "torch==2.5.1", "torchvision", "torchaudio",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(*_COMMON_PIP)
    # build-essential gives gcc/g++; SageAttention's setup.py invokes clang++
    # for the final shared-library link step → must install clang too.
    .apt_install("git", "build-essential", "clang")
    # ── flash-attn: install via DIRECT GITHUB RELEASE WHEEL ──────────────
    # Modal's PyPI mirror only serves the .tar.gz source for flash-attn,
    # which can't build reliably. The wheel URL below is the official
    # Dao-AILab release matching: cu12 + torch 2.5 + cxx11abiFALSE +
    # cp311 + linux_x86_64. Install: ~30 sec (no compile).
    # Verify ABI:  python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"
    # Standard PyPI torch 2.5.1 = False → use cxx11abiFALSE wheel.
    .run_commands(
        "pip install --no-build-isolation "
        "https://github.com/Dao-AILab/flash-attention/releases/download/"
        "v2.7.4.post1/flash_attn-2.7.4.post1+"
        "cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl "
        "|| echo 'flash-attn wheel install failed'"
    )
    # ── SageAttention 2.x: BUILD FROM SOURCE (PyPI has only 1.x) ─────────
    # Critical: TORCH_CUDA_ARCH_LIST="9.0" pins to H100 only — without it,
    # builds for all archs (~30-45 min) and may OOM on Modal default RAM.
    # `python setup.py install` per their README; pip's PEP 517 isolation
    # makes the CUDAExtension build fragile.
    # Estimated build time: ~8-12 min on cuda-devel H100 container.
    .run_commands("pip install ninja wheel setuptools packaging")
    .run_commands(
        "git clone --depth=1 https://github.com/thu-ml/SageAttention.git "
        "/tmp/SageAttention "
        "&& cd /tmp/SageAttention "
        "&& TORCH_CUDA_ARCH_LIST='9.0' "
        "   EXT_PARALLEL=4 "
        "   NVCC_APPEND_FLAGS='--threads 8' "
        "   MAX_JOBS=4 "
        "   CUDA_HOME=/usr/local/cuda "
        "   python setup.py install "
        "|| echo 'SageAttention 2.x source build failed — SDPA fallback only'"
    )
    # ── Sanity checks: build fails LOUD if either kernel didn't install ─
    .run_commands(
        "python -c 'import torch; "
        "print(\"torch:\", torch.__version__); "
        "print(\"torch CXX11 ABI:\", torch._C._GLIBCXX_USE_CXX11_ABI)' "
        "|| echo 'torch import failed'"
    )
    .run_commands(
        "python -c 'import flash_attn; from flash_attn import flash_attn_func; "
        "print(\"flash_attn:\", flash_attn.__version__, \"callable:\", callable(flash_attn_func))' "
        "|| echo '!!! flash_attn IMPORT FAILED !!!'"
    )
    .run_commands(
        "python -c 'import sageattention; "
        "from sageattention import sageattn_qk_int8_pv_fp8_cuda_sm90; "
        "print(\"sageattention v2 SM90 callable:\", callable(sageattn_qk_int8_pv_fp8_cuda_sm90))' "
        "|| echo '!!! sageattention 2.x SM90 IMPORT FAILED !!!'"
    )
)
image_cuda_devel = _add_common_layers(image_cuda_devel_base)

# ---------------------------------------------------------------------------
# Secrets and volumes
# ---------------------------------------------------------------------------

# Users create this once with:
#   modal secret create huggingface-secret HF_TOKEN=hf_...
hf_secret = modal.Secret.from_name("huggingface")

# Persistent volume so model weights are only downloaded once across runs
model_volume = modal.Volume.from_name("worldserve-models", create_if_missing=True)
MODEL_CACHE = "/models"

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = modal.App("worldserve-benchmarks")
