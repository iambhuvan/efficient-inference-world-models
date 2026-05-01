"""
modal_common.py — importable alias for common.py.

Modal functions run inside the container with /root/benchmarks on sys.path
(mounted via add_local_dir on the benchmarks/ directory in common.py),
so every benchmark can do:

    from modal_common import app, image, hf_secret, model_volume, MODEL_CACHE

This file simply re-exports everything from common.py.  Keep both files in
sync; common.py is the canonical definition.
"""

from common import (  # noqa: F401
    app,
    image,
    image_base,
    image_cuda_devel,
    image_cuda_devel_base,
    _add_common_layers,
    hf_secret,
    model_volume,
    MODEL_CACHE,
)

__all__ = [
    "app", "image", "image_base", "image_cuda_devel", "image_cuda_devel_base",
    "_add_common_layers", "hf_secret", "model_volume", "MODEL_CACHE",
]
