"""
WorldServe model wrappers for uniform profiling and optimization.

Two patterns coexist:

* **Class-based wrappers** (``BaseWorldModel`` subclasses in ``base.py``)
  expose a uniform ABC interface — ``load_checkpoint``, ``generate_frame``,
  ``get_kv_cache``, ``profile_forward``, ``reset`` — used by the profiler
  and the OptimizationStack.

* **Functional loaders** (``oasis``, ``helios``) provide ``load_model`` /
  ``generate`` helpers used directly by Modal / GCE benchmark scripts where
  inline loading is preferred over the wrapper boilerplate.

Both patterns load the same checkpoints and share kernel implementations.
"""

from worldserve.models import helios, oasis  # functional loaders
from worldserve.models.base import (
    BaseWorldModel,
    OasisWrapper,
    load_model,
)

__all__ = [
    "BaseWorldModel",
    "OasisWrapper",
    "helios",
    "load_model",
    "oasis",
]
