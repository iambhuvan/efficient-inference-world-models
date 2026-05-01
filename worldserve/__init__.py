"""
WorldServe: Video world model inference optimization.

Combines step caching, KV-cache compression, sparse attention,
and speculative decoding to accelerate video world model inference
on H100 GPUs.
"""

__version__ = "0.1.0"

from pathlib import Path

# Project root directory (parent of worldserve package)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Default paths
CONFIGS_DIR = PROJECT_ROOT / "configs"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
BENCHMARKS_DIR = PROJECT_ROOT / "benchmarks"
LOGS_DIR = PROJECT_ROOT / "logs"


def get_config_path(name: str = "default") -> Path:
    """Return path to a named YAML config file."""
    path = CONFIGS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return path
