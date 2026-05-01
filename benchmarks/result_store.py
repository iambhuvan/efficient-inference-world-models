"""
Utility for saving individual benchmark results to disk.

Every script's local entrypoint calls save_result(result, category, key) so
runs don't get lost even when using individual scripts instead of run_all.py.

Layout:
    modal/runs/
        all_results.json          ← merged master (updated on every save)
        baseline/
            oasis_baseline.json
            ltx_baseline.json
            ...
        optimised_kernels/
            oasis_sageattention2_int4.json
            ltx_sta.json
            ...
"""

import json
import os
from datetime import datetime, timezone

_RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
_MASTER = os.path.join(_RUNS_DIR, "all_results.json")


def save_result(result: dict, category: str, key: str) -> None:
    """
    Persist a single benchmark result.

    Args:
        result:   The dict returned by a Modal benchmark function.
        category: "baseline" or "optimised_kernels".
        key:      Unique key, e.g. "oasis_baseline" or "oasis_sageattention2_int4".
    """
    # Stamp with wall-clock time so runs are traceable
    stamped = {"_saved_at": datetime.now(timezone.utc).isoformat(), **result}

    # Per-run file
    cat_dir = os.path.join(_RUNS_DIR, category)
    os.makedirs(cat_dir, exist_ok=True)
    per_run_path = os.path.join(cat_dir, f"{key}.json")
    with open(per_run_path, "w") as fh:
        json.dump(stamped, fh, indent=2)

    # Update master
    os.makedirs(_RUNS_DIR, exist_ok=True)
    master: dict = {}
    if os.path.exists(_MASTER):
        try:
            with open(_MASTER) as fh:
                master = json.load(fh)
        except (json.JSONDecodeError, OSError):
            master = {}
    master[key] = stamped
    with open(_MASTER, "w") as fh:
        json.dump(master, fh, indent=2)

    print(f"  [saved] {per_run_path}")
    print(f"  [saved] {_MASTER}  (key={key!r})")
