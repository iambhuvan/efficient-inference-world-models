"""
Run all WorldServe benchmarks on Modal H100 and generate a speedup report.

Baselines and their optimised counterparts are dispatched as parallel Modal
remote calls.  Results are collected, written to JSON, and printed as a
formatted speedup table.

Usage:
    modal run modal/run_all.py                  # full suite
    modal run modal/run_all.py --baselines-only # baselines only
    modal run modal/run_all.py --quick          # LTX + Oasis only (fast CI check)

The HuggingFace token secret must be created first:
    modal secret create huggingface-secret HF_TOKEN=hf_...
"""

import sys
sys.path.insert(0, "/root")
sys.path.insert(0, "/root/benchmarks")

import json
import os
from typing import Optional

import modal

# ---------------------------------------------------------------------------
# Import all benchmark functions so Modal can resolve them
# ---------------------------------------------------------------------------

# Baselines
from baseline.ltx_video_modal import run_ltx_baseline
from baseline.cogvideox_modal import run_cogvideox_baseline
from baseline.hunyuan_video_modal import run_hunyuan_baseline
from baseline.flux_dev_modal import run_flux_dev_baseline
from baseline.wan21_modal import run_wan21_baseline
from baseline.oasis_modal import run_oasis_baseline
from baseline.cosmos_modal import run_cosmos_baseline

# Optimised
from optimised.ltx_video_sta_modal import run_ltx_sta
from optimised.ltx_video_sageattention_modal import run_ltx_sageattention
from optimised.cogvideox_sta_modal import run_cogvideox_sta
from optimised.cogvideox_sageattention_modal import run_cogvideox_sageattention
from optimised.hunyuan_sta_modal import run_hunyuan_sta
from optimised.flux_sageattention_modal import run_flux_sageattention
from optimised.flux_prediT_modal import run_flux_prediT
from optimised.wan21_tempache_modal import run_wan21_tempache
from optimised.oasis_all_modal import run_oasis_all
from optimised.oasis_custom_modal import run_oasis_custom
from optimised.cosmos_sta_modal import run_cosmos_sta
from optimised.cosmos_sageattention_modal import run_cosmos_sageattention
from optimised.cosmos_teacache_modal import run_cosmos_teacache
from optimised.cosmos_prediT_modal import run_cosmos_prediT

# Use the app defined in common.py
from modal_common import app

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_get(result: dict, key: str, default=None):
    """Return result[key] if result is a valid dict without an 'error' key."""
    if not isinstance(result, dict) or "error" in result:
        return default
    return result.get(key, default)


def _speedup(baseline_ms: Optional[float], optimised_ms: Optional[float]) -> str:
    if baseline_ms is None or optimised_ms is None or optimised_ms <= 0:
        return "N/A"
    return f"{baseline_ms / optimised_ms:.2f}x"


def _print_speedup_table(results: dict) -> None:
    """Print a human-readable speedup table from the collected results dict."""
    try:
        from tabulate import tabulate
    except ImportError:
        print("tabulate not available; printing raw results")
        print(json.dumps(results, indent=2))
        return

    rows = []

    # Pairs: (baseline_key, optimised_key, display_name)
    pairs = [
        ("ltx_baseline",          "ltx_sta",             "LTX-Video",          "STA"),
        ("ltx_baseline",          "ltx_sageattention",   "LTX-Video",          "SageAttn2"),
        ("cogvideox_baseline",    "cogvideox_sta",        "CogVideoX-5b",       "STA"),
        ("cogvideox_baseline",    "cogvideox_sageattention", "CogVideoX-5b",    "SageAttn2"),
        ("hunyuan_baseline",      "hunyuan_sta",          "HunyuanVideo",       "STA"),
        ("flux_baseline",         "flux_sageattention",   "FLUX.1-dev",         "SageAttn2"),
        ("flux_baseline",         "flux_prediT",          "FLUX.1-dev",         "PrediT AB-2"),
        ("wan21_baseline",        "wan21_tempache",        "Wan2.1-T2V-14B",    "TeaCache"),
        ("oasis_baseline",        "oasis_all",            "Oasis-500M",         "ThirdParty(SageAttn2+INT4)"),
        ("oasis_baseline",        "oasis_custom",         "Oasis-500M",         "Custom(SageTriton+INT4+AdaLN)"),
        # Cosmos group
        ("cosmos_baseline",       "cosmos_sta",           "Cosmos-7B-V2W",      "STA"),
        ("cosmos_baseline",       "cosmos_sageattention", "Cosmos-7B-V2W",      "SageAttn2"),
        ("cosmos_baseline",       "cosmos_teacache",      "Cosmos-7B-V2W",      "TeaCache"),
        ("cosmos_baseline",       "cosmos_prediT",        "Cosmos-7B-V2W",      "PrediT"),
    ]

    for bl_key, opt_key, model, kernel in pairs:
        bl = results.get(bl_key, {})
        opt = results.get(opt_key, {})
        bl_ms = _safe_get(bl, "latency_ms_mean")
        opt_ms = _safe_get(opt, "latency_ms_mean")
        opt_vram = _safe_get(opt, "vram_gb", "—")
        rows.append([
            model,
            kernel,
            f"{bl_ms:.0f}" if bl_ms else "—",
            f"{opt_ms:.0f}" if opt_ms else "—",
            _speedup(bl_ms, opt_ms),
            f"{opt_vram:.2f}" if isinstance(opt_vram, float) else opt_vram,
        ])

    headers = ["Model", "Kernel", "Baseline (ms)", "Optimised (ms)", "Speedup", "VRAM (GB)"]
    print("\n" + "=" * 80)
    print("WorldServe H100 Benchmark Results")
    print("=" * 80)
    print(tabulate(rows, headers=headers, tablefmt="github"))
    print("=" * 80 + "\n")


def _save_results(results: dict, path: str = "modal/runs/all_results.json") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"Results written to {path}")


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

def _run_batch(batch: list, results: dict) -> None:
    """Spawn up to BATCH_SIZE functions concurrently, wait for all, then save."""
    futures = {}
    for key, fn in batch:
        print(f"  [{key}] spawning ...")
        futures[key] = fn.spawn()

    for key, future in futures.items():
        try:
            results[key] = future.get()
            ms = _safe_get(results[key], "latency_ms_mean")
            print(f"  [{key}] done — {ms:.0f} ms" if ms else f"  [{key}] done (no latency captured)")
        except Exception as exc:
            print(f"  [{key}] FAILED — {exc}")
            results[key] = {"error": str(exc)}

    # Persist after every batch so partial results survive a crash
    _save_results(results)


BATCH_SIZE = 4  # H100s to run concurrently


def _run_list(jobs: list, results: dict) -> None:
    """Run a list of (key, fn) jobs in batches of BATCH_SIZE."""
    for i in range(0, len(jobs), BATCH_SIZE):
        batch = jobs[i : i + BATCH_SIZE]
        keys = [k for k, _ in batch]
        print(f"\n  Batch {i // BATCH_SIZE + 1}: {keys}")
        _run_batch(batch, results)


@app.local_entrypoint()
def main(
    baselines_only: bool = False,
    quick: bool = False,
):
    """
    Run all benchmarks in batches of BATCH_SIZE (default 4) concurrent H100s.

    Args:
        baselines_only: Only run the baseline benchmarks (skip optimised).
        quick:          Only run Oasis baseline + Oasis-all (fast smoke test).
    """
    results: dict = {}

    # -----------------------------------------------------------------------
    # Baselines
    # -----------------------------------------------------------------------
    print("=" * 60)
    print(f"WorldServe — baselines (batches of {BATCH_SIZE}) ...")
    print("=" * 60)

    if quick:
        baseline_jobs = [("oasis_baseline", run_oasis_baseline)]
    else:
        baseline_jobs = [
            ("ltx_baseline",      run_ltx_baseline),
            ("cogvideox_baseline", run_cogvideox_baseline),
            ("hunyuan_baseline",   run_hunyuan_baseline),
            ("flux_baseline",      run_flux_dev_baseline),
            ("wan21_baseline",     run_wan21_baseline),
            ("oasis_baseline",     run_oasis_baseline),
            ("cosmos_baseline",    run_cosmos_baseline),
        ]

    _run_list(baseline_jobs, results)

    if baselines_only:
        _print_speedup_table(results)
        _save_results(results)
        return

    # -----------------------------------------------------------------------
    # Optimised
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"WorldServe — optimised (batches of {BATCH_SIZE}) ...")
    print("=" * 60)

    if quick:
        opt_jobs = [("oasis_all", run_oasis_all), ("oasis_custom", run_oasis_custom)]
    else:
        opt_jobs = [
            ("ltx_sta",                 run_ltx_sta),
            ("ltx_sageattention",       run_ltx_sageattention),
            ("cogvideox_sta",           run_cogvideox_sta),
            ("cogvideox_sageattention", run_cogvideox_sageattention),
            ("hunyuan_sta",             run_hunyuan_sta),
            ("flux_sageattention",      run_flux_sageattention),
            ("flux_prediT",             run_flux_prediT),
            ("wan21_tempache",          run_wan21_tempache),
            ("oasis_all",               run_oasis_all),
            ("oasis_custom",            run_oasis_custom),
            ("cosmos_sta",              run_cosmos_sta),
            ("cosmos_sageattention",    run_cosmos_sageattention),
            ("cosmos_teacache",         run_cosmos_teacache),
            ("cosmos_prediT",           run_cosmos_prediT),
        ]

    _run_list(opt_jobs, results)

    # -----------------------------------------------------------------------
    # Final report
    # -----------------------------------------------------------------------
    _print_speedup_table(results)
    _save_results(results)
