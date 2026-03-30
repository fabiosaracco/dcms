"""DECM solver benchmark — Phase 6.

Generates test networks using the Chung-Lu power-law model
(``k_s_generator_pl``), solves them with the DECM alternating GS-Newton
solver, and prints a comparison table.

The **multi-seed variant** runs *n_seeds* independent realisations per node
count and reports:

* convergence rate (%)
* mean iterations ± std
* mean wall-clock time (s) ± std
* mean peak RAM (MB) ± std
* mean Maximum Relative Error ± std

Usage::

    # Single network, N=50
    python -m src.benchmarks.decm_comparison --n 50 --seed 0

    # Multi-seed comparison
    python -m src.benchmarks.decm_comparison --n 1000 --n_seeds 10

    # Multiple sizes
    python -m src.benchmarks.decm_comparison --sizes 100 500 1000

    # Phase 6 focused benchmark (N=1k, 10 seeds)
    python -m src.benchmarks.decm_comparison --phase6

    # Custom tolerance
    python -m src.benchmarks.decm_comparison --n 200 --tol 1e-5
"""
from __future__ import annotations

import argparse
import math
import signal
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from src.models.decm import DECMModel
from src.solvers.base import SolverResult
from src.utils.wng import k_s_generator_pl


# -------------------------------------------------------------------------
# Defaults
# -------------------------------------------------------------------------
DEFAULT_SIZES: list[int] = [100, 500, 1_000]
DEFAULT_RHO: float = 0.001
DEFAULT_TOL: float = 1e-5
SOLVER_TIMEOUT: float = 900.0


# -------------------------------------------------------------------------
# Timeout helper
# -------------------------------------------------------------------------

class _TimeoutError(Exception):
    """Raised when a solver exceeds its wall-clock budget."""


def _call_with_timeout(fn, timeout_s: float):
    """Call ``fn()`` and raise ``_TimeoutError`` if it exceeds *timeout_s* seconds."""
    if not hasattr(signal, "SIGALRM") or timeout_s <= 0:
        return fn()

    def _handler(signum: int, frame: object) -> None:
        raise _TimeoutError(f"Solver exceeded {timeout_s:.0f}s timeout.")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(max(1, int(timeout_s)))
    try:
        result = fn()
        signal.alarm(0)
        return result
    finally:
        signal.signal(signal.SIGALRM, old_handler)


# -------------------------------------------------------------------------
# Solver descriptor
# -------------------------------------------------------------------------

_METHODS: list[dict] = [
    {
        "name": "θ-Newton Anderson(10)",
        "ic": "degrees",
        "variant": "theta-newton",
        "anderson_depth": 10,
    },
]


# -------------------------------------------------------------------------
# Single-run wrapper
# -------------------------------------------------------------------------

def _run_decm_solver(
    model: DECMModel,
    ic: str,
    variant: str,
    anderson_depth: int,
    tol: float,
    max_iter: int,
    timeout: float,
) -> Optional[SolverResult]:
    """Run one DECM solver and return the SolverResult (or None on timeout)."""
    def _fn():
        return model.solve_tool(
            ic=ic,
            tol=tol,
            max_iter=max_iter,
            max_time=timeout if timeout > 0 else 0,
            variant=variant,
            anderson_depth=anderson_depth,
        )

    try:
        return _call_with_timeout(_fn, timeout)
    except _TimeoutError:
        return None
    except Exception as exc:  # pragma: no cover
        print(f"  [ERROR] {exc}", file=sys.stderr)
        return None


# -------------------------------------------------------------------------
# Single-network benchmark
# -------------------------------------------------------------------------

def run_single(
    N: int,
    seed: int,
    rho: float = DEFAULT_RHO,
    tol: float = DEFAULT_TOL,
    timeout: float = SOLVER_TIMEOUT,
    max_iter: int = 5000,
    verbose: bool = True,
) -> list[dict]:
    """Run all DECM solvers on a single network and return results.

    Args:
        N:        Number of nodes.
        seed:     RNG seed for the network generator.
        rho:      Connection density for the Chung-Lu model.
        tol:      Convergence tolerance.
        timeout:  Per-solver wall-clock timeout in seconds.
        max_iter: Maximum iterations per solver.
        verbose:  Print per-solver results to stdout.

    Returns:
        List of dicts with keys: method, converged, iterations, elapsed_time,
        peak_ram_bytes, mre.
    """
    k, s = k_s_generator_pl(N, rho=rho, seed=seed)
    k_out = k[:N].numpy()
    k_in = k[N:].numpy()
    s_out = s[:N].numpy()
    s_in = s[N:].numpy()

    model = DECMModel(k_out, k_in, s_out, s_in)

    if verbose:
        print(f"\nDECM Benchmark | N={N} | seed={seed}")
        print("=" * 70)
        print(f"{'Method':<35} {'Conv':>5} {'Iters':>7} {'Time(s)':>9} {'MRE':>12}")
        print("-" * 70)

    results = []
    for m in _METHODS:
        result = _run_decm_solver(
            model,
            ic=m["ic"],
            variant=m["variant"],
            anderson_depth=m["anderson_depth"],
            tol=tol,
            max_iter=max_iter,
            timeout=timeout,
        )

        if result is None:
            rec = {
                "method": m["name"],
                "converged": False,
                "iterations": -1,
                "elapsed_time": timeout,
                "peak_ram_bytes": 0,
                "mre": float("nan"),
            }
        else:
            mre = model.max_relative_error(result.theta)
            rec = {
                "method": m["name"],
                "converged": result.converged,
                "iterations": result.iterations,
                "elapsed_time": result.elapsed_time,
                "peak_ram_bytes": result.peak_ram_bytes,
                "mre": mre,
            }

        results.append(rec)

        if verbose:
            conv_str = "YES" if rec["converged"] else "NO "
            iters_str = str(rec["iterations"]) if rec["iterations"] >= 0 else "TIMEOUT"
            time_str = f"{rec['elapsed_time']:.2f}"
            mre_str = f"{rec['mre']:.3e}" if math.isfinite(rec["mre"]) else "N/A"
            print(
                f"{rec['method']:<35} {conv_str:>5} {iters_str:>7} {time_str:>9} {mre_str:>12}"
            )

    if verbose:
        print("-" * 70)

    return results


# -------------------------------------------------------------------------
# Multi-seed benchmark
# -------------------------------------------------------------------------

def run_multi_seed(
    N: int,
    n_seeds: int = 10,
    rho: float = DEFAULT_RHO,
    tol: float = DEFAULT_TOL,
    timeout: float = SOLVER_TIMEOUT,
    max_iter: int = 5000,
    verbose: bool = True,
) -> dict[str, list[dict]]:
    """Run all solvers over *n_seeds* networks and collect aggregate statistics.

    Args:
        N:        Number of nodes.
        n_seeds:  Number of independent network realisations.
        rho:      Connection density.
        tol:      Convergence tolerance.
        timeout:  Per-solver wall-clock timeout.
        max_iter: Maximum iterations.
        verbose:  Print progress to stdout.

    Returns:
        Dict mapping method name to a list of per-seed result dicts.
    """
    per_method: dict[str, list[dict]] = {m["name"]: [] for m in _METHODS}

    for seed in range(n_seeds):
        if verbose:
            print(f"  seed {seed + 1}/{n_seeds}", end="\r", flush=True)
        recs = run_single(N, seed, rho=rho, tol=tol, timeout=timeout,
                          max_iter=max_iter, verbose=False)
        for rec in recs:
            per_method[rec["method"]].append(rec)

    if verbose:
        print()  # clear the progress line

    return per_method


def _print_summary_table(
    per_method: dict[str, list[dict]],
    N: int,
    n_seeds: int,
    tol: float,
) -> None:
    """Print a formatted multi-seed summary table.

    Args:
        per_method: Dict mapping method name → list of per-seed result dicts.
        N:          Number of nodes.
        n_seeds:    Number of seeds.
        tol:        Convergence tolerance used.
    """
    header = f"DECM Benchmark | N={N} | {n_seeds} seeds | tol={tol:.0e}"
    sep = "=" * 85
    print(f"\n{header}")
    print(sep)
    print(
        f"{'Method':<35} | {'Conv%':>5} | {'Iters':>14} | {'Time (s)':>14} | {'MRE':>14}"
    )
    print("-" * 85)

    for method_name, recs in per_method.items():
        n_conv = sum(1 for r in recs if r["converged"])
        conv_pct = 100.0 * n_conv / len(recs) if recs else 0.0

        iters_conv = [r["iterations"] for r in recs if r["converged"] and r["iterations"] >= 0]
        times = [r["elapsed_time"] for r in recs]
        mres = [r["mre"] for r in recs if r["converged"] and math.isfinite(r["mre"])]

        def _stats(vals):
            if not vals:
                return "N/A"
            mu = float(np.mean(vals))
            sd = float(np.std(vals))
            return f"{mu:.1f} ± {sd:.1f}"

        def _mre_stats(vals):
            if not vals:
                return "N/A"
            mu = float(np.mean(vals))
            sd = float(np.std(vals))
            return f"{mu:.2e} ± {sd:.2e}"

        print(
            f"{method_name:<35} | {conv_pct:>4.0f}% | {_stats(iters_conv):>14} "
            f"| {_stats(times):>14} | {_mre_stats(mres):>14}"
        )

    print("-" * 85)


# -------------------------------------------------------------------------
# CLI entry point
# -------------------------------------------------------------------------

def main() -> None:
    """Command-line interface for the DECM benchmark."""
    parser = argparse.ArgumentParser(
        description="DECM solver benchmark (Phase 6)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n", type=int, default=100, help="Number of nodes.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (single run).")
    parser.add_argument("--n_seeds", type=int, default=1, help="Number of seeds for multi-seed run.")
    parser.add_argument("--sizes", type=int, nargs="+", default=None, help="List of N values.")
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL, help="Convergence tolerance.")
    parser.add_argument("--timeout", type=float, default=SOLVER_TIMEOUT, help="Per-solver timeout (s).")
    parser.add_argument("--max_iter", type=int, default=5000, help="Maximum iterations.")
    parser.add_argument("--rho", type=float, default=DEFAULT_RHO, help="Connection density.")
    parser.add_argument(
        "--phase6",
        action="store_true",
        help="Run Phase 6 benchmark: N=1000, 10 seeds.",
    )
    args = parser.parse_args()

    if args.phase6:
        sizes = [1_000]
        n_seeds = 10
    elif args.sizes is not None:
        sizes = args.sizes
        n_seeds = args.n_seeds
    elif args.n_seeds > 1:
        sizes = [args.n]
        n_seeds = args.n_seeds
    else:
        sizes = [args.n]
        n_seeds = 1

    for N in sizes:
        if n_seeds == 1:
            run_single(N, args.seed, rho=args.rho, tol=args.tol,
                       timeout=args.timeout, max_iter=args.max_iter, verbose=True)
        else:
            print(f"\nRunning {n_seeds} seeds for N={N} …")
            per_method = run_multi_seed(
                N, n_seeds=n_seeds, rho=args.rho, tol=args.tol,
                timeout=args.timeout, max_iter=args.max_iter, verbose=True,
            )
            _print_summary_table(per_method, N=N, n_seeds=n_seeds, tol=args.tol)


if __name__ == "__main__":
    main()
