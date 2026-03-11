"""DWCM solver comparison benchmark — Phase 4.

Tests all applicable DWCM solvers across 10 random networks for each of
four network sizes (N = 1 000, 5 000, 10 000, 50 000) and prints a
statistics table reporting:

* **Success rate** — fraction of runs where max constraint error < ``tol``.
* **Average MRE** (mean relative error) ± 2σ — only over converged runs.
* **Average wall-clock time** ± 2σ.
* **Average peak RAM** ± 2σ.
* **Average number of iterations** ± 2σ — only over converged runs.

Solvers tested
--------------
* Fixed-point Gauss-Seidel  α = 1.0  (GS1)
* Fixed-point Gauss-Seidel  α = 0.5  (GS05)
* Fixed-point Jacobi        α = 1.0  (Jacobi)
* L-BFGS (m = 20)           — always applicable
* LM diagonal               — always applicable (O(N) RAM)
* Newton (exact Jacobian)   — skipped for N > ``NEWTON_N_MAX``
* Broyden (rank-1 J)        — skipped for N > ``NEWTON_N_MAX``
* LM (full Jacobian)        — skipped for N > ``NEWTON_N_MAX``

Usage::

    python -m src.benchmarks.dwcm_comparison          # default settings
    python -m src.benchmarks.dwcm_comparison --sizes 1000 5000
    python -m src.benchmarks.dwcm_comparison --n_seeds 5 --tol 1e-6
    python -m src.benchmarks.dwcm_comparison --sizes 1000 --seed 42

Reference: Phase 4 of the AGENTS.md operational plan.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from src.models.dwcm import DWCMModel
from src.solvers.base import SolverResult
from src.solvers.fixed_point_dwcm import solve_fixed_point_dwcm
from src.solvers.quasi_newton import solve_lbfgs
from src.solvers.newton import solve_newton
from src.solvers.broyden import solve_broyden
from src.solvers.levenberg_marquardt import solve_lm
from src.utils.wng import k_s_generator_pl

# ---------------------------------------------------------------------------
# Scaling thresholds
# ---------------------------------------------------------------------------

# Newton, Broyden, full-Jacobian LM need O(N²) RAM.
NEWTON_N_MAX: int = 2_000

# Default network sizes to benchmark
DEFAULT_SIZES: list[int] = [1_000, 5_000, 10_000, 50_000]

# Number of independent network realisations per size
DEFAULT_N_SEEDS: int = 10

# Connection density for the Chung-Lu generator
DEFAULT_RHO: float = 0.05

# Convergence tolerance used by all solvers
DEFAULT_TOL: float = 1e-6

# Per-solver wall-clock time limit (seconds)
SOLVER_TIMEOUT: float = 300.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_feasible_dwcm(s_out: np.ndarray, s_in: np.ndarray) -> bool:
    """Return True if the strength sequence is non-trivially feasible.

    Args:
        s_out: Out-strength sequence, shape (N,).
        s_in:  In-strength sequence, shape (N,).

    Returns:
        True when both sequences have at least one positive entry.
    """
    return bool(s_out.max() > 0 and s_in.max() > 0)


def _make_solvers(
    model: DWCMModel,
    theta0: torch.Tensor,
    tol: float,
) -> list[tuple[str, Callable[[], SolverResult]]]:
    """Return a list of (name, callable) solver pairs for *model*.

    Methods requiring O(N²) RAM are omitted for large N.

    Args:
        model:  The DWCMModel instance for this network.
        theta0: Initial parameter vector.
        tol:    Convergence tolerance.

    Returns:
        Ordered list of ``(name, solver_callable)`` pairs.
    """
    N = model.N
    solvers: list[tuple[str, Callable[[], SolverResult]]] = []

    # Fixed-point variants (always applicable; chunked for large N)
    solvers.append((
        "FP GS α=1.0",
        lambda: solve_fixed_point_dwcm(
            model.residual, theta0, model.s_out, model.s_in,
            tol=tol, damping=1.0, variant="gauss-seidel",
        ),
    ))
    solvers.append((
        "FP GS α=0.5",
        lambda: solve_fixed_point_dwcm(
            model.residual, theta0, model.s_out, model.s_in,
            tol=tol, damping=0.5, variant="gauss-seidel",
        ),
    ))
    solvers.append((
        "FP Jacobi α=1.0",
        lambda: solve_fixed_point_dwcm(
            model.residual, theta0, model.s_out, model.s_in,
            tol=tol, damping=1.0, variant="jacobi",
        ),
    ))

    # L-BFGS (always applicable; chunked for large N)
    solvers.append((
        "L-BFGS (m=20)",
        lambda: solve_lbfgs(
            model.residual, theta0, tol=tol, m=20,
            neg_loglik_fn=model.neg_log_likelihood,
        ),
    ))

    # LM diagonal (always applicable — O(N) RAM)
    solvers.append((
        "LM diag",
        lambda: solve_lm(
            model.residual, model.jacobian, theta0, tol=tol,
            diagonal_only=True,
        ),
    ))

    # O(N²) methods — only for small N
    if N <= NEWTON_N_MAX:
        solvers.append((
            "Newton (exact J)",
            lambda: solve_newton(
                model.residual, model.jacobian, theta0, tol=tol,
            ),
        ))
        solvers.append((
            "Broyden (rank-1 J)",
            lambda: solve_broyden(
                model.residual, model.jacobian, theta0, tol=tol,
            ),
        ))
        solvers.append((
            "LM full J",
            lambda: solve_lm(
                model.residual, model.jacobian, theta0, tol=tol,
                diagonal_only=False,
            ),
        ))

    return solvers


# ---------------------------------------------------------------------------
# Single-seed run
# ---------------------------------------------------------------------------

def _run_single_seed(
    N: int,
    seed: int,
    rho: float,
    tol: float,
    timeout: float,
) -> Optional[list[dict]]:
    """Run all applicable solvers on one network realisation.

    Args:
        N:       Number of nodes.
        seed:    Random seed.
        rho:     Edge density for the Chung-Lu generator.
        tol:     Convergence tolerance.
        timeout: Per-solver wall-clock time limit.

    Returns:
        List of result dicts, or ``None`` if the network is infeasible.
    """
    k, s = k_s_generator_pl(N, rho=rho, seed=seed)
    s_out = s[:N].numpy().astype(float)
    s_in = s[N:].numpy().astype(float)

    if not _is_feasible_dwcm(s_out, s_in):
        return None

    model = DWCMModel(s_out, s_in)
    theta0 = model.initial_theta("strengths")
    records: list[dict] = []

    for name, fn in _make_solvers(model, theta0, tol):
        t_start = time.perf_counter()
        try:
            result_sr: SolverResult = fn()
            elapsed = time.perf_counter() - t_start
            if elapsed > timeout:
                status = "TIMEOUT"
                mre = float("nan")
                iters = result_sr.iterations
                ram_mb = result_sr.peak_ram_bytes / 1024 / 1024
            else:
                mre = model.mean_relative_error(result_sr.theta)
                status = "OK" if result_sr.converged else "NO-CONV"
                iters = result_sr.iterations
                ram_mb = result_sr.peak_ram_bytes / 1024 / 1024
        except (MemoryError, RuntimeError) as exc:
            elapsed = time.perf_counter() - t_start
            exc_str = str(exc)
            if "out of memory" in exc_str.lower() or "alloc" in exc_str.lower():
                status = "OOM"
            elif isinstance(exc, MemoryError):
                status = "OOM"
            else:
                status = "ERR"
            mre = float("nan")
            iters = 0
            ram_mb = float("nan")
        except Exception:
            elapsed = time.perf_counter() - t_start
            status = "ERR"
            mre = float("nan")
            iters = 0
            ram_mb = float("nan")

        records.append(dict(
            method=name,
            N=N,
            seed=seed,
            status=status,
            converged=(status == "OK"),
            iterations=iters,
            mre=mre,
            elapsed=elapsed,
            peak_ram_mb=ram_mb,
        ))

    return records


# ---------------------------------------------------------------------------
# Multi-seed benchmark for one size
# ---------------------------------------------------------------------------

def run_dwcm_benchmark(
    N: int,
    n_seeds: int = DEFAULT_N_SEEDS,
    rho: float = DEFAULT_RHO,
    tol: float = DEFAULT_TOL,
    timeout: float = SOLVER_TIMEOUT,
    start_seed: int = 0,
    verbose: bool = True,
) -> list[dict]:
    """Run DWCM solvers on *n_seeds* random networks of size *N*.

    Skips infeasible seeds (where all strengths are zero), trying up to
    ``n_seeds * 20`` candidates before raising ``RuntimeError``.

    Args:
        N:          Number of nodes.
        n_seeds:    Target number of valid network realisations.
        rho:        Edge density.
        tol:        Convergence tolerance.
        timeout:    Per-solver time limit (seconds).
        start_seed: First seed to try.
        verbose:    If True, print per-seed progress.

    Returns:
        List of result dicts (one per solver per seed).

    Raises:
        RuntimeError: If not enough feasible networks are found.
    """
    all_records: list[dict] = []
    valid_count = 0
    candidate_seed = start_seed
    max_attempts = n_seeds * 20

    if verbose:
        print(f"\n{'='*74}")
        print(
            f"DWCM Benchmark  |  N={N:,}  |  {n_seeds} seeds  |  "
            f"rho={rho}  |  tol={tol:.0e}"
        )
        print(f"{'='*74}")

    while valid_count < n_seeds:
        if (candidate_seed - start_seed) >= max_attempts:
            raise RuntimeError(
                f"Could not find {n_seeds} feasible DWCM networks in "
                f"{max_attempts} attempts (N={N})."
            )

        records = _run_single_seed(N, candidate_seed, rho, tol, timeout)
        if records is None:
            candidate_seed += 1
            continue

        all_records.extend(records)
        if verbose:
            print(f"  seed={candidate_seed}", end="")
            for r in records:
                tag = "✓" if r["converged"] else ("✗ " + r["status"])
                print(f"  [{r['method']}: {tag}]", end="")
            print()

        valid_count += 1
        candidate_seed += 1

    return all_records


# ---------------------------------------------------------------------------
# Statistics helper
# ---------------------------------------------------------------------------

def _stats(values: list[float]) -> tuple[float, float]:
    """Return (mean, 2*std) for *values*, or (nan, nan) if empty.

    Args:
        values: List of finite float values.

    Returns:
        ``(mean, 2*std)`` tuple.
    """
    finite = [v for v in values if np.isfinite(v)]
    if not finite:
        return float("nan"), float("nan")
    arr = np.array(finite, dtype=float)
    return float(arr.mean()), float(2.0 * arr.std(ddof=0))


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _print_summary_table(
    all_records: list[dict],
    sizes: list[int],
    tol: float,
    timeout: float,
) -> None:
    """Print the aggregate statistics table for all sizes and methods.

    For each (method, N) combination, reports:

    * **OK%** — percentage of converged runs.
    * **MRE** — mean ± 2σ of the mean relative error (converged runs only).
    * **Time(s)** — mean ± 2σ of wall-clock time (all runs).
    * **RAM(MB)** — mean ± 2σ of peak RAM (all runs).
    * **Steps** — mean ± 2σ of iterations (converged runs only).

    Args:
        all_records: Flat list of result dicts from all seeds and sizes.
        sizes:       List of network sizes benchmarked.
        tol:         Convergence tolerance used.
        timeout:     Per-solver time limit used.
    """
    # Collect all method names (in order of first appearance)
    method_order: list[str] = []
    seen: set[str] = set()
    for r in all_records:
        if r["method"] not in seen:
            method_order.append(r["method"])
            seen.add(r["method"])

    # Group records by (method, N)
    from collections import defaultdict
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in all_records:
        groups[(r["method"], r["N"])].append(r)

    print(f"\n{'='*100}")
    print(f"{'DWCM BENCHMARK SUMMARY':^100}")
    print(f"  tol={tol:.0e}  |  timeout={timeout:.0f}s  |  "
          f"10 random networks per size")
    print(f"  Intervals are mean ± 2σ  (computed over converged runs for MRE & Steps)")
    print(f"{'='*100}")

    for N in sizes:
        col_method = 20
        col_stat = 28
        print(f"\n  N = {N:,}")
        print(f"  {'Method':<{col_method}} "
              f"{'OK%':>6}  "
              f"{'MRE (mean±2σ)':>{col_stat}}  "
              f"{'Time(s) (mean±2σ)':>{col_stat}}  "
              f"{'RAM(MB) (mean±2σ)':>{col_stat}}  "
              f"{'Steps (mean±2σ)':>{col_stat}}")
        print("  " + "-" * (col_method + 4 * (col_stat + 2) + 10))

        for method in method_order:
            key = (method, N)
            if key not in groups:
                continue
            recs = groups[key]
            total = len(recs)
            ok_recs = [r for r in recs if r["converged"]]
            ok_pct = 100.0 * len(ok_recs) / total if total > 0 else 0.0

            mre_mean, mre_2s = _stats([r["mre"] for r in ok_recs])
            time_mean, time_2s = _stats([r["elapsed"] for r in recs])
            ram_mean, ram_2s = _stats([r["peak_ram_mb"] for r in recs])
            steps_mean, steps_2s = _stats(
                [float(r["iterations"]) for r in ok_recs]
            )

            def _fmt(mean: float, two_sigma: float, fmt: str = ".3f") -> str:
                if not np.isfinite(mean):
                    return "—"
                return f"{mean:{fmt}} ± {two_sigma:{fmt}}"

            mre_str = _fmt(mre_mean, mre_2s, ".3e")
            time_str = _fmt(time_mean, time_2s, ".2f")
            ram_str = _fmt(ram_mean, ram_2s, ".1f")
            steps_str = _fmt(steps_mean, steps_2s, ".0f")

            print(
                f"  {method:<{col_method}} "
                f"{ok_pct:>5.0f}%  "
                f"{mre_str:>{col_stat}}  "
                f"{time_str:>{col_stat}}  "
                f"{ram_str:>{col_stat}}  "
                f"{steps_str:>{col_stat}}"
            )
        print()

    print(f"{'='*100}")
    print("Legend:  OK% = fraction of runs that converged  |  — = no converged runs")
    print(f"         Intervals show mean ± 2σ across the {DEFAULT_N_SEEDS} realisations")
    print()


# ---------------------------------------------------------------------------
# Full multi-size comparison
# ---------------------------------------------------------------------------

def run_dwcm_comparison(
    sizes: list[int] = DEFAULT_SIZES,
    n_seeds: int = DEFAULT_N_SEEDS,
    rho: float = DEFAULT_RHO,
    tol: float = DEFAULT_TOL,
    timeout: float = SOLVER_TIMEOUT,
    start_seed: int = 0,
) -> None:
    """Run the DWCM benchmark for all *sizes* and print a summary table.

    Args:
        sizes:      List of node counts to benchmark.
        n_seeds:    Number of random network realisations per size.
        rho:        Edge density for the Chung-Lu generator.
        tol:        Convergence tolerance for all solvers.
        timeout:    Per-solver time limit in seconds.
        start_seed: First random seed to try.
    """
    all_records: list[dict] = []
    for N in sizes:
        records = run_dwcm_benchmark(
            N=N,
            n_seeds=n_seeds,
            rho=rho,
            tol=tol,
            timeout=timeout,
            start_seed=start_seed,
            verbose=True,
        )
        all_records.extend(records)

    _print_summary_table(all_records, sizes, tol, timeout)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DWCM solver performance benchmark"
    )
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=DEFAULT_SIZES,
        metavar="N",
        help="Network sizes to benchmark (default: 1000 5000 10000 50000)",
    )
    parser.add_argument(
        "--n_seeds", type=int, default=DEFAULT_N_SEEDS,
        help=f"Number of realisations per size (default: {DEFAULT_N_SEEDS})",
    )
    parser.add_argument(
        "--rho", type=float, default=DEFAULT_RHO,
        help=f"Edge density (default: {DEFAULT_RHO})",
    )
    parser.add_argument(
        "--tol", type=float, default=DEFAULT_TOL,
        help=f"Convergence tolerance (default: {DEFAULT_TOL})",
    )
    parser.add_argument(
        "--timeout", type=float, default=SOLVER_TIMEOUT,
        help=f"Per-solver time limit in seconds (default: {SOLVER_TIMEOUT})",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Starting random seed (default: 0)",
    )
    args = parser.parse_args()

    run_dwcm_comparison(
        sizes=args.sizes,
        n_seeds=args.n_seeds,
        rho=args.rho,
        tol=args.tol,
        timeout=args.timeout,
        start_seed=args.seed,
    )
