"""Phase 3 — DCM Scaling Benchmark.

Tests all applicable DCM solvers across network sizes from N=1 000 to N=50 000
and prints a comparison table.

Usage::

    python -m src.benchmarks.dcm_scaling               # default sizes
    python -m src.benchmarks.dcm_scaling --sizes 1000 5000
    python -m src.benchmarks.dcm_scaling --sizes 10000 --seed 7
    python -m src.benchmarks.dcm_scaling --rho 0.05    # sparser networks

Reference: AGENTS.md, Phase 3.
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

from dcms.models.dcm import DCMModel
from dcms.solvers.base import SolverResult
from dcms.solvers.fixed_point_dcm import solve_fixed_point_dcm
from dcms.utils.wng import k_s_generator_pl

# ---------------------------------------------------------------------------
# Scaling thresholds
# ---------------------------------------------------------------------------

# Default network sizes to benchmark
DEFAULT_SIZES: list[int] = [1_000, 5_000, 10_000, 50_000]

# Connection density (rho) for the Chung-Lu generator
DEFAULT_RHO: float = 0.05

# Convergence tolerance used by all solvers in this benchmark
DEFAULT_TOL: float = 1e-6

# Maximum solver wall-clock time (seconds) — skip if exceeded
SOLVER_TIMEOUT: float = 300.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_feasible(k_out: np.ndarray, k_in: np.ndarray) -> bool:
    """Return True if the degree sequence is feasible for the DCM.

    Args:
        k_out: Out-degree sequence, shape (N,).
        k_in:  In-degree sequence, shape (N,).

    Returns:
        True when max degree < N (no node requires more connections than available).
    """
    N = len(k_out)
    return bool(k_out.max() < N and k_in.max() < N)


def _find_feasible_seed(
    N: int,
    rho: float,
    start_seed: int = 0,
    max_attempts: int = 50,
) -> Optional[tuple[np.ndarray, np.ndarray, int]]:
    """Find a feasible power-law degree sequence for the given N and rho.

    Args:
        N:            Number of nodes.
        rho:          Target connection density.
        start_seed:   First seed to try.
        max_attempts: Maximum number of candidate seeds.

    Returns:
        ``(k_out, k_in, seed)`` if a feasible network is found, else ``None``.
    """
    for attempt in range(max_attempts):
        seed = start_seed + attempt
        k, _ = k_s_generator_pl(N, rho=rho, seed=seed)
        k_out = k[:N].numpy().astype(float)
        k_in = k[N:].numpy().astype(float)
        if _is_feasible(k_out, k_in):
            return k_out, k_in, seed
    return None


def _make_solvers(
    model: DCMModel,
    theta0: torch.Tensor,
    tol: float,
) -> list[tuple[str, Callable[[], SolverResult]]]:
    """Return a list of (name, callable) pairs for the two kept methods.

    Args:
        model:  The DCMModel instance for this network.
        theta0: Initial parameter vector.
        tol:    Convergence tolerance.

    Returns:
        Ordered list of ``(name, solver_callable)`` pairs.
    """
    solvers: list[tuple[str, Callable[[], SolverResult]]] = []

    solvers.append((
        "FP-GS Anderson(10)",
        lambda: solve_fixed_point_dcm(
            model.residual, theta0, model.k_out, model.k_in,
            tol=tol, variant="gauss-seidel", anderson_depth=10,
        ),
    ))

    solvers.append((
        "θ-Newton Anderson(10)",
        lambda: solve_fixed_point_dcm(
            model.residual, theta0, model.k_out, model.k_in,
            tol=tol, variant="theta-newton", anderson_depth=10,
        ),
    ))

    return solvers


# ---------------------------------------------------------------------------
# Single-size benchmark
# ---------------------------------------------------------------------------

def run_scaling_benchmark(
    N: int,
    rho: float = DEFAULT_RHO,
    seed: int = 0,
    tol: float = DEFAULT_TOL,
    timeout: float = SOLVER_TIMEOUT,
    verbose: bool = True,
) -> list[dict]:
    """Run all applicable solvers on a single network of size N.

    .. note::
        The ``timeout`` parameter is *retrospective*: a solver is marked
        ``TIMEOUT`` only after it completes and its elapsed time is found to
        exceed the limit.  It does **not** interrupt a running solver.  For
        a hard interrupt, run each solver in a separate process.

    Args:
        N:       Number of nodes.
        rho:     Target edge density for the Chung-Lu generator.
        seed:    Random seed (first seed to try; incremented if infeasible).
        tol:     Convergence tolerance for all solvers.
        timeout: Per-solver wall-clock time limit in seconds.  Solvers that
                 finish but exceed this limit are reported as ``TIMEOUT``.
        verbose: If True, print the comparison table.

    Returns:
        List of result dicts with keys: method, N, converged, iterations,
        max_err, elapsed, peak_ram_mb, status.

    Raises:
        RuntimeError: If no feasible network is found after 50 attempts.
    """
    result = _find_feasible_seed(N, rho, start_seed=seed)
    if result is None:
        raise RuntimeError(
            f"Could not find a feasible network for N={N}, rho={rho} in 50 attempts."
        )
    k_out, k_in, actual_seed = result

    model = DCMModel(k_out, k_in)
    theta0 = model.initial_theta("degrees")

    if verbose:
        print(f"\n{'='*74}")
        print(
            f"DCM Scaling Benchmark  |  N={N:,}  |  rho={rho}  |  "
            f"seed={actual_seed}  |  tol={tol:.0e}"
        )
        print(f"{'='*74}")
        print(f"  k_out  min={k_out.min():.0f}  max={k_out.max():.0f}  "
              f"mean={k_out.mean():.1f}")
        print(f"  k_in   min={k_in.min():.0f}  max={k_in.max():.0f}  "
              f"mean={k_in.mean():.1f}")
        print()
        col = [26, 8, 8, 12, 10, 12]
        header = (
            f"{'Method':<{col[0]}} {'Status':>{col[1]}} {'Iters':>{col[2]}} "
            f"{'MaxErr':>{col[3]}} {'Time(s)':>{col[4]}} {'RAM(MB)':>{col[5]}}"
        )
        print(header)
        print("-" * sum(col))

    records: list[dict] = []
    for name, fn in _make_solvers(model, theta0, tol):
        t_start = time.perf_counter()
        try:
            result_sr: SolverResult = fn()
            elapsed = time.perf_counter() - t_start
            if elapsed > timeout:
                status = "TIMEOUT"
                max_err = float("nan")
                iters = result_sr.iterations
                ram_mb = result_sr.peak_ram_bytes / 1024 / 1024
            else:
                max_err = model.constraint_error(result_sr.theta)
                status = "OK" if result_sr.converged else "NO-CONV"
                iters = result_sr.iterations
                ram_mb = result_sr.peak_ram_bytes / 1024 / 1024
        except MemoryError:
            elapsed = time.perf_counter() - t_start
            status = "OOM"
            max_err = float("nan")
            iters = 0
            ram_mb = float("nan")
        except RuntimeError as exc:
            elapsed = time.perf_counter() - t_start
            exc_str = str(exc)
            # PyTorch raises RuntimeError for GPU/CPU allocation failures
            if "out of memory" in exc_str.lower() or "alloc" in exc_str.lower():
                status = "OOM"
            else:
                status = "ERR"
            max_err = float("nan")
            iters = 0
            ram_mb = float("nan")
            if verbose and status == "ERR":
                print(f"  [{name}] RuntimeError: {exc}")
        except Exception as exc:
            elapsed = time.perf_counter() - t_start
            status = "ERR"
            max_err = float("nan")
            iters = 0
            ram_mb = float("nan")
            if verbose:
                print(f"  [{name}] Exception: {exc}")

        records.append(dict(
            method=name,
            N=N,
            status=status,
            converged=(status == "OK"),
            iterations=iters,
            max_err=max_err,
            elapsed=elapsed,
            peak_ram_mb=ram_mb,
        ))

        if verbose:
            col = [26, 8, 8, 12, 10, 12]
            err_str = f"{max_err:.3e}" if np.isfinite(max_err) else "  —"
            ram_str = f"{ram_mb:.1f}" if np.isfinite(ram_mb) else "  —"
            print(
                f"{name:<{col[0]}} {status:>{col[1]}} {iters:>{col[2]}} "
                f"{err_str:>{col[3]}} {elapsed:>{col[4]}.3f} "
                f"{ram_str:>{col[5]}}"
            )

    if verbose:
        print()

    return records


# ---------------------------------------------------------------------------
# Multi-size comparison table
# ---------------------------------------------------------------------------

def run_scaling_comparison(
    sizes: list[int] = DEFAULT_SIZES,
    rho: float = DEFAULT_RHO,
    seed: int = 0,
    tol: float = DEFAULT_TOL,
    timeout: float = SOLVER_TIMEOUT,
) -> None:
    """Run the scaling benchmark for each size in *sizes* and print a summary.

    Args:
        sizes:   List of node counts to benchmark.
        rho:     Target edge density.
        seed:    Base random seed.
        tol:     Convergence tolerance for all solvers.
        timeout: Per-solver time limit in seconds.
    """
    all_records: list[dict] = []
    for N in sizes:
        records = run_scaling_benchmark(N=N, rho=rho, seed=seed, tol=tol,
                                        timeout=timeout, verbose=True)
        all_records.extend(records)

    # -----------------------------------------------------------------------
    # Aggregate summary table
    # -----------------------------------------------------------------------
    print(f"\n{'='*74}")
    print(f"{'SCALING SUMMARY':^74}")
    print(f"{'='*74}")
    col_w = [26] + [max(len(str(N)), 8) + 2 for N in sizes]
    header = f"{'Method':<{col_w[0]}}" + "".join(
        f"  {'N='+str(N):^{col_w[i+1]-2}}" for i, N in enumerate(sizes)
    )
    print(header)
    print("-" * sum(col_w))

    # Group by method name
    all_methods: list[str] = []
    seen: set[str] = set()
    for r in all_records:
        if r["method"] not in seen:
            all_methods.append(r["method"])
            seen.add(r["method"])

    # Build lookup: (method, N) → record
    lookup: dict[tuple[str, int], dict] = {
        (r["method"], r["N"]): r for r in all_records
    }

    for method in all_methods:
        row = f"{method:<{col_w[0]}}"
        for i, N in enumerate(sizes):
            if (method, N) in lookup:
                r = lookup[(method, N)]
                if r["status"] in ("OOM", "ERR", "TIMEOUT"):
                    cell = r["status"]
                elif r["status"] == "OK":
                    cell = f"✓ {r['elapsed']:.1f}s"
                else:
                    cell = f"✗ {r['elapsed']:.1f}s"
            else:
                cell = "—"
            row += f"  {cell:^{col_w[i+1]-2}}"
        print(row)

    print()
    print("Legend: ✓ = converged  ✗ = not converged  — = not applicable")
    print(f"        OOM = out of memory  TIMEOUT = exceeded {timeout:.0f}s limit")
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DCM scaling benchmark (Phase 3)")
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=DEFAULT_SIZES,
        metavar="N", help="Network sizes to benchmark (default: 1000 5000 10000 50000)",
    )
    parser.add_argument(
        "--rho", type=float, default=DEFAULT_RHO,
        help=f"Edge density for the Chung-Lu generator (default: {DEFAULT_RHO})",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Base random seed (default: 0)",
    )
    parser.add_argument(
        "--tol", type=float, default=DEFAULT_TOL,
        help=f"Solver convergence tolerance (default: {DEFAULT_TOL})",
    )
    parser.add_argument(
        "--timeout", type=float, default=SOLVER_TIMEOUT,
        help=f"Per-solver time limit in seconds (default: {SOLVER_TIMEOUT})",
    )
    args = parser.parse_args()

    run_scaling_comparison(
        sizes=args.sizes,
        rho=args.rho,
        seed=args.seed,
        tol=args.tol,
        timeout=args.timeout,
    )
