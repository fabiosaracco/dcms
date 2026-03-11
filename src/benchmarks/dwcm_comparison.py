"""DWCM solver comparison benchmark — Phase 4.

Generates test networks using the Chung-Lu power-law model (``k_s_generator_pl``),
then runs all applicable DWCM solvers and prints a comparison table.

The **multi-seed variant** runs *n_seeds* independent network realisations per
node count and reports aggregate statistics:

* mean calculation time ± 2σ
* mean peak RAM usage ± 2σ
* mean iteration count ± 2σ
* mean Maximum Relative Error at convergence ± 2σ

Usage::

    # Single network, small N
    python -m src.benchmarks.dwcm_comparison --n 50 --seed 42

    # Multi-seed comparison at several sizes (Phase 4 full run)
    python -m src.benchmarks.dwcm_comparison --sizes 1000 5000 10000 50000

    # Quick validation (fewer seeds)
    python -m src.benchmarks.dwcm_comparison --n 100 --n_seeds 3

Memory thresholds
-----------------
* N > ``NEWTON_N_MAX``       → skip Newton and Broyden (O(N²) Jacobian).
* N > ``FULL_JAC_LM_N_MAX`` → use diagonal-only LM instead of full-Jacobian LM.
* N > ``_LARGE_N_THRESHOLD`` (from DWCMModel) → chunked residual / fixed-point.
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

from src.models.dwcm import DWCMModel, _ETA_MIN, _ETA_MAX
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

# Newton and Broyden need the full N×N Jacobian (O(N²) RAM).
NEWTON_N_MAX: int = 2_000

# Full-Jacobian LM shares the same RAM cost as Newton.
FULL_JAC_LM_N_MAX: int = 2_000

# Default network sizes to benchmark
DEFAULT_SIZES: list[int] = [1_000, 5_000, 10_000, 50_000]

# Connection density (rho) for the Chung-Lu generator
DEFAULT_RHO: float = 0.05

# Convergence tolerance used by all solvers in this benchmark
DEFAULT_TOL: float = 1e-6

# Maximum solver wall-clock time (seconds) — skip if exceeded
SOLVER_TIMEOUT: float = 300.0

# Number of seeds for multi-seed comparison
DEFAULT_N_SEEDS: int = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_clamped_residual(model: DWCMModel) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a residual function that clamps θ to valid DWCM range before evaluating."""
    def fn(theta: torch.Tensor) -> torch.Tensor:
        theta_safe = theta.clamp(_ETA_MIN, _ETA_MAX)
        return model.residual(theta_safe)
    return fn


def _make_clamped_nll(model: DWCMModel) -> Callable[[torch.Tensor], float]:
    """Return a neg_log_likelihood function that clamps θ to valid DWCM range."""
    def fn(theta: torch.Tensor) -> float:
        theta_safe = theta.clamp(_ETA_MIN, _ETA_MAX)
        return model.neg_log_likelihood(theta_safe)
    return fn


def _make_clamped_jacobian(model: DWCMModel) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a Jacobian function that clamps θ to valid DWCM range."""
    def fn(theta: torch.Tensor) -> torch.Tensor:
        theta_safe = theta.clamp(_ETA_MIN, _ETA_MAX)
        return model.jacobian(theta_safe)
    return fn


def _check_strength_consistency(
    s_out: np.ndarray,
    s_in: np.ndarray,
) -> bool:
    """Check basic feasibility: all strengths non-negative, sums balance.

    Args:
        s_out: Out-strength sequence, shape (N,).
        s_in:  In-strength sequence, shape (N,).

    Returns:
        True if the sequence is valid for DWCM fitting.
    """
    if (s_out < 0).any() or (s_in < 0).any():
        return False
    # Sum of out-strengths == sum of in-strengths (total weight is shared)
    total_out = s_out.sum()
    total_in = s_in.sum()
    if total_out == 0 and total_in == 0:
        return False
    if total_out > 0 and total_in > 0:
        rel_imbalance = abs(total_out - total_in) / max(total_out, total_in)
        if rel_imbalance > 0.01:  # allow 1% imbalance from rounding
            return False
    return True


def _make_solvers(
    model: DWCMModel,
    theta0: torch.Tensor,
    tol: float,
) -> list[tuple[str, Callable[[], SolverResult]]]:
    """Return a list of (name, callable) solver pairs for *model*.

    Methods requiring O(N²) RAM are omitted for large N.

    Args:
        model:  The DWCMModel instance.
        theta0: Initial parameter vector (all positive).
        tol:    Convergence tolerance.

    Returns:
        Ordered list of ``(name, solver_callable)`` pairs.
    """
    N = model.N
    res_fn = _make_clamped_residual(model)
    nll_fn = _make_clamped_nll(model)
    jac_fn = _make_clamped_jacobian(model)

    # Adaptive max_iter for fixed-point: large N converges slowly so cap iterations
    # to avoid multi-minute runs that are statistically better captured by timeout.
    # Formula: aim for ~5M total element operations, floor at 500.
    max_iter_fp: int = max(500, min(10_000, int(5_000_000 / max(N, 1))))

    solvers: list[tuple[str, Callable[[], SolverResult]]] = []

    # Fixed-point (always applicable; chunked for large N)
    solvers.append((
        "Fixed-point GS α=1.0",
        lambda: solve_fixed_point_dwcm(
            res_fn, theta0, model.s_out, model.s_in,
            tol=tol, damping=1.0, variant="gauss-seidel",
            max_iter=max_iter_fp,
        ),
    ))
    solvers.append((
        "Fixed-point GS α=0.5",
        lambda: solve_fixed_point_dwcm(
            res_fn, theta0, model.s_out, model.s_in,
            tol=tol, damping=0.5, variant="gauss-seidel",
            max_iter=max_iter_fp,
        ),
    ))
    solvers.append((
        "Fixed-point Jacobi",
        lambda: solve_fixed_point_dwcm(
            res_fn, theta0, model.s_out, model.s_in,
            tol=tol, damping=1.0, variant="jacobi",
            max_iter=max_iter_fp,
        ),
    ))

    # L-BFGS (always applicable; chunked residual for large N)
    solvers.append((
        "L-BFGS (m=20)",
        lambda: solve_lbfgs(
            res_fn, theta0, tol=tol, m=20,
            neg_loglik_fn=nll_fn,
        ),
    ))

    # LM diagonal-only (always applicable — O(N) RAM)
    solvers.append((
        "LM (diag Hessian)",
        lambda: solve_lm(
            res_fn, jac_fn, theta0, tol=tol,
            diagonal_only=True,
        ),
    ))

    # Newton, Broyden, full-Jacobian LM — only for small N
    if N <= NEWTON_N_MAX:
        solvers.append((
            "Newton (exact J)",
            lambda: solve_newton(res_fn, jac_fn, theta0, tol=tol),
        ))
        solvers.append((
            "Broyden (rank-1 J)",
            lambda: solve_broyden(res_fn, jac_fn, theta0, tol=tol),
        ))

    if N <= FULL_JAC_LM_N_MAX:
        solvers.append((
            "LM (full Jacobian)",
            lambda: solve_lm(
                res_fn, jac_fn, theta0, tol=tol, diagonal_only=False,
            ),
        ))

    return solvers


# ---------------------------------------------------------------------------
# Single-network comparison
# ---------------------------------------------------------------------------

def run_comparison(N: int = 50, seed: Optional[int] = None, tol: float = DEFAULT_TOL) -> None:
    """Run all DWCM solvers on a single random network and print a comparison table.

    Args:
        N:    Number of nodes.
        seed: Random seed.  ``None`` picks a random seed.
        tol:  Convergence tolerance.
    """
    print(f"\n{'='*74}")
    print(f"DWCM Solver Comparison  |  N={N} nodes  |  seed={seed}  |  tol={tol:.0e}")
    print(f"{'='*74}")

    k, s = k_s_generator_pl(N, rho=DEFAULT_RHO, seed=seed)
    s_out = s[:N].numpy().astype(float)
    s_in = s[N:].numpy().astype(float)
    k_out = k[:N].numpy().astype(float)
    k_in = k[N:].numpy().astype(float)

    print(f"  s_out: min={s_out.min():.0f}  max={s_out.max():.0f}  mean={s_out.mean():.1f}")
    print(f"  s_in:  min={s_in.min():.0f}  max={s_in.max():.0f}  mean={s_in.mean():.1f}")
    print(f"  k_out: min={k_out.min():.0f}  max={k_out.max():.0f}  mean={k_out.mean():.1f}")
    print(f"  k_in:  min={k_in.min():.0f}  max={k_in.max():.0f}  mean={k_in.mean():.1f}")
    print()

    model = DWCMModel(s_out, s_in)
    theta0 = model.initial_theta("strengths")

    col = [24, 8, 8, 14, 10, 12]
    header = (
        f"{'Method':<{col[0]}} {'Conv?':>{col[1]}} {'Iters':>{col[2]}} "
        f"{'MaxRelErr':>{col[3]}} {'Time(s)':>{col[4]}} {'RAM(KB)':>{col[5]}}"
    )
    print(header)
    print("-" * sum(col))

    for name, fn in _make_solvers(model, theta0, tol):
        result: SolverResult = fn()
        max_rel_err = model.max_relative_error(result.theta)
        conv_str = "YES" if result.converged else "NO"
        print(
            f"{name:<{col[0]}} {conv_str:>{col[1]}} {result.iterations:>{col[2]}} "
            f"{max_rel_err:>{col[3]}.3e} {result.elapsed_time:>{col[4]}.3f} "
            f"{result.peak_ram_bytes/1024:>{col[5]}.1f}"
        )
    print()


# ---------------------------------------------------------------------------
# Multi-seed aggregate comparison
# ---------------------------------------------------------------------------

def _run_single_network(
    N: int,
    seed: int,
    tol: float,
    timeout: float,
) -> Optional[dict[str, dict]]:
    """Run all DWCM solvers on one network realisation.

    Args:
        N:       Number of nodes.
        seed:    Random seed.
        tol:     Convergence tolerance.
        timeout: Per-solver time limit (seconds).

    Returns:
        Dict mapping solver name → result dict, or None if the network is invalid.
    """
    k, s = k_s_generator_pl(N, rho=DEFAULT_RHO, seed=seed)
    s_out = s[:N].numpy().astype(float)
    s_in = s[N:].numpy().astype(float)

    if not _check_strength_consistency(s_out, s_in):
        return None

    model = DWCMModel(s_out, s_in)
    theta0 = model.initial_theta("strengths")
    solvers = _make_solvers(model, theta0, tol)

    results: dict[str, dict] = {}
    for name, fn in solvers:
        t_start = time.perf_counter()
        try:
            sr: SolverResult = fn()
            elapsed = time.perf_counter() - t_start
            if elapsed > timeout:
                results[name] = dict(
                    converged=False, iterations=sr.iterations,
                    max_rel_err=float("nan"), elapsed=elapsed,
                    peak_ram_mb=float("nan"), status="TIMEOUT",
                )
            else:
                max_rel_err = model.max_relative_error(sr.theta)
                results[name] = dict(
                    converged=sr.converged, iterations=sr.iterations,
                    max_rel_err=max_rel_err,
                    elapsed=sr.elapsed_time,
                    peak_ram_mb=sr.peak_ram_bytes / 1024 / 1024,
                    status="OK" if sr.converged else "NO-CONV",
                )
        except MemoryError:
            results[name] = dict(
                converged=False, iterations=0, max_rel_err=float("nan"),
                elapsed=time.perf_counter() - t_start,
                peak_ram_mb=float("nan"), status="OOM",
            )
        except RuntimeError as exc:
            exc_str = str(exc)
            status = "OOM" if ("out of memory" in exc_str.lower() or
                               "alloc" in exc_str.lower()) else "ERR"
            results[name] = dict(
                converged=False, iterations=0, max_rel_err=float("nan"),
                elapsed=time.perf_counter() - t_start,
                peak_ram_mb=float("nan"), status=status,
            )
        except Exception:
            results[name] = dict(
                converged=False, iterations=0, max_rel_err=float("nan"),
                elapsed=time.perf_counter() - t_start,
                peak_ram_mb=float("nan"), status="ERR",
            )
    return results


def run_multi_seed_comparison(
    N: int,
    n_seeds: int = DEFAULT_N_SEEDS,
    tol: float = DEFAULT_TOL,
    timeout: float = SOLVER_TIMEOUT,
    start_seed: int = 0,
    verbose: bool = True,
) -> dict[str, dict]:
    """Run all DWCM solvers on *n_seeds* independent network realisations.

    Collects per-run statistics and reports aggregate mean ± 2σ for:
    - calculation time (seconds)
    - peak RAM usage (MB)
    - number of iterations
    - Maximum Relative Error at convergence

    Args:
        N:          Number of nodes.
        n_seeds:    Number of valid realisations to use.
        tol:        Convergence tolerance.
        timeout:    Per-solver time limit in seconds.
        start_seed: First random seed to try.
        verbose:    If True, print detailed output.

    Returns:
        Dict mapping solver_name → aggregate_stats_dict.
    """
    if verbose:
        print(f"\n{'='*74}")
        print(
            f"DWCM Multi-Seed Comparison  |  N={N:,} nodes  |  "
            f"{n_seeds} runs  |  tol={tol:.0e}"
        )
        print(f"{'='*74}")

    # Collect results across seeds
    all_stats: dict[str, list[dict]] = {}
    valid_count = 0
    candidate_seed = start_seed
    max_attempts = n_seeds * 20

    while valid_count < n_seeds:
        if (candidate_seed - start_seed) >= max_attempts:
            raise RuntimeError(
                f"Could not find {n_seeds} valid networks for N={N} "
                f"in {max_attempts} attempts."
            )
        results = _run_single_network(N, candidate_seed, tol, timeout)
        if results is None:
            candidate_seed += 1
            continue

        if verbose:
            print(f"\n  Seed {candidate_seed}:")
            for name, r in results.items():
                tag = "✓" if r["converged"] else "✗"
                rel_err_str = (
                    f"{r['max_rel_err']:.2e}" if np.isfinite(r["max_rel_err"]) else "   —"
                )
                print(
                    f"    {tag} {name:<28} "
                    f"err={rel_err_str}  "
                    f"iters={r['iterations']:>6}  "
                    f"t={r['elapsed']:.2f}s"
                )

        for name, r in results.items():
            if name not in all_stats:
                all_stats[name] = []
            all_stats[name].append(r)

        valid_count += 1
        candidate_seed += 1

    # Compute aggregate statistics
    agg: dict[str, dict] = {}
    for name, runs in all_stats.items():
        times = np.array([r["elapsed"] for r in runs])
        rams = np.array([r["peak_ram_mb"] for r in runs if np.isfinite(r["peak_ram_mb"])])
        iters = np.array([r["iterations"] for r in runs])
        errs = np.array([r["max_rel_err"] for r in runs if np.isfinite(r["max_rel_err"])])
        conv_count = sum(r["converged"] for r in runs)

        agg[name] = {
            "conv_rate": conv_count / len(runs),
            "conv_count": conv_count,
            "n_runs": len(runs),
            "time_mean": times.mean(),
            "time_2sigma": 2 * times.std(ddof=1) if len(times) > 1 else 0.0,
            "ram_mean": rams.mean() if len(rams) > 0 else float("nan"),
            "ram_2sigma": (2 * rams.std(ddof=1) if len(rams) > 1 else 0.0),
            "iter_mean": iters.mean(),
            "iter_2sigma": 2 * iters.std(ddof=1) if len(iters) > 1 else 0.0,
            "err_mean": errs.mean() if len(errs) > 0 else float("nan"),
            "err_2sigma": (2 * errs.std(ddof=1) if len(errs) > 1 else 0.0),
        }

    if verbose:
        _print_aggregate_table(N, agg, n_seeds)

    return agg


def _print_aggregate_table(N: int, agg: dict[str, dict], n_seeds: int) -> None:
    """Print the aggregate statistics table.

    Args:
        N:       Number of nodes.
        agg:     Aggregate statistics from :func:`run_multi_seed_comparison`.
        n_seeds: Number of realisations used.
    """
    print(f"\n{'─'*74}")
    print(f"Aggregate Statistics  |  N={N:,}  |  {n_seeds} runs")
    print(f"{'─'*74}")

    col = [26, 10, 22, 22, 16, 16]
    header = (
        f"{'Method':<{col[0]}} {'Conv%':>{col[1]}} "
        f"{'Time(s) mean±2σ':^{col[2]}} "
        f"{'RAM(MB) mean±2σ':^{col[3]}} "
        f"{'Iters mean±2σ':^{col[4]}} "
        f"{'MaxRelErr mean±2σ':^{col[5]}}"
    )
    print(header)
    print("-" * (sum(col) + len(col) - 1))

    for name, s in agg.items():
        conv_pct = f"{s['conv_rate']:.0%}"
        time_str = f"{s['time_mean']:.3f}±{s['time_2sigma']:.3f}"
        if np.isfinite(s["ram_mean"]):
            ram_str = f"{s['ram_mean']:.1f}±{s['ram_2sigma']:.1f}"
        else:
            ram_str = "   —"
        iter_str = f"{s['iter_mean']:.0f}±{s['iter_2sigma']:.0f}"
        if np.isfinite(s["err_mean"]):
            err_str = f"{s['err_mean']:.2e}±{s['err_2sigma']:.2e}"
        else:
            err_str = "   —"
        print(
            f"{name:<{col[0]}} {conv_pct:>{col[1]}} "
            f"{time_str:^{col[2]}} "
            f"{ram_str:^{col[3]}} "
            f"{iter_str:^{col[4]}} "
            f"{err_str:^{col[5]}}"
        )
    print()


# ---------------------------------------------------------------------------
# Full scaling comparison (N = 1k, 5k, 10k, 50k)
# ---------------------------------------------------------------------------

def run_scaling_comparison(
    sizes: list[int] = DEFAULT_SIZES,
    n_seeds: int = DEFAULT_N_SEEDS,
    tol: float = DEFAULT_TOL,
    timeout: float = SOLVER_TIMEOUT,
    start_seed: int = 0,
) -> None:
    """Run multi-seed DWCM comparison for each size in *sizes*.

    Prints per-size aggregate tables and a final summary.

    Args:
        sizes:      List of node counts to benchmark.
        n_seeds:    Number of realisations per size.
        tol:        Convergence tolerance.
        timeout:    Per-solver time limit in seconds.
        start_seed: Base random seed.
    """
    all_agg: dict[int, dict[str, dict]] = {}

    for N in sizes:
        agg = run_multi_seed_comparison(
            N=N, n_seeds=n_seeds, tol=tol, timeout=timeout,
            start_seed=start_seed, verbose=True,
        )
        all_agg[N] = agg

    # Summary table
    print(f"\n{'='*74}")
    print(f"{'DWCM SCALING SUMMARY — Convergence Rate':^74}")
    print(f"{'='*74}")

    # Collect all method names seen
    all_methods: list[str] = []
    seen: set[str] = set()
    for agg in all_agg.values():
        for name in agg:
            if name not in seen:
                all_methods.append(name)
                seen.add(name)

    col_w = [28] + [max(9, len(f"N={N:,}") + 2) for N in sizes]
    header = f"{'Method':<{col_w[0]}}" + "".join(
        f"  {'N='+f'{N:,}':^{col_w[i+1]-2}}" for i, N in enumerate(sizes)
    )
    print(header)
    print("-" * sum(col_w))

    for method in all_methods:
        row = f"{method:<{col_w[0]}}"
        for i, N in enumerate(sizes):
            if N in all_agg and method in all_agg[N]:
                r = all_agg[N][method]
                conv_rate = f"{r['conv_rate']:.0%}"
                time_str = f"{r['time_mean']:.1f}s"
                cell = f"{conv_rate} {time_str}"
            else:
                cell = "—"
            row += f"  {cell:^{col_w[i+1]-2}}"
        print(row)

    print()
    print("Columns: convergence rate  mean time")
    print(f"Timeout: {timeout:.0f}s per solver")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DWCM solver comparison (Phase 4)")
    parser.add_argument("--n", type=int, default=50, help="Number of nodes (single-run mode)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Single seed (omit for multi-seed mode)")
    parser.add_argument("--n_seeds", type=int, default=DEFAULT_N_SEEDS,
                        help=f"Number of realisations (default: {DEFAULT_N_SEEDS})")
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=None,
        metavar="N",
        help=f"Node counts for scaling comparison (default: {DEFAULT_SIZES})",
    )
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL,
                        help=f"Convergence tolerance (default: {DEFAULT_TOL})")
    parser.add_argument("--timeout", type=float, default=SOLVER_TIMEOUT,
                        help=f"Per-solver time limit in seconds (default: {SOLVER_TIMEOUT})")
    parser.add_argument("--start_seed", type=int, default=0,
                        help="Base random seed (default: 0)")
    args = parser.parse_args()

    if args.sizes is not None:
        # Scaling mode: run multi-seed for each size
        run_scaling_comparison(
            sizes=args.sizes,
            n_seeds=args.n_seeds,
            tol=args.tol,
            timeout=args.timeout,
            start_seed=args.start_seed,
        )
    elif args.seed is not None:
        # Single-network mode
        run_comparison(N=args.n, seed=args.seed, tol=args.tol)
    else:
        # Default: multi-seed on single N
        run_multi_seed_comparison(
            N=args.n, n_seeds=args.n_seeds, tol=args.tol,
            timeout=args.timeout, start_seed=args.start_seed,
        )
