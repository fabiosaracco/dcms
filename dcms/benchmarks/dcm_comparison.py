"""DCM solver comparison benchmark.

Generates test networks using the Chung-Lu power-law model, then runs all
solvers and prints a comparison table.  The multi-seed variant runs
*n_seeds* independent network realisations per node count and reports
aggregate convergence statistics (mean ± 2σ for time, RAM, iterations,
max-relative-error).

Usage::

    python -m src.benchmarks.dcm_comparison                         # default sizes, 10 seeds
    python -m src.benchmarks.dcm_comparison --n 100                 # single size, 10 seeds
    python -m src.benchmarks.dcm_comparison --n 50 --seed 42        # single network
    python -m src.benchmarks.dcm_comparison --sizes 100 1000 10000  # scaling comparison
    python -m src.benchmarks.dcm_comparison --fast                   # fast methods only
"""
from __future__ import annotations

import argparse
import math
import signal
import sys
import time as _time_mod
from pathlib import Path
from typing import Callable, Optional

# Allow running as a script from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from dcms.models.dcm import DCMModel
from dcms.solvers.base import SolverResult
from dcms.solvers.fixed_point_dcm import solve_fixed_point_dcm
from dcms.utils.wng import k_s_generator_pl

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SIZES: list[int] = [100, 1_000, 10_000, 50_000]
DEFAULT_RHO: float = 0.001
DEFAULT_TOL: float = 1e-5
SOLVER_TIMEOUT: float = 300.0
DEFAULT_N_SEEDS: int = 10


# ---------------------------------------------------------------------------
# Timeout helper (POSIX SIGALRM)
# ---------------------------------------------------------------------------

class _TimeoutError(Exception):
    """Raised by the SIGALRM handler when a solver exceeds its wall-clock budget."""


def _call_with_timeout(fn: Callable, timeout_s: float):
    """Call *fn()* and raise _TimeoutError if it does not finish in *timeout_s* seconds.

    Uses POSIX SIGALRM (Linux/macOS only).  On platforms where SIGALRM is
    unavailable the function falls back to running without a timeout.

    Args:
        fn:        Zero-argument callable to invoke.
        timeout_s: Maximum allowed wall-clock seconds (rounded to nearest int).
                   Pass 0 or a negative value to disable the timeout entirely.

    Returns:
        Whatever *fn()* returns.

    Raises:
        _TimeoutError: If *fn()* exceeds *timeout_s* seconds.
    """
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


# ---------------------------------------------------------------------------
# Feasibility check
# ---------------------------------------------------------------------------

def _is_feasible(k_out: np.ndarray, k_in: np.ndarray) -> bool:
    """Return True if the degree sequence is feasible for the DCM.

    A degree sequence is feasible when no node has degree ≥ N (since
    self-loops are forbidden and the maximum possible degree is N−1).

    Args:
        k_out: Out-degree sequence, shape (N,).
        k_in:  In-degree sequence, shape (N,).

    Returns:
        True when the sequence is valid for DCM fitting.
    """
    N = len(k_out)
    return bool(k_out.max() < N and k_in.max() < N)


# ---------------------------------------------------------------------------
# Solver list factory
# ---------------------------------------------------------------------------

def _make_solvers(
    model: DCMModel,
    theta0: torch.Tensor,
    tol: float,
    fast: bool = False,
) -> list[tuple[str, Callable[[], SolverResult]]]:
    """Return the list of (name, callable) solver pairs for *model*.

    Args:
        model:  Fitted DCMModel.
        theta0: Initial parameter vector.
        tol:    Convergence tolerance.
        fast:   Unused; both methods always run.
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
# Single-network run (with timeout)
# ---------------------------------------------------------------------------

def run_comparison(
    N: int = 100,
    seed: Optional[int] = None,
    tol: float = DEFAULT_TOL,
    fast: bool = False,
) -> None:
    """Run all DCM solvers on a single random network and print a comparison table.

    Args:
        N:    Number of nodes.
        seed: Random seed for reproducibility.  ``None`` picks a random seed.
        tol:  Convergence tolerance for all solvers.
        fast: If True, skip slow methods.
    """
    print(f"\n{'='*80}")
    print(f"DCM Solver Comparison  |  N={N:,} nodes  |  seed={seed}  |  tol={tol:.0e}")
    print(f"{'='*80}")

    k, _ = k_s_generator_pl(N, rho=DEFAULT_RHO, seed=seed)
    k_out = k[:N].numpy().astype(float)
    k_in = k[N:].numpy().astype(float)

    if not _is_feasible(k_out, k_in):
        print("  SKIPPED: infeasible degree sequence (max degree ≥ N).")
        return

    model = DCMModel(k_out, k_in)
    theta0 = model.initial_theta("degrees")

    print(f"  k_out: min={k_out.min():.0f}  max={k_out.max():.0f}  mean={k_out.mean():.1f}")
    print(f"  k_in:  min={k_in.min():.0f}  max={k_in.max():.0f}  mean={k_in.mean():.1f}")
    print()

    col = [28, 8, 8, 14, 10, 12]
    header = (
        f"{'Method':<{col[0]}} {'Conv?':>{col[1]}} {'Iters':>{col[2]}} "
        f"{'MaxErr':>{col[3]}} {'Time(s)':>{col[4]}} {'RAM(KB)':>{col[5]}}"
    )
    print(header)
    print("-" * sum(col))

    for name, fn in _make_solvers(model, theta0, tol, fast=fast):
        result: SolverResult = fn()
        max_err = model.constraint_error(result.theta)
        conv_str = "YES" if result.converged else "NO"
        print(
            f"{name:<{col[0]}} {conv_str:>{col[1]}} {result.iterations:>{col[2]}} "
            f"{max_err:>{col[3]}.3e} {result.elapsed_time:>{col[4]}.3f} "
            f"{result.peak_ram_bytes/1024:>{col[5]}.1f}"
        )

    print()


# ---------------------------------------------------------------------------
# Per-network helper used by multi-seed loop
# ---------------------------------------------------------------------------

def _run_single_network(
    N: int,
    seed: int,
    tol: float,
    timeout: float,
    fast: bool = False,
) -> Optional[dict[str, dict]]:
    """Run all solvers on one network realisation with per-solver timeout.

    Returns ``None`` if the degree sequence is infeasible (skipped).
    Otherwise returns a dict mapping solver_name → stats_dict with keys:
    ``converged``, ``iterations``, ``elapsed``, ``peak_ram_mb``,
    ``max_abs_err``, ``max_rel_err``.
    """
    k, _ = k_s_generator_pl(N, rho=DEFAULT_RHO, seed=seed)
    k_out = k[:N].numpy().astype(float)
    k_in = k[N:].numpy().astype(float)

    if not _is_feasible(k_out, k_in):
        return None

    model = DCMModel(k_out, k_in)
    theta0 = model.initial_theta("degrees")

    # Apply the timeout limit.
    effective_timeout = timeout

    results: dict[str, dict] = {}
    for name, fn in _make_solvers(model, theta0, tol, fast=fast):
        try:
            result: SolverResult = _call_with_timeout(fn, effective_timeout)
            max_abs = model.constraint_error(result.theta)
            ref = float(model.k_out.abs().mean().clamp(min=1e-15))
            max_rel = max_abs / ref
            results[name] = {
                "converged": result.converged,
                "iterations": result.iterations,
                "elapsed": result.elapsed_time,
                "peak_ram_mb": result.peak_ram_bytes / 1e6,
                "max_abs_err": max_abs,
                "max_rel_err": max_rel,
            }
        except _TimeoutError:
            results[name] = {
                "converged": False,
                "iterations": 0,
                "elapsed": effective_timeout,
                "peak_ram_mb": float("nan"),
                "max_abs_err": float("nan"),
                "max_rel_err": float("nan"),
            }

    return results


# ---------------------------------------------------------------------------
# Multi-seed comparison
# ---------------------------------------------------------------------------

def run_multi_seed_comparison(
    N: int,
    n_seeds: int = DEFAULT_N_SEEDS,
    tol: float = DEFAULT_TOL,
    timeout: float = SOLVER_TIMEOUT,
    start_seed: int = 0,
    verbose: bool = True,
    fast: bool = False,
) -> tuple[dict[str, dict], list[int]]:
    """Run all DCM solvers on *n_seeds* independent network realisations.

    Collects per-run statistics and reports aggregate mean ± 2σ for:
    - calculation time (seconds)
    - peak RAM usage (MB)
    - number of iterations
    - maximum relative constraint error at convergence

    Args:
        N:          Number of nodes.
        n_seeds:    Number of valid realisations to use.
        tol:        Convergence tolerance.
        timeout:    Per-solver time limit in seconds.
        start_seed: First random seed to try.
        verbose:    If True, print detailed output.
        fast:       If True, skip slow methods (Jacobi, Broyden, LM).

    Returns:
        ``(agg, bad_seeds)`` where *agg* maps solver_name → aggregate_stats_dict
        and *bad_seeds* is a list of seeds where no solver converged.
    """
    if verbose:
        print(f"\n{'='*100}")
        print(
            f"DCM Multi-Seed Comparison  |  N={N:,} nodes  |  "
            f"{n_seeds} runs  |  tol={tol:.0e}  |  start_seed={start_seed}"
        )
        print(f"{'='*100}")

    all_stats: dict[str, list[dict]] = {}
    bad_seeds: list[int] = []
    valid_count = 0
    candidate_seed = start_seed
    max_attempts = n_seeds * 20

    while valid_count < n_seeds:
        if (candidate_seed - start_seed) >= max_attempts:
            raise RuntimeError(
                f"Could not find {n_seeds} valid networks for N={N} "
                f"in {max_attempts} attempts."
            )

        results = _run_single_network(N, candidate_seed, tol, timeout, fast=fast)
        if results is None:
            candidate_seed += 1
            continue

        any_converged = any(r["converged"] for r in results.values())
        if not any_converged:
            bad_seeds.append(candidate_seed)

        if verbose:
            print(f"\n  Seed {candidate_seed}{'' if any_converged else ' ⚠ NO METHOD CONVERGED'}:")
            for name, r in results.items():
                tag = "✓" if r["converged"] else "✗"
                rel_err_str = (
                    f"{r['max_rel_err']:.2e}"
                    if np.isfinite(r["max_rel_err"])
                    else "   —"
                )
                print(
                    f"    {tag} {name:<36} "
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

    if verbose and bad_seeds:
        print(f"\n  ⚠ Seeds where NO method converged: {bad_seeds}")

    # Compute aggregate statistics over converged runs only
    agg: dict[str, dict] = {}
    for name, runs in all_stats.items():
        conv_runs = [r for r in runs if r["converged"]]
        conv_count = len(conv_runs)

        times_all = np.array([r["elapsed"] for r in runs])
        times_conv = np.array([r["elapsed"] for r in conv_runs]) if conv_runs else np.array([])
        rams_conv = np.array(
            [r["peak_ram_mb"] for r in conv_runs if np.isfinite(r["peak_ram_mb"])]
        )
        iters_conv = np.array([r["iterations"] for r in conv_runs]) if conv_runs else np.array([])
        errs_conv = np.array(
            [r["max_rel_err"] for r in conv_runs if np.isfinite(r["max_rel_err"])]
        )

        def _mean2s(arr: np.ndarray) -> tuple[float, float]:
            if len(arr) == 0:
                return float("nan"), float("nan")
            return arr.mean(), (2 * arr.std(ddof=1) if len(arr) > 1 else 0.0)

        t_mean, t_2s = _mean2s(times_conv)
        r_mean, r_2s = _mean2s(rams_conv)
        i_mean, i_2s = _mean2s(iters_conv)
        e_mean, e_2s = _mean2s(errs_conv)

        agg[name] = {
            "conv_rate": conv_count / len(runs),
            "conv_count": conv_count,
            "n_runs": len(runs),
            "time_mean": t_mean,
            "time_2sigma": t_2s,
            "ram_mean": r_mean,
            "ram_2sigma": r_2s,
            "iter_mean": i_mean,
            "iter_2sigma": i_2s,
            "err_mean": e_mean,
            "err_2sigma": e_2s,
            "time_all_mean": times_all.mean(),
        }

    if verbose:
        _print_aggregate_table(N, agg, n_seeds, bad_seeds=bad_seeds)

    return agg, bad_seeds


# ---------------------------------------------------------------------------
# Aggregate table printer
# ---------------------------------------------------------------------------

def _print_aggregate_table(
    N: int,
    agg: dict[str, dict],
    n_seeds: int,
    bad_seeds: Optional[list[int]] = None,
) -> None:
    """Print the aggregate statistics table.

    Performance metrics (Time, RAM, Iters, MaxRelErr) are reported only over
    the runs that converged.

    Args:
        N:         Number of nodes.
        agg:       Aggregate statistics from :func:`run_multi_seed_comparison`.
        n_seeds:   Number of realisations used.
        bad_seeds: List of seeds where no method converged (optional).
    """
    print(f"\n{'─'*100}")
    print(f"Aggregate Statistics  |  N={N:,}  |  {n_seeds} runs")
    print(f"(Performance metrics computed over converged runs only)")
    print(f"{'─'*100}")

    col = [40, 10, 22, 22, 16, 16]
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
        time_str = (
            f"{s['time_mean']:.3f}±{s['time_2sigma']:.3f}"
            if np.isfinite(s["time_mean"]) else "   —"
        )
        ram_str = (
            f"{s['ram_mean']:.1f}±{s['ram_2sigma']:.1f}"
            if np.isfinite(s["ram_mean"]) else "   —"
        )
        iter_str = (
            f"{s['iter_mean']:.0f}±{s['iter_2sigma']:.0f}"
            if np.isfinite(s["iter_mean"]) else "   —"
        )
        err_str = (
            f"{s['err_mean']:.2e}±{s['err_2sigma']:.2e}"
            if np.isfinite(s["err_mean"]) else "   —"
        )
        print(
            f"{name:<{col[0]}} {conv_pct:>{col[1]}} "
            f"{time_str:^{col[2]}} "
            f"{ram_str:^{col[3]}} "
            f"{iter_str:^{col[4]}} "
            f"{err_str:^{col[5]}}"
        )

    if bad_seeds:
        print(f"\n  ⚠ Seeds where no method converged: {bad_seeds}")
    print()


# ---------------------------------------------------------------------------
# Scaling comparison
# ---------------------------------------------------------------------------

def run_scaling_comparison(
    sizes: list[int] = DEFAULT_SIZES,
    n_seeds: int = DEFAULT_N_SEEDS,
    tol: float = DEFAULT_TOL,
    timeout: float = SOLVER_TIMEOUT,
    start_seed: int = 0,
    fast: bool = False,
) -> None:
    """Run multi-seed DCM comparison for each size in *sizes*.

    Args:
        sizes:      List of node counts to benchmark.
        n_seeds:    Number of realisations per size.
        tol:        Convergence tolerance.
        timeout:    Per-solver time limit in seconds.
        start_seed: Base random seed.
        fast:       If True, skip slow methods.
    """
    all_agg: dict[int, dict[str, dict]] = {}
    all_bad: dict[int, list[int]] = {}

    for N in sizes:
        agg, bad = run_multi_seed_comparison(
            N=N, n_seeds=n_seeds, tol=tol, timeout=timeout,
            start_seed=start_seed, verbose=True, fast=fast,
        )
        all_agg[N] = agg
        all_bad[N] = bad

    print(f"\n{'='*74}")
    print(f"{'DCM SCALING SUMMARY — Convergence Rate':^74}")
    print(f"{'='*74}")

    all_methods: list[str] = []
    seen: set[str] = set()
    for agg in all_agg.values():
        for name in agg:
            if name not in seen:
                all_methods.append(name)
                seen.add(name)

    col_w = [38] + [max(9, len(f"N={N:,}") + 2) for N in sizes]
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
                t = r["time_mean"]
                time_str = f"{t:.1f}s" if np.isfinite(t) else "—"
                cell = f"{conv_rate} {time_str}"
            else:
                cell = "—"
            row += f"  {cell:^{col_w[i+1]-2}}"
        print(row)

    print()
    print("Columns: convergence rate  mean time (converged runs only)")
    print(f"Timeout: {timeout:.0f}s per solver")
    any_bad = any(v for v in all_bad.values())
    if any_bad:
        for N, bad in all_bad.items():
            if bad:
                print(f"  ⚠ N={N:,}: seeds where no method converged: {bad}")
    print()


# ---------------------------------------------------------------------------
# Backward-compatible alias
# ---------------------------------------------------------------------------

def run_comparison_multi_seed(
    N: int = 100,
    n_seeds: int = DEFAULT_N_SEEDS,
    tol: float = DEFAULT_TOL,
    start_seed: int = 0,
) -> dict[str, list[bool]]:
    """Backward-compatible alias for :func:`run_multi_seed_comparison`.

    .. deprecated::
        Use :func:`run_multi_seed_comparison` instead.
    """
    agg, _ = run_multi_seed_comparison(
        N=N, n_seeds=n_seeds, tol=tol, start_seed=start_seed,
    )
    return {name: [True] * int(s["conv_count"]) + [False] * (n_seeds - int(s["conv_count"]))
            for name, s in agg.items()}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DCM solver comparison benchmark (Phase 2).")
    parser.add_argument("--n", type=int, default=None,
                        help="Number of nodes (single-size run)")
    parser.add_argument("--sizes", type=int, nargs="+", default=None,
                        help=f"Node counts for scaling comparison (default: {DEFAULT_SIZES})")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (single-network run)")
    parser.add_argument("--n_seeds", type=int, default=DEFAULT_N_SEEDS,
                        help=f"Number of realisations (default: {DEFAULT_N_SEEDS})")
    parser.add_argument("--start_seed", type=int, default=None,
                        help="Base random seed (default: random, from current time)")
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL,
                        help=f"Convergence tolerance (default: {DEFAULT_TOL})")
    parser.add_argument("--timeout", type=float, default=SOLVER_TIMEOUT,
                        help=f"Per-solver timeout in seconds (default: {SOLVER_TIMEOUT})")
    parser.add_argument(
        "--fast", action="store_true",
        help="Skip plain FP variants, Broyden, and LM; run FP-GS, Anderson, θ-Newton, L-BFGS",
    )
    parser.add_argument(
        "--phase2", action="store_true",
        help=(
            "Phase 2 focused mode: N=1k, 10 seeds, all methods.  "
            "Bad seeds (no method converges) are saved to bad_seeds_phase2.txt."
        ),
    )
    args = parser.parse_args()

    effective_start_seed: int = (
        args.start_seed
        if args.start_seed is not None
        else int(_time_mod.time() * 1000) % (2 ** 31)
    )

    if args.phase2:
        _phase2_N = 1_000
        _phase2_seeds = 10
        print(f"\n{'='*74}")
        print(
            f"Phase 2 focused benchmark  |  N={_phase2_N:,} nodes  |  "
            f"{_phase2_seeds} seeds  |  start_seed={effective_start_seed}"
        )
        print(f"{'='*74}")
        agg, bad_seeds = run_multi_seed_comparison(
            N=_phase2_N,
            n_seeds=_phase2_seeds,
            tol=args.tol,
            timeout=args.timeout,
            start_seed=effective_start_seed,
            verbose=True,
            fast=False,
        )
        bad_seeds_path = "bad_seeds_phase2.txt"
        with open(bad_seeds_path, "a") as _f:
            for _s in bad_seeds:
                _f.write(f"{_s}\n")
        if bad_seeds:
            print(f"\nBad seeds saved to {bad_seeds_path}: {bad_seeds}")
        else:
            print(f"\nAll seeds converged. No bad seeds recorded.")
    elif args.sizes is not None:
        run_scaling_comparison(
            sizes=args.sizes,
            n_seeds=args.n_seeds,
            tol=args.tol,
            timeout=args.timeout,
            start_seed=effective_start_seed,
            fast=args.fast,
        )
    elif args.n is not None and args.seed is not None:
        run_comparison(N=args.n, seed=args.seed, tol=args.tol, fast=args.fast)
    elif args.n is not None:
        run_multi_seed_comparison(
            N=args.n,
            n_seeds=args.n_seeds,
            tol=args.tol,
            timeout=args.timeout,
            start_seed=effective_start_seed,
            fast=args.fast,
        )
    else:
        run_scaling_comparison(
            sizes=DEFAULT_SIZES,
            n_seeds=args.n_seeds,
            tol=args.tol,
            timeout=args.timeout,
            start_seed=effective_start_seed,
            fast=args.fast,
        )
