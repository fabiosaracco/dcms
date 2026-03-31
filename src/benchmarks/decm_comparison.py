"""DECM solver benchmark.

Generates test networks using the Chung-Lu power-law model
(``k_s_generator_pl``), solves them with the DECM θ-Newton Anderson(10)
solver, and prints a comparison table identical in format to the
dcm_comparison, dwcm_comparison and daecm_comparison benchmarks.

The **multi-seed variant** runs *n_seeds* independent realisations per node
count and reports aggregate statistics:

* convergence rate (%)
* mean calculation time ± 2σ
* mean peak RAM usage ± 2σ
* mean iteration count ± 2σ
* mean Maximum Relative Error at convergence ± 2σ

Usage::

    # Single network
    python -m src.benchmarks.decm_comparison --n 100 --seed 42

    # Multi-seed comparison
    python -m src.benchmarks.decm_comparison --n 1000 --n_seeds 10

    # Scaling across multiple sizes
    python -m src.benchmarks.decm_comparison --sizes 100 500 1000

    # Fast mode (applies per-solver timeout cap)
    python -m src.benchmarks.decm_comparison --n 1000 --n_seeds 10 --fast

    # No timeout
    python -m src.benchmarks.decm_comparison --sizes 1000 --n_seeds 5 --timeout 0 --fast
"""
from __future__ import annotations

import argparse
import signal
import sys
import math
import time
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from src.models.decm import DECMModel
from src.solvers.base import SolverResult
from src.utils.wng import k_s_generator_pl


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_SIZES: list[int] = [100, 500, 1_000]
DEFAULT_RHO: float = 0.001
DEFAULT_TOL: float = 1e-5
SOLVER_TIMEOUT: float = 900.0
FAST_SOLVER_TIMEOUT_S: float = 150.0
DEFAULT_N_SEEDS: int = 10


# ---------------------------------------------------------------------------
# Timeout helper
# ---------------------------------------------------------------------------

class _TimeoutError(Exception):
    """Raised when a solver exceeds its wall-clock budget."""


def _call_with_timeout(fn: Callable, timeout_s: float):
    """Call *fn()* and raise _TimeoutError if it exceeds *timeout_s* seconds."""
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
# Method descriptors
# ---------------------------------------------------------------------------

# All IC variants to try in multi-start order.  The best-converged result
# is returned, matching the multi-start logic used in the other comparisons.
_ALL_METHODS: list[dict] = [
    {"name": "θ-Newton Anderson(10)", "ic": "degrees",  "variant": "theta-newton", "anderson_depth": 10},
    {"name": "θ-Newton Anderson(10) [daecm]",   "ic": "daecm",   "variant": "theta-newton", "anderson_depth": 10},
    {"name": "θ-Newton Anderson(10) [uniform]", "ic": "uniform", "variant": "theta-newton", "anderson_depth": 10},
    {"name": "θ-Newton Anderson(10) [random]",  "ic": "random",  "variant": "theta-newton", "anderson_depth": 10},
]

# Fast mode: degrees (default) + daecm warm-start (most robust for large N).
_FAST_METHODS: list[dict] = [
    {"name": "θ-Newton Anderson(10)",          "ic": "degrees", "variant": "theta-newton", "anderson_depth": 10},
    {"name": "θ-Newton Anderson(10) [daecm]",  "ic": "daecm",  "variant": "theta-newton", "anderson_depth": 10},
]


# ---------------------------------------------------------------------------
# Per-solver runner
# ---------------------------------------------------------------------------

def _run_one(
    model: DECMModel,
    m: dict,
    tol: float,
    max_iter: int,
    timeout: float,
) -> SolverResult:
    """Run a single DECM solver descriptor and return its SolverResult."""
    def _fn() -> SolverResult:
        model.solve_tool(
            ic=m["ic"],
            tol=tol,
            max_iter=max_iter,
            max_time=timeout if timeout > 0 else 0,
            variant=m["variant"],
            anderson_depth=m["anderson_depth"],
            multi_start=False,
        )
        return model.sol

    return _call_with_timeout(_fn, timeout)


# ---------------------------------------------------------------------------
# Single-network verbose comparison
# ---------------------------------------------------------------------------

def run_comparison(
    N: int = 100,
    seed: Optional[int] = None,
    tol: float = DEFAULT_TOL,
) -> None:
    """Run all DECM solvers on a single random network and print a table.

    Args:
        N:    Number of nodes.
        seed: Random seed.  ``None`` picks a random seed.
        tol:  Convergence tolerance.
    """
    print(f"\n{'='*100}")
    print(f"DECM Solver Comparison  |  N={N} nodes  |  seed={seed}  |  tol={tol:.0e}")
    print(f"{'='*100}")

    k, s = k_s_generator_pl(N, rho=DEFAULT_RHO, seed=seed)
    k_out = k[:N].numpy().astype(float)
    k_in  = k[N:].numpy().astype(float)
    s_out = s[:N].numpy().astype(float)
    s_in  = s[N:].numpy().astype(float)

    print(f"  k_out: min={k_out.min():.0f}  max={k_out.max():.0f}  mean={k_out.mean():.1f}")
    print(f"  k_in:  min={k_in.min():.0f}  max={k_in.max():.0f}  mean={k_in.mean():.1f}")
    print(f"  s_out: min={s_out.min():.0f}  max={s_out.max():.0f}  mean={s_out.mean():.1f}")
    print(f"  s_in:  min={s_in.min():.0f}  max={s_in.max():.0f}  mean={s_in.mean():.1f}")
    print()

    model = DECMModel(k_out, k_in, s_out, s_in)

    col = [50, 8, 8, 14, 10, 12]
    header = (
        f"{'Method':<{col[0]}} {'Conv?':>{col[1]}} {'Iters':>{col[2]}} "
        f"{'MaxRelErr':>{col[3]}} {'Time(s)':>{col[4]}} {'RAM(MB)':>{col[5]}}"
    )
    print(header)
    print("-" * sum(col))

    for m in _ALL_METHODS:
        try:
            sr = _run_one(model, m, tol=tol, max_iter=5000, timeout=SOLVER_TIMEOUT)
            mre = model.max_relative_error(sr.theta)
            conv_str = "YES" if sr.converged else "NO"
            print(
                f"{m['name']:<{col[0]}} {conv_str:>{col[1]}} {sr.iterations:>{col[2]}} "
                f"{mre:>{col[3]}.3e} {sr.elapsed_time:>{col[4]}.3f} "
                f"{sr.peak_ram_bytes/1024/1024:>{col[5]}.2f}"
            )
        except _TimeoutError:
            print(f"{m['name']:<{col[0]}} {'TIMEOUT':>{col[1]+col[2]+col[3]+col[4]+col[5]+4}}")
    print()


# ---------------------------------------------------------------------------
# Per-seed worker
# ---------------------------------------------------------------------------

def _run_single_network(
    N: int,
    seed: int,
    tol: float,
    timeout: float,
    fast: bool = False,
) -> Optional[dict[str, dict]]:
    """Run all DECM solvers on one network realisation.

    Returns:
        Dict mapping solver name → result dict, or None if the network is invalid.
    """
    k, s = k_s_generator_pl(N, rho=DEFAULT_RHO, seed=seed)
    k_out = k[:N].numpy().astype(float)
    k_in  = k[N:].numpy().astype(float)
    s_out = s[:N].numpy().astype(float)
    s_in  = s[N:].numpy().astype(float)

    if s_out.sum() == 0 or s_in.sum() == 0 or k_out.sum() == 0 or k_in.sum() == 0:
        return None

    model = DECMModel(k_out, k_in, s_out, s_in)
    methods = _FAST_METHODS if fast else _ALL_METHODS

    if fast:
        per_solver_timeout = (
            FAST_SOLVER_TIMEOUT_S if timeout <= 0
            else min(timeout, FAST_SOLVER_TIMEOUT_S)
        )
    else:
        per_solver_timeout = timeout

    results: dict[str, dict] = {}
    for m in methods:
        t_start = time.perf_counter()
        try:
            sr = _run_one(model, m, tol=tol, max_iter=5000, timeout=per_solver_timeout)
            mre = model.max_relative_error(sr.theta)
            results[m["name"]] = dict(
                converged=sr.converged,
                iterations=sr.iterations,
                max_rel_err=mre,
                elapsed=sr.elapsed_time,
                peak_ram_mb=sr.peak_ram_bytes / 1024 / 1024,
                status="OK" if sr.converged else "NO-CONV",
            )
        except _TimeoutError:
            results[m["name"]] = dict(
                converged=False, iterations=0, max_rel_err=float("nan"),
                elapsed=time.perf_counter() - t_start,
                peak_ram_mb=float("nan"), status="TIMEOUT",
            )
        except MemoryError:
            results[m["name"]] = dict(
                converged=False, iterations=0, max_rel_err=float("nan"),
                elapsed=time.perf_counter() - t_start,
                peak_ram_mb=float("nan"), status="OOM",
            )
        except RuntimeError as exc:
            exc_str = str(exc)
            status = "OOM" if ("out of memory" in exc_str.lower() or
                               "alloc" in exc_str.lower()) else "ERR"
            results[m["name"]] = dict(
                converged=False, iterations=0, max_rel_err=float("nan"),
                elapsed=time.perf_counter() - t_start,
                peak_ram_mb=float("nan"), status=status,
            )
        except Exception:
            results[m["name"]] = dict(
                converged=False, iterations=0, max_rel_err=float("nan"),
                elapsed=time.perf_counter() - t_start,
                peak_ram_mb=float("nan"), status="ERR",
            )

    return results


# ---------------------------------------------------------------------------
# Multi-seed aggregate comparison
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
    """Run all DECM solvers on *n_seeds* independent network realisations.

    Collects per-run statistics and reports aggregate mean ± 2σ for
    calculation time, peak RAM, iterations, and Maximum Relative Error.

    Args:
        N:          Number of nodes.
        n_seeds:    Number of valid realisations to use.
        tol:        Convergence tolerance.
        timeout:    Per-solver time limit in seconds (0 = no limit).
        start_seed: First random seed to try.
        verbose:    If True, print detailed output.
        fast:       If True, only use the default IC ("degrees"); apply
                    per-solver timeout cap of ``FAST_SOLVER_TIMEOUT_S``.

    Returns:
        ``(agg, bad_seeds)`` where *agg* maps solver name → aggregate stats
        dict and *bad_seeds* is the list of seeds where no method converged.
    """
    if verbose:
        print(f"\n{'='*100}")
        print(
            f"DECM Multi-Seed Comparison  |  N={N:,} nodes  |  "
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
                    f"    {tag} {name:<44} "
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

    # Aggregate statistics over converged runs
    agg: dict[str, dict] = {}
    for name, runs in all_stats.items():
        conv_runs = [r for r in runs if r["converged"]]
        conv_count = len(conv_runs)

        times_conv   = np.array([r["elapsed"]      for r in conv_runs]) if conv_runs else np.array([])
        rams_conv    = np.array([r["peak_ram_mb"]   for r in conv_runs
                                  if np.isfinite(r["peak_ram_mb"])])  if conv_runs else np.array([])
        iters_conv   = np.array([r["iterations"]   for r in conv_runs]) if conv_runs else np.array([])
        errs_conv    = np.array([r["max_rel_err"]   for r in conv_runs
                                  if np.isfinite(r["max_rel_err"])])   if conv_runs else np.array([])

        def _mean2s(arr: np.ndarray) -> tuple[float, float]:
            if len(arr) == 0:
                return float("nan"), float("nan")
            return arr.mean(), (2 * arr.std(ddof=1) if len(arr) > 1 else 0.0)

        t_mean, t_2s = _mean2s(times_conv)
        r_mean, r_2s = _mean2s(rams_conv)
        i_mean, i_2s = _mean2s(iters_conv)
        e_mean, e_2s = _mean2s(errs_conv)

        agg[name] = {
            "conv_rate":   conv_count / len(runs),
            "conv_count":  conv_count,
            "n_runs":      len(runs),
            "time_mean":   t_mean,   "time_2sigma":  t_2s,
            "ram_mean":    r_mean,   "ram_2sigma":   r_2s,
            "iter_mean":   i_mean,   "iter_2sigma":  i_2s,
            "err_mean":    e_mean,   "err_2sigma":   e_2s,
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
    """Print the aggregate statistics table (same format as daecm_comparison).

    Performance metrics are reported only over converged runs.

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

    col = [50, 10, 22, 22, 16, 16]
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
        time_str = (f"{s['time_mean']:.3f}±{s['time_2sigma']:.3f}"
                    if np.isfinite(s["time_mean"]) else "   —")
        ram_str  = (f"{s['ram_mean']:.1f}±{s['ram_2sigma']:.1f}"
                    if np.isfinite(s["ram_mean"]) else "   —")
        iter_str = (f"{s['iter_mean']:.0f}±{s['iter_2sigma']:.0f}"
                    if np.isfinite(s["iter_mean"]) else "   —")
        err_str  = (f"{s['err_mean']:.2e}±{s['err_2sigma']:.2e}"
                    if np.isfinite(s["err_mean"]) else "   —")
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
# Scaling comparison (multiple sizes)
# ---------------------------------------------------------------------------

def run_scaling_comparison(
    sizes: list[int] = DEFAULT_SIZES,
    n_seeds: int = DEFAULT_N_SEEDS,
    tol: float = DEFAULT_TOL,
    timeout: float = SOLVER_TIMEOUT,
    start_seed: int = 0,
    fast: bool = False,
) -> None:
    """Run multi-seed DECM comparison for each size in *sizes*.

    Args:
        sizes:      List of node counts to benchmark.
        n_seeds:    Number of realisations per size.
        tol:        Convergence tolerance.
        timeout:    Per-solver time limit in seconds.
        start_seed: Base random seed.
        fast:       If True, use only the default IC and apply the fast timeout cap.
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
    print(f"{'DECM SCALING SUMMARY — Convergence Rate':^74}")
    print(f"{'='*74}")

    all_methods: list[str] = []
    seen: set[str] = set()
    for agg in all_agg.values():
        for name in agg:
            if name not in seen:
                all_methods.append(name)
                seen.add(name)

    col_w = [50] + [max(9, len(f"N={N:,}") + 2) for N in sizes]
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
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Command-line interface for the DECM benchmark."""
    import time as _time_mod

    parser = argparse.ArgumentParser(
        description="DECM solver benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n", type=int, default=None,
                        help="Number of nodes (single-size run)")
    parser.add_argument("--sizes", type=int, nargs="+", default=None,
                        help="Multiple node counts for scaling comparison")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (single-network run)")
    parser.add_argument("--n_seeds", type=int, default=DEFAULT_N_SEEDS,
                        help=f"Number of realisations (default: {DEFAULT_N_SEEDS})")
    parser.add_argument("--start_seed", type=int, default=None,
                        help="Base random seed (default: time-based)")
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL,
                        help=f"Convergence tolerance (default: {DEFAULT_TOL})")
    parser.add_argument("--timeout", type=float, default=SOLVER_TIMEOUT,
                        help=f"Per-solver timeout in seconds (default: {SOLVER_TIMEOUT})")
    parser.add_argument(
        "--fast", action="store_true",
        help=(
            "Fast mode: use only the default IC ('degrees') and cap each "
            f"solver at {FAST_SOLVER_TIMEOUT_S:.0f}s."
        ),
    )
    args = parser.parse_args()

    effective_start_seed: int = (
        args.start_seed
        if args.start_seed is not None
        else int(_time_mod.time() * 1000) % (2 ** 31)
    )

    if args.sizes is not None:
        run_scaling_comparison(
            sizes=args.sizes,
            n_seeds=args.n_seeds,
            tol=args.tol,
            timeout=args.timeout,
            start_seed=effective_start_seed,
            fast=args.fast,
        )
    elif args.n is not None and args.seed is not None:
        run_comparison(N=args.n, seed=args.seed, tol=args.tol)
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


if __name__ == "__main__":
    main()
