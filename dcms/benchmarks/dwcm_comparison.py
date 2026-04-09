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

    # Quick validation (fewer seeds, explicit start seed for reproducibility)
    python -m src.benchmarks.dwcm_comparison --n 100 --n_seeds 3 --start_seed 42

Memory thresholds
-----------------
* N > ``NEWTON_N_MAX``       → skip Newton and Broyden (O(N²) Jacobian).
* N > ``LBFGS_N_MAX``        → skip L-BFGS (O(N²) residual cost per step makes timeout impractical).
* N > ``FULL_JAC_LM_N_MAX`` → use diagonal-only LM instead of full-Jacobian LM.
* N > ``_LARGE_N_THRESHOLD`` (from DWCMModel) → chunked residual / fixed-point.
"""
from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from dcms.models.dwcm import DWCMModel, _ETA_MIN, _ETA_MAX
from dcms.solvers.base import SolverResult
from dcms.solvers.fixed_point_dwcm import solve_fixed_point_dwcm
from dcms.utils.wng import k_s_generator_pl


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
        # No timeout: Windows fallback, or caller explicitly disabled it.
        return fn()

    def _handler(signum: int, frame: object) -> None:
        raise _TimeoutError(f"Solver exceeded {timeout_s:.0f}s timeout.")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(max(1, int(timeout_s)))
    try:
        result = fn()
        signal.alarm(0)  # cancel alarm on success
        return result
    finally:
        signal.signal(signal.SIGALRM, old_handler)


# ---------------------------------------------------------------------------
# Scaling thresholds
# ---------------------------------------------------------------------------

# Default network sizes to benchmark
DEFAULT_SIZES: list[int] = [1_000, 5_000, 10_000, 50_000]

# Connection density (rho) for the Chung-Lu generator.
# Using the wng default (0.001) produces networks with moderate degree heterogeneity
# that are numerically tractable for DWCM. Denser networks (rho > 0.02) can push
# β values close to 1, making all solver methods numerically challenging.
DEFAULT_RHO: float = 0.001

# Convergence tolerance used by all solvers in this benchmark
#DEFAULT_TOL: float = 1e-6
DEFAULT_TOL: float = 1e-5

# Maximum solver wall-clock time (seconds) — skip if exceeded
#SOLVER_TIMEOUT: float = 300.0
SOLVER_TIMEOUT: float = 900.0

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
    total_out = s_out.sum()
    total_in = s_in.sum()
    if total_out == 0 and total_in == 0:
        return False
    if total_out > 0 and total_in > 0:
        rel_imbalance = abs(total_out - total_in) / max(total_out, total_in)
        if rel_imbalance > 0.01:
            return False
    return True


def _make_solvers(
    model: DWCMModel,
    theta0: torch.Tensor,
    tol: float,
    timeout: float = SOLVER_TIMEOUT,
    fast: bool = False,
) -> list[tuple[str, Callable[[], SolverResult]]]:
    """Return a list of (name, callable) solver pairs for *model*.

    Args:
        model:   The DWCMModel instance.
        theta0:  Initial parameter vector (all positive).
        tol:     Convergence tolerance.
        timeout: Per-solver hard timeout in seconds (used to set iteration budgets).
        fast:    Unused; both methods always run.

    Returns:
        Ordered list of ``(name, solver_callable)`` pairs.
    """
    N = model.N
    res_fn = _make_clamped_residual(model)

    # Per-method iteration budgets based on residual evaluation cost
    if N <= 5_000:
        residual_s = max(3e-4, (N / 1_000) ** 2 * 8e-3)
    else:
        residual_s = max(3e-4, (N / 1_000) ** 2 * 15e-3)

    full_budget_s = timeout if timeout > 0 else 300.0

    if N > 5_000:
        max_anderson_cap = 500
    elif N >= 1_000:
        max_anderson_cap = 1_000
    else:
        max_anderson_cap = 5_000
    MAX_FP_ANDERSON_ITER: int = max(100, min(max_anderson_cap, int(full_budget_s / residual_s)))

    _N_ANDERSON_INITS = 4
    _ITER_PER_ANDERSON_INIT = max(50, MAX_FP_ANDERSON_ITER // _N_ANDERSON_INITS)
    _TIME_PER_INIT: float = min(60.0, full_budget_s / _N_ANDERSON_INITS)

    solvers: list[tuple[str, Callable[[], SolverResult]]] = []

    # ── Fixed-point GS + Anderson depth=10, multi-start ─────────────────────
    def _fp_anderson_multistart() -> SolverResult:
        import time as _t
        from dcms.solvers.base import SolverResult as _SR
        inits = [theta0,
                 model.initial_theta("normalized"),
                 model.initial_theta("uniform"),
                 model.initial_theta("random")]
        best: Optional[SolverResult] = None
        best_err = float("inf")
        total_iters = 0
        peak_ram = 0
        t0_wall = _t.perf_counter()
        for t0_cand in inits:
            if best is not None and not best.converged:
                t0_cand = torch.tensor(best.theta, dtype=torch.float64)
            r = solve_fixed_point_dwcm(
                res_fn, t0_cand, model.s_out, model.s_in,
                tol=tol, damping=1.0, variant="gauss-seidel",
                max_iter=_ITER_PER_ANDERSON_INIT, anderson_depth=10,
                max_time=_TIME_PER_INIT,
            )
            total_iters += r.iterations
            peak_ram = max(peak_ram, r.peak_ram_bytes)
            err = model.max_relative_error(r.theta)
            if err < best_err:
                best_err = err
                best = r
            if r.converged:
                break
        assert best is not None
        return _SR(
            theta=best.theta,
            converged=best.converged,
            iterations=total_iters,
            residuals=best.residuals,
            elapsed_time=_t.perf_counter() - t0_wall,
            peak_ram_bytes=peak_ram,
            message=best.message,
        )

    solvers.append(("FP-GS Anderson(10) multi-init", _fp_anderson_multistart))

    # ── θ-space coordinate Newton + Anderson(10), multi-start ───────────────
    _ITER_PER_TN_INIT = max(50, MAX_FP_ANDERSON_ITER // _N_ANDERSON_INITS)

    def _theta_newton_multistart() -> SolverResult:
        import time as _t
        from dcms.solvers.base import SolverResult as _SR
        inits = [theta0,
                 model.initial_theta("normalized"),
                 model.initial_theta("uniform"),
                 model.initial_theta("random")]
        best: Optional[SolverResult] = None
        best_err = float("inf")
        total_iters = 0
        peak_ram = 0
        t0_wall = _t.perf_counter()
        for t0_cand in inits:
            if best is not None and not best.converged:
                t0_cand = torch.tensor(best.theta, dtype=torch.float64)
            r = solve_fixed_point_dwcm(
                res_fn, t0_cand, model.s_out, model.s_in,
                tol=tol, variant="theta-newton",
                max_iter=_ITER_PER_TN_INIT, anderson_depth=10, max_step=1.0,
                max_time=_TIME_PER_INIT,
            )
            total_iters += r.iterations
            peak_ram = max(peak_ram, r.peak_ram_bytes)
            err = model.max_relative_error(r.theta)
            if err < best_err:
                best_err = err
                best = r
            if r.converged:
                break
        assert best is not None
        return _SR(
            theta=best.theta,
            converged=best.converged,
            iterations=total_iters,
            residuals=best.residuals,
            elapsed_time=_t.perf_counter() - t0_wall,
            peak_ram_bytes=peak_ram,
            message=best.message,
        )

    solvers.append(("θ-Newton Anderson(10) multi-init", _theta_newton_multistart))

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
    print(f"\n{'='*100}")
    print(f"DWCM Solver Comparison  |  N={N} nodes  |  seed={seed}  |  tol={tol:.0e}")
    print(f"{'='*100}")

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

    col = [50, 8, 8, 14, 10, 12]
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
    fast: bool = False,
) -> Optional[dict[str, dict]]:
    """Run all DWCM solvers on one network realisation.

    Args:
        N:       Number of nodes.
        seed:    Random seed.
        tol:     Convergence tolerance.
        timeout: Per-solver time limit (seconds).
        fast:    If True, skip plain FP/Jacobi and LM; only Anderson and L-BFGS.

    Returns:
        Dict mapping solver name → result dict, or None if the network is invalid.
    """
    #t_start = time.perf_counter()
    k, s = k_s_generator_pl(N, rho=DEFAULT_RHO, seed=seed)
    #elapsed = time.perf_counter() - t_start
    #print(f'k_s_generator_pl, elapsed={elapsed:.2f} s')
    s_out = s[:N].numpy().astype(float)
    s_in = s[N:].numpy().astype(float)

    if not _check_strength_consistency(s_out, s_in):
        return None

    model = DWCMModel(s_out, s_in)
    theta0 = model.initial_theta("strengths")
    solvers = _make_solvers(model, theta0, tol, timeout=timeout, fast=fast)

    results: dict[str, dict] = {}
    for name, fn in solvers:
        t_start = time.perf_counter()
        try:
            sr: SolverResult = _call_with_timeout(fn, timeout)
            elapsed = time.perf_counter() - t_start
            max_rel_err = model.max_relative_error(sr.theta)
            results[name] = dict(
                converged=sr.converged, iterations=sr.iterations,
                max_rel_err=max_rel_err,
                elapsed=sr.elapsed_time,
                peak_ram_mb=sr.peak_ram_bytes / 1024 / 1024,
                status="OK" if sr.converged else "NO-CONV",
            )
        except _TimeoutError:
            results[name] = dict(
                converged=False, iterations=0,
                max_rel_err=float("nan"),
                elapsed=time.perf_counter() - t_start,
                peak_ram_mb=float("nan"), status="TIMEOUT",
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
    fast: bool = False,
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
        fast:       If True, skip plain FP/Jacobi and LM; only Anderson and L-BFGS.

    Returns:
        Dict mapping solver_name → aggregate_stats_dict.
    """
    if verbose:
        print(f"\n{'='*100}")
        print(
            f"DWCM Multi-Seed Comparison  |  N={N:,} nodes  |  "
            f"{n_seeds} runs  |  tol={tol:.0e}  |  start_seed={start_seed}"
        )
        print(f"{'='*100}")

    # Collect results across seeds
    all_stats: dict[str, list[dict]] = {}
    # Track seeds where no method converged (for diagnostic follow-up)
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

    if verbose and bad_seeds:
        print(f"\n  ⚠ Seeds where NO method converged: {bad_seeds}")

    # Compute aggregate statistics.
    # Performance metrics (time, iters, RAM, error) are computed only over
    # the converged runs so that the table reflects actual solver performance
    # rather than being diluted by timeouts/failures.
    agg: dict[str, dict] = {}
    for name, runs in all_stats.items():
        conv_runs = [r for r in runs if r["converged"]]
        conv_count = len(conv_runs)

        # All runs (converged + not) for time — so total wall time is visible
        times_all = np.array([r["elapsed"] for r in runs])
        # Converged-only for performance metrics
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
            # Performance statistics computed over converged runs only
            "time_mean": t_mean,
            "time_2sigma": t_2s,
            "ram_mean": r_mean,
            "ram_2sigma": r_2s,
            "iter_mean": i_mean,
            "iter_2sigma": i_2s,
            "err_mean": e_mean,
            "err_2sigma": e_2s,
            # Also expose total-run time for informational purposes
            "time_all_mean": times_all.mean(),
        }

    if verbose:
        _print_aggregate_table(N, agg, n_seeds, bad_seeds=bad_seeds)

    return agg, bad_seeds


def _print_aggregate_table(
    N: int,
    agg: dict[str, dict],
    n_seeds: int,
    bad_seeds: Optional[list[int]] = None,
) -> None:
    """Print the aggregate statistics table.

    Performance metrics (Time, RAM, Iters, MaxRelErr) are reported only over
    the runs that converged.  This keeps the table interpretable: a method
    that converges on 2 out of 5 seeds reports the mean time *for those 2 runs*,
    not for all 5 (where the failed ones may be cut short by a timeout or
    iteration cap).

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
        if np.isfinite(s["time_mean"]):
            time_str = f"{s['time_mean']:.3f}±{s['time_2sigma']:.3f}"
        else:
            time_str = "   —"
        if np.isfinite(s["ram_mean"]):
            ram_str = f"{s['ram_mean']:.1f}±{s['ram_2sigma']:.1f}"
        else:
            ram_str = "   —"
        if np.isfinite(s["iter_mean"]):
            iter_str = f"{s['iter_mean']:.0f}±{s['iter_2sigma']:.0f}"
        else:
            iter_str = "   —"
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

    if bad_seeds:
        print(f"\n  ⚠ Seeds where no method converged: {bad_seeds}")
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
    fast: bool = False,
) -> None:
    """Run multi-seed DWCM comparison for each size in *sizes*.

    Prints per-size aggregate tables and a final summary.

    Args:
        sizes:      List of node counts to benchmark.
        n_seeds:    Number of realisations per size.
        tol:        Convergence tolerance.
        timeout:    Per-solver time limit in seconds.
        start_seed: Base random seed.
        fast:       If True, skip plain FP/Jacobi and LM; only Anderson and L-BFGS.
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
                # Show converged-run mean time (nan → —)
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
    # Report bad seeds per size
    any_bad = any(v for v in all_bad.values())
    if any_bad:
        print("\nSeeds where no method converged:")
        for N, seeds in all_bad.items():
            if seeds:
                print(f"  N={N:,}: {seeds}")


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
    parser.add_argument("--start_seed", type=int, default=None,
                        help="Base random seed (default: random, from current time)")
    parser.add_argument(
        "--fast", action="store_true",
        help="Skip plain FP/Jacobi and LM; run only Anderson and L-BFGS methods",
    )
    parser.add_argument(
        "--phase4", action="store_true",
        help=(
            "Phase 4 focused mode: N=5k, 5 seeds, FP-GS α∈{1,0.5,0.3}, "
            "FP-GS Anderson(10), θ-Newton Anderson(10), and L-BFGS.  "
            "Bad seeds (no method converges) are saved to bad_seeds_phase4.txt."
        ),
    )
    args = parser.parse_args()

    # If start_seed is not provided, use a time-based random seed so that
    # different invocations naturally sample different realisations.
    import time as _time_mod
    effective_start_seed: int = (
        args.start_seed
        if args.start_seed is not None
        else int(_time_mod.time() * 1000) % (2 ** 31)
    )

    if args.phase4:
        # Phase-4 focused mode: N=5k, 5 seeds, key methods only.
        _phase4_N = 5_000
        _phase4_seeds = 10
        print(f"\n{'='*74}")
        print(
            f"Phase 4 focused benchmark  |  N={_phase4_N:,} nodes  |  "
            f"{_phase4_seeds} seeds  |  start_seed={effective_start_seed}"
        )
        print(f"{'='*74}")
        agg, bad_seeds = run_multi_seed_comparison(
            N=_phase4_N,
            n_seeds=_phase4_seeds,
            tol=args.tol,
            timeout=args.timeout,
            start_seed=effective_start_seed,
            verbose=True,
            fast=False,
        )
        # Save bad seeds to file for diagnostic follow-up
        bad_seeds_path = "bad_seeds_phase4.txt"
        with open(bad_seeds_path, "a</28>") as _f:
            #_f.write(f"# Phase4 bad seeds (N={_phase4_N}, start_seed={effective_start_seed})\n")
            for _s in bad_seeds:
                _f.write(f"{_s}\n")
        if bad_seeds:
            print(f"\nBad seeds saved to {bad_seeds_path}: {bad_seeds}")
        else:
            print(f"\nAll seeds converged. No bad seeds recorded.")
    elif args.sizes is not None:
        # Scaling mode: run multi-seed for each size
        run_scaling_comparison(
            sizes=args.sizes,
            n_seeds=args.n_seeds,
            tol=args.tol,
            timeout=args.timeout,
            start_seed=effective_start_seed,
            fast=args.fast,
        )
    elif args.seed is not None:
        # Single-network mode
        run_comparison(N=args.n, seed=args.seed, tol=args.tol)
    else:
        # Default: multi-seed on single N
        run_multi_seed_comparison(
            N=args.n, n_seeds=args.n_seeds, tol=args.tol,
            timeout=args.timeout, start_seed=effective_start_seed,
            fast=args.fast,
        )
