"""DaECM solver comparison benchmark — Phase 5.

Generates test networks using the Chung-Lu power-law model (``k_s_generator_pl``),
then runs all applicable DaECM two-step solvers and prints a comparison table.

The **multi-seed variant** runs *n_seeds* independent network realisations per
node count and reports aggregate statistics:

* convergence rate (%)
* mean calculation time ± 2σ
* mean peak RAM usage ± 2σ
* mean iteration count ± 2σ
* mean Maximum Relative Error at convergence ± 2σ

Usage::

    # Single network
    python -m src.benchmarks.daecm_comparison --n 100 --seed 42

    # Multi-seed comparison (Phase 5 full run)
    python -m src.benchmarks.daecm_comparison --n 1000 --n_seeds 10

    # Both N=1k and N=5k
    python -m src.benchmarks.daecm_comparison --sizes 1000 5000

    # Fast mode: Newton / L-BFGS only (skip FP-GS Anderson which never converges for DaECM)
    python -m src.benchmarks.daecm_comparison --n 1000 --fast

    # Phase 5 focused benchmark (N=1k, 10 seeds, saves bad seeds)
    python -m src.benchmarks.daecm_comparison --phase5

Memory thresholds
-----------------
* N > ``NEWTON_N_MAX`` → skip Newton and Broyden weight solvers (2N×2N Jacobian, ~32 MB at N=1k).
* N > ``LBFGS_N_MAX``  → skip L-BFGS weight solver (O(N²) cost per gradient eval).
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

from src.models.daecm import DaECMModel, _ETA_MIN, _ETA_MAX, _LARGE_N_THRESHOLD as _DAECM_LARGE_N
from src.models.dcm import DCMModel
from src.solvers.base import SolverResult
from src.solvers.fixed_point_daecm import solve_fixed_point_daecm
from src.solvers.fixed_point_dcm import solve_fixed_point_dcm
from src.utils.wng import k_s_generator_pl


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
# Scaling thresholds
# ---------------------------------------------------------------------------

# Default benchmark sizes
DEFAULT_SIZES: list[int] = [1_000, 5_000]

# Connection density for the Chung-Lu generator.
DEFAULT_RHO: float = 0.001

# Convergence tolerance for all solvers in this benchmark
DEFAULT_TOL: float = 1e-5

# Solver wall-clock timeout (seconds)
SOLVER_TIMEOUT: float = 900.0

# Per-solver cap in fast mode: kills any solver that exceeds this wall time.
# At N=200 Newton needs ~100s for hard seeds (slow linear convergence phase after
# the initial quadratic descent); at N=1000 L-BFGS line search stalls at ~25ms/call.
# 150 s covers: Newton at N≤200 (340 iters × 0.3s) and L-BFGS at N≤500.
FAST_SOLVER_TIMEOUT_S: float = 150.0

# Number of seeds for multi-seed comparison
DEFAULT_N_SEEDS: int = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_strength_residual_fn(
    model: DaECMModel,
    theta_topo: torch.Tensor,
    P: Optional[torch.Tensor] = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a clamped residual function for the weight step.

    If ``P`` is provided (pre-computed DCM probability matrix), it is used
    instead of recomputing ``p_ij`` at every call, reducing per-eval cost
    from ~12ms to ~9ms at N=1000.
    """
    def fn(theta_w: torch.Tensor) -> torch.Tensor:
        theta_w_safe = theta_w.clamp(_ETA_MIN, _ETA_MAX)
        return model.residual_strength(theta_topo, theta_w_safe, P=P)
    return fn


def _run_topo_step(
    model: DaECMModel,
    tol: float,
    timeout: float,
    theta_topo0: Optional[torch.Tensor] = None,
) -> tuple[Optional[torch.Tensor], SolverResult]:
    """Run the topology (DCM) step with FP-GS Anderson(10) then θ-Newton Anderson(10).

    Mirrors the two-stage approach used for DWCM: try FP-GS A10 first (fast on
    well-conditioned networks), fall back to θ-Newton A10 if not converged.

    Args:
        model:       DaECMModel instance.
        tol:         Convergence tolerance.
        timeout:     Total wall-clock time budget for the topology step.
        theta_topo0: Initial guess; if None, uses model.initial_theta_topo().

    Returns:
        ``(theta_topo, result)`` where theta_topo is the solved topology
        parameters (or best found), and result contains solver statistics.
    """
    dcm = model._dcm
    if theta_topo0 is None:
        theta_topo0 = model.initial_theta_topo("degrees")

    k_out = dcm.k_out
    k_in = dcm.k_in
    half_budget = max(10.0, timeout / 2.0)

    t_start = time.perf_counter()

    # Stage 1: FP-GS Anderson(10)
    r1 = solve_fixed_point_dcm(
        dcm.residual, theta_topo0,
        k_out=k_out, k_in=k_in,
        tol=tol, max_iter=500,
        variant="gauss-seidel", anderson_depth=10,
        max_time=half_budget,
    )

    if r1.converged:
        result = SolverResult(
            theta=r1.theta, converged=True, iterations=r1.iterations,
            residuals=r1.residuals, elapsed_time=time.perf_counter() - t_start,
            peak_ram_bytes=r1.peak_ram_bytes, message=r1.message,
        )
        return torch.tensor(r1.theta, dtype=torch.float64), result

    # Stage 2: θ-Newton Anderson(10), warm-started from best FP result
    remaining = max(5.0, timeout - (time.perf_counter() - t_start))
    theta1 = torch.tensor(r1.theta, dtype=torch.float64)
    r2 = solve_fixed_point_dcm(
        dcm.residual, theta1,
        k_out=k_out, k_in=k_in,
        tol=tol, max_iter=500,
        variant="theta-newton", anderson_depth=10,
        max_time=remaining,
    )

    best = r2 if r2.converged or (r2.residuals and r2.residuals[-1] <
                                   (r1.residuals[-1] if r1.residuals else float("inf")))  \
           else r1
    result = SolverResult(
        theta=best.theta, converged=best.converged,
        iterations=r1.iterations + r2.iterations,
        residuals=best.residuals,
        elapsed_time=time.perf_counter() - t_start,
        peak_ram_bytes=max(r1.peak_ram_bytes, r2.peak_ram_bytes),
        message=best.message,
    )
    return torch.tensor(best.theta, dtype=torch.float64), result


def _fp_weight_multistart(
    model: DaECMModel,
    theta_topo: torch.Tensor,
    theta_weight0: torch.Tensor,
    tol: float,
    max_iter: int,
    variant: str = "gauss-seidel",
    damping: float = 1.0,
    anderson_depth: int = 0,
    max_step: float = 1.0,
    chunk_size: int = 0,
    n_starts: int = 4,
    max_time_per_init: float = 0.0,
) -> SolverResult:
    """FP weight solver with multiple initialisations.

    Args:
        model:              DaECMModel instance.
        theta_topo:         Fixed topology parameters.
        theta_weight0:      Default initial weight parameters.
        tol:                Convergence tolerance.
        max_iter:           Maximum iterations per start.
        variant:            ``"gauss-seidel"``, ``"jacobi"``, or ``"theta-newton"``.
        damping:            Damping factor.
        anderson_depth:     Anderson acceleration depth.
        max_step:           Max Newton step (θ-Newton only).
        chunk_size:         0 = auto.
        n_starts:           Total starting points to try.
        max_time_per_init:  Wall-clock time limit (seconds) per starting point.
                            0 = no per-init limit.

    Returns:
        :class:`~src.solvers.base.SolverResult` with the best solution found.
    """
    import time as _t
    from src.solvers.base import SolverResult as _SR

    # Pre-compute p_ij for dense path
    N = model.N
    if N <= _DAECM_LARGE_N:
        topo_out = theta_topo[:N]
        topo_in = theta_topo[N:]
        log_xy = -topo_out[:, None] - topo_in[None, :]
        P_mat: Optional[torch.Tensor] = torch.sigmoid(log_xy)
        P_mat.fill_diagonal_(0.0)
    else:
        P_mat = None

    res_fn = _make_strength_residual_fn(model, theta_topo, P=P_mat)

    starts = [theta_weight0]
    for method in ("topology", "normalized", "uniform"):
        starts.append(model.initial_theta_weight(theta_topo, method))
    for i in range(max(0, n_starts - len(starts))):
        torch.manual_seed(i)
        starts.append(model.initial_theta_weight(theta_topo, "random"))

    iter_per_start = max(50, max_iter // n_starts)
    best_result: Optional[SolverResult] = None
    best_err = float("inf")
    total_iters = 0
    peak_ram = 0
    t_start = _t.perf_counter()

    for t0 in starts[:n_starts]:
        # Warm-start: if a previous init didn't converge, begin from the best
        # θ found so far rather than a fresh initialisation.
        if best_result is not None and not best_result.converged:
            t0 = torch.tensor(best_result.theta, dtype=torch.float64)
        r = solve_fixed_point_daecm(
            res_fn, t0,
            model.s_out, model.s_in,
            theta_topo=theta_topo,
            P=P_mat,
            tol=tol, max_iter=iter_per_start,
            damping=damping, variant=variant,
            anderson_depth=anderson_depth, max_step=max_step,
            chunk_size=chunk_size,
            max_time=max_time_per_init,
        )
        total_iters += r.iterations
        peak_ram = max(peak_ram, r.peak_ram_bytes)
        err = model.constraint_error_strength(theta_topo, r.theta)
        if err < best_err:
            best_err = err
            best_result = r
        if r.converged:
            break

    assert best_result is not None
    elapsed = _t.perf_counter() - t_start
    return _SR(
        theta=best_result.theta,
        converged=best_result.converged,
        iterations=total_iters,
        residuals=best_result.residuals,
        elapsed_time=elapsed,
        peak_ram_bytes=peak_ram,
        message=best_result.message,
    )


def _make_solvers(
    model: DaECMModel,
    theta_topo: torch.Tensor,
    theta_weight0: torch.Tensor,
    tol: float,
    timeout: float = SOLVER_TIMEOUT,
    fast: bool = False,
) -> list[tuple[str, Callable[[], tuple[bool, int, float, float]]]]:
    """Return a list of (name, callable) weight solver pairs.

    Each callable returns ``(converged, iterations, elapsed_s, mre)``.

    Args:
        model:          DaECMModel instance.
        theta_topo:     Fixed topology parameters (from DCM step).
        theta_weight0:  Default initial weight parameters.
        tol:            Convergence tolerance.
        timeout:        Per-solver time limit in seconds.
        fast:           Unused; both methods always run.

    Returns:
        Ordered list of ``(name, solver_callable)`` pairs.
    """
    N = model.N
    # Pre-compute p_ij once and reuse in all solvers
    if N <= _DAECM_LARGE_N:
        topo_out = theta_topo[:N]
        topo_in = theta_topo[N:]
        log_xy = -topo_out[:, None] - topo_in[None, :]
        P_mat: Optional[torch.Tensor] = torch.sigmoid(log_xy)
        P_mat.fill_diagonal_(0.0)
    else:
        P_mat = None

    res_fn = _make_strength_residual_fn(model, theta_topo, P=P_mat)

    # Per-method iteration budgets based on residual evaluation cost
    if N <= _DAECM_LARGE_N:
        residual_s = max(3e-4, (N / 1_000) ** 2 * 8e-3)
    else:
        residual_s = max(3e-4, (N / 1_000) ** 2 * 15e-3)

    full_budget_s = timeout if timeout > 0 else 300.0
    MAX_FP_ANDERSON_ITER: int = max(100, min(2_000, int(full_budget_s / residual_s)))

    _N_INITS = 4
    _ITER_PER_INIT_ANDERSON = max(50, MAX_FP_ANDERSON_ITER // _N_INITS)
    _TIME_PER_INIT: float = min(60.0, full_budget_s / _N_INITS)

    solvers: list[tuple[str, Callable]] = []

    # ── FP-GS + Anderson(10) multi-init ─────────────────────────────────────
    def _fp_anderson_multistart() -> SolverResult:
        return _fp_weight_multistart(
            model, theta_topo, theta_weight0, tol=tol,
            max_iter=_ITER_PER_INIT_ANDERSON * _N_INITS,
            variant="gauss-seidel", anderson_depth=10,
            n_starts=_N_INITS,
            max_time_per_init=_TIME_PER_INIT,
        )

    solvers.append(("FP-GS Anderson(10) multi-init", _fp_anderson_multistart))

    # ── θ-Newton + Anderson(10) multi-init ──────────────────────────────────
    def _theta_newton_multistart() -> SolverResult:
        return _fp_weight_multistart(
            model, theta_topo, theta_weight0, tol=tol,
            max_iter=_ITER_PER_INIT_ANDERSON * _N_INITS,
            variant="theta-newton", anderson_depth=10, max_step=1.0,
            n_starts=_N_INITS,
            max_time_per_init=_TIME_PER_INIT,
        )

    solvers.append(("θ-Newton Anderson(10) multi-init", _theta_newton_multistart))

    return solvers


# ---------------------------------------------------------------------------
# Single-network comparison
# ---------------------------------------------------------------------------

def run_comparison(
    N: int = 50,
    seed: Optional[int] = None,
    tol: float = DEFAULT_TOL,
) -> None:
    """Run all DaECM weight solvers on a single random network.

    Args:
        N:    Number of nodes.
        seed: Random seed.  ``None`` picks a random seed.
        tol:  Convergence tolerance.
    """
    print(f"\n{'='*100}")
    print(f"DaECM Solver Comparison  |  N={N} nodes  |  seed={seed}  |  tol={tol:.0e}")
    print(f"{'='*100}")

    k, s = k_s_generator_pl(N, rho=DEFAULT_RHO, seed=seed)
    k_out = k[:N].numpy().astype(float)
    k_in = k[N:].numpy().astype(float)
    s_out = s[:N].numpy().astype(float)
    s_in = s[N:].numpy().astype(float)

    print(f"  k_out: min={k_out.min():.0f}  max={k_out.max():.0f}  mean={k_out.mean():.1f}")
    print(f"  k_in:  min={k_in.min():.0f}  max={k_in.max():.0f}  mean={k_in.mean():.1f}")
    print(f"  s_out: min={s_out.min():.0f}  max={s_out.max():.0f}  mean={s_out.mean():.1f}")
    print(f"  s_in:  min={s_in.min():.0f}  max={s_in.max():.0f}  mean={s_in.mean():.1f}")
    print()

    model = DaECMModel(k_out, k_in, s_out, s_in)
    theta_topo0 = model.initial_theta_topo("degrees")

    # Step 1: solve topology
    print("  Step 1: Solving DCM topology...")
    theta_topo, topo_res = _run_topo_step(model, tol=tol, timeout=300.0,
                                          theta_topo0=theta_topo0)
    topo_err = model._dcm.constraint_error(topo_res.theta)
    print(f"  Topo: converged={topo_res.converged}, err={topo_err:.2e}, "
          f"iters={topo_res.iterations}, t={topo_res.elapsed_time:.3f}s")
    print()

    theta_weight0 = model.initial_theta_weight(theta_topo, method="topology")

    col = [50, 8, 8, 14, 10, 12]
    header = (
        f"{'Method':<{col[0]}} {'Conv?':>{col[1]}} {'Iters':>{col[2]}} "
        f"{'MaxRelErr':>{col[3]}} {'Time(s)':>{col[4]}} {'RAM(KB)':>{col[5]}}"
    )
    print("  Step 2: Weight solvers comparison:")
    print(header)
    print("-" * sum(col))

    for name, fn in _make_solvers(model, theta_topo, theta_weight0, tol):
        result: SolverResult = fn()
        mre = model.max_relative_error(topo_res.theta, result.theta)
        conv_str = "YES" if result.converged else "NO"
        print(
            f"{name:<{col[0]}} {conv_str:>{col[1]}} {result.iterations:>{col[2]}} "
            f"{mre:>{col[3]}.3e} {result.elapsed_time:>{col[4]}.3f} "
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
    """Run all DaECM weight solvers on one network realisation.

    Returns:
        Dict mapping solver name → result dict, or None if the network is invalid.
    """
    k, s = k_s_generator_pl(N, rho=DEFAULT_RHO, seed=seed)
    k_out = k[:N].numpy().astype(float)
    k_in = k[N:].numpy().astype(float)
    s_out = s[:N].numpy().astype(float)
    s_in = s[N:].numpy().astype(float)

    # Basic validity check
    if s_out.sum() == 0 or s_in.sum() == 0:
        return None
    if k_out.sum() == 0 or k_in.sum() == 0:
        return None

    model = DaECMModel(k_out, k_in, s_out, s_in)
    theta_topo0 = model.initial_theta_topo("degrees")

    # Step 1: topology (DCM) — FP-GS A10 then θ-Newton A10
    topo_budget = min(timeout * 0.3, 120.0) if timeout > 0 else 120.0
    try:
        theta_topo, topo_sr = _call_with_timeout(
            lambda: _run_topo_step(model, tol=tol, timeout=topo_budget,
                                   theta_topo0=theta_topo0),
            topo_budget + 5.0,
        )
    except (_TimeoutError, Exception):
        return None

    # Proceed even if topology didn't fully converge — use best θ found.
    # Poor topology convergence will manifest as high max_rel_err in weight results.

    theta_weight0 = model.initial_theta_weight(theta_topo, method="topology")
    weight_timeout = (timeout - topo_sr.elapsed_time) if timeout > 0 else 0.0

    # In fast mode, cap each individual solver to avoid runaway iterations.
    # At N=1000 the residual is ~25 ms; without a cap a single L-BFGS multistart
    # can take 4+ minutes even in fast mode.
    # NOTE: weight_timeout == 0 means "no outer timeout" — in fast mode we still
    # apply the per-solver cap (FAST_SOLVER_TIMEOUT_S) so that --timeout 0 --fast
    # does not accidentally disable all solver timeouts.
    if fast:
        per_solver_timeout = (FAST_SOLVER_TIMEOUT_S if weight_timeout <= 0
                              else min(weight_timeout, FAST_SOLVER_TIMEOUT_S))
    else:
        per_solver_timeout = weight_timeout

    solvers = _make_solvers(model, theta_topo, theta_weight0, tol,
                            timeout=weight_timeout, fast=fast)

    results: dict[str, dict] = {}
    for name, fn in solvers:
        t_start = time.perf_counter()
        try:
            sr: SolverResult = _call_with_timeout(fn, per_solver_timeout)
            mre = model.max_relative_error(topo_sr.theta, sr.theta)
            results[name] = dict(
                converged=sr.converged,
                iterations=sr.iterations,
                max_rel_err=mre,
                elapsed=topo_sr.elapsed_time + sr.elapsed_time,
                peak_ram_mb=sr.peak_ram_bytes / 1024 / 1024,
                status="OK" if sr.converged else "NO-CONV",
            )
        except _TimeoutError:
            results[name] = dict(
                converged=False, iterations=0, max_rel_err=float("nan"),
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
) -> tuple[dict[str, dict], list[int]]:
    """Run all DaECM weight solvers on *n_seeds* independent network realisations.

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
        fast:       If True, skip FP-GS/θ-Newton (non-converging) and LM; use Newton/Broyden/L-BFGS.

    Returns:
        ``(agg, bad_seeds)`` where agg maps solver_name → aggregate_stats_dict.
    """
    if verbose:
        print(f"\n{'='*100}")
        print(
            f"DaECM Multi-Seed Comparison  |  N={N:,} nodes  |  "
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
                    f"    {tag} {name:<34} "
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

    # Compute aggregate statistics over converged runs
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
            # Performance statistics computed over converged runs only
            "time_mean": t_mean,
            "time_2sigma": t_2s,
            "ram_mean": r_mean,
            "ram_2sigma": r_2s,
            "iter_mean": i_mean,
            "iter_2sigma": i_2s,
            "err_mean": e_mean,
            "err_2sigma": e_2s,
            # Total-run time for informational purposes
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
    not for all 5.

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


def run_scaling_comparison(
    sizes: list[int] = DEFAULT_SIZES,
    n_seeds: int = DEFAULT_N_SEEDS,
    tol: float = DEFAULT_TOL,
    timeout: float = SOLVER_TIMEOUT,
    start_seed: int = 0,
    fast: bool = False,
) -> None:
    """Run multi-seed DaECM comparison for each size in *sizes*.

    Args:
        sizes:      List of node counts to benchmark.
        n_seeds:    Number of realisations per size.
        tol:        Convergence tolerance.
        timeout:    Per-solver time limit in seconds.
        start_seed: Base random seed.
        fast:       If True, skip FP-GS/θ-Newton (non-converging) and LM; use Newton/Broyden/L-BFGS.
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
    print(f"{'DaECM SCALING SUMMARY — Convergence Rate':^74}")
    print(f"{'='*74}")

    all_methods: list[str] = []
    seen: set[str] = set()
    for agg in all_agg.values():
        for name in agg:
            if name not in seen:
                all_methods.append(name)
                seen.add(name)

    col_w = [34] + [max(9, len(f"N={N:,}") + 2) for N in sizes]
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
    import time as _time_mod

    parser = argparse.ArgumentParser(
        description="DaECM solver comparison benchmark (Phase 5)."
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
                        help="Base random seed (default: random, from current time)")
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL,
                        help=f"Convergence tolerance (default: {DEFAULT_TOL})")
    parser.add_argument("--timeout", type=float, default=SOLVER_TIMEOUT,
                        help=f"Per-solver timeout in seconds (default: {SOLVER_TIMEOUT})")
    parser.add_argument(
        "--fast", action="store_true",
        help="Skip plain FP and LM; run only Anderson, θ-Newton and L-BFGS methods",
    )
    parser.add_argument(
        "--phase5", action="store_true",
        help=(
            "Phase 5 focused mode: N=1k, 10 seeds, Anderson and L-BFGS methods.  "
            "Bad seeds (no method converges) are saved to bad_seeds_phase5.txt."
        ),
    )
    args = parser.parse_args()

    # If start_seed is not provided, use a time-based random seed so that
    # different invocations naturally sample different realisations.
    effective_start_seed: int = (
        args.start_seed
        if args.start_seed is not None
        else int(_time_mod.time() * 1000) % (2 ** 31)
    )

    if args.phase5:
        _phase5_N = 1_000
        _phase5_seeds = 10
        print(f"\n{'='*74}")
        print(
            f"Phase 5 focused benchmark  |  N={_phase5_N:,} nodes  |  "
            f"{_phase5_seeds} seeds  |  start_seed={effective_start_seed}"
        )
        print(f"{'='*74}")
        agg, bad_seeds = run_multi_seed_comparison(
            N=_phase5_N,
            n_seeds=_phase5_seeds,
            tol=args.tol,
            timeout=args.timeout,
            start_seed=effective_start_seed,
            verbose=True,
            fast=False,
        )
        bad_seeds_path = "bad_seeds_phase5.txt"
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
