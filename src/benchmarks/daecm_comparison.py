"""DaECM solver comparison benchmark — Phase 5.

Generates test networks using the Chung-Lu power-law model (``k_s_generator_pl``),
then runs all applicable DaECM two-step solvers and prints a comparison table.

The **multi-seed variant** runs *n_seeds* independent network realisations per
node count and reports aggregate statistics:

* convergence rate (%)
* mean iteration count ± 2σ  (topology + weight steps combined)
* mean calculation time ± 2σ
* mean Maximum Relative Error at convergence ± 2σ

Usage::

    # Single network
    python -m src.benchmarks.daecm_comparison --n 100 --seed 42

    # Multi-seed comparison (Phase 5 full run)
    python -m src.benchmarks.daecm_comparison --n 1000 --n_seeds 10

    # Both N=1k and N=5k
    python -m src.benchmarks.daecm_comparison --sizes 1000 5000

Memory thresholds
-----------------
* N > ``NEWTON_N_MAX`` → skip Newton and Broyden weight solvers (O(N²) Jacobian).
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

from src.models.daecm import DaECMModel, _ETA_MIN, _ETA_MAX
from src.models.dcm import DCMModel
from src.solvers.base import SolverResult
from src.solvers.daecm_solver import DaECMResult, solve_daecm, _solve_lm_diag_daecm, solve_daecm_joint_lbfgs
from src.solvers.fixed_point_daecm import solve_fixed_point_daecm
from src.solvers.quasi_newton import solve_lbfgs
from src.solvers.newton import solve_newton
from src.solvers.broyden import solve_broyden
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

# Weight Newton/Broyden need full N×N Jacobian (O(N²) RAM).
NEWTON_N_MAX: int = 500

# L-BFGS weight step is O(N²) cost per iteration (residual evaluation).
# Practical for small-medium N; skip for large N.
LBFGS_N_MAX: int = 10_000

# Default benchmark sizes
DEFAULT_SIZES: list[int] = [1_000, 5_000]

# Connection density for the Chung-Lu generator.
DEFAULT_RHO: float = 0.001

# Convergence tolerance for all solvers in this benchmark
DEFAULT_TOL: float = 1e-5

# Solver wall-clock timeout (seconds)
SOLVER_TIMEOUT: float = 900.0

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


def _make_strength_nll_fn(
    model: DaECMModel,
    theta_topo: torch.Tensor,
) -> Callable[[torch.Tensor], float]:
    """Return a clamped neg_log_likelihood function for the weight step."""
    def fn(theta_w: torch.Tensor) -> float:
        theta_w_safe = theta_w.clamp(_ETA_MIN, _ETA_MAX)
        return model.neg_log_likelihood_strength(theta_topo, theta_w_safe)
    return fn


def _make_strength_jacobian_fn(
    model: DaECMModel,
    theta_topo: torch.Tensor,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a clamped Jacobian function for the weight step."""
    def fn(theta_w: torch.Tensor) -> torch.Tensor:
        theta_w_safe = theta_w.clamp(_ETA_MIN, _ETA_MAX)
        return model.jacobian_strength(theta_topo, theta_w_safe)
    return fn


def _run_topo_step(
    model: DaECMModel,
    tol: float,
    timeout: float,
    theta_topo0: Optional[torch.Tensor] = None,
) -> tuple[Optional[torch.Tensor], SolverResult]:
    """Run the topology (DCM) step with L-BFGS (best method for topology).

    Args:
        model:      DaECMModel instance.
        tol:        Convergence tolerance.
        timeout:    Solver time limit.
        theta_topo0: Initial guess; if None, uses model.initial_theta_topo().

    Returns:
        ``(theta_topo, result)`` where theta_topo is the solved topology
        parameters (or best found), and result contains solver statistics.
    """
    from src.solvers.quasi_newton import solve_lbfgs as _lbfgs

    dcm = model._dcm
    if theta_topo0 is None:
        theta_topo0 = model.initial_theta_topo("degrees")

    res_fn = dcm.residual
    nll_fn = dcm.neg_log_likelihood

    t_start = time.perf_counter()
    result = _lbfgs(
        res_fn, theta_topo0,
        tol=tol, max_iter=5_000, m=20,
        neg_loglik_fn=nll_fn,
        theta_bounds=(-_ETA_MAX, _ETA_MAX),
    )
    result = SolverResult(
        theta=result.theta,
        converged=result.converged,
        iterations=result.iterations,
        residuals=result.residuals,
        elapsed_time=time.perf_counter() - t_start,
        peak_ram_bytes=result.peak_ram_bytes,
        message=result.message,
    )
    theta_topo = torch.tensor(result.theta, dtype=torch.float64)
    return theta_topo, result


def _lbfgs_weight_multistart(
    model: DaECMModel,
    theta_topo: torch.Tensor,
    theta_weight0: torch.Tensor,
    tol: float,
    max_iter: int,
    n_starts: int = 4,
) -> SolverResult:
    """L-BFGS weight solver with multiple initialisations.

    Args:
        model:         DaECMModel instance.
        theta_topo:    Fixed topology parameters.
        theta_weight0: Default initial weight parameters.
        tol:           Convergence tolerance.
        max_iter:      Maximum iterations per start.
        n_starts:      Total starting points to try.

    Returns:
        :class:`~src.solvers.base.SolverResult` with the best solution found.
    """
    import time as _t
    from src.solvers.base import SolverResult as _SR

    res_fn = _make_strength_residual_fn(model, theta_topo)
    nll_fn = _make_strength_nll_fn(model, theta_topo)

    best_result: Optional[SolverResult] = None
    best_err = float("inf")
    total_iters = 0
    combined_ram = 0
    t_start = _t.perf_counter()

    starts = [theta_weight0]
    for method in ("normalized", "uniform"):
        starts.append(model.initial_theta_weight(theta_topo, method))
    for i in range(max(0, n_starts - len(starts))):
        torch.manual_seed(i)
        starts.append(model.initial_theta_weight(theta_topo, "random"))

    iter_per_start = max(30, max_iter // n_starts)
    for t0 in starts[:n_starts]:
        torch.manual_seed(0)
        result = solve_lbfgs(
            res_fn, t0, tol=tol, m=20, max_iter=iter_per_start,
            neg_loglik_fn=nll_fn, theta_bounds=(_ETA_MIN, _ETA_MAX),
        )
        total_iters += result.iterations
        combined_ram = max(combined_ram, result.peak_ram_bytes)
        err = model.constraint_error_strength(theta_topo, result.theta)
        if err < best_err:
            best_err = err
            best_result = result
        if result.converged:
            break

    assert best_result is not None
    elapsed = _t.perf_counter() - t_start
    return _SR(
        theta=best_result.theta,
        converged=best_result.converged,
        iterations=total_iters,
        residuals=best_result.residuals,
        elapsed_time=elapsed,
        peak_ram_bytes=combined_ram,
        message=best_result.message,
    )


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
) -> SolverResult:
    """FP weight solver with multiple initialisations.

    Args:
        model:         DaECMModel instance.
        theta_topo:    Fixed topology parameters.
        theta_weight0: Default initial weight parameters.
        tol:           Convergence tolerance.
        max_iter:      Maximum iterations per start.
        variant:       ``"gauss-seidel"``, ``"jacobi"``, or ``"theta-newton"``.
        damping:       Damping factor.
        anderson_depth: Anderson acceleration depth.
        max_step:      Max Newton step (θ-Newton only).
        chunk_size:    0 = auto.
        n_starts:      Total starting points to try.

    Returns:
        :class:`~src.solvers.base.SolverResult` with the best solution found.
    """
    import time as _t
    from src.solvers.base import SolverResult as _SR

    # Pre-compute p_ij for dense path
    N = model.N
    if N <= 5_000:
        topo_out = theta_topo[:N]
        topo_in = theta_topo[N:]
        log_xy = -topo_out[:, None] - topo_in[None, :]
        P_mat: Optional[torch.Tensor] = torch.sigmoid(log_xy)
        P_mat.fill_diagonal_(0.0)
    else:
        P_mat = None

    res_fn = _make_strength_residual_fn(model, theta_topo, P=P_mat)

    starts = [theta_weight0]
    for method in ("normalized", "uniform"):
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
        r = solve_fixed_point_daecm(
            res_fn, t0,
            model.s_out, model.s_in,
            theta_topo=theta_topo,
            P=P_mat,
            tol=tol, max_iter=iter_per_start,
            damping=damping, variant=variant,
            anderson_depth=anderson_depth, max_step=max_step,
            chunk_size=chunk_size,
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
) -> list[tuple[str, Callable[[], tuple[bool, int, float, float]]]]:
    """Return a list of (name, callable) weight solver pairs.

    Each callable returns ``(converged, iterations, elapsed_s, mre)``.

    Args:
        model:          DaECMModel instance.
        theta_topo:     Fixed topology parameters (from DCM step).
        theta_weight0:  Default initial weight parameters.
        tol:            Convergence tolerance.
        timeout:        Per-solver time limit in seconds.

    Returns:
        Ordered list of ``(name, solver_callable)`` pairs.
    """
    N = model.N
    # Pre-compute p_ij once and reuse in all solvers
    if N <= 5_000:
        topo_out = theta_topo[:N]
        topo_in = theta_topo[N:]
        log_xy = -topo_out[:, None] - topo_in[None, :]
        P_mat: Optional[torch.Tensor] = torch.sigmoid(log_xy)
        P_mat.fill_diagonal_(0.0)
    else:
        P_mat = None

    res_fn = _make_strength_residual_fn(model, theta_topo, P=P_mat)
    nll_fn = _make_strength_nll_fn(model, theta_topo)
    jac_fn = _make_strength_jacobian_fn(model, theta_topo)

    # Per-method iteration budgets based on residual evaluation cost
    if N <= 5_000:
        residual_s = max(3e-4, (N / 1_000) ** 2 * 8e-3)
    else:
        residual_s = max(3e-4, (N / 1_000) ** 2 * 15e-3)

    full_budget_s = timeout if timeout > 0 else 1e9
    plain_budget_s = min(full_budget_s * 0.2, 30.0)
    MAX_FP_PLAIN_ITER: int = max(50, min(500, int(plain_budget_s / residual_s)))
    # Cap Anderson at 500 per init to avoid extremely long non-converging runs
    MAX_FP_ANDERSON_ITER: int = max(100, min(2_000, int(full_budget_s / residual_s)))
    MAX_LBFGS_ITER: int = max(50, min(1_000, int(full_budget_s / (10 * residual_s))))
    MAX_LM_ITER: int = max(50, min(1_000, int(full_budget_s / (3 * residual_s))))

    _N_INITS = 4
    _ITER_PER_INIT_ANDERSON = max(50, MAX_FP_ANDERSON_ITER // _N_INITS)
    _ITER_PER_INIT_LBFGS = max(30, MAX_LBFGS_ITER // _N_INITS)

    solvers: list[tuple[str, Callable]] = []

    # ── FP-GS α=1.0 (plain) ────────────────────────────────────────────────
    solvers.append((
        "FP-GS α=1.0",
        lambda: solve_fixed_point_daecm(
            res_fn, theta_weight0, model.s_out, model.s_in,
            theta_topo=theta_topo, P=P_mat,
            tol=tol, damping=1.0, variant="gauss-seidel",
            max_iter=MAX_FP_PLAIN_ITER, anderson_depth=0,
        ),
    ))

    # ── FP-GS α=0.5 ────────────────────────────────────────────────────────
    solvers.append((
        "FP-GS α=0.5",
        lambda: solve_fixed_point_daecm(
            res_fn, theta_weight0, model.s_out, model.s_in,
            theta_topo=theta_topo, P=P_mat,
            tol=tol, damping=0.5, variant="gauss-seidel",
            max_iter=MAX_FP_PLAIN_ITER, anderson_depth=0,
        ),
    ))

    # ── FP-GS + Anderson(10) multi-init ─────────────────────────────────────
    def _fp_anderson_multistart() -> SolverResult:
        return _fp_weight_multistart(
            model, theta_topo, theta_weight0, tol=tol,
            max_iter=_ITER_PER_INIT_ANDERSON * _N_INITS,
            variant="gauss-seidel", anderson_depth=10,
            n_starts=_N_INITS,
        )

    solvers.append(("FP-GS Anderson(10) multi-init", _fp_anderson_multistart))

    # ── θ-Newton + Anderson(10) multi-init ──────────────────────────────────
    def _theta_newton_multistart() -> SolverResult:
        return _fp_weight_multistart(
            model, theta_topo, theta_weight0, tol=tol,
            max_iter=_ITER_PER_INIT_ANDERSON * _N_INITS,
            variant="theta-newton", anderson_depth=10, max_step=1.0,
            n_starts=_N_INITS,
        )

    solvers.append(("θ-Newton Anderson(10) multi-init", _theta_newton_multistart))

    # ── L-BFGS multi-start ──────────────────────────────────────────────────
    if N <= LBFGS_N_MAX:
        def _lbfgs_ms() -> SolverResult:
            return _lbfgs_weight_multistart(
                model, theta_topo, theta_weight0, tol=tol,
                max_iter=_ITER_PER_INIT_LBFGS * _N_INITS, n_starts=_N_INITS,
            )

        solvers.append(("L-BFGS (multi-start)", _lbfgs_ms))

    # ── Diagonal LM (O(N) RAM) ──────────────────────────────────────────────
    solvers.append((
        "LM (diag Hessian)",
        lambda: _solve_lm_diag_daecm(
            model, theta_topo, theta_weight0,
            tol=tol, theta_bounds=(_ETA_MIN, _ETA_MAX),
            max_iter=MAX_LM_ITER,
        ),
    ))

    # ── Newton / Broyden (only for small N) ──────────────────────────────────
    if N <= NEWTON_N_MAX:
        solvers.append((
            "Newton (exact J)",
            lambda: solve_newton(
                res_fn, jac_fn, theta_weight0, tol=tol, max_iter=200,
                theta_bounds=(_ETA_MIN, _ETA_MAX),
            ),
        ))
        solvers.append((
            "Broyden (rank-1 J)",
            lambda: solve_broyden(
                res_fn, jac_fn, theta_weight0, tol=tol, max_iter=500,
                theta_bounds=(_ETA_MIN, _ETA_MAX),
            ),
        ))

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

    theta_weight0 = model.initial_theta_weight(theta_topo, method="strengths")

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

    # Joint L-BFGS (full 4N)
    jr = solve_daecm_joint_lbfgs(model, tol=tol, max_iter=2_000, m=20)
    mre_j = model.max_relative_error(jr.theta_topo, jr.theta_weight)
    conv_j = "YES" if jr.converged else "NO"
    print(
        f"{'L-BFGS joint (4N)':<{col[0]}} {conv_j:>{col[1]}} {jr.topo_iterations:>{col[2]}} "
        f"{mre_j:>{col[3]}.3e} {jr.elapsed_time:>{col[4]}.3f} "
        f"{jr.peak_ram_bytes/1024:>{col[5]}.1f}"
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

    # Step 1: topology (DCM) — use L-BFGS with time limit
    try:
        theta_topo, topo_sr = _call_with_timeout(
            lambda: _run_topo_step(model, tol=tol, timeout=min(timeout * 0.3, 120.0),
                                   theta_topo0=theta_topo0),
            timeout * 0.3,
        )
    except (_TimeoutError, Exception):
        return None

    if not topo_sr.converged:
        # Topology did not converge; skip this network
        return None

    theta_weight0 = model.initial_theta_weight(theta_topo, method="strengths")
    weight_timeout = timeout - topo_sr.elapsed_time

    solvers = _make_solvers(model, theta_topo, theta_weight0, tol,
                            timeout=weight_timeout)

    results: dict[str, dict] = {}
    for name, fn in solvers:
        t_start = time.perf_counter()
        try:
            sr: SolverResult = _call_with_timeout(fn, weight_timeout)
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

    # ── Joint L-BFGS (full 4N optimisation, independent of the two-step topo) ──
    t_start = time.perf_counter()
    try:
        jr: DaECMResult = _call_with_timeout(
            lambda: solve_daecm_joint_lbfgs(
                model, tol=tol, max_iter=2_000, m=20,
            ),
            timeout,
        )
        mre = model.max_relative_error(jr.theta_topo, jr.theta_weight)
        results["L-BFGS joint (4N)"] = dict(
            converged=jr.converged,
            iterations=jr.topo_iterations,
            max_rel_err=mre,
            elapsed=jr.elapsed_time,
            peak_ram_mb=jr.peak_ram_bytes / 1024 / 1024,
            status="OK" if jr.converged else "NO-CONV",
        )
    except _TimeoutError:
        results["L-BFGS joint (4N)"] = dict(
            converged=False, iterations=0, max_rel_err=float("nan"),
            elapsed=time.perf_counter() - t_start,
            peak_ram_mb=float("nan"), status="TIMEOUT",
        )
    except Exception:
        results["L-BFGS joint (4N)"] = dict(
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
) -> tuple[dict[str, dict], list[int]]:
    """Run all DaECM weight solvers on *n_seeds* independent network realisations.

    Args:
        N:          Number of nodes.
        n_seeds:    Number of valid realisations to use.
        tol:        Convergence tolerance.
        timeout:    Per-solver time limit in seconds.
        start_seed: First random seed to try.
        verbose:    If True, print detailed output.

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
        results = _run_single_network(N, candidate_seed, tol, timeout)
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
    """Print the aggregate statistics table."""
    print(f"\n{'─'*100}")
    print(f"Aggregate Statistics  |  N={N:,}  |  {n_seeds} runs")
    print(f"(Performance metrics computed over converged runs only)")
    print(f"{'─'*100}")

    col = [50, 10, 22, 16, 16]
    header = (
        f"{'Method':<{col[0]}} {'Conv%':>{col[1]}} "
        f"{'Time(s) mean±2σ':^{col[2]}} "
        f"{'Iters mean±2σ':^{col[3]}} "
        f"{'MRE mean±2σ':^{col[4]}}"
    )
    print(header)
    print("-" * (sum(col) + len(col) - 1))

    for name, s in agg.items():
        conv_pct = f"{s['conv_rate']:.0%}"
        if np.isfinite(s["time_mean"]):
            time_str = f"{s['time_mean']:.3f}±{s['time_2sigma']:.3f}"
        else:
            time_str = "   —"
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
            f"{iter_str:^{col[3]}} "
            f"{err_str:^{col[4]}}"
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
) -> None:
    """Run multi-seed DaECM comparison for each size in *sizes*.

    Args:
        sizes:      List of node counts to benchmark.
        n_seeds:    Number of realisations per size.
        tol:        Convergence tolerance.
        timeout:    Per-solver time limit in seconds.
        start_seed: Base random seed.
    """
    all_agg: dict[int, dict[str, dict]] = {}
    all_bad: dict[int, list[int]] = {}

    for N in sizes:
        agg, bad = run_multi_seed_comparison(
            N=N, n_seeds=n_seeds, tol=tol, timeout=timeout,
            start_seed=start_seed, verbose=True,
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
                        help="Number of seeds for multi-seed run")
    parser.add_argument("--start_seed", type=int, default=0,
                        help="Starting seed for multi-seed run")
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL,
                        help="Convergence tolerance")
    parser.add_argument("--timeout", type=float, default=SOLVER_TIMEOUT,
                        help="Per-solver timeout in seconds")
    args = parser.parse_args()

    if args.sizes:
        run_scaling_comparison(
            sizes=args.sizes,
            n_seeds=args.n_seeds,
            tol=args.tol,
            timeout=args.timeout,
            start_seed=args.start_seed,
        )
    elif args.n is not None and args.seed is not None:
        run_comparison(N=args.n, seed=args.seed, tol=args.tol)
    elif args.n is not None:
        run_multi_seed_comparison(
            N=args.n,
            n_seeds=args.n_seeds,
            tol=args.tol,
            timeout=args.timeout,
            start_seed=args.start_seed,
        )
    else:
        run_scaling_comparison(
            sizes=DEFAULT_SIZES,
            n_seeds=args.n_seeds,
            tol=args.tol,
            timeout=args.timeout,
            start_seed=args.start_seed,
        )


if __name__ == "__main__":
    main()
