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
import math
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


class _TimeoutError(Exception):
    """Raised by the SIGALRM handler when a solver exceeds its wall-clock budget."""


def _call_with_timeout(fn: Callable, timeout_s: float):
    """Call *fn()* and raise _TimeoutError if it does not finish in *timeout_s* seconds.

    Uses POSIX SIGALRM (Linux/macOS only).  On platforms where SIGALRM is
    unavailable the function falls back to running without a timeout.

    Args:
        fn:        Zero-argument callable to invoke.
        timeout_s: Maximum allowed wall-clock seconds (rounded to nearest int).

    Returns:
        Whatever *fn()* returns.

    Raises:
        _TimeoutError: If *fn()* exceeds *timeout_s* seconds.
    """
    if not hasattr(signal, "SIGALRM"):  # Windows fallback
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

# Newton and Broyden need the full N×N Jacobian (O(N²) RAM).
NEWTON_N_MAX: int = 500

# Full-Jacobian LM shares the same RAM cost as Newton.
FULL_JAC_LM_N_MAX: int = 500

# L-BFGS is O(N) RAM per step but each step calls the residual (O(N²) cost).
# For N > LBFGS_N_MAX the cost per function evaluation is so high that L-BFGS
# becomes impractical within a reasonable timeout.
LBFGS_N_MAX: int = 5_000

# LM-diagonal (using hessian_diag, O(N) RAM) — applicable up to this N.
# For N > DIAG_LM_N_MAX, the diagonal hessian LM is still O(N) but may
# converge poorly due to ill-conditioning; we keep it for all sizes.
DIAG_LM_N_MAX: int = 200_000

# Default network sizes to benchmark
DEFAULT_SIZES: list[int] = [1_000, 5_000, 10_000, 50_000]

# Connection density (rho) for the Chung-Lu generator.
# Using the wng default (0.001) produces networks with moderate degree heterogeneity
# that are numerically tractable for DWCM. Denser networks (rho > 0.02) can push
# β values close to 1, making all solver methods numerically challenging.
DEFAULT_RHO: float = 0.001

# Convergence tolerance used by all solvers in this benchmark
DEFAULT_TOL: float = 1e-6

# Maximum solver wall-clock time (seconds) — skip if exceeded
SOLVER_TIMEOUT: float = 300.0

# Number of seeds for multi-seed comparison
DEFAULT_N_SEEDS: int = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _solve_lm_diag_dwcm(
    model: DWCMModel,
    theta0: torch.Tensor,
    tol: float = DEFAULT_TOL,
    theta_bounds: tuple[float, float] = (_ETA_MIN, _ETA_MAX),
    max_iter: int = 500,
    lam0: float = 1e-3,
    lam_up: float = 10.0,
    lam_down: float = 0.1,
    lam_max: float = 1e10,
) -> SolverResult:
    """LM with O(N) diagonal Hessian (no full Jacobian materialised).

    Uses model.hessian_diag() and model.residual() only, avoiding the O(N²)
    Jacobian allocation.  The normal equations reduce to:

        (diag(H) + λ) δ = −F

    where H = ∂F/∂θ = Hess(L) (negative semi-definite) and diag(H) ≤ 0.

    We regularise as: (−diag(H) + λ) δ = F, so δ = F / (−diag(H) + λ).

    Args:
        model:        DWCMModel instance.
        theta0:       Initial parameter vector, shape (2N,).
        tol:          Convergence tolerance.
        theta_bounds: (theta_lo, theta_hi) clamp applied at every step.
        max_iter:     Maximum iterations.
        lam0:         Initial damping λ.
        lam_up:       λ increase factor on rejection.
        lam_down:     λ decrease factor on acceptance.
        lam_max:      Maximum λ before failure.

    Returns:
        :class:`~src.solvers.base.SolverResult` instance.
    """
    import tracemalloc as _tm
    import time as _t
    from src.solvers.base import SolverResult as _SR

    _tm.start()
    t0 = _t.perf_counter()

    theta_lo, theta_hi = theta_bounds
    theta = theta0.clone().to(dtype=torch.float64).clamp(theta_lo, theta_hi)

    F = model.residual(theta.clamp(theta_lo, theta_hi))
    cost = F.dot(F).item()
    lam = lam0

    residuals: list[float] = []
    converged = False
    n_iter = 0
    message = "Maximum iterations reached without convergence."

    try:
        for _ in range(max_iter):
            res_norm = F.abs().max().item()
            if not math.isfinite(res_norm):
                message = f"NaN/Inf at iteration {n_iter}."
                break
            if res_norm < tol:
                converged = True
                message = f"Converged in {n_iter} iteration(s)."
                break

            # Diagonal of Hess(L) = ∂F/∂θ — all entries ≤ 0
            h_diag = model.hessian_diag(theta.clamp(theta_lo, theta_hi))
            neg_h = -h_diag  # ≥ 0

            # LM step: δ = F / (−diag(H) + λ)
            delta = F / (neg_h + lam)

            theta_new = (theta + delta).clamp(theta_lo, theta_hi)
            F_new = model.residual(theta_new)
            cost_new = F_new.dot(F_new).item()

            if cost_new < cost:
                theta = theta_new
                F = F_new
                cost = cost_new
                lam = max(lam * lam_down, 1e-14)
                n_iter += 1
                residuals.append(F.abs().max().item())
            else:
                lam *= lam_up

            if lam > lam_max:
                message = f"Damping λ={lam:.2e} exceeded maximum."
                break
    finally:
        elapsed = _t.perf_counter() - t0
        _, peak_ram = _tm.get_traced_memory()
        _tm.stop()

    return _SR(
        theta=theta.detach().numpy(),
        converged=converged,
        iterations=n_iter,
        residuals=residuals,
        elapsed_time=elapsed,
        peak_ram_bytes=peak_ram,
        message=message,
    )


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


def _lbfgs_multistart(
    model: DWCMModel,
    theta0: torch.Tensor,
    tol: float,
    max_iter: int,
    n_starts: int = 4,
) -> SolverResult:
    """L-BFGS with multiple initialisations if the default init fails.

    Tries starting points in this order, stopping as soon as one converges:

    1. ``theta0`` — the "strengths" mean-field approximation (default).
    2. ``"normalized"`` — β_i = s_i / Σ_j s_j (Squartini & Garlaschelli 2011).
    3. ``"uniform"``    — all betas equal to the median of the strengths init.
    4. ``"random"``     — uniform random θ ∈ [0.1, 2.0] (with torch.manual_seed).

    Extra random restarts fill up to ``n_starts`` total attempts.  Returns the
    result with the lowest MaxRelError across all starts.

    Args:
        model:    DWCMModel instance.
        theta0:   Default initial parameter vector (from model.initial_theta).
        tol:      Convergence tolerance.
        max_iter: Maximum iterations per L-BFGS run.
        n_starts: Total number of starting points to try (including theta0).

    Returns:
        :class:`~src.solvers.base.SolverResult` with the best solution found.
    """
    import time as _t
    from src.solvers.base import SolverResult as _SR

    res_fn = _make_clamped_residual(model)
    nll_fn = _make_clamped_nll(model)

    best_result: Optional[SolverResult] = None
    best_err = float("inf")
    total_iters = 0
    combined_ram = 0
    t_start = _t.perf_counter()

    # Build ordered list of starting points
    starts: list[torch.Tensor] = [theta0]
    for method in ("normalized", "uniform"):
        starts.append(model.initial_theta(method))
    # Fill remaining slots with random restarts
    for i in range(max(0, n_starts - len(starts))):
        torch.manual_seed(i)
        starts.append(model.initial_theta("random"))

    for i, t0 in enumerate(starts[:n_starts]):
        torch.manual_seed(i)
        result = solve_lbfgs(
            res_fn, t0, tol=tol, m=20, max_iter=max_iter,
            neg_loglik_fn=nll_fn, theta_bounds=(_ETA_MIN, _ETA_MAX),
        )
        total_iters += result.iterations
        combined_ram = max(combined_ram, result.peak_ram_bytes)
        err = model.max_relative_error(result.theta)
        if err < best_err:
            best_err = err
            best_result = result
        if result.converged:
            break

    elapsed = _t.perf_counter() - t_start
    assert best_result is not None
    return _SR(
        theta=best_result.theta,
        converged=best_result.converged,
        iterations=total_iters,
        residuals=best_result.residuals,
        elapsed_time=elapsed,
        peak_ram_bytes=combined_ram,
        message=best_result.message,
    )


def _make_solvers(
    model: DWCMModel,
    theta0: torch.Tensor,
    tol: float,
    timeout: float = SOLVER_TIMEOUT,
) -> list[tuple[str, Callable[[], SolverResult]]]:
    """Return a list of (name, callable) solver pairs for *model*.

    Methods requiring O(N²) RAM are omitted for large N.

    Args:
        model:   The DWCMModel instance.
        theta0:  Initial parameter vector (all positive).
        tol:     Convergence tolerance.
        timeout: Per-solver hard timeout in seconds (used to set iteration budgets).

    Returns:
        Ordered list of ``(name, solver_callable)`` pairs.
    """
    N = model.N
    res_fn = _make_clamped_residual(model)
    nll_fn = _make_clamped_nll(model)
    jac_fn = _make_clamped_jacobian(model)

    # ---------------------------------------------------------------------------
    # Per-method iteration budgets.
    # The residual cost is O(N²) per evaluation (chunked for N > _LARGE_N_THRESHOLD).
    # Empirical calibration (chunked, single CPU core):
    #   N=100 →   0.3 ms,  N=1k →  8 ms,  N=5k → 200 ms,
    #   N=10k → ~1.5  s,  N=50k → ~15 s
    # The calibration formula uses a piecewise fit:
    #   - for N ≤ 5k:  cost ≈ (N/1k)² × 8 ms  (GPU-friendly tensor ops)
    #   - for N > 5k:  cost ≈ (N/1k)² × 15 ms  (chunked overhead dominates)
    # ---------------------------------------------------------------------------
    if N <= 5_000:
        residual_s = max(3e-4, (N / 1_000) ** 2 * 8e-3)
    else:
        residual_s = max(3e-4, (N / 1_000) ** 2 * 15e-3)

    # Plain FP-GS/Jacobi: only ~1 s budget (enough to see it won't converge)
    MAX_FP_PLAIN_ITER: int = max(10, min(5_000, int(1.0 / residual_s)))
    # Anderson-accelerated FP: budget = min(timeout, 300 s)
    # At N=10k (residual≈1.5s) this gives ~200 iterations within 300s
    anderson_budget_s = min(timeout, 300.0) if timeout > 0 else 300.0
    MAX_FP_ANDERSON_ITER: int = max(20, min(10_000, int(anderson_budget_s / residual_s)))
    # L-BFGS: each iter costs ~3-5 residuals (gradient + line search)
    MAX_LBFGS_ITER: int = max(20, min(2_000, int(anderson_budget_s / (5 * residual_s))))
    # Diagonal LM: ~10 s budget (it rarely converges for DWCM anyway)
    MAX_LM_ITER: int = max(10, min(500, int(10.0 / (3 * residual_s))))

    solvers: list[tuple[str, Callable[[], SolverResult]]] = []

    # ── Fixed-point GS α=1.0 (plain, fast) ─────────────────────────────────
    solvers.append((
        "FP-GS α=1.0",
        lambda: solve_fixed_point_dwcm(
            res_fn, theta0, model.s_out, model.s_in,
            tol=tol, damping=1.0, variant="gauss-seidel",
            max_iter=MAX_FP_PLAIN_ITER, anderson_depth=0,
        ),
    ))

    # ── Fixed-point GS α=0.5 (damped) ───────────────────────────────────────
    solvers.append((
        "FP-GS α=0.5",
        lambda: solve_fixed_point_dwcm(
            res_fn, theta0, model.s_out, model.s_in,
            tol=tol, damping=0.5, variant="gauss-seidel",
            max_iter=MAX_FP_PLAIN_ITER, anderson_depth=0,
        ),
    ))

    # ── Fixed-point GS + Anderson depth=5, multi-start ──────────────────────
    # Tries "strengths", "normalized", "uniform", and "random" initialisations
    # in sequence, stopping as soon as one converges.  The Anderson mixing
    # (Walker & Ni 2011) extrapolates from the last 5 FP iterates and typically
    # achieves 100% convergence for Pareto networks regardless of the init.
    def _fp_anderson_multistart() -> SolverResult:
        import time as _t
        from src.solvers.base import SolverResult as _SR
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
            r = solve_fixed_point_dwcm(
                res_fn, t0_cand, model.s_out, model.s_in,
                tol=tol, damping=1.0, variant="gauss-seidel",
                max_iter=MAX_FP_ANDERSON_ITER, anderson_depth=5,
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

    solvers.append(("FP-GS Anderson(5) multi-init", _fp_anderson_multistart))

    # ── Fixed-point Jacobi ──────────────────────────────────────────────────
    solvers.append((
        "FP-Jacobi",
        lambda: solve_fixed_point_dwcm(
            res_fn, theta0, model.s_out, model.s_in,
            tol=tol, damping=1.0, variant="jacobi",
            max_iter=MAX_FP_PLAIN_ITER, anderson_depth=0,
        ),
    ))

    # ── L-BFGS multi-start (skipped for N > LBFGS_N_MAX) ───────────────────
    # At large N each gradient evaluation costs O(N²) = ~1.5s at N=10k.
    # With 5 evals/iter, a 300s timeout allows only ~40 L-BFGS steps — often
    # insufficient.  Skip for N > LBFGS_N_MAX and rely on Anderson FP instead.
    if N <= LBFGS_N_MAX:
        solvers.append((
            "L-BFGS (multi-start)",
            lambda: _lbfgs_multistart(model, theta0, tol=tol,
                                      max_iter=MAX_LBFGS_ITER, n_starts=4),
        ))

    # ── Diagonal LM (O(N) RAM, always applicable) ───────────────────────────
    solvers.append((
        "LM (diag Hessian)",
        lambda: _solve_lm_diag_dwcm(model, theta0, tol=tol,
                                     theta_bounds=(_ETA_MIN, _ETA_MAX),
                                     max_iter=MAX_LM_ITER),
    ))

    # ── Newton, Broyden, full-Jacobian LM — only for small N (O(N²) RAM) ──
    if N <= NEWTON_N_MAX:
        solvers.append((
            "Newton (exact J)",
            lambda: solve_newton(
                res_fn, jac_fn, theta0, tol=tol, max_iter=200,
                theta_bounds=(_ETA_MIN, _ETA_MAX),
            ),
        ))
        solvers.append((
            "Broyden (rank-1 J)",
            lambda: solve_broyden(
                res_fn, jac_fn, theta0, tol=tol, max_iter=500,
                theta_bounds=(_ETA_MIN, _ETA_MAX),
            ),
        ))

    if N <= FULL_JAC_LM_N_MAX:
        solvers.append((
            "LM (full Jacobian)",
            lambda: solve_lm(
                res_fn, jac_fn, theta0, tol=tol, diagonal_only=False,
                max_iter=500, theta_bounds=(_ETA_MIN, _ETA_MAX),
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
    solvers = _make_solvers(model, theta0, tol, timeout=timeout)

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
            f"{n_seeds} runs  |  tol={tol:.0e}  |  start_seed={start_seed}"
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
    parser.add_argument("--start_seed", type=int, default=None,
                        help="Base random seed (default: random, from current time)")
    args = parser.parse_args()

    # If start_seed is not provided, use a time-based random seed so that
    # different invocations naturally sample different realisations.
    import time as _time_mod
    effective_start_seed: int = (
        args.start_seed
        if args.start_seed is not None
        else int(_time_mod.time() * 1000) % (2 ** 31)
    )

    if args.sizes is not None:
        # Scaling mode: run multi-seed for each size
        run_scaling_comparison(
            sizes=args.sizes,
            n_seeds=args.n_seeds,
            tol=args.tol,
            timeout=args.timeout,
            start_seed=effective_start_seed,
        )
    elif args.seed is not None:
        # Single-network mode
        run_comparison(N=args.n, seed=args.seed, tol=args.tol)
    else:
        # Default: multi-seed on single N
        run_multi_seed_comparison(
            N=args.n, n_seeds=args.n_seeds, tol=args.tol,
            timeout=args.timeout, start_seed=effective_start_seed,
        )
