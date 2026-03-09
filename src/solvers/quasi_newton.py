"""Quasi-Newton (L-BFGS) solver for MaxEnt network models.

The solver minimises −L(θ) (equivalently drives F(θ) → 0) using the
limited-memory BFGS algorithm with a Wolfe-condition line search.

The gradient is ∇(−L) = F(θ) (the residual).

Reference:
    Nocedal & Wright, *Numerical Optimization*, 2nd ed., Chapter 7.
"""
from __future__ import annotations

import time
import tracemalloc
from typing import Callable, Optional

import numpy as np

from .base import SolverResult

_THETA_CLAMP = 50.0


def _wolfe_line_search(
    f: Callable[[np.ndarray], float],
    g: Callable[[np.ndarray], np.ndarray],
    theta: np.ndarray,
    p: np.ndarray,
    f0: float,
    g0: np.ndarray,
    c1: float = 1e-4,
    c2: float = 0.9,
    max_ls: int = 50,
) -> tuple[float, np.ndarray, np.ndarray, bool]:
    """Wolfe condition line search.

    Args:
        f:       Scalar objective function.
        g:       Gradient function (returns np.ndarray).
        theta:   Current parameter vector.
        p:       Search direction.
        f0:      f(theta).
        g0:      g(theta).
        c1:      Sufficient decrease constant (Armijo).
        c2:      Curvature constant.
        max_ls:  Max number of function evaluations.

    Returns:
        ``(alpha, theta_new, g_new, success)``
    """
    alpha = 1.0
    alpha_lo, alpha_hi = 0.0, np.inf
    f_lo = f0
    derphi0 = g0 @ p  # should be negative for descent direction

    for _ in range(max_ls):
        theta_new = np.clip(theta + alpha * p, -_THETA_CLAMP, _THETA_CLAMP)
        f_new = f(theta_new)
        g_new = g(theta_new)

        if f_new > f0 + c1 * alpha * derphi0 or (f_new >= f_lo and alpha_lo > 0):
            alpha_hi = alpha
            alpha = 0.5 * (alpha_lo + alpha_hi)
        else:
            derphi = g_new @ p
            if abs(derphi) <= -c2 * derphi0:
                return alpha, theta_new, g_new, True
            if derphi >= 0:
                alpha_hi = alpha
            alpha_lo = alpha
            f_lo = f_new
            if alpha_hi < np.inf:
                alpha = 0.5 * (alpha_lo + alpha_hi)
            else:
                alpha = 2.0 * alpha_lo

        if alpha < 1e-14:
            break

    return alpha, theta_new, g_new, False


def solve_lbfgs(
    residual_fn: Callable[[np.ndarray], np.ndarray],
    theta0: np.ndarray,
    tol: float = 1e-8,
    max_iter: int = 1_000,
    m: int = 20,
    neg_loglik_fn: Optional[Callable[[np.ndarray], float]] = None,
) -> SolverResult:
    """L-BFGS quasi-Newton solver.

    Minimises −L(θ) where L is the DCM log-likelihood.  The gradient of
    −L is −F(θ) = −residual_fn(θ).  If *neg_loglik_fn* is not provided,
    the objective value is approximated as ½‖F‖² for line-search purposes
    (which still gives a consistent descent direction).

    Args:
        residual_fn:   Function F(θ) → residual vector, shape (2N,).
        theta0:        Initial parameter vector, shape (2N,).
        tol:           Convergence tolerance on the ℓ∞ residual.
        max_iter:      Maximum number of L-BFGS iterations.
        m:             Number of curvature pairs to store.
        neg_loglik_fn: Optional callable −L(θ) → float for exact line search.
                       If None, ½‖F‖² is used as a surrogate.

    Returns:
        :class:`~src.solvers.base.SolverResult` instance.
    """
    tracemalloc.start()
    t0 = time.perf_counter()

    theta = np.array(theta0, dtype=np.float64)

    # Gradient of −L is −F (we minimise −L)
    def grad_neg_L(th: np.ndarray) -> np.ndarray:
        return -residual_fn(th)

    if neg_loglik_fn is not None:
        def objective(th: np.ndarray) -> float:
            return neg_loglik_fn(th)
    else:
        # Surrogate: ½‖F‖² has the same zero but may differ in shape.
        # Use it only for the Armijo check in line search.
        def objective(th: np.ndarray) -> float:
            F = residual_fn(th)
            return 0.5 * float(F @ F)

    g = grad_neg_L(theta)   # gradient of −L = −F
    f = objective(theta)
    # g = −F, so F = −g; track ℓ∞ of F = ℓ∞ of g
    residuals: list[float] = [float(np.max(np.abs(g)))]

    converged = False
    message = "Maximum iterations reached without convergence."

    # L-BFGS history
    s_hist: list[np.ndarray] = []
    y_hist: list[np.ndarray] = []
    rho_hist: list[float] = []

    for iteration in range(max_iter):
        # g = −F at current theta; check convergence using ‖g‖∞ = ‖F‖∞
        res_norm = float(np.max(np.abs(g)))
        if res_norm < tol:
            converged = True
            message = f"Converged in {iteration} iteration(s)."
            break

        if not np.isfinite(res_norm):
            message = f"NaN/Inf detected at iteration {iteration}."
            break

        # ---------------------------------------------------------------
        # L-BFGS two-loop recursion to compute search direction p = -H g
        # ---------------------------------------------------------------
        k = len(s_hist)
        q = g.copy()
        alphas = np.zeros(k)
        for i in range(k - 1, -1, -1):
            alphas[i] = rho_hist[i] * (s_hist[i] @ q)
            q -= alphas[i] * y_hist[i]

        if k > 0:
            s_k = s_hist[-1]
            y_k = y_hist[-1]
            gamma = (s_k @ y_k) / (y_k @ y_k + 1e-300)
            r = gamma * q
        else:
            r = q.copy()

        for i in range(k):
            beta = rho_hist[i] * (y_hist[i] @ r)
            r += (alphas[i] - beta) * s_hist[i]

        p = -r  # descent direction for −L

        # ---------------------------------------------------------------
        # Wolfe line search
        # ---------------------------------------------------------------
        alpha_step, theta_new, g_new, ls_ok = _wolfe_line_search(
            objective, grad_neg_L, theta, p, f, g
        )

        if not ls_ok or not np.isfinite(theta_new).all():
            # Fallback: simple steepest descent step
            alpha_step = min(1e-3, 1.0 / (np.linalg.norm(g) + 1e-14))
            theta_new = np.clip(theta - alpha_step * g, -_THETA_CLAMP, _THETA_CLAMP)
            g_new = grad_neg_L(theta_new)

        # ---------------------------------------------------------------
        # Update L-BFGS history
        # ---------------------------------------------------------------
        s_k = theta_new - theta
        y_k = g_new - g
        sy = float(s_k @ y_k)
        if sy > 1e-14:
            if len(s_hist) == m:
                s_hist.pop(0)
                y_hist.pop(0)
                rho_hist.pop(0)
            s_hist.append(s_k)
            y_hist.append(y_k)
            rho_hist.append(1.0 / sy)

        theta = theta_new
        g = g_new
        f = objective(theta)
        residuals.append(float(np.max(np.abs(g))))

    elapsed = time.perf_counter() - t0
    _, peak_ram = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return SolverResult(
        theta=theta,
        converged=converged,
        iterations=len(residuals),
        residuals=residuals,
        elapsed_time=elapsed,
        peak_ram_bytes=peak_ram,
        message=message,
    )
