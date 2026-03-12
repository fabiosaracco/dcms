"""Quasi-Newton (L-BFGS) solver for MaxEnt network models.

The solver minimises −L(θ) (equivalently drives F(θ) → 0) using the
limited-memory BFGS algorithm with a Wolfe-condition line search.

The gradient of −L is ∇(−L) = −F(θ) = −residual(θ).

Reference:
    Nocedal & Wright, *Numerical Optimization*, 2nd ed., Chapter 7.
"""
from __future__ import annotations

import math
import time
import tracemalloc
from typing import Callable, Optional

import torch

from .base import SolverResult

_THETA_CLAMP = 50.0


def _wolfe_line_search(
    f: Callable[[torch.Tensor], float],
    g: Callable[[torch.Tensor], torch.Tensor],
    theta: torch.Tensor,
    p: torch.Tensor,
    f0: float,
    g0: torch.Tensor,
    c1: float = 1e-4,
    c2: float = 0.9,
    max_ls: int = 50,
    theta_lo: float = -_THETA_CLAMP,
    theta_hi: float = _THETA_CLAMP,
) -> tuple[float, torch.Tensor, torch.Tensor, bool]:
    """Wolfe condition line search.

    Args:
        f:        Scalar objective function.
        g:        Gradient function (returns torch.Tensor).
        theta:    Current parameter vector.
        p:        Search direction.
        f0:       f(theta).
        g0:       g(theta).
        c1:       Sufficient decrease constant (Armijo).
        c2:       Curvature constant.
        max_ls:   Max number of function evaluations.
        theta_lo: Lower bound for each θ component (default: -50).
        theta_hi: Upper bound for each θ component (default: +50).

    Returns:
        ``(alpha, theta_new, g_new, success)``
    """
    alpha = 1.0
    alpha_lo, alpha_hi = 0.0, math.inf
    f_lo = f0
    derphi0 = g0.dot(p).item()  # should be negative for descent direction

    theta_new = theta
    g_new = g0

    for _ in range(max_ls):
        theta_new = torch.clamp(theta + alpha * p, theta_lo, theta_hi)
        f_new = f(theta_new)
        g_new = g(theta_new)

        if f_new > f0 + c1 * alpha * derphi0 or (f_new >= f_lo and alpha_lo > 0):
            alpha_hi = alpha
            alpha = 0.5 * (alpha_lo + alpha_hi)
        else:
            derphi = g_new.dot(p).item()
            if abs(derphi) <= -c2 * derphi0:
                return alpha, theta_new, g_new, True
            if derphi >= 0:
                alpha_hi = alpha
            alpha_lo = alpha
            f_lo = f_new
            if alpha_hi < math.inf:
                alpha = 0.5 * (alpha_lo + alpha_hi)
            else:
                alpha = 2.0 * alpha_lo

        if alpha < 1e-14:
            break

    return alpha, theta_new, g_new, False


def solve_lbfgs(
    residual_fn: Callable[[torch.Tensor], torch.Tensor],
    theta0: "ArrayLike",  # type: ignore[name-defined]
    tol: float = 1e-8,
    max_iter: int = 1_000,
    m: int = 20,
    neg_loglik_fn: Optional[Callable[["ArrayLike"], float]] = None,  # type: ignore[name-defined]
    theta_bounds: Optional[tuple[float, float]] = None,
) -> SolverResult:
    """L-BFGS quasi-Newton solver.

    Minimises −L(θ) where L is the DCM (or DWCM) log-likelihood.  The
    gradient of −L is −F(θ) = −residual_fn(θ).  If *neg_loglik_fn* is not
    provided, the objective value is approximated as ½‖F‖² for line-search
    purposes (same zero, consistent descent direction).

    Args:
        residual_fn:   Function F(θ) → residual tensor, shape (2N,).
        theta0:        Initial parameter vector, shape (2N,).
        tol:           Convergence tolerance on the ℓ∞ residual.
        max_iter:      Maximum number of L-BFGS iterations.
        m:             Number of curvature pairs to store.
        neg_loglik_fn: Optional callable −L(θ) → float for exact line search.
                       If None, ½‖F‖² is used as a surrogate.
        theta_bounds:  Optional ``(theta_lo, theta_hi)`` box constraint applied
                       at every step (clamp).  Useful for DWCM where θ > 0 is
                       required for feasibility.  Defaults to
                       ``(-_THETA_CLAMP, +_THETA_CLAMP)`` = ``(-50, +50)``.

    Returns:
        :class:`~src.solvers.base.SolverResult` instance.
    """
    tracemalloc.start()
    t0 = time.perf_counter()

    if not isinstance(theta0, torch.Tensor):
        theta = torch.tensor(theta0, dtype=torch.float64)
    else:
        theta = theta0.clone().to(dtype=torch.float64)

    # Resolve and validate clamp bounds
    if theta_bounds is None:
        theta_lo: float = -_THETA_CLAMP
        theta_hi: float = _THETA_CLAMP
    else:
        if not isinstance(theta_bounds, (tuple, list)) or len(theta_bounds) != 2:
            raise ValueError(
                f"theta_bounds must be a 2-element (lo, hi) sequence or None; got {theta_bounds!r}"
            )
        try:
            theta_lo = float(theta_bounds[0])
            theta_hi = float(theta_bounds[1])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"theta_bounds values must be numeric; got {theta_bounds!r}"
            ) from exc
        if not (math.isfinite(theta_lo) and math.isfinite(theta_hi)):
            raise ValueError(
                f"theta_bounds values must be finite; got ({theta_lo}, {theta_hi})"
            )
        if theta_lo >= theta_hi:
            raise ValueError(
                f"theta_bounds must satisfy lo < hi; got ({theta_lo}, {theta_hi})"
            )

    # Clamp initial theta to the feasible box
    theta = theta.clamp(theta_lo, theta_hi)

    # Gradient of −L is −F (we minimise −L)
    def grad_neg_L(th: torch.Tensor) -> torch.Tensor:
        return -residual_fn(th)

    if neg_loglik_fn is not None:
        def objective(th: torch.Tensor) -> float:
            return neg_loglik_fn(th)
    else:
        # Surrogate: ½‖F‖² has the same zero but may differ in shape.
        def objective(th: torch.Tensor) -> float:
            F = residual_fn(th)
            return 0.5 * F.dot(F).item()

    n_iter = 0
    g = grad_neg_L(theta)   # gradient of −L = −F; ‖g‖∞ = ‖F‖∞
    f = objective(theta)
    residuals: list[float] = []

    converged = False
    message = "Maximum iterations reached without convergence."

    # L-BFGS history (all torch tensors)
    s_hist: list[torch.Tensor] = []
    y_hist: list[torch.Tensor] = []
    rho_hist: list[float] = []

    try:
        for _ in range(max_iter):
            # g = −F at current theta; check convergence using ‖g‖∞ = ‖F‖∞
            res_norm = g.abs().max().item()
            if res_norm < tol:
                converged = True
                message = f"Converged in {n_iter} iteration(s)."
                break

            if not math.isfinite(res_norm):
                message = f"NaN/Inf detected at iteration {n_iter}."
                break

            # ---------------------------------------------------------------
            # L-BFGS two-loop recursion to compute search direction p = -H g
            # ---------------------------------------------------------------
            k = len(s_hist)
            q = g.clone()
            alphas = torch.zeros(k, dtype=torch.float64)
            for i in range(k - 1, -1, -1):
                alphas[i] = rho_hist[i] * s_hist[i].dot(q).item()
                q -= alphas[i] * y_hist[i]

            if k > 0:
                s_k = s_hist[-1]
                y_k = y_hist[-1]
                gamma = s_k.dot(y_k).item() / (y_k.dot(y_k).item() + 1e-300)
                r = gamma * q
            else:
                r = q.clone()

            for i in range(k):
                beta = rho_hist[i] * y_hist[i].dot(r).item()
                r += (alphas[i].item() - beta) * s_hist[i]

            p = -r  # descent direction for −L

            # ---------------------------------------------------------------
            # Wolfe line search
            # ---------------------------------------------------------------
            _, theta_new, g_new, ls_ok = _wolfe_line_search(
                objective, grad_neg_L, theta, p, f, g,
                theta_lo=theta_lo, theta_hi=theta_hi,
            )

            if not ls_ok or not torch.isfinite(theta_new).all():
                # Fallback: simple steepest descent step
                alpha_fb = min(1e-3, 1.0 / (g.norm().item() + 1e-14))
                theta_new = torch.clamp(theta - alpha_fb * g, theta_lo, theta_hi)
                g_new = grad_neg_L(theta_new)

            # ---------------------------------------------------------------
            # Update L-BFGS history
            # ---------------------------------------------------------------
            s_k = theta_new - theta
            y_k = g_new - g
            sy = s_k.dot(y_k).item()
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
            n_iter += 1
            residuals.append(g.abs().max().item())
    finally:
        elapsed = time.perf_counter() - t0
        _, peak_ram = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    return SolverResult(
        theta=theta.detach().numpy(),
        converged=converged,
        iterations=n_iter,
        residuals=residuals,
        elapsed_time=elapsed,
        peak_ram_bytes=peak_ram,
        message=message,
    )
