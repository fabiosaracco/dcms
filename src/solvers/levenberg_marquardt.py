"""Levenberg-Marquardt solver.

Solves the non-linear least-squares problem min ½ ‖F(θ)‖² using the
Levenberg-Marquardt algorithm:

    (JᵀJ + λI) δθ = −Jᵀ F(θ)

where λ is adapted each iteration:

* If the step *decreases* ‖F‖, accept it and decrease λ (more Newton-like).
* If the step *increases* ‖F‖, reject it and increase λ (more gradient-like).

For large N the full Jacobian is expensive (O(N²) RAM).  In that regime the
solver falls back to using only the diagonal of JᵀJ — controlled by the
``diagonal_only`` flag.

Reference:
    Moré, J.J. (1978). The Levenberg-Marquardt algorithm. In *Numerical
    Analysis*, Lecture Notes in Mathematics 630.
"""
from __future__ import annotations

import time
import tracemalloc
from typing import Callable, Optional

import numpy as np

from .base import SolverResult

_THETA_CLAMP = 50.0


def solve_lm(
    residual_fn: Callable[[np.ndarray], np.ndarray],
    jacobian_fn: Callable[[np.ndarray], np.ndarray],
    theta0: np.ndarray,
    tol: float = 1e-8,
    max_iter: int = 500,
    lam0: float = 1e-3,
    lam_up: float = 10.0,
    lam_down: float = 0.1,
    lam_max: float = 1e10,
    diagonal_only: bool = False,
) -> SolverResult:
    """Levenberg-Marquardt with adaptive damping.

    Args:
        residual_fn:   F(θ) → residual vector, shape (2N,).
        jacobian_fn:   J(θ) → Jacobian matrix, shape (2N, 2N).
        theta0:        Initial parameter vector, shape (2N,).
        tol:           Convergence tolerance (ℓ∞ residual).
        max_iter:      Maximum iterations.
        lam0:          Initial damping parameter λ.
        lam_up:        Multiplicative factor to increase λ on rejection.
        lam_down:      Multiplicative factor to decrease λ on acceptance.
        lam_max:       Maximum λ before declaring failure.
        diagonal_only: If True, use only diag(JᵀJ) instead of the full JᵀJ.
                       Cheaper for large N.

    Returns:
        :class:`~src.solvers.base.SolverResult` instance.
    """
    tracemalloc.start()
    t0 = time.perf_counter()

    theta = np.array(theta0, dtype=np.float64)
    F = residual_fn(theta)
    cost = float(F @ F)
    residuals: list[float] = [float(np.max(np.abs(F)))]
    lam = lam0
    converged = False
    message = "Maximum iterations reached without convergence."
    n2 = len(theta)

    for iteration in range(max_iter):
        res_norm = float(np.max(np.abs(F)))
        if res_norm < tol:
            converged = True
            message = f"Converged in {iteration} iteration(s)."
            break

        if not np.isfinite(res_norm):
            message = f"NaN/Inf detected at iteration {iteration}."
            break

        J = jacobian_fn(theta)  # (2N, 2N), negative semi-definite Hess(L)
        # LM normal equations: (JᵀJ + λI) δ = Jᵀ(−F) = −JᵀF
        # δ solves the linearised least-squares min ½‖Jδ+F‖² + ½λ‖δ‖²
        JtF = J.T @ F           # (2N,)

        if diagonal_only:
            JtJ_diag = np.sum(J ** 2, axis=0)
            delta = -JtF / (JtJ_diag + lam)
        else:
            JtJ = J.T @ J       # (2N, 2N)
            A = JtJ + lam * np.eye(n2)
            try:
                delta = np.linalg.solve(A, -JtF)
            except np.linalg.LinAlgError:
                lam *= lam_up
                continue

        theta_new = np.clip(theta + delta, -_THETA_CLAMP, _THETA_CLAMP)
        F_new = residual_fn(theta_new)
        cost_new = float(F_new @ F_new)

        if cost_new < cost:
            # Accept step, reduce damping
            theta = theta_new
            F = F_new
            cost = cost_new
            lam = max(lam * lam_down, 1e-14)
            residuals.append(float(np.max(np.abs(F))))
        else:
            # Reject step, increase damping (no new residual entry)
            lam = lam * lam_up

        if lam > lam_max:
            message = f"Damping λ={lam:.2e} exceeded maximum. Stopping."
            break

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
