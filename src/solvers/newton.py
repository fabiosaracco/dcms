"""Full Newton solver with exact Jacobian.

At each step solves the linear system

    J(θ) · δθ = −F(θ)

where J = ∂F/∂θ is the exact Jacobian provided by the model.

Tikhonov regularisation (``J + ε I``) is applied when the Jacobian is
poorly conditioned.  A simple backtracking Armijo line search controls
step size.

Cost per iteration: O(N³) (linear solve); RAM: O(N²).
Recommended only for N ≲ 5 000.
"""
from __future__ import annotations

import time
import tracemalloc
from typing import Callable, Optional

import numpy as np

from .base import SolverResult

_THETA_CLAMP = 50.0


def solve_newton(
    residual_fn: Callable[[np.ndarray], np.ndarray],
    jacobian_fn: Callable[[np.ndarray], np.ndarray],
    theta0: np.ndarray,
    tol: float = 1e-8,
    max_iter: int = 200,
    reg: float = 1e-8,
    max_reg: float = 1e2,
    armijo_c: float = 1e-4,
    max_ls: int = 30,
) -> SolverResult:
    """Full Newton method with Tikhonov regularisation and Armijo line search.

    The DCM Jacobian J = ∂F/∂θ = Hess(L) is negative semi-definite (L is
    concave).  We exploit this by solving the equivalent positive system

        −J(θ) · δθ = F(θ)

    so that −J (≈ positive definite) can be regularised as −J + εI.

    Args:
        residual_fn: F(θ) → residual vector, shape (2N,).
        jacobian_fn: J(θ) → Jacobian matrix, shape (2N, 2N).
        theta0:      Initial parameter vector, shape (2N,).
        tol:         Convergence tolerance (ℓ∞ residual).
        max_iter:    Maximum Newton iterations.
        reg:         Initial Tikhonov regularisation ε applied to −J.
        max_reg:     Maximum regularisation (triggers failure if exceeded).
        armijo_c:    Armijo sufficient-decrease constant.
        max_ls:      Max backtracking steps.

    Returns:
        :class:`~src.solvers.base.SolverResult` instance.
    """
    tracemalloc.start()
    t0 = time.perf_counter()

    theta = np.array(theta0, dtype=np.float64)
    F = residual_fn(theta)
    residuals: list[float] = [float(np.max(np.abs(F)))]
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

        J = jacobian_fn(theta)

        # Tikhonov regularisation on −J (which is PSD).
        # Since J is negative semi-definite, −J is PSD; we solve −J δθ = F.
        neg_J = -J
        eps = reg
        delta = None
        while eps <= max_reg:
            neg_J_reg = neg_J + eps * np.eye(n2)
            try:
                delta = np.linalg.solve(neg_J_reg, F)
                break
            except np.linalg.LinAlgError:
                eps *= 10.0

        if delta is None:
            message = "Jacobian singular even after heavy regularisation."
            break

        # Armijo backtracking line search on ‖F‖²
        alpha = 1.0
        f0 = float(F @ F)
        for _ in range(max_ls):
            theta_new = np.clip(theta + alpha * delta, -_THETA_CLAMP, _THETA_CLAMP)
            F_new = residual_fn(theta_new)
            if float(F_new @ F_new) <= f0 * (1.0 - 2.0 * armijo_c * alpha):
                break
            alpha *= 0.5

        theta = theta_new
        F = F_new
        residuals.append(float(np.max(np.abs(F))))

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
