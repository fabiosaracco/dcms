"""Broyden's method (rank-1 approximate Jacobian update).

The exact Jacobian is computed only at the first iteration.  Subsequent
iterations use the Sherman-Morrison formula to update J⁻¹ in O(N²):

    J⁻¹_{k+1} = J⁻¹_k + (s_k − J⁻¹_k y_k) sᵀ_k J⁻¹_k / (sᵀ_k J⁻¹_k y_k)

where  s_k = θ_{k+1} − θ_k  and  y_k = F_{k+1} − F_k.

A simple Armijo backtracking line search is included.

Cost per iteration: O(N²) (matrix-vector product); RAM: O(N²).
"""
from __future__ import annotations

import time
import tracemalloc
from typing import Callable

import numpy as np

from .base import SolverResult

_THETA_CLAMP = 50.0


def solve_broyden(
    residual_fn: Callable[[np.ndarray], np.ndarray],
    jacobian_fn: Callable[[np.ndarray], np.ndarray],
    theta0: np.ndarray,
    tol: float = 1e-8,
    max_iter: int = 500,
    reg: float = 1e-8,
    armijo_c: float = 1e-4,
    max_ls: int = 30,
) -> SolverResult:
    """Broyden's good-Broyden method with Sherman-Morrison updates.

    Args:
        residual_fn: F(θ) → residual vector, shape (2N,).
        jacobian_fn: J(θ) → exact Jacobian at θ₀ only, shape (2N, 2N).
        theta0:      Initial parameter vector, shape (2N,).
        tol:         Convergence tolerance (ℓ∞ residual).
        max_iter:    Maximum iterations.
        reg:         Tikhonov regularisation for the initial Jacobian.
        armijo_c:    Armijo constant for line search.
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

    # Compute initial J^{-1} from the exact Jacobian (negative semi-definite)
    J0 = jacobian_fn(theta) + reg * np.eye(n2)
    try:
        J_inv = np.linalg.inv(J0)
    except np.linalg.LinAlgError:
        J_inv = np.eye(n2)

    for iteration in range(max_iter):
        res_norm = float(np.max(np.abs(F)))
        if res_norm < tol:
            converged = True
            message = f"Converged in {iteration} iteration(s)."
            break

        if not np.isfinite(res_norm):
            message = f"NaN/Inf detected at iteration {iteration}."
            break

        # Newton-like step: δ = J^{-1}(−F), with J negative-definite
        delta = J_inv @ (-F)

        # Armijo backtracking on ‖F‖²
        alpha = 1.0
        f0 = float(F @ F)
        for _ in range(max_ls):
            theta_new = np.clip(theta + alpha * delta, -_THETA_CLAMP, _THETA_CLAMP)
            F_new = residual_fn(theta_new)
            if float(F_new @ F_new) <= f0 * (1.0 - 2.0 * armijo_c * alpha):
                break
            alpha *= 0.5

        s = theta_new - theta   # actual step taken
        y = F_new - F           # change in residual

        # Sherman-Morrison update of J^{-1}
        # Good-Broyden: B_{k+1} = B_k + (y - B_k s) s^T / ||s||^2
        # Inverse update: B^{-1}_{k+1} = B^{-1}_k
        #     + (s - B^{-1}_k y) (s^T B^{-1}_k) / (s^T B^{-1}_k y)
        Jinv_y = J_inv @ y
        denom = float(s @ Jinv_y)
        if abs(denom) > 1e-14:
            numer = np.outer(s - Jinv_y, s @ J_inv)
            J_inv = J_inv + numer / denom

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
