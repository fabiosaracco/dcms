"""Fixed-point iteration solver for MaxEnt network models.

Two variants are implemented:

* **Jacobi** — all multipliers are updated simultaneously using values from
  the *previous* iteration.
* **Gauss-Seidel** — out-multipliers are updated first; the updated values
  are immediately used when computing new in-multipliers.

Both variants support an optional damping factor α ∈ (0, 1].  The update
rule in physical space (x = exp(−θ)) is:

    x_i^new = k_out_i / Σ_{j≠i} y_j / (1 + x_i * y_j)

followed by a damped step:

    x_i^(t+1) = (x_i^new)^α · (x_i^(t))^(1−α)

or equivalently, in θ-space:

    θ_i^(t+1) = α · θ_i^new + (1−α) · θ_i^(t)
"""
from __future__ import annotations

import time
import tracemalloc
from typing import Callable, Optional

import numpy as np

from .base import SolverResult

# Clamp θ to avoid exp overflow/underflow
_THETA_CLAMP = 50.0


def solve_fixed_point(
    residual_fn: Callable[[np.ndarray], np.ndarray],
    theta0: np.ndarray,
    k_out: np.ndarray,
    k_in: np.ndarray,
    tol: float = 1e-8,
    max_iter: int = 10_000,
    damping: float = 1.0,
    variant: str = "gauss-seidel",
) -> SolverResult:
    """Fixed-point iteration for the DCM.

    Args:
        residual_fn: Function F(θ) → residual vector (used only for logging).
        theta0: Initial parameter vector [θ_out | θ_in], shape (2N,).
        k_out:  Observed out-degree sequence, shape (N,).
        k_in:   Observed in-degree sequence, shape (N,).
        tol:    Convergence tolerance on the ℓ∞ residual norm.
        max_iter: Maximum number of iterations.
        damping: Damping factor α ∈ (0, 1].  α=1 → no damping.
        variant: ``"jacobi"`` or ``"gauss-seidel"``.

    Returns:
        :class:`~src.solvers.base.SolverResult` instance.
    """
    if variant not in ("jacobi", "gauss-seidel"):
        raise ValueError(f"Unknown variant {variant!r}. Choose 'jacobi' or 'gauss-seidel'.")
    if not (0.0 < damping <= 1.0):
        raise ValueError(f"damping must be in (0, 1], got {damping}")

    k_out = np.asarray(k_out, dtype=np.float64)
    k_in = np.asarray(k_in, dtype=np.float64)
    N = len(k_out)

    tracemalloc.start()
    t0 = time.perf_counter()

    theta = np.array(theta0, dtype=np.float64)
    residuals: list[float] = []
    converged = False
    message = "Maximum iterations reached without convergence."

    for iteration in range(max_iter):
        x = np.exp(-theta[:N])
        y = np.exp(-theta[N:])

        # ---------------------------------------------------------------
        # Compute the denominator sums D_out_i = Σ_{j≠i} y_j/(1+x_i*y_j)
        # and D_in_i = Σ_{j≠i} x_j/(1+x_j*y_i)
        # ---------------------------------------------------------------
        # xy[i, j] = x_i * y_j, shape (N, N)
        xy = x[:, None] * y[None, :]
        xy_diag_zero = xy.copy()
        np.fill_diagonal(xy_diag_zero, 0.0)

        # Denominators for the out-degree update
        # D_out[i] = Σ_{j≠i} y_j / (1 + x_i * y_j)
        D_out = (y[None, :] / (1.0 + xy)).sum(axis=1)
        diag_correction_out = y / (1.0 + np.diag(xy))
        D_out -= diag_correction_out  # remove self-loop term

        # New x values
        x_new = np.where(D_out > 0, k_out / D_out, x)

        if variant == "gauss-seidel":
            # Use updated x values immediately
            x_upd = x_new
        else:
            x_upd = x  # Jacobi: keep old values

        # Denominators for the in-degree update using (possibly updated) x
        xy2 = x_upd[:, None] * y[None, :]
        D_in = (x_upd[None, :] / (1.0 + xy2.T)).sum(axis=1)
        diag_correction_in = x_upd / (1.0 + np.diag(xy2))
        D_in -= diag_correction_in

        y_new = np.where(D_in > 0, k_in / D_in, y)

        # ---------------------------------------------------------------
        # Damped update in θ-space
        # ---------------------------------------------------------------
        theta_out_new = -np.log(np.clip(x_new, 1e-300, None))
        theta_in_new = -np.log(np.clip(y_new, 1e-300, None))

        theta_out_new = np.clip(theta_out_new, -_THETA_CLAMP, _THETA_CLAMP)
        theta_in_new = np.clip(theta_in_new, -_THETA_CLAMP, _THETA_CLAMP)

        theta_new = np.concatenate([theta_out_new, theta_in_new])
        theta = damping * theta_new + (1.0 - damping) * theta

        # ---------------------------------------------------------------
        # Convergence check (ℓ∞ norm of residual)
        # ---------------------------------------------------------------
        res = residual_fn(theta)
        res_norm = float(np.max(np.abs(res)))

        if not np.isfinite(res_norm):
            message = f"NaN/Inf detected at iteration {iteration}."
            break

        residuals.append(res_norm)

        if res_norm < tol:
            converged = True
            message = f"Converged in {iteration + 1} iteration(s)."
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
