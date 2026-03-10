"""Fixed-point iteration solver for MaxEnt network models.

Two variants are implemented:

* **Jacobi** — all multipliers are updated simultaneously using values from
  the *previous* iteration.
* **Gauss-Seidel** — out-multipliers are updated first; the updated values
  are immediately used when computing new in-multipliers.

Both variants support an optional damping factor α ∈ (0, 1].  The update
rule in physical space (x = exp(−θ)) is:

    x_i^new = k_out_i / Σ_{j≠i} y_j / (1 + x_i * y_j)

followed by a damped step in θ-space:

    θ_i^(t+1) = α · θ_i^new + (1−α) · θ_i^(t)
"""
from __future__ import annotations

import math
import time
import tracemalloc
from typing import Callable

import torch

from .base import SolverResult

# Clamp θ to avoid exp overflow/underflow
_THETA_CLAMP = 50.0


def solve_fixed_point(
    residual_fn: Callable[[torch.Tensor], torch.Tensor],
    theta0: "ArrayLike",  # type: ignore[name-defined]
    k_out: "ArrayLike",  # type: ignore[name-defined]
    k_in: "ArrayLike",  # type: ignore[name-defined]
    tol: float = 1e-8,
    max_iter: int = 10_000,
    damping: float = 1.0,
    variant: str = "gauss-seidel",
) -> SolverResult:
    """Fixed-point iteration for the DCM.

    Args:
        residual_fn: Function F(θ) → residual tensor (used for convergence check).
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

    if not isinstance(k_out, torch.Tensor):
        k_out = torch.tensor(k_out, dtype=torch.float64)
    else:
        k_out = k_out.to(dtype=torch.float64)
    if not isinstance(k_in, torch.Tensor):
        k_in = torch.tensor(k_in, dtype=torch.float64)
    else:
        k_in = k_in.to(dtype=torch.float64)

    N = k_out.shape[0]

    if not isinstance(theta0, torch.Tensor):
        theta = torch.tensor(theta0, dtype=torch.float64)
    else:
        theta = theta0.clone().to(dtype=torch.float64)

    tracemalloc.start()
    t0 = time.perf_counter()

    n_iter = 0
    residuals: list[float] = []
    converged = False
    message = "Maximum iterations reached without convergence."

    try:
        for _ in range(max_iter):
            x = torch.exp(-theta[:N])
            y = torch.exp(-theta[N:])

            # xy[i, j] = x_i * y_j, shape (N, N)
            xy = x[:, None] * y[None, :]

            # D_out[i] = Σ_{j≠i} y_j / (1 + x_i * y_j)
            D_out = (y[None, :] / (1.0 + xy)).sum(dim=1) - y / (1.0 + xy.diagonal())

            x_new = torch.where(D_out > 0, k_out / D_out, x)

            if variant == "gauss-seidel":
                x_upd = x_new   # use updated x immediately
            else:
                x_upd = x       # Jacobi: keep old values

            # D_in[i] = Σ_{j≠i} x_j / (1 + x_j * y_i)
            xy2 = x_upd[:, None] * y[None, :]
            D_in = (x_upd[None, :] / (1.0 + xy2.T)).sum(dim=1) - x_upd / (1.0 + xy2.diagonal())

            y_new = torch.where(D_in > 0, k_in / D_in, y)

            # Damped update in θ-space
            theta_out_new = -torch.log(x_new.clamp(min=1e-300))
            theta_in_new = -torch.log(y_new.clamp(min=1e-300))
            theta_out_new = theta_out_new.clamp(-_THETA_CLAMP, _THETA_CLAMP)
            theta_in_new = theta_in_new.clamp(-_THETA_CLAMP, _THETA_CLAMP)

            theta_new = torch.cat([theta_out_new, theta_in_new])
            theta = damping * theta_new + (1.0 - damping) * theta

            # Convergence check (ℓ∞ norm of residual)
            res = residual_fn(theta)
            res_norm = res.abs().max().item()

            if not math.isfinite(res_norm):
                message = f"NaN/Inf detected at iteration {n_iter}."
                break

            n_iter += 1
            residuals.append(res_norm)

            if res_norm < tol:
                converged = True
                message = f"Converged in {n_iter} iteration(s)."
                break
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
