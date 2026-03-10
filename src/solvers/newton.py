"""Full Newton solver with exact Jacobian.

At each step solves the linear system

    −J(θ) · δθ = F(θ)

where J = ∂F/∂θ = Hess(L) is negative semi-definite.  We regularise −J
(which is PSD) as −J + εI and use ``torch.linalg.solve``.

A simple backtracking Armijo line search controls the step size.

Cost per iteration: O(N³) (linear solve); RAM: O(N²).
Recommended only for N ≲ 5 000.
"""
from __future__ import annotations

import math
import time
import tracemalloc
from typing import Callable

import torch

from .base import SolverResult

_THETA_CLAMP = 50.0


def solve_newton(
    residual_fn: Callable[[torch.Tensor], torch.Tensor],
    jacobian_fn: Callable[[torch.Tensor], torch.Tensor],
    theta0: "ArrayLike",  # type: ignore[name-defined]
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

    so that −J + εI (PSD) can be safely regularised.
    ``torch.linalg.solve`` is used instead of explicit matrix inversion.

    Args:
        residual_fn: F(θ) → residual tensor, shape (2N,).
        jacobian_fn: J(θ) → Jacobian tensor, shape (2N, 2N).
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

    if not isinstance(theta0, torch.Tensor):
        theta = torch.tensor(theta0, dtype=torch.float64)
    else:
        theta = theta0.clone().to(dtype=torch.float64)

    F = residual_fn(theta)
    n2 = theta.shape[0]
    eye = torch.eye(n2, dtype=torch.float64)

    n_iter = 0
    residuals: list[float] = []
    converged = False
    message = "Maximum iterations reached without convergence."

    try:
        for _ in range(max_iter):
            res_norm = F.abs().max().item()
            if res_norm < tol:
                converged = True
                message = f"Converged in {n_iter} iteration(s)."
                break

            if not math.isfinite(res_norm):
                message = f"NaN/Inf detected at iteration {n_iter}."
                break

            J = jacobian_fn(theta)

            # Tikhonov regularisation on −J (PSD).
            # Solve (−J + εI) δθ = F  →  δθ = (−J + εI)^{-1} F
            neg_J = -J
            eps = reg
            delta = None
            while eps <= max_reg:
                try:
                    delta = torch.linalg.solve(neg_J + eps * eye, F)
                    break
                except RuntimeError:
                    eps *= 10.0

            if delta is None:
                message = "Jacobian singular even after heavy regularisation."
                break

            # Armijo backtracking line search on ‖F‖²
            alpha = 1.0
            f0 = F.dot(F).item()
            for _ in range(max_ls):
                theta_new = torch.clamp(theta + alpha * delta, -_THETA_CLAMP, _THETA_CLAMP)
                F_new = residual_fn(theta_new)
                if F_new.dot(F_new).item() <= f0 * (1.0 - 2.0 * armijo_c * alpha):
                    break
                alpha *= 0.5

            theta = theta_new
            F = F_new
            n_iter += 1
            residuals.append(F.abs().max().item())
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
