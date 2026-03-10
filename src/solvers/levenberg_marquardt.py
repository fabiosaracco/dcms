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

import math
import time
import tracemalloc
from typing import Callable

import torch

from .base import SolverResult

_THETA_CLAMP = 50.0


def solve_lm(
    residual_fn: Callable[[torch.Tensor], torch.Tensor],
    jacobian_fn: Callable[[torch.Tensor], torch.Tensor],
    theta0: "ArrayLike",  # type: ignore[name-defined]
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
        residual_fn:   F(θ) → residual tensor, shape (2N,).
        jacobian_fn:   J(θ) → Jacobian tensor, shape (2N, 2N).
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

    if not isinstance(theta0, torch.Tensor):
        theta = torch.tensor(theta0, dtype=torch.float64)
    else:
        theta = theta0.clone().to(dtype=torch.float64)

    F = residual_fn(theta)
    cost = F.dot(F).item()
    n2 = theta.shape[0]
    lam = lam0

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

            J = jacobian_fn(theta)  # (2N, 2N), negative semi-definite
            # LM normal equations: (JᵀJ + λI) δ = −Jᵀ F
            JtF = J.T @ F  # (2N,)

            if diagonal_only:
                JtJ_diag = (J ** 2).sum(dim=0)
                delta = -JtF / (JtJ_diag + lam)
            else:
                JtJ = J.T @ J  # (2N, 2N)
                A = JtJ + lam * torch.eye(n2, dtype=torch.float64)
                try:
                    delta = torch.linalg.solve(A, -JtF)
                except RuntimeError:
                    lam *= lam_up
                    continue

            theta_new = torch.clamp(theta + delta, -_THETA_CLAMP, _THETA_CLAMP)
            F_new = residual_fn(theta_new)
            cost_new = F_new.dot(F_new).item()

            if cost_new < cost:
                # Accept step, reduce damping
                theta = theta_new
                F = F_new
                cost = cost_new
                lam = max(lam * lam_down, 1e-14)
                n_iter += 1
                residuals.append(F.abs().max().item())
            else:
                # Reject step, increase damping (no new residual entry)
                lam = lam * lam_up

            if lam > lam_max:
                message = f"Damping λ={lam:.2e} exceeded maximum. Stopping."
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
