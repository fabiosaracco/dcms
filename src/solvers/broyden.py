"""Broyden's method (rank-1 approximate Jacobian update).

The exact Jacobian is computed only at the first iteration.  Subsequent
iterations use the Sherman-Morrison formula to update H⁻¹ (where H = −J)
in O(N²):

    H⁻¹_{k+1} = H⁻¹_k + (s_k − H⁻¹_k ỹ_k)(sᵀ_k H⁻¹_k) / (sᵀ_k H⁻¹_k ỹ_k)

where  s_k = θ_{k+1} − θ_k,  ỹ_k = −(F_{k+1} − F_k)  (sign flip because
H = −J approximates −J, so H s ≈ −y).

The step is δ = H⁻¹ F (solving H δ = F, equiv. to J δ = −F).

A simple Armijo backtracking line search is included.

Cost per iteration: O(N²) (matrix-vector product); RAM: O(N²).
"""
from __future__ import annotations

import math
import time
import tracemalloc
from typing import Callable

import torch

from .base import SolverResult

_THETA_CLAMP = 50.0


def solve_broyden(
    residual_fn: Callable[[torch.Tensor], torch.Tensor],
    jacobian_fn: Callable[[torch.Tensor], torch.Tensor],
    theta0: "ArrayLike",  # type: ignore[name-defined]
    tol: float = 1e-8,
    max_iter: int = 500,
    reg: float = 1e-8,
    armijo_c: float = 1e-4,
    max_ls: int = 30,
) -> SolverResult:
    """Broyden's good-Broyden method with Sherman-Morrison H⁻¹ updates.

    Uses H = −J (which is PSD) as the Broyden matrix so that the inverse
    H⁻¹ is also PSD and the step δ = H⁻¹ F has a well-defined descent
    property.

    Args:
        residual_fn: F(θ) → residual tensor, shape (2N,).
        jacobian_fn: J(θ) → exact Jacobian at θ₀ only, shape (2N, 2N).
        theta0:      Initial parameter vector, shape (2N,).
        tol:         Convergence tolerance (ℓ∞ residual).
        max_iter:    Maximum iterations.
        reg:         Tikhonov regularisation for the initial (−J + εI)⁻¹.
        armijo_c:    Armijo constant for line search.
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

    # Initialise H⁻¹ where H = −J + reg·I (always PSD).
    # H⁻¹ ≈ (−J)⁻¹ for small reg; step δ = H⁻¹ F solves J δ = −F.
    J0 = jacobian_fn(theta)
    H0 = -J0 + reg * eye
    try:
        H_inv = torch.linalg.inv(H0)
    except RuntimeError:
        # Increase regularisation until invertible
        for reg_mul in [10, 100, 1000]:
            try:
                H_inv = torch.linalg.inv(-J0 + reg * reg_mul * eye)
                break
            except RuntimeError:
                pass
        else:
            H_inv = eye.clone()

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

            # Step: solve H δ = F, i.e. δ = H⁻¹ F
            delta = H_inv @ F

            # Armijo backtracking on ‖F‖²
            alpha = 1.0
            f0 = F.dot(F).item()
            for _ in range(max_ls):
                theta_new = torch.clamp(theta + alpha * delta, -_THETA_CLAMP, _THETA_CLAMP)
                F_new = residual_fn(theta_new)
                if F_new.dot(F_new).item() <= f0 * (1.0 - 2.0 * armijo_c * alpha):
                    break
                alpha *= 0.5

            s = theta_new - theta          # actual step taken
            y_F = F_new - F               # change in residual
            y_H = -y_F                    # Broyden cond. for H: H s ≈ -y_F

            # Sherman-Morrison update of H⁻¹:
            # B^{-1}_{k+1} = B^{-1}_k + (s − B^{-1}_k y)(sᵀ B^{-1}_k) / (sᵀ B^{-1}_k y)
            Hinv_yH = H_inv @ y_H
            denom = s.dot(Hinv_yH).item()
            if abs(denom) > 1e-14:
                numer = torch.outer(s - Hinv_yH, s @ H_inv)
                H_inv = H_inv + numer / denom

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
