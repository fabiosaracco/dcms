"""Two-step solver orchestration for the DaECM model.

The DaECM is solved in two sequential steps:

1. **Topology step** — solve the DCM using any available DCM solver to find
   the out/in-degree Lagrange multipliers ``θ_topo = [θ_out | θ_in]``.

2. **Weight step** — using the DCM probability matrix ``p_ij = sigmoid(-θ_out_i
   - θ_in_j)`` as a fixed topology, solve the conditioned strength equations to
   find the weight Lagrange multipliers ``θ_weight = [θ_β_out | θ_β_in]``.

The entry point is :func:`solve_daecm`, which accepts a :class:`DaECMModel`
instance, initial guesses and method names, and returns a combined
:class:`DaECMResult` dataclass containing both steps' results.
"""
from __future__ import annotations

import math
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch

from src.models.daecm import DaECMModel, _ETA_MIN, _ETA_MAX
from src.models.dcm import DCMModel
from src.solvers.base import SolverResult
from src.solvers.fixed_point import solve_fixed_point
from src.solvers.fixed_point_daecm import solve_fixed_point_daecm
from src.solvers.quasi_newton import solve_lbfgs
from src.solvers.newton import solve_newton
from src.solvers.broyden import solve_broyden
from src.solvers.levenberg_marquardt import solve_lm


@dataclass
class DaECMResult:
    """Combined result for a DaECM two-step solve.

    Attributes:
        theta_topo:        Final topology parameters [θ_out | θ_in], shape (2N,).
        theta_weight:      Final weight parameters [θ_β_out | θ_β_in], shape (2N,).
        topo_converged:    True if the topology step converged.
        weight_converged:  True if the weight step converged.
        converged:         True if *both* steps converged.
        topo_iterations:   Number of topology-step iterations.
        weight_iterations: Number of weight-step iterations.
        topo_residuals:    History of ℓ∞ residuals for topology step.
        weight_residuals:  History of ℓ∞ residuals for weight step.
        elapsed_time:      Total wall-clock seconds.
        peak_ram_bytes:    Peak RAM usage in bytes (both steps combined).
        message:           Human-readable status string.
    """

    theta_topo: np.ndarray
    theta_weight: np.ndarray
    topo_converged: bool
    weight_converged: bool
    converged: bool
    topo_iterations: int
    weight_iterations: int
    topo_residuals: list[float] = field(default_factory=list)
    weight_residuals: list[float] = field(default_factory=list)
    elapsed_time: float = 0.0
    peak_ram_bytes: int = 0
    message: str = ""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_dcm_residual(
    dcm: DCMModel,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a residual function for the DCM."""
    return dcm.residual


def _make_dcm_jacobian(
    dcm: DCMModel,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a Jacobian function for the DCM."""
    return dcm.jacobian


def _make_dcm_nll(
    dcm: DCMModel,
) -> Callable[[torch.Tensor], float]:
    """Return a neg_log_likelihood function for the DCM."""
    return dcm.neg_log_likelihood


def _make_strength_residual(
    model: DaECMModel,
    theta_topo: torch.Tensor,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a residual function for the weight step given fixed theta_topo."""
    def fn(theta_w: torch.Tensor) -> torch.Tensor:
        theta_w_safe = theta_w.clamp(_ETA_MIN, _ETA_MAX)
        return model.residual_strength(theta_topo, theta_w_safe)
    return fn


def _make_strength_jacobian(
    model: DaECMModel,
    theta_topo: torch.Tensor,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a Jacobian function for the weight step given fixed theta_topo."""
    def fn(theta_w: torch.Tensor) -> torch.Tensor:
        theta_w_safe = theta_w.clamp(_ETA_MIN, _ETA_MAX)
        return model.jacobian_strength(theta_topo, theta_w_safe)
    return fn


def _make_strength_nll(
    model: DaECMModel,
    theta_topo: torch.Tensor,
) -> Callable[[torch.Tensor], float]:
    """Return a neg_log_likelihood function for the weight step."""
    def fn(theta_w: torch.Tensor) -> float:
        theta_w_safe = theta_w.clamp(_ETA_MIN, _ETA_MAX)
        return model.neg_log_likelihood_strength(theta_topo, theta_w_safe)
    return fn


def _solve_lm_diag_daecm(
    model: DaECMModel,
    theta_topo: torch.Tensor,
    theta_weight0: torch.Tensor,
    tol: float = 1e-5,
    theta_bounds: tuple[float, float] = (_ETA_MIN, _ETA_MAX),
    max_iter: int = 500,
    lam0: float = 1e-3,
    lam_up: float = 10.0,
    lam_down: float = 0.1,
    lam_max: float = 1e10,
) -> SolverResult:
    """LM with O(N) diagonal Hessian for the DaECM weight step.

    Uses ``model.hessian_diag_strength()`` and ``model.residual_strength()``
    only, avoiding the O(N²) Jacobian allocation.

    Args:
        model:        DaECMModel instance.
        theta_topo:   Fixed topology parameters, shape (2N,).
        theta_weight0: Initial weight parameter vector, shape (2N,).
        tol:          Convergence tolerance.
        theta_bounds: (theta_lo, theta_hi) clamp applied at every step.
        max_iter:     Maximum iterations.
        lam0:         Initial damping λ.
        lam_up:       λ increase factor on rejection.
        lam_down:     λ decrease factor on acceptance.
        lam_max:      Maximum λ before failure.

    Returns:
        :class:`~src.solvers.base.SolverResult` instance.
    """
    tracemalloc.start()
    t0 = time.perf_counter()

    theta_lo, theta_hi = theta_bounds
    theta = theta_weight0.clone().to(dtype=torch.float64).clamp(theta_lo, theta_hi)

    F = model.residual_strength(theta_topo, theta.clamp(theta_lo, theta_hi))
    cost = F.dot(F).item()
    lam = lam0

    residuals: list[float] = []
    converged = False
    n_iter = 0
    message = "Maximum iterations reached without convergence."

    try:
        for _ in range(max_iter):
            res_norm = F.abs().max().item()
            if not math.isfinite(res_norm):
                message = f"NaN/Inf at iteration {n_iter}."
                break
            if res_norm < tol:
                converged = True
                message = f"Converged in {n_iter} iteration(s)."
                break

            h_diag = model.hessian_diag_strength(
                theta_topo, theta.clamp(theta_lo, theta_hi)
            )
            neg_h = -h_diag
            delta = F / (neg_h + lam)

            theta_new = (theta + delta).clamp(theta_lo, theta_hi)
            F_new = model.residual_strength(theta_topo, theta_new)
            cost_new = F_new.dot(F_new).item()

            if cost_new < cost:
                theta = theta_new
                F = F_new
                cost = cost_new
                lam = max(lam * lam_down, 1e-14)
                n_iter += 1
                residuals.append(F.abs().max().item())
            else:
                lam *= lam_up

            if lam > lam_max:
                message = f"Damping λ={lam:.2e} exceeded maximum."
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


# ---------------------------------------------------------------------------
# Main two-step solver
# ---------------------------------------------------------------------------

def solve_daecm(
    model: DaECMModel,
    theta_topo0: Optional[torch.Tensor] = None,
    theta_weight0: Optional[torch.Tensor] = None,
    topo_method: str = "lbfgs",
    weight_method: str = "fp-gs",
    tol: float = 1e-5,
    topo_max_iter: int = 5_000,
    weight_max_iter: int = 10_000,
    damping: float = 1.0,
    anderson_depth: int = 0,
    max_step: float = 1.0,
    chunk_size: int = 0,
    n_topo_starts: int = 1,
    n_weight_starts: int = 1,
) -> DaECMResult:
    """Solve the DaECM model in two sequential steps.

    **Step 1 — Topology:** solve the DCM to find ``θ_topo``.
    **Step 2 — Weight:** using fixed ``p_ij`` from step 1, solve the
    conditioned strength equations to find ``θ_weight``.

    Args:
        model:            DaECMModel instance.
        theta_topo0:      Initial topology guess, shape (2N,).
                          Defaults to ``model.initial_theta_topo("degrees")``.
        theta_weight0:    Initial weight guess, shape (2N,).
                          Defaults to ``model.initial_theta_weight(theta_topo0)``.
        topo_method:      Topology solver: ``"fp-gs"``, ``"lbfgs"``,
                          ``"newton"``, ``"broyden"``, ``"lm"``.
        weight_method:    Weight solver: ``"fp-gs"``, ``"fp-gs-anderson"``,
                          ``"theta-newton"``, ``"theta-newton-anderson"``,
                          ``"lbfgs"``, ``"newton"``, ``"broyden"``, ``"lm"``,
                          ``"lm-diag"``.
        tol:              Convergence tolerance for both steps.
        topo_max_iter:    Maximum iterations for the topology step.
        weight_max_iter:  Maximum iterations for the weight step.
        damping:          Damping factor for β-space FP variants.
        anderson_depth:   Anderson acceleration depth for weight FP variants.
        max_step:         Max |Δθ| for θ-Newton weight variants.
        chunk_size:       0 = auto (dense for N ≤ 5000, chunked otherwise).
        n_topo_starts:    Number of topology initialisation attempts (multi-start).
        n_weight_starts:  Number of weight initialisation attempts (multi-start).

    Returns:
        :class:`DaECMResult` with all statistics from both steps.
    """
    t_total = time.perf_counter()
    N = model.N

    # ------------------------------------------------------------------
    # Initial guesses
    # ------------------------------------------------------------------
    if theta_topo0 is None:
        theta_topo0 = model.initial_theta_topo("degrees")
    else:
        theta_topo0 = _to_tensor(theta_topo0)

    dcm = model._dcm

    # ------------------------------------------------------------------
    # Step 1: Topology (DCM)
    # ------------------------------------------------------------------
    topo_res_fn = _make_dcm_residual(dcm)
    topo_jac_fn = _make_dcm_jacobian(dcm)
    topo_nll_fn = _make_dcm_nll(dcm)

    topo_result: Optional[SolverResult] = None
    best_topo_err = float("inf")

    topo_starts = [theta_topo0]
    if n_topo_starts > 1:
        topo_starts.append(dcm.initial_theta("random"))

    for t0_topo in topo_starts[:n_topo_starts]:
        if topo_method == "fp-gs":
            r = solve_fixed_point(
                topo_res_fn, t0_topo,
                dcm.k_out, dcm.k_in,
                tol=tol, max_iter=topo_max_iter,
                damping=1.0, variant="gauss-seidel",
            )
        elif topo_method == "lbfgs":
            r = solve_lbfgs(
                topo_res_fn, t0_topo,
                tol=tol, max_iter=topo_max_iter,
                m=20, neg_loglik_fn=topo_nll_fn,
                theta_bounds=(-_ETA_MAX, _ETA_MAX),
            )
        elif topo_method == "newton":
            r = solve_newton(
                topo_res_fn, topo_jac_fn, t0_topo,
                tol=tol, max_iter=min(topo_max_iter, 500),
                theta_bounds=(-_ETA_MAX, _ETA_MAX),
            )
        elif topo_method == "broyden":
            r = solve_broyden(
                topo_res_fn, topo_jac_fn, t0_topo,
                tol=tol, max_iter=min(topo_max_iter, 500),
                theta_bounds=(-_ETA_MAX, _ETA_MAX),
            )
        elif topo_method == "lm":
            r = solve_lm(
                topo_res_fn, topo_jac_fn, t0_topo,
                tol=tol, max_iter=min(topo_max_iter, 500),
                diagonal_only=True,
                theta_bounds=(-_ETA_MAX, _ETA_MAX),
            )
        else:
            raise ValueError(f"Unknown topo_method: {topo_method!r}")

        err = dcm.constraint_error(r.theta)
        if err < best_topo_err:
            best_topo_err = err
            topo_result = r
        if r.converged:
            break

    assert topo_result is not None
    theta_topo_final = torch.tensor(topo_result.theta, dtype=torch.float64)

    # ------------------------------------------------------------------
    # Step 2: Weight (conditioned DWCM)
    # ------------------------------------------------------------------
    if theta_weight0 is None:
        theta_weight0 = model.initial_theta_weight(
            theta_topo_final, method="strengths"
        )
    else:
        theta_weight0 = _to_tensor(theta_weight0)

    w_res_fn = _make_strength_residual(model, theta_topo_final)
    w_jac_fn = _make_strength_jacobian(model, theta_topo_final)
    w_nll_fn = _make_strength_nll(model, theta_topo_final)

    weight_result: Optional[SolverResult] = None
    best_weight_err = float("inf")

    weight_starts = [theta_weight0]
    if n_weight_starts > 1:
        weight_starts.append(
            model.initial_theta_weight(theta_topo_final, method="normalized")
        )
        weight_starts.append(
            model.initial_theta_weight(theta_topo_final, method="uniform")
        )
        for i in range(max(0, n_weight_starts - len(weight_starts))):
            torch.manual_seed(i)
            weight_starts.append(
                model.initial_theta_weight(theta_topo_final, method="random")
            )

    # Pre-compute p_ij for dense path (includes zero-degree masks)
    if N <= 5_000:
        P_mat = model.pij_matrix(theta_topo_final)
    else:
        P_mat = None

    for tw0 in weight_starts[:n_weight_starts]:
        if weight_method in ("fp-gs", "fp-jacobi"):
            var = "gauss-seidel" if weight_method == "fp-gs" else "jacobi"
            r = solve_fixed_point_daecm(
                w_res_fn, tw0,
                model.s_out, model.s_in,
                theta_topo=theta_topo_final,
                P=P_mat,
                tol=tol, max_iter=weight_max_iter,
                damping=damping, variant=var,
                chunk_size=chunk_size, anderson_depth=0,
            )
        elif weight_method == "fp-gs-anderson":
            r = solve_fixed_point_daecm(
                w_res_fn, tw0,
                model.s_out, model.s_in,
                theta_topo=theta_topo_final,
                P=P_mat,
                tol=tol, max_iter=weight_max_iter,
                damping=1.0, variant="gauss-seidel",
                chunk_size=chunk_size, anderson_depth=anderson_depth or 10,
            )
        elif weight_method == "theta-newton":
            r = solve_fixed_point_daecm(
                w_res_fn, tw0,
                model.s_out, model.s_in,
                theta_topo=theta_topo_final,
                P=P_mat,
                tol=tol, max_iter=weight_max_iter,
                variant="theta-newton",
                chunk_size=chunk_size, anderson_depth=0,
                max_step=max_step,
            )
        elif weight_method == "theta-newton-anderson":
            r = solve_fixed_point_daecm(
                w_res_fn, tw0,
                model.s_out, model.s_in,
                theta_topo=theta_topo_final,
                P=P_mat,
                tol=tol, max_iter=weight_max_iter,
                variant="theta-newton",
                chunk_size=chunk_size,
                anderson_depth=anderson_depth or 10,
                max_step=max_step,
            )
        elif weight_method == "lbfgs":
            r = solve_lbfgs(
                w_res_fn, tw0,
                tol=tol, max_iter=weight_max_iter,
                m=20, neg_loglik_fn=w_nll_fn,
                theta_bounds=(_ETA_MIN, _ETA_MAX),
            )
        elif weight_method == "newton":
            r = solve_newton(
                w_res_fn, w_jac_fn, tw0,
                tol=tol, max_iter=min(weight_max_iter, 500),
                theta_bounds=(_ETA_MIN, _ETA_MAX),
            )
        elif weight_method == "broyden":
            r = solve_broyden(
                w_res_fn, w_jac_fn, tw0,
                tol=tol, max_iter=min(weight_max_iter, 500),
                theta_bounds=(_ETA_MIN, _ETA_MAX),
            )
        elif weight_method == "lm":
            r = solve_lm(
                w_res_fn, w_jac_fn, tw0,
                tol=tol, max_iter=min(weight_max_iter, 500),
                diagonal_only=False,
                theta_bounds=(_ETA_MIN, _ETA_MAX),
            )
        elif weight_method == "lm-diag":
            r = _solve_lm_diag_daecm(
                model, theta_topo_final, tw0,
                tol=tol, theta_bounds=(_ETA_MIN, _ETA_MAX),
                max_iter=min(weight_max_iter, 2_000),
            )
        else:
            raise ValueError(f"Unknown weight_method: {weight_method!r}")

        err = model.constraint_error_strength(theta_topo_final, r.theta)
        if err < best_weight_err:
            best_weight_err = err
            weight_result = r
        if r.converged:
            break

    assert weight_result is not None

    peak_ram = max(topo_result.peak_ram_bytes, weight_result.peak_ram_bytes)
    elapsed = time.perf_counter() - t_total

    topo_ok = topo_result.converged
    weight_ok = weight_result.converged
    msg_parts = []
    if not topo_ok:
        msg_parts.append(f"topology did NOT converge ({topo_result.message})")
    if not weight_ok:
        msg_parts.append(f"weight did NOT converge ({weight_result.message})")
    if not msg_parts:
        msg_parts.append(
            f"Both steps converged (topo {topo_result.iterations} iters, "
            f"weight {weight_result.iterations} iters)."
        )
    message = "; ".join(msg_parts)

    return DaECMResult(
        theta_topo=topo_result.theta,
        theta_weight=weight_result.theta,
        topo_converged=topo_ok,
        weight_converged=weight_ok,
        converged=(topo_ok and weight_ok),
        topo_iterations=topo_result.iterations,
        weight_iterations=weight_result.iterations,
        topo_residuals=topo_result.residuals,
        weight_residuals=weight_result.residuals,
        elapsed_time=elapsed,
        peak_ram_bytes=peak_ram,
        message=message,
    )


# ---------------------------------------------------------------------------
# Joint (full 4N) L-BFGS solver
# ---------------------------------------------------------------------------

def solve_daecm_joint_lbfgs(
    model: DaECMModel,
    theta_topo0: Optional[torch.Tensor] = None,
    theta_weight0: Optional[torch.Tensor] = None,
    tol: float = 1e-5,
    max_iter: int = 5_000,
    m: int = 20,
) -> DaECMResult:
    """Solve the DaECM by jointly optimising all 4N parameters with L-BFGS.

    Uses a two-phase strategy:

    1. **Warm-start** — solve the two-step DaECM (LBFGS topology, then LBFGS
       weight) to obtain a good initial point.
    2. **Joint refinement** — minimise the combined negative log-likelihood
       ``−L_topo(θ_topo) − L_weight(θ_topo, θ_weight)`` over the full 4N
       parameter vector ``[θ_out | θ_in | θ_β_out | θ_β_in]``.

    This avoids the slow convergence of pure joint optimisation due to the
    different NLL scales between topology and weight terms.

    Args:
        model:         DaECMModel instance.
        theta_topo0:   Initial topology guess, shape (2N,).  Defaults to
                       ``model.initial_theta_topo("degrees")``.
        theta_weight0: Initial weight guess, shape (2N,).  Defaults to
                       ``model.initial_theta_weight(theta_topo0)``.
        tol:           Convergence tolerance on the ℓ∞ residual norm.
        max_iter:      Maximum L-BFGS iterations for the joint phase.
        m:             Number of stored curvature pairs.

    Returns:
        :class:`DaECMResult` with combined statistics.
    """
    t_total = time.perf_counter()
    N = model.N

    # Phase 1: warm-start via two-step solve
    warmup_result = solve_daecm(
        model,
        theta_topo0=theta_topo0,
        theta_weight0=theta_weight0,
        topo_method="lbfgs",
        weight_method="lbfgs",
        tol=tol,
        topo_max_iter=min(max_iter, 5_000),
        weight_max_iter=min(max_iter, 3_000),
    )

    warmup_iters = warmup_result.topo_iterations + warmup_result.weight_iterations

    # If warm-start already converged, return early
    if warmup_result.converged:
        elapsed = time.perf_counter() - t_total
        return DaECMResult(
            theta_topo=warmup_result.theta_topo,
            theta_weight=warmup_result.theta_weight,
            topo_converged=warmup_result.topo_converged,
            weight_converged=warmup_result.weight_converged,
            converged=True,
            topo_iterations=warmup_iters,
            weight_iterations=0,
            topo_residuals=warmup_result.topo_residuals,
            weight_residuals=warmup_result.weight_residuals,
            elapsed_time=elapsed,
            peak_ram_bytes=warmup_result.peak_ram_bytes,
            message="Joint L-BFGS: converged in warm-start phase.",
        )

    # Phase 2: joint refinement over all 4N parameters
    theta_topo_ws = torch.tensor(warmup_result.theta_topo, dtype=torch.float64)
    theta_weight_ws = torch.tensor(warmup_result.theta_weight, dtype=torch.float64)
    theta0 = torch.cat([theta_topo_ws, theta_weight_ws])

    theta_lo = torch.cat([
        torch.full((2 * N,), -_ETA_MAX, dtype=torch.float64),
        torch.full((2 * N,), _ETA_MIN, dtype=torch.float64),
    ])
    theta_hi = torch.full((4 * N,), _ETA_MAX, dtype=torch.float64)

    def _clamp(t: torch.Tensor) -> torch.Tensor:
        return torch.max(torch.min(t, theta_hi), theta_lo)

    def residual_fn(theta: torch.Tensor) -> torch.Tensor:
        return model.residual_joint(_clamp(theta))

    def nll_fn(theta: torch.Tensor) -> float:
        return model.neg_log_likelihood_joint(_clamp(theta))

    joint_result = solve_lbfgs(
        residual_fn, _clamp(theta0),
        tol=tol, max_iter=max_iter, m=m,
        neg_loglik_fn=nll_fn,
        theta_bounds=None,  # we handle clamping ourselves
    )

    elapsed = time.perf_counter() - t_total
    peak_ram = max(warmup_result.peak_ram_bytes, joint_result.peak_ram_bytes)

    theta_final = _clamp(torch.tensor(joint_result.theta, dtype=torch.float64))
    theta_topo_final = theta_final[:2 * N].numpy()
    theta_weight_final = theta_final[2 * N:].numpy()

    topo_err = model.constraint_error_topo(theta_topo_final)
    weight_err = model.constraint_error_strength(theta_topo_final, theta_weight_final)
    topo_ok = topo_err < tol
    weight_ok = weight_err < tol

    total_iters = warmup_iters + joint_result.iterations

    return DaECMResult(
        theta_topo=theta_topo_final,
        theta_weight=theta_weight_final,
        topo_converged=topo_ok,
        weight_converged=weight_ok,
        converged=(topo_ok and weight_ok),
        topo_iterations=total_iters,
        weight_iterations=0,
        topo_residuals=warmup_result.topo_residuals + joint_result.residuals,
        weight_residuals=warmup_result.weight_residuals,
        elapsed_time=elapsed,
        peak_ram_bytes=peak_ram,
        message=f"Joint L-BFGS: {joint_result.message}",
    )


def _to_tensor(
    x: "numpy.ndarray | torch.Tensor",  # type: ignore[name-defined]
) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(dtype=torch.float64)
    return torch.tensor(x, dtype=torch.float64)
