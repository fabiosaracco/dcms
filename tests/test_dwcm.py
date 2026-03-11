"""Tests for the DWCMModel and fixed-point DWCM solver.

Tests cover:
- wij_matrix: correct shape, zero diagonal, non-negative values.
- residual: correct value (≈ 0) at the true solution.
- jacobian: finite-difference consistency and correct structure.
- neg_log_likelihood: gradient consistency with residual.
- initial_theta: sensible starting point.
- mean_relative_error: correct computation.
- solve_fixed_point_dwcm: convergence on small synthetic problems.
- solve_lbfgs (with DWCMModel): convergence on small synthetic problems.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.dwcm import DWCMModel
from src.solvers.base import SolverResult
from src.solvers.fixed_point_dwcm import solve_fixed_point_dwcm
from src.solvers.quasi_newton import solve_lbfgs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CONV_TOL = 1e-5  # acceptance threshold for constraint error


def make_dwcm_model(N: int = 6, seed: int = 0) -> tuple[DWCMModel, np.ndarray]:
    """Return a DWCMModel and the corresponding exact θ_true (numpy).

    Generates a random θ_true, computes z_ij = exp(-θ_out_i - θ_in_j),
    then sets s_out and s_in from the expected weight matrix.

    Args:
        N:    Number of nodes.
        seed: RNG seed.

    Returns:
        (model, theta_true) such that model.residual(theta_true) ≈ 0.
    """
    rng = np.random.default_rng(seed)
    # θ must be strictly positive (so z_ij < 1)
    theta_true = rng.uniform(0.5, 3.0, size=2 * N)
    beta_out = np.exp(-theta_true[:N])
    beta_in = np.exp(-theta_true[N:])
    z = beta_out[:, None] * beta_in[None, :]  # (N, N)
    np.fill_diagonal(z, 0.0)
    W = z / (1.0 - z)
    np.fill_diagonal(W, 0.0)
    s_out = W.sum(axis=1)
    s_in = W.sum(axis=0)
    model = DWCMModel(s_out, s_in)
    return model, theta_true


# ---------------------------------------------------------------------------
# DWCMModel tests
# ---------------------------------------------------------------------------

class TestWijMatrix:
    def test_shape(self) -> None:
        model, theta = make_dwcm_model(N=6)
        W = model.wij_matrix(theta)
        assert W.shape == torch.Size([6, 6])

    def test_zero_diagonal(self) -> None:
        model, theta = make_dwcm_model(N=6)
        W = model.wij_matrix(theta)
        assert W.diagonal().abs().max().item() == 0.0

    def test_non_negative(self) -> None:
        model, theta = make_dwcm_model(N=8)
        W = model.wij_matrix(theta)
        assert (W >= 0.0).all()

    def test_dtype(self) -> None:
        model, theta = make_dwcm_model(N=4)
        W = model.wij_matrix(theta)
        assert W.dtype == torch.float64


class TestResidualDWCM:
    def test_zero_at_true_solution(self) -> None:
        model, theta_true = make_dwcm_model(N=8, seed=1)
        F = model.residual(theta_true)
        assert F.abs().max().item() < 1e-10, (
            f"Max residual at true solution: {F.abs().max().item():.3e}"
        )

    def test_shape(self) -> None:
        model, theta = make_dwcm_model(N=5)
        F = model.residual(theta)
        assert F.shape == torch.Size([10])

    def test_nonzero_far_from_solution(self) -> None:
        model, theta_true = make_dwcm_model(N=5)
        theta_bad = theta_true + 1.0
        F = model.residual(theta_bad)
        assert F.abs().max().item() > 1e-6

    def test_chunked_matches_dense(self) -> None:
        """Chunked residual must agree with dense residual to machine precision."""
        model, theta = make_dwcm_model(N=10, seed=3)
        F_dense = model.residual(theta)
        F_chunked = model._residual_chunked(theta, chunk_size=3)
        assert torch.allclose(F_dense, F_chunked, atol=1e-12), (
            f"Max diff: {(F_dense - F_chunked).abs().max().item():.3e}"
        )


class TestJacobianDWCM:
    def test_shape(self) -> None:
        model, theta = make_dwcm_model(N=4)
        J = model.jacobian(theta)
        assert J.shape == torch.Size([8, 8])

    def test_fd_consistency(self) -> None:
        """Jacobian must be consistent with finite-difference approximation."""
        model, theta = make_dwcm_model(N=4, seed=2)
        theta_t = torch.tensor(theta, dtype=torch.float64)
        J = model.jacobian(theta_t)
        eps = 1e-6
        n = len(theta)
        J_fd = torch.zeros(n, n, dtype=torch.float64)
        for j in range(n):
            dth = theta_t.clone()
            dth[j] += eps
            J_fd[:, j] = (model.residual(dth) - model.residual(theta_t)) / eps
        assert torch.allclose(J, J_fd, atol=1e-4), (
            f"Max FD diff: {(J - J_fd).abs().max().item():.3e}"
        )

    def test_diagonal_negative(self) -> None:
        """Diagonal entries of J must be ≤ 0 (Hessian of concave L)."""
        model, theta = make_dwcm_model(N=6, seed=1)
        J = model.jacobian(theta)
        assert (J.diagonal() <= 0).all()


class TestNegLogLikelihoodDWCM:
    def test_gradient_is_residual(self) -> None:
        """∇L = F(θ) ↔ ∇(−L) = −F(θ).  Check via finite differences."""
        model, theta = make_dwcm_model(N=5, seed=4)
        theta_t = torch.tensor(theta, dtype=torch.float64)
        eps = 1e-6
        n = len(theta)
        grad_neg_L_fd = torch.zeros(n, dtype=torch.float64)
        f0 = model.neg_log_likelihood(theta_t)
        for j in range(n):
            dth = theta_t.clone()
            dth[j] += eps
            grad_neg_L_fd[j] = (model.neg_log_likelihood(dth) - f0) / eps
        F = model.residual(theta_t)
        assert torch.allclose(-F, grad_neg_L_fd, atol=1e-4), (
            f"Max grad diff: {(-F - grad_neg_L_fd).abs().max().item():.3e}"
        )

    def test_chunked_matches_dense(self) -> None:
        model, theta = make_dwcm_model(N=8, seed=5)
        theta_t = torch.tensor(theta, dtype=torch.float64)
        nll_dense = model.neg_log_likelihood(theta_t)
        nll_chunked = model._neg_log_likelihood_chunked(theta_t, chunk_size=3)
        assert abs(nll_dense - nll_chunked) < 1e-10


class TestInitialThetaDWCM:
    def test_shape(self) -> None:
        model, _ = make_dwcm_model(N=8)
        theta0 = model.initial_theta("strengths")
        assert theta0.shape == torch.Size([16])

    def test_feasibility(self) -> None:
        """All β_out_i * β_in_j must be < 1 at the initial θ."""
        model, _ = make_dwcm_model(N=10)
        theta0 = model.initial_theta("strengths")
        N = model.N
        beta_out = torch.exp(-theta0[:N])
        beta_in = torch.exp(-theta0[N:])
        z = beta_out[:, None] * beta_in[None, :]
        assert (z < 1.0).all(), (
            f"Infeasible initial θ: max z = {z.max().item():.4f}"
        )

    def test_random_init_positive(self) -> None:
        model, _ = make_dwcm_model(N=6)
        theta0 = model.initial_theta("random")
        assert theta0.shape == torch.Size([12])
        assert (theta0 > 0).all()  # random init in [0.5, 3.0] → always positive


class TestMeanRelativeError:
    def test_zero_at_true_solution(self) -> None:
        model, theta_true = make_dwcm_model(N=6, seed=0)
        mre = model.mean_relative_error(theta_true)
        assert mre < 1e-10

    def test_positive_elsewhere(self) -> None:
        model, theta_true = make_dwcm_model(N=6, seed=0)
        mre = model.mean_relative_error(theta_true + 1.0)
        assert mre > 1e-6


# ---------------------------------------------------------------------------
# Solver tests
# ---------------------------------------------------------------------------

class TestFixedPointDWCM:
    @pytest.mark.parametrize("N,seed", [(4, 0), (8, 1)])
    def test_gauss_seidel_converges(self, N: int, seed: int) -> None:
        model, _ = make_dwcm_model(N=N, seed=seed)
        theta0 = model.initial_theta("strengths")
        result = solve_fixed_point_dwcm(
            model.residual, theta0, model.s_out, model.s_in,
            tol=1e-10, max_iter=10_000, damping=1.0, variant="gauss-seidel",
        )
        err = model.constraint_error(result.theta)
        assert err < CONV_TOL, f"N={N} GS error={err:.3e}"

    @pytest.mark.parametrize("N,seed", [(4, 2), (8, 3)])
    def test_gauss_seidel_damped_converges(self, N: int, seed: int) -> None:
        model, _ = make_dwcm_model(N=N, seed=seed)
        theta0 = model.initial_theta("strengths")
        result = solve_fixed_point_dwcm(
            model.residual, theta0, model.s_out, model.s_in,
            tol=1e-10, max_iter=10_000, damping=0.5, variant="gauss-seidel",
        )
        err = model.constraint_error(result.theta)
        assert err < CONV_TOL, f"N={N} GS-0.5 error={err:.3e}"

    @pytest.mark.parametrize("N,seed", [(4, 0), (8, 1)])
    def test_jacobi_damped_converges(self, N: int, seed: int) -> None:
        model, _ = make_dwcm_model(N=N, seed=seed)
        theta0 = model.initial_theta("strengths")
        result = solve_fixed_point_dwcm(
            model.residual, theta0, model.s_out, model.s_in,
            tol=1e-10, max_iter=10_000, damping=0.5, variant="jacobi",
        )
        err = model.constraint_error(result.theta)
        assert err < CONV_TOL, f"N={N} Jacobi(damped) error={err:.3e}"

    def test_solver_result_fields(self) -> None:
        model, _ = make_dwcm_model(N=4, seed=0)
        theta0 = model.initial_theta("strengths")
        result = solve_fixed_point_dwcm(
            model.residual, theta0, model.s_out, model.s_in,
            tol=1e-8, max_iter=1000,
        )
        assert isinstance(result, SolverResult)
        assert result.iterations > 0
        assert result.elapsed_time >= 0.0
        assert result.peak_ram_bytes >= 0
        assert len(result.residuals) == result.iterations

    def test_chunked_matches_dense(self) -> None:
        """Chunked step must give same result as dense step for small N."""
        model, _ = make_dwcm_model(N=6, seed=7)
        theta0 = model.initial_theta("strengths")
        result_dense = solve_fixed_point_dwcm(
            model.residual, theta0, model.s_out, model.s_in,
            tol=1e-10, max_iter=5000, chunk_size=0,
        )
        result_chunked = solve_fixed_point_dwcm(
            model.residual, theta0, model.s_out, model.s_in,
            tol=1e-10, max_iter=5000, chunk_size=2,
        )
        err_dense = model.constraint_error(result_dense.theta)
        err_chunked = model.constraint_error(result_chunked.theta)
        assert err_chunked < CONV_TOL, f"Chunked error={err_chunked:.3e}"
        assert err_dense < CONV_TOL, f"Dense error={err_dense:.3e}"

    def test_invalid_variant_raises(self) -> None:
        model, _ = make_dwcm_model(N=4)
        theta0 = model.initial_theta("strengths")
        with pytest.raises(ValueError, match="variant"):
            solve_fixed_point_dwcm(
                model.residual, theta0, model.s_out, model.s_in,
                variant="bad-variant",
            )

    def test_invalid_damping_raises(self) -> None:
        model, _ = make_dwcm_model(N=4)
        theta0 = model.initial_theta("strengths")
        with pytest.raises(ValueError, match="damping"):
            solve_fixed_point_dwcm(
                model.residual, theta0, model.s_out, model.s_in,
                damping=1.5,
            )


class TestLBFGSOnDWCM:
    @pytest.mark.parametrize("N,seed", [(4, 0), (10, 1)])
    def test_lbfgs_converges(self, N: int, seed: int) -> None:
        model, _ = make_dwcm_model(N=N, seed=seed)
        theta0 = model.initial_theta("strengths")
        result = solve_lbfgs(
            model.residual, theta0, tol=1e-10, m=20,
            neg_loglik_fn=model.neg_log_likelihood,
        )
        err = model.constraint_error(result.theta)
        assert err < CONV_TOL, f"N={N} L-BFGS error={err:.3e}"
