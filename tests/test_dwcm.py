"""Tests for the DWCM model equations and all five solver methods.

Tests cover:
- wij_matrix: correct shape, zero diagonal, non-negative values.
- residual:   correct value at the true solution.
- gradient:   equals +residual (∇L = F).
- hessian_diag: all entries negative (concave log-likelihood).
- jacobian:   FD consistency and negative diagonal.
- initial_theta: sensible starting point.
- neg_log_likelihood: finite at valid theta.
- constraint_error / max_relative_error: correct values.
- Zero-strength node handling.
- Solver convergence (N=4, N=10).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.dwcm import DWCMModel, _ETA_MAX, _ETA_MIN
from src.solvers.base import SolverResult
from src.solvers.fixed_point_dwcm import solve_fixed_point_dwcm
from src.solvers.quasi_newton import solve_lbfgs
from src.solvers.newton import solve_newton
from src.solvers.broyden import solve_broyden
from src.solvers.levenberg_marquardt import solve_lm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CONV_TOL = 1e-5  # acceptance threshold for constraint error


def make_dwcm_model(N: int = 6, seed: int = 0) -> tuple[DWCMModel, np.ndarray]:
    """Return a small DWCMModel and a synthetic θ vector (numpy).

    Generates a random θ_true (all positive, in [0.5, 2.0]) and computes
    the corresponding s_out / s_in analytically so the exact solution is known.

    Args:
        N:    Number of nodes.
        seed: RNG seed.

    Returns:
        (model, theta_true)
    """
    rng = np.random.default_rng(seed)
    theta_true = rng.uniform(0.5, 2.0, size=2 * N)
    beta_out = np.exp(-theta_true[:N])
    beta_in = np.exp(-theta_true[N:])
    # W_ij = β_out_i * β_in_j / (1 - β_out_i * β_in_j)  for i ≠ j
    beta_mat = beta_out[:, None] * beta_in[None, :]      # (N, N)
    W = beta_mat / (1.0 - beta_mat)
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

    def test_nonnegative(self) -> None:
        model, theta = make_dwcm_model(N=10)
        W = model.wij_matrix(theta)
        assert (W >= 0.0).all()

    def test_dtype(self) -> None:
        model, theta = make_dwcm_model(N=4)
        W = model.wij_matrix(theta)
        assert W.dtype == torch.float64

    def test_value_formula(self) -> None:
        """W_ij = β_out_i * β_in_j / (1 - β_out_i * β_in_j) for i ≠ j."""
        N = 4
        model, theta = make_dwcm_model(N=N)
        W = model.wij_matrix(theta)
        beta_out = np.exp(-theta[:N])
        beta_in = np.exp(-theta[N:])
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                bij = beta_out[i] * beta_in[j]
                expected = bij / (1.0 - bij)
                assert abs(W[i, j].item() - expected) < 1e-12, (
                    f"W[{i},{j}] = {W[i,j].item():.6f}, expected {expected:.6f}"
                )


class TestResidual:
    def test_zero_at_true_solution(self) -> None:
        """Residual must be ≈ 0 when θ is the true solution."""
        model, theta_true = make_dwcm_model(N=8, seed=1)
        F = model.residual(theta_true)
        assert F.abs().max().item() < 1e-10, (
            f"Max residual: {F.abs().max().item():.3e}"
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
        """Chunked residual must match the dense version exactly."""
        model, theta = make_dwcm_model(N=12)
        F_dense = model.residual(torch.tensor(theta, dtype=torch.float64))
        F_chunked = model._residual_chunked(
            torch.tensor(theta, dtype=torch.float64), chunk_size=4
        )
        assert torch.allclose(F_dense, F_chunked, atol=1e-13), (
            f"Max diff: {(F_dense - F_chunked).abs().max().item():.3e}"
        )


class TestGradient:
    def test_gradient_is_residual(self) -> None:
        """∇L = F(θ) (not −F)."""
        model, theta = make_dwcm_model(N=6)
        grad = model.gradient(theta)
        F = model.residual(theta)
        assert torch.allclose(grad, F), "gradient() must equal +residual()"


class TestHessianDiag:
    def test_all_negative(self) -> None:
        """Diagonal Hessian entries must be negative (concave log-likelihood)."""
        model, theta = make_dwcm_model(N=8, seed=2)
        h = model.hessian_diag(theta)
        assert (h < 0).all(), (
            f"Some Hessian diag entries non-negative: {h[h >= 0]}"
        )

    def test_shape(self) -> None:
        model, theta = make_dwcm_model(N=5)
        h = model.hessian_diag(theta)
        assert h.shape == torch.Size([10])


class TestJacobian:
    def test_shape(self) -> None:
        model, theta = make_dwcm_model(N=4)
        J = model.jacobian(theta)
        assert J.shape == torch.Size([8, 8])

    def test_negative_diagonal(self) -> None:
        """Jacobian diagonal entries must be negative (L is concave)."""
        model, theta = make_dwcm_model(N=6, seed=3)
        J = model.jacobian(theta)
        diag = J.diagonal()
        assert (diag < 0).all(), (
            f"Some Jacobian diag entries non-negative: {diag[diag >= 0]}"
        )

    def test_finite_difference_consistency(self) -> None:
        """Jacobian must match finite-difference approximation."""
        model, theta = make_dwcm_model(N=5, seed=4)
        theta_t = torch.tensor(theta, dtype=torch.float64)
        J_exact = model.jacobian(theta_t)
        n2 = len(theta)
        J_fd = torch.zeros(n2, n2, dtype=torch.float64)
        eps = 1e-5
        for j in range(n2):
            dth = torch.zeros(n2, dtype=torch.float64)
            dth[j] = eps
            J_fd[:, j] = (
                model.residual(theta_t + dth) - model.residual(theta_t - dth)
            ) / (2 * eps)
        assert torch.allclose(J_exact, J_fd, atol=1e-6), (
            f"Max FD error: {(J_exact - J_fd).abs().max().item():.3e}"
        )


class TestInitialTheta:
    def test_strengths_method(self) -> None:
        model, _ = make_dwcm_model(N=10)
        theta0 = model.initial_theta("strengths")
        assert theta0.shape == torch.Size([20])
        assert torch.isfinite(theta0).all()
        assert (theta0 > 0).all(), "All initial θ must be strictly positive for DWCM"

    def test_random_method(self) -> None:
        model, _ = make_dwcm_model(N=10)
        theta0 = model.initial_theta("random")
        assert theta0.shape == torch.Size([20])
        assert torch.isfinite(theta0).all()

    def test_unknown_method_raises(self) -> None:
        model, _ = make_dwcm_model(N=4)
        with pytest.raises(ValueError):
            model.initial_theta("unknown")

    def test_feasibility_of_initial_theta(self) -> None:
        """Initial θ must satisfy the feasibility constraint β_i * β_j < 1."""
        model, _ = make_dwcm_model(N=20)
        theta0 = model.initial_theta("strengths")
        N = model.N
        beta_out = torch.exp(-theta0[:N])
        beta_in = torch.exp(-theta0[N:])
        max_product = (beta_out[:, None] * beta_in[None, :]).fill_diagonal_(0.0).max().item()
        assert max_product < 1.0, (
            f"Feasibility violated: max(β_out_i * β_in_j) = {max_product:.4f}"
        )


class TestConstraintError:
    def test_zero_at_solution(self) -> None:
        model, theta_true = make_dwcm_model(N=6, seed=5)
        err = model.constraint_error(theta_true)
        assert err < 1e-10

    def test_positive_away_from_solution(self) -> None:
        model, theta_true = make_dwcm_model(N=6, seed=5)
        err = model.constraint_error(theta_true + 0.5)
        assert err > 0.0


class TestMaxRelativeError:
    def test_zero_at_solution(self) -> None:
        model, theta_true = make_dwcm_model(N=6, seed=5)
        err = model.max_relative_error(theta_true)
        assert err < 1e-9

    def test_nonnegative(self) -> None:
        model, theta_true = make_dwcm_model(N=6, seed=5)
        err = model.max_relative_error(theta_true + 0.5)
        assert err >= 0.0


class TestNegLogLikelihood:
    def test_finite_at_valid_theta(self) -> None:
        model, theta = make_dwcm_model(N=8)
        nll = model.neg_log_likelihood(theta)
        assert np.isfinite(nll)

    def test_chunked_matches_dense(self) -> None:
        """Chunked neg_log_likelihood must match the dense version."""
        model, theta = make_dwcm_model(N=12)
        theta_t = torch.tensor(theta, dtype=torch.float64)
        nll_dense = model.neg_log_likelihood(theta_t)
        nll_chunked = model._neg_log_likelihood_chunked(theta_t, chunk_size=4)
        assert abs(nll_dense - nll_chunked) < 1e-10, (
            f"Diff: {abs(nll_dense - nll_chunked):.3e}"
        )


# ---------------------------------------------------------------------------
# Zero-strength node handling
# ---------------------------------------------------------------------------

class TestZeroStrengthBehavior:
    """Tests for the exact handling of nodes with s_out=0 or s_in=0."""

    def _make_zero_strength_model(self) -> DWCMModel:
        """Build a 6-node model where node 0 has s_out=0 and node 5 has s_in=0."""
        s_out = np.array([0.0, 3.0, 2.0, 1.5, 2.0, 1.5])
        s_in  = np.array([2.0, 1.5, 2.0, 2.5, 2.0, 0.0])
        return DWCMModel(s_out, s_in)

    def test_zero_out_mask(self) -> None:
        model = self._make_zero_strength_model()
        assert model.zero_out[0].item()
        assert not model.zero_out[1:].any().item()

    def test_zero_in_mask(self) -> None:
        model = self._make_zero_strength_model()
        N = model.N
        assert model.zero_in[N - 1].item()
        assert not model.zero_in[: N - 1].any().item()

    def test_wij_row_zero_for_zero_out(self) -> None:
        """W_0j = 0 for all j when s_out[0] = 0."""
        model = self._make_zero_strength_model()
        theta0 = model.initial_theta("strengths")
        W = model.wij_matrix(theta0)
        assert W[0].abs().max().item() == 0.0, "Row 0 must be exactly zero"

    def test_wij_col_zero_for_zero_in(self) -> None:
        """W_i5 = 0 for all i when s_in[5] = 0."""
        model = self._make_zero_strength_model()
        N = model.N
        theta0 = model.initial_theta("strengths")
        W = model.wij_matrix(theta0)
        assert W[:, N - 1].abs().max().item() == 0.0, "Last column must be exactly zero"

    def test_residual_exact_zero_for_zero_strength(self) -> None:
        """Residual components for zero-strength nodes must be exactly 0."""
        model = self._make_zero_strength_model()
        N = model.N
        theta0 = model.initial_theta("strengths")
        F = model.residual(theta0)
        assert F[0].abs().item() == 0.0
        assert F[N + N - 1].abs().item() == 0.0

    def test_initial_theta_zero_strength_nodes_large(self) -> None:
        """initial_theta must set θ = _ETA_MAX for zero-strength nodes."""
        model = self._make_zero_strength_model()
        N = model.N
        theta0 = model.initial_theta("strengths")
        assert theta0[0].item() == _ETA_MAX, (
            f"θ_out[0] must equal _ETA_MAX={_ETA_MAX}"
        )
        assert theta0[N + N - 1].item() == _ETA_MAX, (
            f"θ_in[N-1] must equal _ETA_MAX={_ETA_MAX}"
        )

    def test_initial_theta_random_zero_strength_nodes_large(self) -> None:
        """initial_theta('random') must also set θ = _ETA_MAX for zero-strength nodes."""
        model = self._make_zero_strength_model()
        N = model.N
        theta0 = model.initial_theta("random")
        assert theta0[0].item() == _ETA_MAX
        assert theta0[N + N - 1].item() == _ETA_MAX


# ---------------------------------------------------------------------------
# Solver convergence tests
# ---------------------------------------------------------------------------

class TestFixedPointDWCM:
    @pytest.mark.parametrize("N,seed", [(4, 0), (10, 1)])
    def test_gauss_seidel_converges(self, N: int, seed: int) -> None:
        model, _ = make_dwcm_model(N=N, seed=seed)
        theta0 = model.initial_theta("strengths")
        result = solve_fixed_point_dwcm(
            model.residual, theta0, model.s_out, model.s_in,
            tol=1e-10, max_iter=5000, damping=1.0, variant="gauss-seidel",
        )
        err = model.constraint_error(result.theta)
        assert err < CONV_TOL, f"N={N} GS error={err:.3e}"

    @pytest.mark.parametrize("N,seed", [(4, 2), (10, 3)])
    def test_jacobi_converges(self, N: int, seed: int) -> None:
        model, _ = make_dwcm_model(N=N, seed=seed)
        theta0 = model.initial_theta("strengths")
        result = solve_fixed_point_dwcm(
            model.residual, theta0, model.s_out, model.s_in,
            tol=1e-10, max_iter=5000, damping=0.5, variant="jacobi",
        )
        err = model.constraint_error(result.theta)
        assert err < CONV_TOL, f"N={N} Jacobi error={err:.3e}"

    def test_result_has_residual_history(self) -> None:
        model, _ = make_dwcm_model(N=4, seed=0)
        theta0 = model.initial_theta("strengths")
        result = solve_fixed_point_dwcm(
            model.residual, theta0, model.s_out, model.s_in, max_iter=20,
        )
        assert len(result.residuals) > 0

    def test_invalid_variant_raises(self) -> None:
        model, _ = make_dwcm_model(N=4)
        theta0 = model.initial_theta()
        with pytest.raises(ValueError):
            solve_fixed_point_dwcm(
                model.residual, theta0, model.s_out, model.s_in, variant="bad"
            )

    def test_invalid_damping_raises(self) -> None:
        model, _ = make_dwcm_model(N=4)
        theta0 = model.initial_theta()
        with pytest.raises(ValueError):
            solve_fixed_point_dwcm(
                model.residual, theta0, model.s_out, model.s_in, damping=0.0
            )

    def test_elapsed_time_positive(self) -> None:
        model, _ = make_dwcm_model(N=4)
        theta0 = model.initial_theta()
        result = solve_fixed_point_dwcm(
            model.residual, theta0, model.s_out, model.s_in, max_iter=5,
        )
        assert result.elapsed_time >= 0.0

    def test_peak_ram_nonnegative(self) -> None:
        model, _ = make_dwcm_model(N=4)
        theta0 = model.initial_theta()
        result = solve_fixed_point_dwcm(
            model.residual, theta0, model.s_out, model.s_in, max_iter=5,
        )
        assert result.peak_ram_bytes >= 0

    def test_chunked_step_matches_dense_step(self) -> None:
        """Chunked fixed-point step must produce the same result as the dense path."""
        from src.solvers.fixed_point_dwcm import _fp_step_chunked_dwcm
        model, theta = make_dwcm_model(N=8)
        theta_t = torch.tensor(theta, dtype=torch.float64)
        N = model.N
        beta_out = torch.exp(-theta_t[:N])
        beta_in = torch.exp(-theta_t[N:])

        # Dense path (using DWCMModel internals)
        xy = beta_out[:, None] * beta_in[None, :]
        denom = (1.0 - xy).clamp(min=1e-15)
        D_out_mat = beta_in[None, :] / denom
        D_out_mat.fill_diagonal_(0.0)
        D_out = D_out_mat.sum(dim=1)
        beta_out_new_dense = torch.where(D_out > 0, model.s_out / D_out, beta_out)

        beta_out_new_chunked, _ = _fp_step_chunked_dwcm(
            beta_out, beta_in, model.s_out, model.s_in, chunk_size=3, variant="gauss-seidel"
        )
        assert torch.allclose(beta_out_new_dense, beta_out_new_chunked, atol=1e-13), (
            f"Max diff: {(beta_out_new_dense - beta_out_new_chunked).abs().max().item():.3e}"
        )


class TestLBFGSDWCM:
    """Tests for L-BFGS solver on DWCM problems."""

    def _make_clamped_residual(self, model: DWCMModel):
        """Wrap residual to clamp theta to valid DWCM range."""
        def fn(theta: torch.Tensor) -> torch.Tensor:
            theta_safe = theta.clamp(_ETA_MIN, _ETA_MAX)
            return model.residual(theta_safe)
        return fn

    def _make_clamped_nll(self, model: DWCMModel):
        """Wrap neg_log_likelihood to clamp theta to valid DWCM range."""
        def fn(theta: torch.Tensor) -> float:
            theta_safe = theta.clamp(_ETA_MIN, _ETA_MAX)
            return model.neg_log_likelihood(theta_safe)
        return fn

    @pytest.mark.parametrize("N,seed", [(4, 0), (10, 1)])
    def test_converges(self, N: int, seed: int) -> None:
        model, _ = make_dwcm_model(N=N, seed=seed)
        theta0 = model.initial_theta("strengths")
        result = solve_lbfgs(
            self._make_clamped_residual(model), theta0, tol=1e-10, max_iter=2000,
            neg_loglik_fn=self._make_clamped_nll(model),
        )
        err = model.constraint_error(result.theta)
        assert err < CONV_TOL, f"N={N} LBFGS error={err:.3e}"


class TestNewtonDWCM:
    """Tests for full Newton solver on DWCM problems."""

    def _make_clamped_residual(self, model: DWCMModel):
        """Wrap residual to clamp theta to valid DWCM range."""
        def fn(theta: torch.Tensor) -> torch.Tensor:
            theta_safe = theta.clamp(_ETA_MIN, _ETA_MAX)
            return model.residual(theta_safe)
        return fn

    def _make_clamped_jacobian(self, model: DWCMModel):
        """Wrap jacobian to clamp theta to valid DWCM range."""
        def fn(theta: torch.Tensor) -> torch.Tensor:
            theta_safe = theta.clamp(_ETA_MIN, _ETA_MAX)
            return model.jacobian(theta_safe)
        return fn

    @pytest.mark.parametrize("N,seed", [(4, 0), (10, 1)])
    def test_converges(self, N: int, seed: int) -> None:
        model, _ = make_dwcm_model(N=N, seed=seed)
        theta0 = model.initial_theta("strengths")
        result = solve_newton(
            self._make_clamped_residual(model),
            self._make_clamped_jacobian(model),
            theta0, tol=1e-10, max_iter=100,
        )
        err = model.constraint_error(result.theta)
        assert err < CONV_TOL, f"N={N} Newton error={err:.3e}"


class TestBroydenDWCM:
    """Tests for Broyden solver on DWCM problems."""

    def _make_clamped_residual(self, model: DWCMModel):
        def fn(theta: torch.Tensor) -> torch.Tensor:
            theta_safe = theta.clamp(_ETA_MIN, _ETA_MAX)
            return model.residual(theta_safe)
        return fn

    def _make_clamped_jacobian(self, model: DWCMModel):
        def fn(theta: torch.Tensor) -> torch.Tensor:
            theta_safe = theta.clamp(_ETA_MIN, _ETA_MAX)
            return model.jacobian(theta_safe)
        return fn

    @pytest.mark.parametrize("N,seed", [(4, 0), (10, 1)])
    def test_converges(self, N: int, seed: int) -> None:
        model, _ = make_dwcm_model(N=N, seed=seed)
        theta0 = model.initial_theta("strengths")
        result = solve_broyden(
            self._make_clamped_residual(model),
            self._make_clamped_jacobian(model),
            theta0, tol=1e-10, max_iter=500,
        )
        err = model.constraint_error(result.theta)
        assert err < CONV_TOL, f"N={N} Broyden error={err:.3e}"


class TestLevenbergMarquardtDWCM:
    """Tests for LM solver on DWCM problems."""

    def _make_clamped_residual(self, model: DWCMModel):
        def fn(theta: torch.Tensor) -> torch.Tensor:
            theta_safe = theta.clamp(_ETA_MIN, _ETA_MAX)
            return model.residual(theta_safe)
        return fn

    def _make_clamped_jacobian(self, model: DWCMModel):
        def fn(theta: torch.Tensor) -> torch.Tensor:
            theta_safe = theta.clamp(_ETA_MIN, _ETA_MAX)
            return model.jacobian(theta_safe)
        return fn

    @pytest.mark.parametrize("N,seed", [(4, 0), (10, 1)])
    def test_converges(self, N: int, seed: int) -> None:
        model, _ = make_dwcm_model(N=N, seed=seed)
        theta0 = model.initial_theta("strengths")
        result = solve_lm(
            self._make_clamped_residual(model),
            self._make_clamped_jacobian(model),
            theta0, tol=1e-10, max_iter=500,
        )
        err = model.constraint_error(result.theta)
        assert err < CONV_TOL, f"N={N} LM error={err:.3e}"
