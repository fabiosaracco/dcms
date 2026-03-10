"""Tests for the DCM model equations.

Tests cover:
- pij_matrix: correct shape, zero diagonal, values in [0,1].
- residual:   correct value at the true solution.
- gradient:   equals +residual (∇L = F).
- hessian_diag: all entries negative (Hessian of maximisation problem).
- jacobian:   FD consistency and negative diagonal.
- initial_theta: sensible starting point.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.dcm import DCMModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_simple_model(N: int = 4, seed: int = 0) -> tuple[DCMModel, np.ndarray]:
    """Return a small DCMModel and a synthetic θ vector (numpy)."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.5, 2.0, size=2 * N)
    # Build k_out and k_in from the true theta
    x = np.exp(-theta[:N])
    y = np.exp(-theta[N:])
    xy = x[:, None] * y[None, :]
    P = xy / (1.0 + xy)
    np.fill_diagonal(P, 0.0)
    k_out = P.sum(axis=1)
    k_in = P.sum(axis=0)
    model = DCMModel(k_out, k_in)
    return model, theta


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPijMatrix:
    def test_shape(self) -> None:
        model, theta = make_simple_model(N=6)
        P = model.pij_matrix(theta)
        assert P.shape == torch.Size([6, 6])

    def test_zero_diagonal(self) -> None:
        model, theta = make_simple_model(N=6)
        P = model.pij_matrix(theta)
        assert P.diagonal().abs().max().item() == 0.0

    def test_probabilities_in_range(self) -> None:
        model, theta = make_simple_model(N=10)
        P = model.pij_matrix(theta)
        assert (P >= 0.0).all()
        assert (P <= 1.0).all()

    def test_dtype(self) -> None:
        model, theta = make_simple_model(N=4)
        P = model.pij_matrix(theta)
        assert P.dtype == torch.float64


class TestResidual:
    def test_zero_at_true_solution(self) -> None:
        """Residual must be ≈ 0 when θ is the true solution."""
        model, theta_true = make_simple_model(N=8, seed=1)
        F = model.residual(theta_true)
        assert F.abs().max().item() < 1e-10, f"Max residual: {F.abs().max().item():.3e}"

    def test_shape(self) -> None:
        model, theta = make_simple_model(N=5)
        F = model.residual(theta)
        assert F.shape == torch.Size([10])

    def test_nonzero_far_from_solution(self) -> None:
        model, theta_true = make_simple_model(N=5)
        theta_bad = theta_true + 1.0
        F = model.residual(theta_bad)
        assert F.abs().max().item() > 1e-6


class TestGradient:
    def test_gradient_is_residual(self) -> None:
        """∇L = F(θ) (not −F)."""
        model, theta = make_simple_model(N=6)
        grad = model.gradient(theta)
        F = model.residual(theta)
        assert torch.allclose(grad, F), "gradient() must equal +residual()"


class TestHessianDiag:
    def test_all_negative(self) -> None:
        """Diagonal Hessian entries must be negative (concave log-likelihood)."""
        model, theta = make_simple_model(N=8, seed=2)
        h = model.hessian_diag(theta)
        assert (h < 0).all(), f"Some Hessian diag entries non-negative: {h[h >= 0]}"

    def test_shape(self) -> None:
        model, theta = make_simple_model(N=5)
        h = model.hessian_diag(theta)
        assert h.shape == torch.Size([10])


class TestJacobian:
    def test_shape(self) -> None:
        model, theta = make_simple_model(N=4)
        J = model.jacobian(theta)
        assert J.shape == torch.Size([8, 8])

    def test_negative_diagonal(self) -> None:
        """Jacobian diagonal entries must be negative (L is concave)."""
        model, theta = make_simple_model(N=6, seed=3)
        J = model.jacobian(theta)
        diag = J.diagonal()
        assert (diag < 0).all(), f"Some Jacobian diag entries non-negative: {diag[diag >= 0]}"

    def test_finite_difference_consistency(self) -> None:
        """Jacobian must match finite-difference approximation."""
        model, theta = make_simple_model(N=5, seed=4)
        theta_t = torch.tensor(theta, dtype=torch.float64)
        J_exact = model.jacobian(theta_t)
        n2 = len(theta)
        J_fd = torch.zeros(n2, n2, dtype=torch.float64)
        eps = 1e-5
        for j in range(n2):
            dth = torch.zeros(n2, dtype=torch.float64)
            dth[j] = eps
            J_fd[:, j] = (model.residual(theta_t + dth) - model.residual(theta_t - dth)) / (2 * eps)
        assert torch.allclose(J_exact, J_fd, atol=1e-6), (
            f"Max FD error: {(J_exact - J_fd).abs().max().item():.3e}"
        )


class TestInitialTheta:
    def test_degrees_method(self) -> None:
        model, _ = make_simple_model(N=10)
        theta0 = model.initial_theta("degrees")
        assert theta0.shape == torch.Size([20])
        assert torch.isfinite(theta0).all()

    def test_random_method(self) -> None:
        model, _ = make_simple_model(N=10)
        theta0 = model.initial_theta("random")
        assert theta0.shape == torch.Size([20])
        assert torch.isfinite(theta0).all()

    def test_unknown_method_raises(self) -> None:
        model, _ = make_simple_model(N=4)
        with pytest.raises(ValueError):
            model.initial_theta("unknown")


class TestConstraintError:
    def test_zero_at_solution(self) -> None:
        model, theta_true = make_simple_model(N=6, seed=5)
        err = model.constraint_error(theta_true)
        assert err < 1e-10

    def test_positive_away_from_solution(self) -> None:
        model, theta_true = make_simple_model(N=6, seed=5)
        err = model.constraint_error(theta_true + 0.5)
        assert err > 0.0


class TestZeroDegreeBehavior:
    """Tests for the exact handling of nodes with k_out=0 or k_in=0."""

    def _make_zero_degree_model(self) -> DCMModel:
        """Build a 6-node model where node 0 has k_out=0 and node 5 has k_in=0.

        The degree sequences are balanced (Σ k_out = Σ k_in = 6.0) so the
        system is feasible for any solver.
        """
        k_out = np.array([0.0, 2.0, 1.0, 1.5, 1.0, 0.5])
        k_in  = np.array([1.5, 1.0, 1.0, 1.0, 1.5, 0.0])
        return DCMModel(k_out, k_in)

    def test_zero_out_mask(self) -> None:
        model = self._make_zero_degree_model()
        assert model.zero_out[0].item()
        assert not model.zero_out[1:].any().item()

    def test_zero_in_mask(self) -> None:
        model = self._make_zero_degree_model()
        N = model.N
        assert model.zero_in[N - 1].item()
        assert not model.zero_in[: N - 1].any().item()

    def test_pij_row_zero_for_zero_out(self) -> None:
        """p_0j = 0 for all j when k_out[0] = 0."""
        model = self._make_zero_degree_model()
        theta0 = model.initial_theta("degrees")
        P = model.pij_matrix(theta0)
        assert P[0].abs().max().item() == 0.0, "Row 0 must be exactly zero"

    def test_pij_col_zero_for_zero_in(self) -> None:
        """p_i5 = 0 for all i when k_in[5] = 0."""
        model = self._make_zero_degree_model()
        N = model.N
        theta0 = model.initial_theta("degrees")
        P = model.pij_matrix(theta0)
        assert P[:, N - 1].abs().max().item() == 0.0, "Last column must be exactly zero"

    def test_residual_exact_zero_for_zero_degree(self) -> None:
        """Residual components for zero-degree nodes must be exactly 0."""
        model = self._make_zero_degree_model()
        N = model.N
        theta0 = model.initial_theta("degrees")
        F = model.residual(theta0)
        # F[0] corresponds to k_out[0] = 0
        assert F[0].abs().item() == 0.0
        # F[N + N-1] corresponds to k_in[N-1] = 0
        assert F[N + N - 1].abs().item() == 0.0

    def test_initial_theta_zero_degree_nodes_large(self) -> None:
        """initial_theta must set θ = _THETA_MAX for zero-degree nodes."""
        from src.models.dcm import _THETA_MAX
        model = self._make_zero_degree_model()
        N = model.N
        theta0 = model.initial_theta("degrees")
        assert theta0[0].item() == _THETA_MAX, "θ_out[0] must equal _THETA_MAX"
        assert theta0[N + N - 1].item() == _THETA_MAX, "θ_in[N-1] must equal _THETA_MAX"

    def test_initial_theta_random_zero_degree_nodes_large(self) -> None:
        """initial_theta('random') must also set θ = _THETA_MAX for zero-degree nodes."""
        from src.models.dcm import _THETA_MAX
        model = self._make_zero_degree_model()
        N = model.N
        theta0 = model.initial_theta("random")
        assert theta0[0].item() == _THETA_MAX
        assert theta0[N + N - 1].item() == _THETA_MAX

    def test_solver_converges_with_zero_degree_nodes(self) -> None:
        """Zero-degree nodes must not prevent solver convergence."""
        from src.solvers.newton import solve_newton
        model = self._make_zero_degree_model()
        theta0 = model.initial_theta("degrees")
        result = solve_newton(
            model.residual, model.jacobian, theta0, tol=1e-10, max_iter=100,
        )
        err = model.constraint_error(result.theta)
        assert err < 1e-5, f"Newton error with zero-degree nodes: {err:.3e}"


class TestSaturatedNodeBehavior:
    """Tests for the exact handling of nodes with k_out=N-1 or k_in=N-1."""

    def _make_saturated_model(self) -> DCMModel:
        """Build a 6-node model where node 0 has k_out=5=N-1 and node 5 has k_in=5=N-1.

        The degree sequences are balanced (Σ k_out = Σ k_in = 11.0).

        Feasibility requires:
        - k_out[i] ≥ 1 for i ≠ 5 (node 5's in-saturation forces p_i5 ≈ 1).
        - k_in[j] ≥ 1 for j ≠ 0 (node 0's out-saturation forces p_0j ≈ 1).
        Both constraints are satisfied by the chosen sequences.
        """
        # N=6, so N-1=5.  Node 0 is out-saturated; node 5 is in-saturated.
        k_out = np.array([5.0, 2.0, 1.0, 1.0, 1.0, 1.0])
        k_in  = np.array([1.0, 1.0, 1.5, 1.5, 1.0, 5.0])
        return DCMModel(k_out, k_in)

    def test_initial_theta_saturated_out_nodes_negative(self) -> None:
        """initial_theta('degrees') must set θ_out = -_THETA_MAX for k_out = N-1."""
        from src.models.dcm import _THETA_MAX
        model = self._make_saturated_model()
        theta0 = model.initial_theta("degrees")
        assert theta0[0].item() == -_THETA_MAX, (
            f"θ_out[0] should be -{_THETA_MAX}, got {theta0[0].item()}"
        )

    def test_initial_theta_saturated_in_nodes_negative(self) -> None:
        """initial_theta('degrees') must set θ_in = -_THETA_MAX for k_in = N-1."""
        from src.models.dcm import _THETA_MAX
        model = self._make_saturated_model()
        N = model.N
        theta0 = model.initial_theta("degrees")
        assert theta0[N + N - 1].item() == -_THETA_MAX, (
            f"θ_in[N-1] should be -{_THETA_MAX}, got {theta0[N + N - 1].item()}"
        )

    def test_initial_theta_random_saturated_nodes_negative(self) -> None:
        """initial_theta('random') must also set θ = -_THETA_MAX for saturated nodes."""
        from src.models.dcm import _THETA_MAX
        model = self._make_saturated_model()
        N = model.N
        theta0 = model.initial_theta("random")
        assert theta0[0].item() == -_THETA_MAX
        assert theta0[N + N - 1].item() == -_THETA_MAX

    def test_pij_row_near_one_for_saturated_out(self) -> None:
        """p_0j ≈ 1 for all j≠0 when k_out[0] = N-1 (connects to every other node)."""
        model = self._make_saturated_model()
        theta0 = model.initial_theta("degrees")
        P = model.pij_matrix(theta0)
        # Off-diagonal entries in row 0 should be close to 1.0
        off_diag = torch.cat([P[0, :0], P[0, 1:]])  # skip diagonal (index 0)
        assert (off_diag > 0.99).all(), (
            f"Row 0 off-diagonal entries should be > 0.99; got min={off_diag.min():.4f}"
        )

    def test_pij_col_near_one_for_saturated_in(self) -> None:
        """p_i5 ≈ 1 for all i≠5 when k_in[5] = N-1 (receives from every other node)."""
        model = self._make_saturated_model()
        N = model.N
        theta0 = model.initial_theta("degrees")
        P = model.pij_matrix(theta0)
        # Off-diagonal entries in column N-1 should be close to 1.0
        off_diag = torch.cat([P[:N - 1, N - 1], P[N:, N - 1]])  # skip diagonal (index N-1)
        assert (off_diag > 0.99).all(), (
            f"Column N-1 off-diagonal entries should be > 0.99; got min={off_diag.min():.4f}"
        )

    def test_residual_near_zero_for_saturated_nodes(self) -> None:
        """Residual components for saturated nodes must be near 0 at initial_theta."""
        model = self._make_saturated_model()
        N = model.N
        theta0 = model.initial_theta("degrees")
        F = model.residual(theta0)
        # F[0] corresponds to k_out[0] = N-1 = 5
        assert F[0].abs().item() < 1e-6, (
            f"Residual for saturated k_out[0]: {F[0].item():.3e}"
        )
        # F[N + N-1] corresponds to k_in[N-1] = N-1 = 5
        assert F[N + N - 1].abs().item() < 1e-6, (
            f"Residual for saturated k_in[N-1]: {F[N + N - 1].item():.3e}"
        )

    def test_solver_converges_with_saturated_nodes(self) -> None:
        """Saturated nodes must not prevent solver convergence."""
        from src.solvers.newton import solve_newton
        model = self._make_saturated_model()
        theta0 = model.initial_theta("degrees")
        result = solve_newton(
            model.residual, model.jacobian, theta0, tol=1e-10, max_iter=200,
        )
        err = model.constraint_error(result.theta)
        assert err < 1e-5, f"Newton error with saturated nodes: {err:.3e}"
