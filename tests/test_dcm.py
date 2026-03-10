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
