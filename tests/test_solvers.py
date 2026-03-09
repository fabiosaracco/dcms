"""Tests for all 5 DCM solver methods.

Each solver is tested on small networks (N=4, 10) using synthetically
generated θ_true so that the exact solution is known.

Convergence criterion: max absolute constraint error < 1e-5.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.dcm import DCMModel
from src.solvers.base import SolverResult
from src.solvers.fixed_point import solve_fixed_point
from src.solvers.quasi_newton import solve_lbfgs
from src.solvers.newton import solve_newton
from src.solvers.broyden import solve_broyden
from src.solvers.levenberg_marquardt import solve_lm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CONV_TOL = 1e-5  # acceptance threshold for constraint error


def make_test_problem(N: int = 10, seed: int = 0) -> tuple[DCMModel, np.ndarray]:
    """Create a DCM test problem with a known solution.

    Returns:
        (model, theta_true)
    """
    rng = np.random.default_rng(seed)
    theta_true = rng.uniform(0.3, 3.0, size=2 * N)
    x = np.exp(-theta_true[:N])
    y = np.exp(-theta_true[N:])
    xy = x[:, None] * y[None, :]
    P = xy / (1.0 + xy)
    np.fill_diagonal(P, 0.0)
    k_out = P.sum(axis=1)
    k_in = P.sum(axis=0)
    model = DCMModel(k_out, k_in)
    return model, theta_true


# ---------------------------------------------------------------------------
# SolverResult dataclass tests
# ---------------------------------------------------------------------------

class TestSolverResult:
    def test_x_y_properties(self) -> None:
        theta = np.array([1.0, 2.0, 3.0, 4.0])
        result = SolverResult(theta=theta, converged=True, iterations=5)
        np.testing.assert_allclose(result.x, np.exp(-theta[:2]))
        np.testing.assert_allclose(result.y, np.exp(-theta[2:]))

    def test_repr(self) -> None:
        result = SolverResult(
            theta=np.zeros(4), converged=True, iterations=10,
            residuals=[1.0, 0.1, 0.01], elapsed_time=0.5, peak_ram_bytes=1024
        )
        s = repr(result)
        assert "CONVERGED" in s
        assert "iters=10" in s


# ---------------------------------------------------------------------------
# Fixed-Point Iteration
# ---------------------------------------------------------------------------

class TestFixedPoint:
    @pytest.mark.parametrize("N,seed", [(4, 0), (10, 1)])
    def test_gauss_seidel_converges(self, N: int, seed: int) -> None:
        model, _ = make_test_problem(N=N, seed=seed)
        theta0 = model.initial_theta("degrees")
        result = solve_fixed_point(
            model.residual, theta0, model.k_out, model.k_in,
            tol=1e-10, max_iter=5000, damping=1.0, variant="gauss-seidel",
        )
        err = model.constraint_error(result.theta)
        assert err < CONV_TOL, f"N={N} GS error={err:.3e}"

    @pytest.mark.parametrize("N,seed", [(4, 2), (10, 3)])
    def test_jacobi_converges(self, N: int, seed: int) -> None:
        model, _ = make_test_problem(N=N, seed=seed)
        theta0 = model.initial_theta("degrees")
        result = solve_fixed_point(
            model.residual, theta0, model.k_out, model.k_in,
            tol=1e-10, max_iter=5000, damping=0.5, variant="jacobi",
        )
        err = model.constraint_error(result.theta)
        assert err < CONV_TOL, f"N={N} Jacobi error={err:.3e}"

    def test_result_has_residual_history(self) -> None:
        model, _ = make_test_problem(N=4, seed=0)
        theta0 = model.initial_theta("degrees")
        result = solve_fixed_point(
            model.residual, theta0, model.k_out, model.k_in, max_iter=20,
        )
        assert len(result.residuals) > 0

    def test_invalid_variant_raises(self) -> None:
        model, _ = make_test_problem(N=4)
        theta0 = model.initial_theta()
        with pytest.raises(ValueError):
            solve_fixed_point(model.residual, theta0, model.k_out, model.k_in,
                              variant="bad")

    def test_invalid_damping_raises(self) -> None:
        model, _ = make_test_problem(N=4)
        theta0 = model.initial_theta()
        with pytest.raises(ValueError):
            solve_fixed_point(model.residual, theta0, model.k_out, model.k_in,
                              damping=0.0)

    def test_elapsed_time_positive(self) -> None:
        model, _ = make_test_problem(N=4)
        theta0 = model.initial_theta()
        result = solve_fixed_point(
            model.residual, theta0, model.k_out, model.k_in, max_iter=5,
        )
        assert result.elapsed_time >= 0.0

    def test_peak_ram_nonnegative(self) -> None:
        model, _ = make_test_problem(N=4)
        theta0 = model.initial_theta()
        result = solve_fixed_point(
            model.residual, theta0, model.k_out, model.k_in, max_iter=5,
        )
        assert result.peak_ram_bytes >= 0


# ---------------------------------------------------------------------------
# L-BFGS
# ---------------------------------------------------------------------------

class TestLBFGS:
    @pytest.mark.parametrize("N,seed", [(4, 0), (10, 1)])
    def test_converges(self, N: int, seed: int) -> None:
        model, _ = make_test_problem(N=N, seed=seed)
        theta0 = model.initial_theta("degrees")
        result = solve_lbfgs(model.residual, theta0, tol=1e-10, max_iter=2000)
        err = model.constraint_error(result.theta)
        assert err < CONV_TOL, f"N={N} LBFGS error={err:.3e}"

    def test_result_fields(self) -> None:
        model, _ = make_test_problem(N=4)
        theta0 = model.initial_theta()
        result = solve_lbfgs(model.residual, theta0, max_iter=50)
        assert isinstance(result.converged, bool)
        assert isinstance(result.iterations, int)
        assert result.elapsed_time >= 0.0


# ---------------------------------------------------------------------------
# Full Newton
# ---------------------------------------------------------------------------

class TestNewton:
    @pytest.mark.parametrize("N,seed", [(4, 0), (10, 1)])
    def test_converges(self, N: int, seed: int) -> None:
        model, _ = make_test_problem(N=N, seed=seed)
        theta0 = model.initial_theta("degrees")
        result = solve_newton(
            model.residual, model.jacobian, theta0, tol=1e-10, max_iter=100,
        )
        err = model.constraint_error(result.theta)
        assert err < CONV_TOL, f"N={N} Newton error={err:.3e}"

    def test_fast_convergence(self) -> None:
        """Newton should converge in ≤ 50 iterations for small N."""
        model, _ = make_test_problem(N=10, seed=2)
        theta0 = model.initial_theta("degrees")
        result = solve_newton(model.residual, model.jacobian, theta0, tol=1e-10)
        assert result.converged
        assert result.iterations <= 50


# ---------------------------------------------------------------------------
# Broyden
# ---------------------------------------------------------------------------

class TestBroyden:
    @pytest.mark.parametrize("N,seed", [(4, 0), (10, 1)])
    def test_converges(self, N: int, seed: int) -> None:
        model, _ = make_test_problem(N=N, seed=seed)
        theta0 = model.initial_theta("degrees")
        result = solve_broyden(
            model.residual, model.jacobian, theta0, tol=1e-10, max_iter=500,
        )
        err = model.constraint_error(result.theta)
        assert err < CONV_TOL, f"N={N} Broyden error={err:.3e}"


# ---------------------------------------------------------------------------
# Levenberg-Marquardt
# ---------------------------------------------------------------------------

class TestLevenbergMarquardt:
    @pytest.mark.parametrize("N,seed", [(4, 0), (10, 1)])
    def test_converges(self, N: int, seed: int) -> None:
        model, _ = make_test_problem(N=N, seed=seed)
        theta0 = model.initial_theta("degrees")
        result = solve_lm(
            model.residual, model.jacobian, theta0, tol=1e-10, max_iter=500,
        )
        err = model.constraint_error(result.theta)
        assert err < CONV_TOL, f"N={N} LM error={err:.3e}"

    @pytest.mark.parametrize("N,seed", [(4, 3), (10, 4)])
    def test_diagonal_only_mode(self, N: int, seed: int) -> None:
        model, _ = make_test_problem(N=N, seed=seed)
        theta0 = model.initial_theta("degrees")
        result = solve_lm(
            model.residual, model.jacobian, theta0,
            tol=1e-10, max_iter=1000, diagonal_only=True,
        )
        err = model.constraint_error(result.theta)
        assert err < CONV_TOL, f"N={N} LM-diag error={err:.3e}"
