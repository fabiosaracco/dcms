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
        assert result.peak_ram_bytes > 0


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


# ---------------------------------------------------------------------------
# RAM usage tests (all 5 solvers)
# ---------------------------------------------------------------------------

def _fmt_bytes(n: int) -> str:
    """Format byte count as a human-readable string."""
    _KB = 1024
    _MB = 1024 * 1024
    if n < _KB:
        return f"{n} B"
    elif n < _MB:
        return f"{n / _KB:.1f} KB"
    return f"{n / _MB:.2f} MB"


class TestRAMUsage:
    """Verify that every solver measures and reports peak RAM consumption.

    Each test asserts:
    - ``peak_ram_bytes > 0`` — tracemalloc captured at least some allocation.
    - ``peak_ram_bytes`` is printed so it appears in ``pytest -s`` / CI logs.

    The final ``test_ram_summary`` test collects all five solvers on the same
    N=10 problem and prints a comparison table (visible with ``pytest -s``).
    """

    N: int = 10

    def _model(self) -> tuple[DCMModel, np.ndarray]:
        return make_test_problem(N=self.N, seed=7)

    def test_fixed_point_ram(self) -> None:
        model, _ = self._model()
        theta0 = model.initial_theta("degrees")
        result = solve_fixed_point(
            model.residual, theta0, model.k_out, model.k_in,
            tol=1e-8, max_iter=5000,
        )
        assert result.peak_ram_bytes > 0, "Fixed-point must report non-zero RAM"
        print(f"\n[RAM] Fixed-point (GS): {_fmt_bytes(result.peak_ram_bytes)}")

    def test_lbfgs_ram(self) -> None:
        model, _ = self._model()
        theta0 = model.initial_theta("degrees")
        result = solve_lbfgs(model.residual, theta0, tol=1e-8, max_iter=2000)
        assert result.peak_ram_bytes > 0, "L-BFGS must report non-zero RAM"
        print(f"\n[RAM] L-BFGS (m=20):   {_fmt_bytes(result.peak_ram_bytes)}")

    def test_newton_ram(self) -> None:
        model, _ = self._model()
        theta0 = model.initial_theta("degrees")
        result = solve_newton(
            model.residual, model.jacobian, theta0, tol=1e-8, max_iter=100,
        )
        assert result.peak_ram_bytes > 0, "Newton must report non-zero RAM"
        print(f"\n[RAM] Full Newton:      {_fmt_bytes(result.peak_ram_bytes)}")

    def test_broyden_ram(self) -> None:
        model, _ = self._model()
        theta0 = model.initial_theta("degrees")
        result = solve_broyden(
            model.residual, model.jacobian, theta0, tol=1e-8, max_iter=500,
        )
        assert result.peak_ram_bytes > 0, "Broyden must report non-zero RAM"
        print(f"\n[RAM] Broyden:          {_fmt_bytes(result.peak_ram_bytes)}")

    def test_lm_ram(self) -> None:
        model, _ = self._model()
        theta0 = model.initial_theta("degrees")
        result = solve_lm(
            model.residual, model.jacobian, theta0, tol=1e-8, max_iter=500,
        )
        assert result.peak_ram_bytes > 0, "LM must report non-zero RAM"
        print(f"\n[RAM] Levenberg-Marquardt: {_fmt_bytes(result.peak_ram_bytes)}")

    def test_ram_summary(self) -> None:
        """Print a comparison table of RAM and timing for all 5 solvers (N=10).

        Run with ``pytest -s`` to see the table in stdout.
        The test always passes as long as all solvers converge and report RAM.
        """
        model, _ = self._model()
        theta0 = model.initial_theta("degrees")

        solvers = [
            ("Fixed-point (GS)", lambda: solve_fixed_point(
                model.residual, theta0, model.k_out, model.k_in,
                tol=1e-8, max_iter=5000, variant="gauss-seidel",
            )),
            ("L-BFGS (m=20)", lambda: solve_lbfgs(
                model.residual, theta0, tol=1e-8, max_iter=2000,
            )),
            ("Newton (exact J)", lambda: solve_newton(
                model.residual, model.jacobian, theta0, tol=1e-8, max_iter=100,
            )),
            ("Broyden (rank-1)", lambda: solve_broyden(
                model.residual, model.jacobian, theta0, tol=1e-8, max_iter=500,
            )),
            ("Levenberg-Marquardt", lambda: solve_lm(
                model.residual, model.jacobian, theta0, tol=1e-8, max_iter=500,
            )),
        ]

        header = f"\n{'Method':<24} {'Conv?':<7} {'Iters':>6} {'MaxErr':>10} {'Time(s)':>9} {'Peak RAM':>10}"
        rows = [header, "-" * len(header.rstrip())]
        for name, run in solvers:
            r = run()
            err = model.constraint_error(r.theta)
            conv = "YES" if r.converged else "NO"
            rows.append(
                f"{name:<24} {conv:<7} {r.iterations:>6} {err:>10.2e} "
                f"{r.elapsed_time:>9.3f} {_fmt_bytes(r.peak_ram_bytes):>10}"
            )
            assert r.peak_ram_bytes > 0, f"{name} must report non-zero RAM"
        print("\n".join(rows))
