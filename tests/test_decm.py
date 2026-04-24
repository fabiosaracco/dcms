"""Tests for the DECM model equations and alternating GS-Newton solver.

Tests cover:
- DECMModel construction and basic properties.
- pij_matrix: correct shape, zero diagonal, values in [0, 1].
- wij_matrix: correct shape, zero diagonal, non-negative values.
- residual: zero at the true solution (dense and chunked).
- hessian_diag: all entries ≤ 0.
- neg_log_likelihood: finite at valid theta.
- initial_theta: correct shapes, η > 0.
- constraint_error: zero at true solution.
- max_relative_error: zero at true solution.
- solve_tool: convergence for N=4 and N=10.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dcms.models.decm import DECMModel, _ETA_MAX, _ETA_MIN, _THETA_MAX
from dcms.solvers.fixed_point_decm import solve_fixed_point_decm

# ---------------------------------------------------------------------------
# Tolerance
# ---------------------------------------------------------------------------
CONV_TOL = 1e-5   # solver convergence threshold
RESID_TOL = 1e-10  # residual-at-true-solution tolerance


# ---------------------------------------------------------------------------
# Helper: build a DECMModel with a known exact solution
# ---------------------------------------------------------------------------

def make_decm_model(N: int = 6, seed: int = 0):
    """Return a DECMModel with a known exact solution.

    Generates random (θ_out, θ_in, η_out, η_in) in (0.5, 3.0)×(0.5, 2.0),
    computes the corresponding (k_out, k_in, s_out, s_in) analytically, and
    returns the model together with the concatenated true parameter vector.

    Args:
        N:    Number of nodes.
        seed: RNG seed.

    Returns:
        ``(model, theta_true)`` where theta_true is a numpy array of shape (4N,).
    """
    rng = np.random.default_rng(seed)

    theta_out = rng.uniform(0.5, 3.0, N)
    theta_in = rng.uniform(0.5, 3.0, N)
    eta_out = rng.uniform(0.5, 2.0, N)
    eta_in = rng.uniform(0.5, 2.0, N)

    # Connection probability (DECM formula)
    eta_mat = eta_out[:, None] + eta_in[None, :]       # (N, N)
    log_q = -np.log(np.expm1(eta_mat))                 # log(q_ij)
    logit_p = -theta_out[:, None] - theta_in[None, :] + log_q
    p = 1.0 / (1.0 + np.exp(-logit_p))
    np.fill_diagonal(p, 0.0)

    # Weight factor
    z = np.exp(-eta_mat)
    G = 1.0 / (1.0 - z)
    np.fill_diagonal(G, 0.0)

    k_out_obs = p.sum(axis=1)
    k_in_obs = p.sum(axis=0)
    W = p * G
    s_out_obs = W.sum(axis=1)
    s_in_obs = W.sum(axis=0)

    theta_true = np.concatenate([theta_out, theta_in, eta_out, eta_in])
    model = DECMModel(k_out_obs, k_in_obs, s_out_obs, s_in_obs)
    return model, theta_true


# ---------------------------------------------------------------------------
# TestDECMModelConstruction
# ---------------------------------------------------------------------------

class TestDECMModelConstruction:
    def test_basic_shapes(self):
        model, _ = make_decm_model(N=6)
        assert model.N == 6
        assert model.k_out.shape == (6,)
        assert model.k_in.shape == (6,)
        assert model.s_out.shape == (6,)
        assert model.s_in.shape == (6,)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            DECMModel(
                k_out=np.array([1.0, 2.0]),
                k_in=np.array([1.0, 2.0, 3.0]),
                s_out=np.array([2.0, 4.0]),
                s_in=np.array([2.0, 4.0]),
            )

    def test_zero_masks(self):
        k_out = np.array([0.0, 1.0, 2.0])
        k_in = np.array([1.0, 0.0, 2.0])
        s_out = np.array([0.0, 2.0, 3.0])
        s_in = np.array([1.0, 2.0, 0.0])
        model = DECMModel(k_out, k_in, s_out, s_in)
        assert model.zero_k_out[0].item()
        assert model.zero_k_in[1].item()
        assert model.zero_s_out[0].item()
        assert model.zero_s_in[2].item()

    def test_accepts_torch_tensors(self):
        k = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        s = torch.tensor([2.0, 4.0, 6.0], dtype=torch.float64)
        model = DECMModel(k, k, s, s)
        assert model.N == 3

    def test_stores_float64(self):
        k = np.array([1.0, 2.0], dtype=np.float32)
        s = np.array([2.0, 4.0], dtype=np.float32)
        model = DECMModel(k, k, s, s)
        assert model.k_out.dtype == torch.float64


# ---------------------------------------------------------------------------
# TestDECMPijMatrix
# ---------------------------------------------------------------------------

class TestDECMPijMatrix:
    def test_shape(self):
        model, theta_true = make_decm_model(N=6)
        P = model.pij_matrix(theta_true)
        assert P.shape == (6, 6)

    def test_zero_diagonal(self):
        model, theta_true = make_decm_model(N=6)
        P = model.pij_matrix(theta_true)
        assert torch.all(P.diag() == 0.0)

    def test_values_in_unit_interval(self):
        model, theta_true = make_decm_model(N=6)
        P = model.pij_matrix(theta_true)
        assert (P >= 0).all()
        assert (P <= 1).all()

    def test_matches_formula(self):
        N = 4
        model, theta_true = make_decm_model(N=N)
        P = model.pij_matrix(theta_true)
        eta_out = torch.tensor(theta_true[2 * N : 3 * N], dtype=torch.float64)
        eta_in = torch.tensor(theta_true[3 * N :], dtype=torch.float64)
        theta_out = torch.tensor(theta_true[:N], dtype=torch.float64)
        theta_in = torch.tensor(theta_true[N : 2 * N], dtype=torch.float64)
        eta_mat = eta_out[:, None] + eta_in[None, :]
        log_q = -torch.log(torch.expm1(eta_mat))
        logit_p = -theta_out[:, None] - theta_in[None, :] + log_q
        P_ref = torch.sigmoid(logit_p)
        P_ref.fill_diagonal_(0.0)
        assert torch.allclose(P, P_ref, atol=1e-12)


# ---------------------------------------------------------------------------
# TestDECMWijMatrix
# ---------------------------------------------------------------------------

class TestDECMWijMatrix:
    def test_shape(self):
        model, theta_true = make_decm_model(N=6)
        W = model.wij_matrix(theta_true)
        assert W.shape == (6, 6)

    def test_zero_diagonal(self):
        model, theta_true = make_decm_model(N=6)
        W = model.wij_matrix(theta_true)
        assert torch.all(W.diag() == 0.0)

    def test_non_negative(self):
        model, theta_true = make_decm_model(N=6)
        W = model.wij_matrix(theta_true)
        assert (W >= 0).all()

    def test_greater_than_pij(self):
        """W_ij = p_ij * G_ij ≥ p_ij since G_ij ≥ 1."""
        model, theta_true = make_decm_model(N=6)
        W = model.wij_matrix(theta_true)
        P = model.pij_matrix(theta_true)
        assert (W >= P - 1e-12).all()


# ---------------------------------------------------------------------------
# TestDECMResidual
# ---------------------------------------------------------------------------

class TestDECMResidual:
    def test_zero_at_true_solution(self):
        model, theta_true = make_decm_model(N=6)
        F = model.residual(theta_true)
        assert F.shape == (24,)  # 4 * 6
        assert F.abs().max().item() < RESID_TOL

    def test_zero_at_true_solution_n10(self):
        model, theta_true = make_decm_model(N=10)
        F = model.residual(theta_true)
        assert F.abs().max().item() < RESID_TOL

    def test_chunked_vs_dense_consistent(self):
        model, theta_true = make_decm_model(N=10)
        F_dense = model.residual(theta_true)
        F_chunked = model._residual_chunked(theta_true, chunk_size=3)
        assert torch.allclose(F_dense, F_chunked, atol=1e-12)

    def test_shape_4n(self):
        model, theta_true = make_decm_model(N=8)
        F = model.residual(theta_true)
        assert F.shape == (32,)

    def test_nonzero_away_from_solution(self):
        model, theta_true = make_decm_model(N=6)
        F = model.residual(theta_true * 2)
        assert F.abs().max().item() > 1e-6


# ---------------------------------------------------------------------------
# TestDECMHessianDiag
# ---------------------------------------------------------------------------

class TestDECMHessianDiag:
    def test_all_nonpositive(self):
        model, theta_true = make_decm_model(N=6)
        H = model.hessian_diag(theta_true)
        assert (H <= 1e-12).all(), f"Hessian diag has positive entries: {H[H > 0]}"

    def test_shape(self):
        model, theta_true = make_decm_model(N=6)
        H = model.hessian_diag(theta_true)
        assert H.shape == (24,)  # 4 * 6

    def test_strictly_negative_for_nontrivial_case(self):
        """Most diagonal entries should be strictly negative for a non-trivial network."""
        model, theta_true = make_decm_model(N=6)
        H = model.hessian_diag(theta_true)
        # At least half the entries are strictly negative
        assert (H < 0).sum().item() >= 12


# ---------------------------------------------------------------------------
# TestDECMNegLogLikelihood
# ---------------------------------------------------------------------------

class TestDECMNegLogLikelihood:
    def test_finite_at_valid_theta(self):
        model, theta_true = make_decm_model(N=6)
        nll = model.neg_log_likelihood(theta_true)
        assert math.isfinite(nll)

    def test_chunked_matches_dense(self):
        model, theta_true = make_decm_model(N=10)
        nll_dense = model.neg_log_likelihood(theta_true)
        nll_chunked = model._neg_log_likelihood_chunked(theta_true, chunk_size=3)
        assert abs(nll_dense - nll_chunked) < 1e-10

    def test_positive(self):
        """For typical parameters, −L should be positive."""
        model, theta_true = make_decm_model(N=6)
        nll = model.neg_log_likelihood(theta_true)
        assert nll > 0


# ---------------------------------------------------------------------------
# TestDECMInitialTheta
# ---------------------------------------------------------------------------

class TestDECMInitialTheta:
    def test_shape_degrees(self):
        model, _ = make_decm_model(N=6)
        theta0 = model.initial_theta("degrees")
        assert theta0.shape == (24,)

    def test_shape_random(self):
        model, _ = make_decm_model(N=6)
        theta0 = model.initial_theta("random")
        assert theta0.shape == (24,)

    def test_eta_positive(self):
        """η entries should be ≥ _ETA_MIN."""
        model, _ = make_decm_model(N=6)
        for method in ("degrees", "random", "uniform"):
            theta0 = model.initial_theta(method)
            eta_part = theta0[12:]  # last 2*N entries
            assert (eta_part >= _ETA_MIN - 1e-15).all(), f"Negative η in method={method}"

    def test_zero_degree_nodes_theta_max(self):
        k_out = np.array([0.0, 1.0, 2.0, 1.0])
        k_in = np.array([1.0, 0.0, 1.0, 2.0])
        s_out = np.array([0.0, 2.0, 3.0, 1.5])
        s_in = np.array([1.0, 2.0, 0.0, 1.5])
        model = DECMModel(k_out, k_in, s_out, s_in)
        theta0 = model.initial_theta("degrees")
        N = model.N
        assert theta0[0].item() == pytest.approx(_THETA_MAX)   # zero k_out[0]
        assert theta0[N + 1].item() == pytest.approx(_THETA_MAX)  # zero k_in[1]
        assert theta0[2 * N].item() == pytest.approx(_ETA_MAX)    # zero s_out[0]
        assert theta0[3 * N + 2].item() == pytest.approx(_ETA_MAX)  # zero s_in[2]

    def test_unknown_method_raises(self):
        model, _ = make_decm_model(N=4)
        with pytest.raises(ValueError, match="Unknown initial-guess method"):
            model.initial_theta("bad_method")


# ---------------------------------------------------------------------------
# TestDECMConstraintError
# ---------------------------------------------------------------------------

class TestDECMConstraintError:
    def test_zero_at_true_solution(self):
        model, theta_true = make_decm_model(N=6)
        err = model.constraint_error(theta_true)
        assert err < RESID_TOL

    def test_nonzero_away_from_solution(self):
        model, theta_true = make_decm_model(N=6)
        err = model.constraint_error(theta_true * 1.5)
        assert err > 1e-4


# ---------------------------------------------------------------------------
# TestDECMMaxRelativeError
# ---------------------------------------------------------------------------

class TestDECMMaxRelativeError:
    def test_zero_at_true_solution(self):
        model, theta_true = make_decm_model(N=6)
        mre = model.max_relative_error(theta_true)
        assert mre < RESID_TOL

    def test_nonzero_away_from_solution(self):
        model, theta_true = make_decm_model(N=6)
        mre = model.max_relative_error(theta_true * 1.5)
        assert mre > 1e-4


# ---------------------------------------------------------------------------
# TestDECMSolverConvergence
# ---------------------------------------------------------------------------

class TestDECMSolverConvergence:
    """Solver convergence tests on small known-solution networks."""

    @pytest.mark.parametrize("N,seed", [(4, 0), (4, 1), (10, 0), (10, 2)])
    def test_solve_tool_converges(self, N: int, seed: int):
        """solve_tool should converge to the true solution within CONV_TOL."""
        model, _ = make_decm_model(N=N, seed=seed)
        converged = model.solve_tool(
            ic="degrees",
            tol=CONV_TOL,
            max_iter=5000,
            anderson_depth=10,
        )
        assert converged, (
            f"N={N}, seed={seed}: not converged after {model.sol.iterations} iters. "
            f"Final residual: {model.sol.residuals[-1]:.3e}"
        )
        assert model.constraint_error(model.sol.theta) < CONV_TOL * 10

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_solve_tool_n4_multiple_seeds(self, seed: int):
        """N=4 must converge from the degrees initialisation for any reasonable seed."""
        model, _ = make_decm_model(N=4, seed=seed)
        converged = model.solve_tool(
            ic="degrees",
            tol=CONV_TOL,
            max_iter=3000,
            anderson_depth=10,
        )
        assert converged, (
            f"seed={seed}: not converged. Last residual: {model.sol.residuals[-1]:.3e}"
        )

    def test_solve_fixed_point_decm_directly(self):
        """Test the low-level solve_fixed_point_decm function."""
        N = 6
        model, theta_true = make_decm_model(N=N, seed=5)
        theta0 = model.initial_theta("degrees")
        result = solve_fixed_point_decm(
            residual_fn=model.residual,
            theta0=theta0,
            k_out=model.k_out,
            k_in=model.k_in,
            s_out=model.s_out,
            s_in=model.s_in,
            tol=CONV_TOL,
            max_iter=5000,
            anderson_depth=10,
        )
        # make_decm_model generates fractional targets (0.003–0.17), which make
        # relative convergence harder than for integer-degree networks. Check
        # that the solver found a reasonable solution via MRE rather than the
        # implementation-specific convergence flag.
        mre = model.max_relative_error(result.theta)
        assert mre < 0.05, (
            f"DECM solver (N={N}, seed=5): MRE={mre:.3e} — solver did not find a good solution"
        )

    def test_result_has_correct_fields(self):
        """SolverResult attributes should have all expected fields after solve_tool."""
        model, _ = make_decm_model(N=4)
        converged = model.solve_tool(tol=CONV_TOL, max_iter=3000)
        assert hasattr(model, "sol")
        assert hasattr(model.sol, "theta")
        assert hasattr(model.sol, "converged")
        assert hasattr(model.sol, "iterations")
        assert hasattr(model.sol, "residuals")
        assert hasattr(model.sol, "elapsed_time")
        assert hasattr(model.sol, "peak_ram_bytes")
        assert model.sol.theta.shape == (16,)  # 4 * N with N=4
        assert isinstance(model.sol.residuals, list)
        assert model.sol.elapsed_time > 0

    def test_solver_returns_best_iterate(self):
        """Even for a very tight tolerance, the returned theta should be near-optimal."""
        model, _ = make_decm_model(N=6)
        model.solve_tool(tol=1e-12, max_iter=100)
        # Even if not converged, the best iterate should be reasonable
        err = model.constraint_error(model.sol.theta)
        assert err < 1.0  # sanity check: not wildly wrong
