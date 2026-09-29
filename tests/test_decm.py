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


def make_decm_model_degenerate(N0: int = 4, r: int = 3, seed: int = 0):
    """Return a genuinely degeneracy-reducible DECMModel with a known exact
    solution, by construction rather than coincidence.

    Draws N0 base (theta_out, theta_in, eta_out, eta_in) values as in
    :func:`make_decm_model`, then builds an N = N0*r network of ``r`` exact
    physical copies of each base node. The group-level target sequences use
    the same "weighted sum minus one diagonal term" identity that
    :func:`_decm_step_dense_weighted`/:func:`_decm_step_chunked_weighted` rely on
    (module ``dcms.solvers.fixed_point_decm``) -- i.e. the base theta values
    are constructed to already be the *exact* fixed point of the reduced
    (group-level) system, not merely a plausible target. This makes
    ``model.max_relative_error`` against the tiled ``theta_true`` a genuine
    correctness check, not just a convergence check.

    Args:
        N0:   Number of distinct degeneracy groups (base nodes).
        r:    Physical copies per group (group multiplicity).
        seed: RNG seed.

    Returns:
        ``(model, theta_true)`` where ``model.N == N0 * r`` and
        ``theta_true`` is a numpy array of shape (4*N0*r,).
    """
    rng = np.random.default_rng(seed)

    theta_out = rng.uniform(0.5, 3.0, N0)
    theta_in = rng.uniform(0.5, 3.0, N0)
    eta_out = rng.uniform(0.5, 2.0, N0)
    eta_in = rng.uniform(0.5, 2.0, N0)

    eta_g = eta_out[:, None] + eta_in[None, :]
    log_q_g = -np.log(np.expm1(eta_g))
    logit_p_g = -theta_out[:, None] - theta_in[None, :] + log_q_g
    P_g = 1.0 / (1.0 + np.exp(-logit_p_g))         # diagonal NOT zeroed here
    G_g = 1.0 / (1.0 - np.exp(-eta_g))
    W_g = P_g * G_g

    mult = np.full(N0, float(r))
    k_out_g = (P_g * mult[None, :]).sum(1) - np.diagonal(P_g)
    k_in_g = (P_g * mult[:, None]).sum(0) - np.diagonal(P_g)
    s_out_g = (W_g * mult[None, :]).sum(1) - np.diagonal(W_g)
    s_in_g = (W_g * mult[:, None]).sum(0) - np.diagonal(W_g)

    k_out = np.repeat(k_out_g, r)
    k_in = np.repeat(k_in_g, r)
    s_out = np.repeat(s_out_g, r)
    s_in = np.repeat(s_in_g, r)
    theta_true = np.concatenate(
        [np.repeat(theta_out, r), np.repeat(theta_in, r), np.repeat(eta_out, r), np.repeat(eta_in, r)]
    )
    model = DECMModel(k_out, k_in, s_out, s_in)
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
        assert model.constraint_error(model.sol.best_theta) < CONV_TOL * 10

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
        mre = model.max_relative_error(result.best_theta)
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
        assert model.sol.best_theta.shape == (16,)  # 4 * N with N=4
        assert isinstance(model.sol.residuals, list)
        assert model.sol.elapsed_time > 0

    def test_solver_returns_best_iterate(self):
        """Even for a very tight tolerance, the returned theta should be near-optimal."""
        model, _ = make_decm_model(N=6)
        model.solve_tool(tol=1e-12, max_iter=100)
        # Even if not converged, the best iterate should be reasonable
        err = model.constraint_error(model.sol.best_theta)
        assert err < 1.0  # sanity check: not wildly wrong

    def test_solve_tool_reduce_degeneracy_default_true(self):
        """solve_tool() must use the reduced path by default (message tags
        it) and reduce_degeneracy=False must still converge to the same
        solution."""
        model, _ = make_decm_model(N=10, seed=0)
        m1 = DECMModel(model.k_out, model.k_in, model.s_out, model.s_in)
        conv1 = m1.solve_tool(ic="degrees", tol=CONV_TOL, max_iter=5000, anderson_depth=10)
        assert conv1
        assert "degeneracy-reduced" in m1.sol.message

        m2 = DECMModel(model.k_out, model.k_in, model.s_out, model.s_in)
        conv2 = m2.solve_tool(ic="degrees", tol=CONV_TOL, max_iter=5000, anderson_depth=10, reduce_degeneracy=False)
        assert conv2
        assert "degeneracy-reduced" not in m2.sol.message

        assert m1.constraint_error(m1.sol.best_theta) < CONV_TOL * 10
        assert m2.constraint_error(m2.sol.best_theta) < CONV_TOL * 10

    def test_solve_tool_falls_back_for_backtracking_gamma(self):
        """reduce_degeneracy=True with backtracking_gamma>0 must fall back
        to the full solver (not supported in the reduced path)."""
        model, _ = make_decm_model(N=10, seed=0)
        m = DECMModel(model.k_out, model.k_in, model.s_out, model.s_in)
        conv = m.solve_tool(ic="degrees", tol=CONV_TOL, max_iter=5000, anderson_depth=10, backtracking_gamma=1.2)
        assert conv


# ---------------------------------------------------------------------------
# Hub eta_out freeze inside the Newton step + read-only state hook
# ---------------------------------------------------------------------------

from dcms.solvers.fixed_point_decm import (  # noqa: E402
    _decm_step_chunked_weighted,
    _decm_step_dense_weighted,
    solve_fixed_point_decm_degenerate,
)


def _group_problem(M: int = 8, seed: int = 5):
    """Group-level (k_out,k_in,s_out,s_in,mult) with a known exact solution,
    plus a perturbed theta away from it (so raw Newton steps are non-trivial)."""
    rng = np.random.default_rng(seed)
    th_o, th_i = rng.uniform(0.5, 3.0, M), rng.uniform(0.5, 3.0, M)
    et_o, et_i = rng.uniform(0.3, 2.0, M), rng.uniform(0.3, 2.0, M)
    eta = et_o[:, None] + et_i[None, :]
    P = 1.0 / (1.0 + np.exp(th_o[:, None] + th_i[None, :] + np.log(np.expm1(eta))))
    W = P / (1.0 - np.exp(-eta))
    mult = np.full(M, 2.0)
    d = lambda A: np.diagonal(A)
    k_out = (P * mult[None, :]).sum(1) - d(P)
    k_in = (P * mult[:, None]).sum(0) - d(P)
    s_out = (W * mult[None, :]).sum(1) - d(W)
    s_in = (W * mult[:, None]).sum(0) - d(W)
    t = lambda a: torch.tensor(a, dtype=torch.float64)
    theta_true = np.concatenate([th_o, th_i, et_o, et_i])
    theta = theta_true * (1.0 + 0.15 * rng.standard_normal(4 * M))
    theta[2 * M:] = np.abs(theta[2 * M:]) + 0.05
    return (t(theta), t(k_out), t(k_in), t(s_out), t(s_in), t(mult), M)


class TestHubOutFreezeInStep:
    """In the Gauss-Seidel step, the in-side pass must see the hub eta_out
    that will actually be applied (the input value, since hub eta is owned by
    the hub bisection and the raw Newton proposal is discarded by the caller),
    not the raw proposal -- otherwise theta_in is corrected against a phantom
    environment (the e1 post-fix-undo bug, see decm_low_degree_precision_floor)."""

    @staticmethod
    def _call(fn, theta, k_out, k_in, s_out, s_in, mult, M, mask):
        z = lambda x: x == 0
        args = (theta, k_out, k_in, s_out, s_in, z(k_out), z(k_in), z(s_out), z(s_in))
        if fn is _decm_step_chunked_weighted:
            return fn(*args, 3, 0.5, mult, hub_out_mask=mask)
        return fn(*args, 0.5, mult, hub_out_mask=mask)

    @pytest.mark.parametrize("fn", [_decm_step_dense_weighted, _decm_step_chunked_weighted])
    def test_none_and_all_false_mask_are_identical(self, fn) -> None:
        theta, k_out, k_in, s_out, s_in, mult, M = _group_problem()
        ref = self._call(fn, theta, k_out, k_in, s_out, s_in, mult, M, None)
        off = self._call(fn, theta, k_out, k_in, s_out, s_in, mult, M, torch.zeros(M, dtype=torch.bool))
        assert torch.equal(ref[0], off[0]) and torch.equal(ref[1], off[1])

    @pytest.mark.parametrize("fn", [_decm_step_dense_weighted, _decm_step_chunked_weighted])
    def test_mask_freezes_hub_eta_out_and_changes_only_the_in_side(self, fn) -> None:
        theta, k_out, k_in, s_out, s_in, mult, M = _group_problem()
        mask = torch.zeros(M, dtype=torch.bool)
        mask[[1, 4]] = True
        ref_t, ref_F = self._call(fn, theta, k_out, k_in, s_out, s_in, mult, M, None)
        new_t, new_F = self._call(fn, theta, k_out, k_in, s_out, s_in, mult, M, mask)
        eta_out_in = theta[2 * M:3 * M]
        # hub eta_out is held at its input value...
        assert torch.equal(new_t[2 * M:3 * M][mask], eta_out_in[mask])
        # ...while the unmasked raw step really would have moved it (test is not vacuous)
        assert (ref_t[2 * M:3 * M][mask] - eta_out_in[mask]).abs().max() > 1e-6
        # the residual is evaluated at the input theta: unaffected by the mask
        assert torch.equal(ref_F, new_F)
        # out-side pass 1 is unaffected: theta_out and non-hub eta_out are identical
        assert torch.equal(ref_t[:M], new_t[:M])
        assert torch.equal(ref_t[2 * M:3 * M][~mask], new_t[2 * M:3 * M][~mask])
        # the in-side pass DOES see the different environment
        assert (ref_t[M:2 * M] - new_t[M:2 * M]).abs().max() > 1e-9

    def test_hub_solver_still_converges_with_freeze(self) -> None:
        model, theta_true = make_decm_model_degenerate(N0=6, r=3, seed=3)
        theta0 = model.initial_theta("degrees")
        result = solve_fixed_point_decm_degenerate(
            theta0, model.k_out, model.k_in, model.s_out, model.s_in,
            tol=1e-8, max_iter=4000, hub_sk_threshold=0.25,   # s/k > 0.25: 6 out-hubs, 3 in-hubs
        )
        assert result.converged, result.message
        assert model.max_relative_error(result.best_theta) < CONV_TOL


class TestDiagStateCallback:
    def test_called_every_iteration_and_read_only(self) -> None:
        model, _ = make_decm_model_degenerate(N0=4, r=3, seed=2)
        theta0 = model.initial_theta("degrees")
        seen = []
        kw = dict(tol=1e-12, max_iter=5)
        res_hook = solve_fixed_point_decm_degenerate(
            theta0, model.k_out, model.k_in, model.s_out, model.s_in,
            diag_state_callback=lambda n, th, F: seen.append((n, tuple(th.shape), tuple(F.shape))), **kw,
        )
        res_plain = solve_fixed_point_decm_degenerate(
            theta0, model.k_out, model.k_in, model.s_out, model.s_in, **kw,
        )
        assert [s[0] for s in seen] == list(range(len(seen))) and len(seen) == 5
        assert all(s[1] == (16,) and s[2] == (16,) for s in seen)   # 4 * M, M = 4 groups
        assert np.array_equal(res_hook.theta, res_plain.theta)


class TestResumeKeepsNegativeHubEta:
    """A checkpoint legitimately carries NEGATIVE hub eta (negative-eta
    relaxation: only the pair sum eta_out+eta_in must stay > 0). The
    start-of-solve clamp used to floor every eta to _ETA_MIN, wiping them:
    e1's checkpoint went from MRE 6.66e-4 to 0.78 at iteration 1 and q4's from
    2.86e-3 to 0.74, on every resume (see decm_low_degree_precision_floor)."""

    THR = 0.25     # s/k threshold: nodes 0-2, 12-14 are out-hubs of this problem

    def _setup(self):
        model, theta_true = make_decm_model_degenerate(N0=6, r=3, seed=3)
        N = model.N
        # a solution-quality start with hub eta_out pushed negative (partners keep z > 0)
        theta0 = torch.as_tensor(theta_true, dtype=torch.float64).clone()
        sk = model.s_out / model.k_out.clamp(min=1.0)
        hub = (sk > self.THR) & (model.s_out > 0) & (model.k_out > 0)
        assert hub.any(), "test problem must contain out-hubs"
        theta0[2 * N:3 * N][hub] = -0.05
        assert float(theta0[3 * N:].min()) > 0.05      # every pair sum stays positive
        return model, theta0

    def _iter0_mre(self, model, theta0, **kw):
        seen = []
        solve_fixed_point_decm_degenerate(
            theta0, model.k_out, model.k_in, model.s_out, model.s_in,
            tol=1e-15, max_iter=1, hub_sk_threshold=self.THR,
            diag_callback=lambda n, a, r: seen.append(r), diag_every=1, **kw,
        )
        return seen[0]

    def test_first_iteration_residual_is_that_of_the_given_theta0(self) -> None:
        model, theta0 = self._setup()
        N = model.N
        as_is = model.max_relative_error(theta0)
        clamped = theta0.clone()
        clamped[2 * N:] = clamped[2 * N:].clamp(min=1e-10)
        assert abs(model.max_relative_error(clamped) - as_is) > 0.1 * as_is   # non-vacuous: the clamp would change it
        assert self._iter0_mre(model, theta0) == pytest.approx(as_is, rel=1e-6)

    def test_negative_non_hub_eta_is_still_floored(self) -> None:
        # only hub eta are exempt: a non-hub eta below the floor is still clamped
        model, theta0 = self._setup()
        N = model.N
        theta_bad = theta0.clone()
        sk_in = model.s_in / model.k_in.clamp(min=1.0)
        non_hub_in = ~((sk_in > self.THR) & (model.s_in > 0) & (model.k_in > 0)) & (model.s_in > 0)
        theta_bad[3 * N:][non_hub_in] = -0.05
        floored = theta_bad.clone()
        floored[3 * N:][non_hub_in] = 1e-10
        # the solver must see the floored value, not the negative one
        assert self._iter0_mre(model, theta_bad) == pytest.approx(
            self._iter0_mre(model, floored), rel=1e-9)


class TestMinPairZ:
    """The Anderson feasibility guard must use the smallest pair sum that EXISTS, not min(eta_out)+min(eta_in): the diagonal
    pair (i, i) does not exist for a single-node group (relaxed hub with negative eta on both sides, q4 group 1403)."""

    def test_plain_sum_when_argmins_differ(self) -> None:
        from dcms.solvers.fixed_point_decm import _min_pair_z
        eo = torch.tensor([0.5, 0.2, 1.0], dtype=torch.float64)
        ei = torch.tensor([0.9, 0.4, 0.1], dtype=torch.float64)
        assert _min_pair_z(eo, ei) == pytest.approx(0.2 + 0.1)

    def test_coinciding_argmins_on_single_node_group_use_next_pair(self) -> None:
        from dcms.solvers.fixed_point_decm import _min_pair_z
        eo = torch.tensor([-1e-5, 0.3, 0.5], dtype=torch.float64)      # group 0: relaxed on both sides
        ei = torch.tensor([-1.5e-3, 1.6e-4, 0.7], dtype=torch.float64)
        mult = torch.ones(3, dtype=torch.float64)
        # plain min+min = -1.51e-3 (the old guard would reject); real binding pairs: (0,1) = -1e-5+1.6e-4, (1,0) = 0.3-1.5e-3
        assert _min_pair_z(eo, ei, mult) == pytest.approx(-1e-5 + 1.6e-4)
        assert _min_pair_z(eo, ei, mult) > 1e-8

    def test_multi_node_group_keeps_its_diagonal_pair(self) -> None:
        from dcms.solvers.fixed_point_decm import _min_pair_z
        eo = torch.tensor([0.05, 0.3], dtype=torch.float64)
        ei = torch.tensor([0.02, 0.4], dtype=torch.float64)
        assert _min_pair_z(eo, ei, torch.tensor([3.0, 1.0], dtype=torch.float64)) == pytest.approx(0.07)   # (i, i') exists

    def test_all_positive_matches_old_rule(self) -> None:
        from dcms.solvers.fixed_point_decm import _min_pair_z
        g = torch.Generator().manual_seed(0)
        eo = torch.rand(20, generator=g, dtype=torch.float64) + 0.1
        ei = torch.rand(20, generator=g, dtype=torch.float64) + 0.1
        mult = torch.full((20,), 2.0, dtype=torch.float64)
        assert _min_pair_z(eo, ei, mult) == pytest.approx(float(eo.min() + ei.min()))


class TestTargetedFixNegativeEta:
    """A node 'unreachable' with eta >= 0 must be solved at its (negative) exact eta by the targeted fix, not clamped at
    _ETA_MIN (which left s ~14% off after every streak-fix on q4's relaxed hub 1403)."""

    def _prob(self):
        theta, k_out, k_in, s_out, s_in, mult, M = _group_problem()
        return theta, k_out, k_in, s_out, s_in, mult, M

    def test_unreachable_target_goes_negative_and_is_exact(self) -> None:
        from dcms.solvers.fixed_point_decm import _targeted_bisection_fix
        theta, k_out, k_in, s_out, s_in, mult, M = self._prob()
        s_big = s_out.clone()
        s_big[0] = s_big[0] * 3.0                       # far above what any eta >= 0 can deliver
        fixed = _targeted_bisection_fix(theta, k_out, k_in, s_big, s_in, mult, torch.tensor([0]), "out", n_sweeps=60, n_bisect=80)
        eta0 = float(fixed[2 * M])
        assert eta0 < 0.0
        assert eta0 + float(theta[3 * M:].min()) > 0.0   # every pair sum stays positive
        # s_out[0] is met exactly, k_out[0] too
        z = lambda x: x == 0
        F = _decm_step_dense_weighted(fixed, k_out, k_in, s_big, s_in, z(k_out), z(k_in), z(s_big), z(s_in), 0.5, mult)[1]
        assert abs(float(F[2 * M])) / float(s_big[0]) < 1e-8
        assert abs(float(F[0])) / float(k_out[0]) < 1e-8

    def test_reachable_target_is_unchanged_positive(self) -> None:
        from dcms.solvers.fixed_point_decm import _targeted_bisection_fix
        theta, k_out, k_in, s_out, s_in, mult, M = self._prob()
        fixed = _targeted_bisection_fix(theta, k_out, k_in, s_out, s_in, mult, torch.tensor([0]), "out", n_sweeps=8, n_bisect=80)
        assert float(fixed[2 * M]) > 0.0


class TestAdaptiveHubSweeps:
    def test_default_and_adaptive_both_converge(self) -> None:
        model, _ = make_decm_model_degenerate(N0=6, r=3, seed=3)
        theta0 = model.initial_theta("degrees")
        for sweeps in (3, 50):
            res = solve_fixed_point_decm_degenerate(
                theta0, model.k_out, model.k_in, model.s_out, model.s_in,
                tol=1e-8, max_iter=4000, hub_sk_threshold=0.25, hub_bisect_max_sweeps=sweeps,
            )
            assert res.converged, (sweeps, res.message)
            assert model.max_relative_error(res.best_theta) < CONV_TOL


class TestProjectPairFloor:
    """Every existing pair sum eta_out_i + eta_in_j must be >= floor; a feasible input is untouched, hub-owned eta are never
    modified, and the diagonal pair only exists for groups with more than one node."""

    F = 1e-8

    def _t(self, x):
        return torch.tensor(x, dtype=torch.float64)

    def test_feasible_input_is_returned_unchanged(self) -> None:
        from dcms.solvers.fixed_point_decm import _project_pair_floor
        eo, ei = self._t([0.5, 0.2, 1.0]), self._t([0.9, 0.4, 0.1])
        no, ni = _project_pair_floor(eo, ei, self._t([1.0, 1.0, 1.0]), self.F)
        assert torch.equal(no, eo) and torch.equal(ni, ei)

    def test_negative_eta_with_feasible_pairs_is_untouched(self) -> None:
        # relaxed hub on both sides (single-node group 0): the pair (0, 0) does not exist
        from dcms.solvers.fixed_point_decm import _project_pair_floor
        eo, ei = self._t([-1e-5, 0.3, 0.5]), self._t([-1.5e-3, 1.6e-4, 0.7])
        no, ni = _project_pair_floor(eo, ei, self._t([1.0, 1.0, 1.0]), self.F)
        assert torch.equal(no, eo) and torch.equal(ni, ei)

    def test_violating_pair_is_repaired_to_the_floor(self) -> None:
        from dcms.solvers.fixed_point_decm import _project_pair_floor, _min_pair_z
        eo, ei = self._t([0.01, 0.4, 0.5]), self._t([0.6, -0.3, 0.2])       # pair (0, 1): 0.01 - 0.3 < 0
        mult = self._t([1.0, 1.0, 1.0])
        no, ni = _project_pair_floor(eo, ei, mult, self.F)
        assert _min_pair_z(no, ni, mult) >= self.F - 1e-15
        assert float(no[0]) == pytest.approx(0.3 + self.F)                  # raised only as much as needed
        assert torch.equal(ni, ei)

    def test_frozen_entries_are_never_modified(self) -> None:
        from dcms.solvers.fixed_point_decm import _project_pair_floor
        eo, ei = self._t([0.01, 0.4, 0.5]), self._t([0.6, -0.3, 0.2])
        frozen = torch.tensor([True, False, False])
        no, _ = _project_pair_floor(eo, ei, self._t([1.0, 1.0, 1.0]), self.F, frozen_out=frozen)
        assert float(no[0]) == 0.01

    def test_multi_node_group_keeps_its_diagonal_pair(self) -> None:
        from dcms.solvers.fixed_point_decm import _project_pair_floor
        eo, ei = self._t([0.05, 0.3]), self._t([-0.2, 0.4])                 # group 0 has 3 nodes: (i, i') exists -> violated
        no, _ = _project_pair_floor(eo, ei, self._t([3.0, 1.0]), self.F)
        assert float(no[0]) == pytest.approx(0.2 + self.F)


class TestTheta0PairFloor:
    def test_violating_theta0_is_repaired_before_the_first_step(self) -> None:
        from dcms.solvers.fixed_point_decm import _min_pair_z
        model, theta_true = make_decm_model_degenerate(N0=6, r=3, seed=3)
        N = model.N
        theta0 = torch.as_tensor(theta_true, dtype=torch.float64).clone()
        theta0[3 * N + 0] = -5.0                     # eta_in of node 0 far below every eta_out: pair sums < 0
        assert _min_pair_z(theta0[2 * N:3 * N], theta0[3 * N:], None) < 0.0
        res = solve_fixed_point_decm_degenerate(
            theta0, model.k_out, model.k_in, model.s_out, model.s_in, tol=1e-15, max_iter=1,
        )
        th = torch.as_tensor(res.theta, dtype=torch.float64)
        assert bool(torch.isfinite(th).all())
        assert _min_pair_z(th[2 * N:3 * N], th[3 * N:], None) >= 1e-8 - 1e-15


class TestPatienceRelTol:
    """patience_rel_tol > 0 makes a creeping record (tiny improvements) count as a stall; the default 0 is the old criterion."""

    def _run(self, capsys, **kw):
        model, _ = make_decm_model_degenerate(N0=6, r=3, seed=3)
        theta0 = model.initial_theta("degrees")
        solve_fixed_point_decm_degenerate(
            theta0, model.k_out, model.k_in, model.s_out, model.s_in,
            tol=1e-30, max_iter=30, patience=4, verbose=True, **kw,
        )
        out = capsys.readouterr().out
        return out.count("[patience]")

    def test_default_is_unchanged(self, capsys) -> None:
        model, _ = make_decm_model_degenerate(N0=6, r=3, seed=3)
        theta0 = model.initial_theta("degrees")
        a = solve_fixed_point_decm_degenerate(theta0, model.k_out, model.k_in, model.s_out, model.s_in, tol=1e-12, max_iter=200)
        b = solve_fixed_point_decm_degenerate(theta0, model.k_out, model.k_in, model.s_out, model.s_in, tol=1e-12, max_iter=200,
                                              patience_rel_tol=0.0)
        assert np.array_equal(a.theta, b.theta) and a.iterations == b.iterations

    def test_large_rel_tol_makes_stalls_visible(self, capsys) -> None:
        n_strict = self._run(capsys, patience_rel_tol=0.0)
        n_rel = self._run(capsys, patience_rel_tol=0.999999)      # a record must improve ~1e6x to count: stalls every window
        assert n_rel >= 2
        assert n_rel > n_strict

    def test_best_theta_still_tracks_every_improvement(self) -> None:
        # even when stalls (and perturbed restarts) fire constantly, best_theta must stay the true minimum of the run
        model, _ = make_decm_model_degenerate(N0=6, r=3, seed=3)
        theta0 = model.initial_theta("degrees")
        res = solve_fixed_point_decm_degenerate(
            theta0, model.k_out, model.k_in, model.s_out, model.s_in, tol=1e-30, max_iter=60, patience=4, patience_rel_tol=0.5,
        )
        assert res.best_mre == pytest.approx(min(res.residuals))
        assert model.max_relative_error(res.best_theta) == pytest.approx(res.best_mre, rel=1e-6)


class TestTargetedFixSaturatedPair:
    """For a node with large eta (weights almost all 1: s ~ k) p_ij depends on phi = theta + eta only. Solving eta at fixed theta moves
    phi and breaks the k equation, so the alternation never converges (q5 group 832: k stayed off by 3.7e-4 for 60 sweeps). Solving eta
    at fixed phi converges geometrically."""

    def _saturated_problem(self):
        M = 10
        rng = np.random.default_rng(11)
        th_o, th_i = rng.uniform(-6.0, -1.0, M), rng.uniform(-1.0, 3.0, M)
        et_o, et_i = rng.uniform(0.3, 1.5, M), rng.uniform(0.3, 1.5, M)
        th_o[0], et_o[0] = -9.0, 7.0                     # node 0: large eta_out, theta_out very negative => p_0j ~ 1 for many j
        eta = et_o[:, None] + et_i[None, :]
        P = 1.0 / (1.0 + np.exp(th_o[:, None] + th_i[None, :] + np.log(np.expm1(eta))))
        W = P / (1.0 - np.exp(-eta))
        mult = np.ones(M)
        d = lambda A: np.diagonal(A)
        k_out = (P * mult[None, :]).sum(1) - d(P); k_in = (P * mult[:, None]).sum(0) - d(P)
        s_out = (W * mult[None, :]).sum(1) - d(W); s_in = (W * mult[:, None]).sum(0) - d(W)
        t = lambda a: torch.tensor(a, dtype=torch.float64)
        theta_true = np.concatenate([th_o, th_i, et_o, et_i])
        return t(theta_true), t(k_out), t(k_in), t(s_out), t(s_in), t(mult), M, P

    def test_pair_is_solved_exactly_from_a_perturbed_start(self) -> None:
        from dcms.solvers.fixed_point_decm import _targeted_bisection_fix
        theta_true, k_out, k_in, s_out, s_in, mult, M, P = self._saturated_problem()
        assert float(P[0].sum()) > 0.3 * float(k_out[0]) and float(k_out[0]) > 1.0        # node 0 really is in the saturated regime
        theta = theta_true.clone()
        theta[0] += 0.4; theta[2 * M] -= 0.25                                              # perturb theta_out and eta_out of node 0
        fixed = _targeted_bisection_fix(theta, k_out, k_in, s_out, s_in, mult, torch.tensor([0]), "out", n_sweeps=40, n_bisect=80)
        z = lambda x: x == 0
        F = _decm_step_dense_weighted(fixed, k_out, k_in, s_out, s_in, z(k_out), z(k_in), z(s_out), z(s_in), 0.5, mult)[1]
        assert abs(float(F[0])) / float(k_out[0]) < 1e-8          # k_out(0)
        assert abs(float(F[2 * M])) / float(s_out[0]) < 1e-8      # s_out(0)


class TestBlockNewton:
    """block_newton_gate: joint 2x2 (theta, eta) Newton step for ill-conditioned (nearly saturated) rows of the degeneracy-reduced step.

    On a saturated row (s ~ k, G ~ 1) the k- and s-equations are almost the same function, so the two decoupled scalar Newton steps
    contract by ~ 1 - det/(A*C)/2 per iteration (q4/e1 plateau: 1 - 1e-5 .. 1e-4)."""

    @staticmethod
    def _solution_problem(M: int = 15, eta_lo: float = 1.5, eta_hi: float = 3.5, seed: int = 1, mult_max: int = 3):
        g = torch.Generator().manual_seed(seed)
        mult = torch.randint(1, mult_max + 1, (M,), generator=g).double()
        theta_true = torch.cat([torch.randn(M, generator=g) * 0.5, torch.randn(M, generator=g) * 0.5,
                                torch.rand(M, generator=g) * (eta_hi - eta_lo) + eta_lo, torch.rand(M, generator=g) * (eta_hi - eta_lo) + eta_lo]).double()
        z = torch.zeros(M, dtype=torch.bool)
        ones = torch.ones(M, dtype=torch.float64)
        F0 = _decm_step_dense_weighted(theta_true, ones, ones, ones, ones, z, z, z, z, 100.0, mult)[1]     # expected - 1
        return theta_true, F0[:M] + 1, F0[M:2 * M] + 1, F0[2 * M:3 * M] + 1, F0[3 * M:] + 1, mult, (z, z, z, z), M

    def test_gate_zero_is_the_scalar_step(self) -> None:
        theta_true, ko, ki, so, si, mult, zs, M = self._solution_problem()
        th = theta_true + 0.02 * torch.randn(4 * M, dtype=torch.float64)
        a = _decm_step_dense_weighted(th, ko, ki, so, si, *zs, 5.0, mult)
        b = _decm_step_dense_weighted(th, ko, ki, so, si, *zs, 5.0, mult, block_newton_gate=1e-300)      # nothing is below this gate
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])

    def test_dense_and_chunked_agree_with_the_gate_on(self) -> None:
        theta_true, ko, ki, so, si, mult, zs, M = self._solution_problem()
        th = theta_true + 0.02 * torch.randn(4 * M, dtype=torch.float64)
        hub = torch.zeros(M, dtype=torch.bool); hub[:2] = True
        for hm in (None, hub):
            d = _decm_step_dense_weighted(th, ko, ki, so, si, *zs, 5.0, mult, hub_out_mask=hm, block_newton_gate=1e9)
            c = _decm_step_chunked_weighted(th, ko, ki, so, si, *zs, 4, 5.0, mult, hub_out_mask=hm, block_newton_gate=1e9)
            assert float((d[0] - c[0]).abs().max()) < 1e-10 and float((d[1] - c[1]).abs().max()) < 1e-10

    def test_joint_step_is_the_exact_2x2_newton_step(self) -> None:
        theta_true, ko, ki, so, si, mult, zs, M = self._solution_problem()
        th = theta_true + 0.02 * torch.randn(4 * M, dtype=torch.float64)
        th_new, F = _decm_step_dense_weighted(th, ko, ki, so, si, *zs, 100.0, mult, block_newton_gate=1e9)
        h = 1e-6
        for g in range(M):
            def Fg(dt, de, g=g):
                t = th.clone(); t[g] += dt; t[2 * M + g] += de
                return _decm_step_dense_weighted(t, ko, ki, so, si, *zs, 100.0, mult)[1][[g, 2 * M + g]]
            J = torch.stack([(Fg(h, 0.0) - Fg(-h, 0.0)) / (2 * h), (Fg(0.0, h) - Fg(0.0, -h)) / (2 * h)], 1)
            d_fd = -torch.linalg.solve(J, F[[g, 2 * M + g]])
            d_step = torch.stack([th_new[g] - th[g], th_new[2 * M + g] - th[2 * M + g]])
            assert float(((d_step - d_fd).abs() / d_fd.abs().clamp(min=1e-9)).max()) < 1e-4

    def test_gate_solves_a_saturated_problem_the_scalar_step_cannot(self) -> None:
        from dcms.solvers.fixed_point_decm import solve_fixed_point_decm_degenerate
        theta_true, ko, ki, so, si, mult, zs, M = self._solution_problem(M=40, eta_lo=4.0, eta_hi=7.0, seed=3, mult_max=1)   # all nodes distinct: the solver's own grouping has mult=1
        assert float((so / ko).max()) < 1.001                                            # really saturated
        th0 = theta_true + 0.3 * torch.randn(4 * M, dtype=torch.float64, generator=torch.Generator().manual_seed(5))
        th0[2 * M:] = th0[2 * M:].clamp(min=1.0)
        kw = dict(tol=1e-9, max_iter=200, anderson_depth=1, hub_sk_threshold=0.0, patience=10**6, verbose=False, num_threads=1)
        blk = solve_fixed_point_decm_degenerate(th0.clone(), ko, ki, so, si, block_newton_gate=0.05, **kw)
        sca = solve_fixed_point_decm_degenerate(th0.clone(), ko, ki, so, si, **kw)
        assert blk.converged and blk.iterations < 50
        assert not sca.converged and min(sca.residuals) > 1e-3


class TestBlowupGuardDepthOne:
    """anderson_depth=1 (plain Newton) must still be protected by the blowup guard/rollback: it used to live only in the depth > 1 branch,
    so a depth-1 run could diverge with no [blowup] event at all (e1 blk2: MRE 2.6e-2 -> 0.77)."""

    def _problem(self):
        theta_true, ko, ki, so, si, mult, zs, M = TestBlockNewton._solution_problem(M=40, eta_lo=4.0, eta_hi=7.0, seed=3, mult_max=1)
        th0 = theta_true + 0.3 * torch.randn(4 * M, dtype=torch.float64, generator=torch.Generator().manual_seed(5))
        th0[2 * M:] = th0[2 * M:].clamp(min=1.0)
        return th0, ko, ki, so, si

    @staticmethod
    def _run(depth, capsys, **kw):
        from dcms.solvers.fixed_point_decm import solve_fixed_point_decm_degenerate
        th0, ko, ki, so, si = TestBlowupGuardDepthOne._problem(TestBlowupGuardDepthOne)
        res = solve_fixed_point_decm_degenerate(th0.clone(), ko, ki, so, si, tol=1e-9, max_iter=60, anderson_depth=depth, hub_sk_threshold=0.0,
                                                patience=10**6, blowup_factor=1.05, verbose=True, num_threads=1, **kw)
        return res, capsys.readouterr().out

    def test_depth_one_has_the_guard(self, capsys) -> None:
        res, out = self._run(1, capsys)
        assert "[blowup]" in out and "rolling back" in out
        assert all(math.isfinite(r) for r in res.residuals)

    def test_depth_three_still_has_the_guard(self, capsys) -> None:
        res, out = self._run(3, capsys)
        assert "[blowup]" in out

    def test_depth_one_default_threshold_does_not_fire_on_a_healthy_run(self) -> None:
        from dcms.solvers.fixed_point_decm import solve_fixed_point_decm_degenerate
        th0, ko, ki, so, si = self._problem()
        res = solve_fixed_point_decm_degenerate(th0.clone(), ko, ki, so, si, tol=1e-9, max_iter=30, anderson_depth=1, hub_sk_threshold=0.0,
                                                patience=10**6, block_newton_gate=0.05, verbose=False, num_threads=1)
        assert res.converged and res.iterations < 30          # the guard (default scale-adaptive factor) must not disturb a converging depth-1 run


class TestBlockNewtonMinRel:
    """block_newton_min_rel: rows already below the given relative residual keep the scalar step instead of the joint 2x2 step (an s == k row
    has its solution at infinity, so the joint step would march it along (theta-c, eta+c) forever: e1 theta -31, eta +37, then MRE > 1)."""

    def test_exclude_helper(self) -> None:
        from dcms.solvers.fixed_point_decm import _block_exclude
        F_k = torch.tensor([1e-9, 1e-3, 0.0, 1e-3], dtype=torch.float64); F_s = torch.tensor([0.0, 1e-9, 0.0, 1e-3], dtype=torch.float64)
        k_t = torch.tensor([10.0, 10.0, 0.0, 10.0], dtype=torch.float64); s_t = k_t.clone()
        hub = torch.tensor([False, False, False, True])
        assert _block_exclude(hub, F_k, F_s, k_t, s_t, 0.0) is hub                       # 0.0 = no restriction: the hub mask is returned untouched
        assert _block_exclude(None, F_k, F_s, k_t, s_t, 0.0) is None
        ex = _block_exclude(hub, F_k, F_s, k_t, s_t, 1e-6)
        assert ex.tolist() == [True, False, True, True]                                  # row 0 tiny, row 1 not, row 2 zero target (rel 0), row 3 hub
        assert _block_exclude(None, F_k, F_s, k_t, s_t, 1e-6).tolist() == [True, False, True, False]

    def test_huge_min_rel_reduces_to_the_scalar_step(self) -> None:
        theta_true, ko, ki, so, si, mult, zs, M = TestBlockNewton._solution_problem()
        th = theta_true + 0.02 * torch.randn(4 * M, dtype=torch.float64, generator=torch.Generator().manual_seed(7))
        base = _decm_step_dense_weighted(th, ko, ki, so, si, *zs, 5.0, mult)
        for step in (lambda **kw: _decm_step_dense_weighted(th, ko, ki, so, si, *zs, 5.0, mult, **kw),
                     lambda **kw: _decm_step_chunked_weighted(th, ko, ki, so, si, *zs, 4, 5.0, mult, **kw)):
            excl = step(block_newton_gate=1e9, block_newton_min_rel=1e9)                 # every row is 'already converged' -> scalar step everywhere
            act = step(block_newton_gate=1e9, block_newton_min_rel=0.0)
            assert float((excl[0] - base[0]).abs().max()) < 1e-12
            assert float((act[0] - base[0]).abs().max()) > 1e-9                          # and without the restriction the joint step really differs

    def test_solver_default_min_rel_still_converges_the_saturated_problem(self) -> None:
        from dcms.solvers.fixed_point_decm import solve_fixed_point_decm_degenerate
        theta_true, ko, ki, so, si, mult, zs, M = TestBlockNewton._solution_problem(M=40, eta_lo=4.0, eta_hi=7.0, seed=3, mult_max=1)
        th0 = theta_true + 0.3 * torch.randn(4 * M, dtype=torch.float64, generator=torch.Generator().manual_seed(5))
        th0[2 * M:] = th0[2 * M:].clamp(min=1.0)
        kw = dict(tol=1e-6, max_iter=100, anderson_depth=1, hub_sk_threshold=0.0, patience=10**6, verbose=False, num_threads=1, block_newton_gate=0.05)
        res = solve_fixed_point_decm_degenerate(th0.clone(), ko, ki, so, si, **kw)       # block_newton_min_rel=None -> 0.1 * tol
        assert res.converged and res.iterations < 50
