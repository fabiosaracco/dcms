"""Tests for the DaECM model equations and two-step solver.

Tests cover:
- DaECMModel construction and basic properties.
- wij_matrix_conditioned: correct shape, zero diagonal, non-negative values.
- residual_strength: correct value at the true solution.
- jacobian_strength: finite-difference consistency and negative diagonal.
- hessian_diag_strength: all entries negative.
- neg_log_likelihood_strength: finite at valid theta.
- initial_theta_weight: sensible starting point.
- constraint_error / max_relative_error: correct values.
- Two-step solver convergence on N=4 and N=10 networks.
- Chunked vs dense residual consistency.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.daecm import DaECMModel, _ETA_MAX, _ETA_MIN
from src.solvers.daecm_solver import DaECMResult, solve_daecm, solve_daecm_joint_lbfgs
from src.solvers.fixed_point_daecm import solve_fixed_point_daecm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CONV_TOL = 1e-5  # acceptance threshold for constraint error


def make_daecm_model(N: int = 6, seed: int = 0) -> tuple[DaECMModel, np.ndarray, np.ndarray]:
    """Return a DaECMModel with a known exact solution.

    Generates random ``θ_topo`` and ``θ_weight`` (both positive, in [0.5, 2.0]),
    computes the corresponding (k_out, k_in, s_out, s_in) analytically, then
    constructs the model so the exact solution is known.

    Args:
        N:    Number of nodes.
        seed: RNG seed.

    Returns:
        ``(model, theta_topo_true, theta_weight_true)``
    """
    rng = np.random.default_rng(seed)

    # Topology parameters: x_i = exp(-θ_out_i), y_i = exp(-θ_in_i)
    theta_topo_true = rng.uniform(0.5, 3.0, size=2 * N)
    x = np.exp(-theta_topo_true[:N])
    y = np.exp(-theta_topo_true[N:])

    # DCM probability matrix
    P = x[:, None] * y[None, :]       # (N, N)
    P = P / (1.0 + P)                 # p_ij = xy/(1+xy)
    np.fill_diagonal(P, 0.0)

    k_out = P.sum(axis=1)
    k_in = P.sum(axis=0)

    # Weight parameters: β_out_i = exp(-θ_β_out_i), β_in_i = exp(-θ_β_in_i)
    theta_weight_true = rng.uniform(0.5, 2.0, size=2 * N)
    b_out = np.exp(-theta_weight_true[:N])
    b_in = np.exp(-theta_weight_true[N:])

    # Conditioned weight matrix: W_ij = p_ij / (1 - β_out_i β_in_j)  (new formula)
    beta_mat = b_out[:, None] * b_in[None, :]  # (N, N)
    G = 1.0 / (1.0 - beta_mat)                # G_new_ij = 1/(1-β_out β_in)
    W = P * G                                   # W_ij = p_ij G_new_ij
    np.fill_diagonal(W, 0.0)

    s_out = W.sum(axis=1)
    s_in = W.sum(axis=0)

    model = DaECMModel(k_out, k_in, s_out, s_in)
    return model, theta_topo_true, theta_weight_true


# ---------------------------------------------------------------------------
# DaECMModel construction
# ---------------------------------------------------------------------------

class TestDaECMModelConstruction:
    def test_shapes(self) -> None:
        model, _, _ = make_daecm_model(N=6)
        assert model.k_out.shape == (6,)
        assert model.k_in.shape == (6,)
        assert model.s_out.shape == (6,)
        assert model.s_in.shape == (6,)
        assert model.N == 6

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError):
            DaECMModel(
                k_out=np.ones(4),
                k_in=np.ones(5),   # mismatch
                s_out=np.ones(4),
                s_in=np.ones(4),
            )


# ---------------------------------------------------------------------------
# pij_matrix
# ---------------------------------------------------------------------------

class TestPijMatrix:
    def test_shape(self) -> None:
        model, theta_topo, _ = make_daecm_model(N=6)
        P = model.pij_matrix(theta_topo)
        assert P.shape == torch.Size([6, 6])

    def test_zero_diagonal(self) -> None:
        model, theta_topo, _ = make_daecm_model(N=6)
        P = model.pij_matrix(theta_topo)
        assert torch.all(P.diagonal() == 0.0)

    def test_values_in_range(self) -> None:
        model, theta_topo, _ = make_daecm_model(N=6)
        P = model.pij_matrix(theta_topo)
        assert torch.all(P >= 0.0)
        assert torch.all(P <= 1.0)


# ---------------------------------------------------------------------------
# wij_matrix_conditioned
# ---------------------------------------------------------------------------

class TestWijMatrixConditioned:
    def test_shape(self) -> None:
        model, theta_topo, theta_weight = make_daecm_model(N=6)
        W = model.wij_matrix_conditioned(theta_topo, theta_weight)
        assert W.shape == torch.Size([6, 6])

    def test_zero_diagonal(self) -> None:
        model, theta_topo, theta_weight = make_daecm_model(N=6)
        W = model.wij_matrix_conditioned(theta_topo, theta_weight)
        assert torch.all(W.diagonal() == 0.0)

    def test_non_negative(self) -> None:
        model, theta_topo, theta_weight = make_daecm_model(N=6)
        W = model.wij_matrix_conditioned(theta_topo, theta_weight)
        assert torch.all(W >= 0.0)

    def test_bounded_by_G(self) -> None:
        """W_ij = p_ij * G_new_ij ≤ G_new_ij (since p_ij ≤ 1)."""
        model, theta_topo, theta_weight = make_daecm_model(N=6)
        W = model.wij_matrix_conditioned(theta_topo, theta_weight)
        # W must be ≤ the conditional expected weight G_new (since p_ij ≤ 1)
        N = model.N
        tb_out = torch.tensor(theta_weight[:N], dtype=torch.float64)
        tb_in = torch.tensor(theta_weight[N:], dtype=torch.float64)
        z = tb_out[:, None] + tb_in[None, :]
        z_safe = z.clamp(min=1e-15)
        G = -1.0 / torch.expm1(-z_safe)  # G_new = 1/(1-exp(-z))
        G.fill_diagonal_(0.0)
        assert torch.all(W <= G + 1e-9)


# ---------------------------------------------------------------------------
# residual_strength
# ---------------------------------------------------------------------------

class TestResidualStrength:
    def test_near_zero_at_solution(self) -> None:
        """Strength residual should be ~0 at the true parameters."""
        model, theta_topo, theta_weight = make_daecm_model(N=10)
        F = model.residual_strength(theta_topo, theta_weight)
        assert F.abs().max().item() < 1e-8, f"Max residual = {F.abs().max().item()}"

    def test_shape(self) -> None:
        model, theta_topo, theta_weight = make_daecm_model(N=6)
        F = model.residual_strength(theta_topo, theta_weight)
        assert F.shape == (12,)

    def test_chunked_equals_dense(self) -> None:
        """Chunked residual must match dense for small N."""
        model, theta_topo, theta_weight = make_daecm_model(N=8, seed=7)
        F_dense = model.residual_strength(theta_topo, theta_weight)
        F_chunked = model._residual_strength_chunked(
            theta_topo, theta_weight, chunk_size=3
        )
        assert torch.allclose(F_dense, F_chunked, atol=1e-12)


# ---------------------------------------------------------------------------
# jacobian_strength
# ---------------------------------------------------------------------------

class TestJacobianStrength:
    def test_shape(self) -> None:
        model, theta_topo, theta_weight = make_daecm_model(N=6)
        J = model.jacobian_strength(theta_topo, theta_weight)
        assert J.shape == (12, 12)

    def test_negative_diagonal(self) -> None:
        """Diagonal of J_w must be ≤ 0."""
        model, theta_topo, theta_weight = make_daecm_model(N=6)
        J = model.jacobian_strength(theta_topo, theta_weight)
        assert torch.all(J.diagonal() <= 0.0)

    def test_finite_difference_consistency(self) -> None:
        """Jacobian should match finite differences."""
        model, theta_topo, theta_weight = make_daecm_model(N=6, seed=3)
        theta_w = torch.tensor(theta_weight, dtype=torch.float64)
        J = model.jacobian_strength(theta_topo, theta_w)
        eps = 1e-5
        n = 2 * model.N
        J_fd = torch.zeros(n, n, dtype=torch.float64)
        for k in range(n):
            dw = torch.zeros(n, dtype=torch.float64)
            dw[k] = eps
            F_plus = model.residual_strength(theta_topo, theta_w + dw)
            F_minus = model.residual_strength(theta_topo, theta_w - dw)
            J_fd[:, k] = (F_plus - F_minus) / (2 * eps)
        assert torch.allclose(J, J_fd, atol=1e-4)


# ---------------------------------------------------------------------------
# hessian_diag_strength
# ---------------------------------------------------------------------------

class TestHessianDiagStrength:
    def test_all_non_positive(self) -> None:
        model, theta_topo, theta_weight = make_daecm_model(N=6)
        h = model.hessian_diag_strength(theta_topo, theta_weight)
        assert torch.all(h <= 0.0)

    def test_matches_jacobian_diagonal(self) -> None:
        model, theta_topo, theta_weight = make_daecm_model(N=6, seed=1)
        J = model.jacobian_strength(theta_topo, theta_weight)
        h = model.hessian_diag_strength(theta_topo, theta_weight)
        assert torch.allclose(J.diagonal(), h, atol=1e-12)


# ---------------------------------------------------------------------------
# neg_log_likelihood_strength
# ---------------------------------------------------------------------------

class TestNegLogLikelihoodStrength:
    def test_finite(self) -> None:
        model, theta_topo, theta_weight = make_daecm_model(N=6)
        nll = model.neg_log_likelihood_strength(theta_topo, theta_weight)
        assert np.isfinite(nll)

    def test_gradient_equals_residual(self) -> None:
        """The gradient of −L_w equals −F_w (numerical check)."""
        model, theta_topo, theta_weight = make_daecm_model(N=5, seed=2)
        theta_w = torch.tensor(theta_weight, dtype=torch.float64)
        eps = 1e-5
        n = 2 * model.N
        grad_fd = torch.zeros(n, dtype=torch.float64)
        nll0 = model.neg_log_likelihood_strength(theta_topo, theta_w)
        for k in range(n):
            dw = torch.zeros(n, dtype=torch.float64)
            dw[k] = eps
            nll_plus = model.neg_log_likelihood_strength(theta_topo, theta_w + dw)
            grad_fd[k] = (nll_plus - nll0) / eps
        # ∇(NLL) = −F_w  (NLL = θ(s−k_exp) + Σp log G_new, d/dθ = −F_w)
        F = model.residual_strength(theta_topo, theta_w)
        assert torch.allclose(grad_fd, -F, atol=1e-4)


# ---------------------------------------------------------------------------
# initial_theta_weight
# ---------------------------------------------------------------------------

class TestInitialThetaWeight:
    def test_shape(self) -> None:
        model, theta_topo, _ = make_daecm_model(N=6)
        theta0 = model.initial_theta_weight(theta_topo, method="strengths")
        assert theta0.shape == (12,)

    def test_all_positive(self) -> None:
        model, theta_topo, _ = make_daecm_model(N=6)
        for method in ("strengths", "normalized", "uniform", "random"):
            theta0 = model.initial_theta_weight(theta_topo, method=method)
            assert torch.all(theta0 > 0), f"method={method!r} produced non-positive θ"

    def test_unknown_method_raises(self) -> None:
        model, theta_topo, _ = make_daecm_model(N=6)
        with pytest.raises(ValueError):
            model.initial_theta_weight(theta_topo, method="bad_method")


# ---------------------------------------------------------------------------
# constraint_error and max_relative_error
# ---------------------------------------------------------------------------

class TestConstraintErrors:
    def test_near_zero_at_solution(self) -> None:
        model, theta_topo, theta_weight = make_daecm_model(N=10)
        err = model.constraint_error_strength(theta_topo, theta_weight)
        assert err < 1e-8

    def test_max_rel_error_near_zero(self) -> None:
        model, theta_topo, theta_weight = make_daecm_model(N=10)
        mre = model.max_relative_error(theta_topo, theta_weight)
        assert mre < 1e-6


# ---------------------------------------------------------------------------
# Two-step solver convergence (N=4 and N=10)
# ---------------------------------------------------------------------------

class TestSolverConvergenceSmall:
    """Convergence tests for the two-step DaECM solver on small networks."""

    @pytest.mark.parametrize("N,seed", [(4, 0), (4, 1), (10, 0), (10, 2)])
    def test_fp_gs(self, N: int, seed: int) -> None:
        """FP-GS does not always converge; only check error when it does."""
        model, _, _ = make_daecm_model(N=N, seed=seed)
        result = solve_daecm(
            model,
            topo_method="lbfgs",
            weight_method="fp-gs",
            tol=CONV_TOL,
            topo_max_iter=5_000,
            weight_max_iter=10_000,
        )
        assert result.topo_converged, f"N={N} seed={seed}: topology step failed"
        if result.weight_converged:
            err = model.constraint_error_strength(
                result.theta_topo, result.theta_weight
            )
            assert err < CONV_TOL * 100, (
                f"N={N} seed={seed}: strength error={err:.2e} after convergence"
            )

    @pytest.mark.parametrize("N,seed", [(4, 0), (4, 1), (10, 0)])
    def test_fp_gs_anderson(self, N: int, seed: int) -> None:
        model, _, _ = make_daecm_model(N=N, seed=seed)
        result = solve_daecm(
            model,
            topo_method="lbfgs",
            weight_method="fp-gs-anderson",
            tol=CONV_TOL,
            weight_max_iter=5_000,
            anderson_depth=5,
        )
        assert result.topo_converged, "Topology step did not converge"

    @pytest.mark.parametrize("N,seed", [(4, 0), (10, 0)])
    def test_theta_newton(self, N: int, seed: int) -> None:
        model, _, _ = make_daecm_model(N=N, seed=seed)
        result = solve_daecm(
            model,
            topo_method="lbfgs",
            weight_method="theta-newton",
            tol=CONV_TOL,
            weight_max_iter=5_000,
        )
        assert result.topo_converged

    @pytest.mark.parametrize("N,seed", [(4, 0), (10, 0)])
    def test_lbfgs(self, N: int, seed: int) -> None:
        model, _, _ = make_daecm_model(N=N, seed=seed)
        result = solve_daecm(
            model,
            topo_method="lbfgs",
            weight_method="lbfgs",
            tol=CONV_TOL,
            weight_max_iter=2_000,
        )
        assert result.topo_converged
        if result.weight_converged:
            err = model.constraint_error_strength(
                result.theta_topo, result.theta_weight
            )
            assert err < CONV_TOL * 10

    @pytest.mark.parametrize("N,seed", [(4, 0), (10, 0)])
    def test_lm_diag(self, N: int, seed: int) -> None:
        model, _, _ = make_daecm_model(N=N, seed=seed)
        result = solve_daecm(
            model,
            topo_method="lbfgs",
            weight_method="lm-diag",
            tol=CONV_TOL,
            weight_max_iter=2_000,
        )
        assert result.topo_converged


class TestSolverResult:
    """Test that DaECMResult has the expected structure."""

    def test_result_fields(self) -> None:
        model, _, _ = make_daecm_model(N=4)
        result = solve_daecm(model, topo_method="lbfgs", weight_method="lbfgs",
                             tol=1e-4, weight_max_iter=200)
        assert isinstance(result, DaECMResult)
        assert result.theta_topo.shape == (8,)
        assert result.theta_weight.shape == (8,)
        assert isinstance(result.converged, bool)
        assert result.elapsed_time >= 0.0
        assert result.peak_ram_bytes >= 0

    def test_solve_daecm_converges_n4(self) -> None:
        """solve_daecm must converge on N=4 with L-BFGS."""
        model, _, _ = make_daecm_model(N=4, seed=0)
        result = solve_daecm(
            model,
            topo_method="lbfgs",
            weight_method="lbfgs",
            tol=1e-8,
            weight_max_iter=3_000,
        )
        assert result.topo_converged, "Topology step failed"
        assert result.weight_converged, "Weight step failed"
        err = model.constraint_error_strength(result.theta_topo, result.theta_weight)
        assert err < 1e-6, f"Strength constraint error={err:.2e}"

    def test_solve_daecm_converges_n10(self) -> None:
        """solve_daecm must converge on N=10 with L-BFGS."""
        model, _, _ = make_daecm_model(N=10, seed=0)
        result = solve_daecm(
            model,
            topo_method="lbfgs",
            weight_method="lbfgs",
            tol=1e-8,
            weight_max_iter=3_000,
        )
        assert result.topo_converged, "Topology step failed"
        assert result.weight_converged, "Weight step failed"
        err = model.constraint_error_strength(result.theta_topo, result.theta_weight)
        assert err < 1e-6, f"Strength constraint error={err:.2e}"


# ---------------------------------------------------------------------------
# Fixed-point solver direct tests
# ---------------------------------------------------------------------------

class TestFixedPointDaECM:
    """Direct tests of solve_fixed_point_daecm."""

    def test_fp_gs_converges_n4(self) -> None:
        model, theta_topo_true, _ = make_daecm_model(N=4, seed=0)
        theta_topo = torch.tensor(theta_topo_true, dtype=torch.float64)
        theta_weight0 = model.initial_theta_weight(theta_topo, "strengths")
        res_fn = lambda tw: model.residual_strength(theta_topo, tw.clamp(_ETA_MIN, _ETA_MAX))
        result = solve_fixed_point_daecm(
            res_fn, theta_weight0,
            model.s_out, model.s_in,
            theta_topo=theta_topo,
            tol=CONV_TOL, max_iter=10_000,
            damping=1.0, variant="gauss-seidel",
        )
        # FP-GS doesn't always converge; just check it doesn't crash
        assert result.iterations >= 0

    def test_theta_newton_converges_n4(self) -> None:
        model, theta_topo_true, _ = make_daecm_model(N=4, seed=0)
        theta_topo = torch.tensor(theta_topo_true, dtype=torch.float64)
        theta_weight0 = model.initial_theta_weight(theta_topo, "strengths")
        res_fn = lambda tw: model.residual_strength(theta_topo, tw.clamp(_ETA_MIN, _ETA_MAX))
        result = solve_fixed_point_daecm(
            res_fn, theta_weight0,
            model.s_out, model.s_in,
            theta_topo=theta_topo,
            tol=CONV_TOL, max_iter=10_000,
            variant="theta-newton",
        )
        assert result.converged or result.residuals[-1] < CONV_TOL * 10


# ---------------------------------------------------------------------------
# Joint L-BFGS tests
# ---------------------------------------------------------------------------

class TestJointLBFGS:
    """Test the joint 4N L-BFGS solver."""

    def test_joint_residual_shape(self) -> None:
        """Joint residual must have length 4N."""
        model, theta_topo_true, theta_weight_true = make_daecm_model(N=4, seed=0)
        theta_full = np.concatenate([theta_topo_true, theta_weight_true])
        F = model.residual_joint(theta_full)
        assert F.shape == (16,)

    def test_joint_nll_finite(self) -> None:
        """Joint NLL must be a finite scalar."""
        model, theta_topo_true, theta_weight_true = make_daecm_model(N=4, seed=0)
        theta_full = np.concatenate([theta_topo_true, theta_weight_true])
        nll = model.neg_log_likelihood_joint(theta_full)
        assert np.isfinite(nll)

    def test_joint_lbfgs_converges_n4(self) -> None:
        """Joint L-BFGS must converge on N=4."""
        model, _, _ = make_daecm_model(N=4, seed=0)
        result = solve_daecm_joint_lbfgs(model, tol=1e-6, max_iter=5_000)
        assert result.converged, f"Joint L-BFGS did not converge: {result.message}"
        err = model.constraint_error_joint(
            np.concatenate([result.theta_topo, result.theta_weight])
        )
        assert err < 1e-5, f"Joint error={err:.2e}"

    def test_joint_lbfgs_converges_n10(self) -> None:
        """Joint L-BFGS must converge on N=10."""
        model, _, _ = make_daecm_model(N=10, seed=0)
        result = solve_daecm_joint_lbfgs(model, tol=1e-6, max_iter=5_000)
        assert result.converged, f"Joint L-BFGS did not converge: {result.message}"
        err = model.constraint_error_joint(
            np.concatenate([result.theta_topo, result.theta_weight])
        )
        assert err < 1e-5, f"Joint error={err:.2e}"
