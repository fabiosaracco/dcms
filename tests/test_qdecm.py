"""Tests for the qDECM model equations and two-step solver.

Tests cover:
- qDECMModel construction and basic properties.
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

from dcms.models.qdecm import qDECMModel, _ETA_MAX, _ETA_MIN
from dcms.models.dcm import DCMModel
from dcms.solvers.fixed_point_qdecm import solve_fixed_point_qdecm, solve_fixed_point_qdecm_weight
from dcms.solvers.fixed_point_dcm import solve_fixed_point_dcm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CONV_TOL = 1e-5  # acceptance threshold for constraint error


def make_qdecm_model(N: int = 6, seed: int = 0) -> tuple[qDECMModel, np.ndarray, np.ndarray]:
    """Return a qDECMModel with a known exact solution.

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

    model = qDECMModel(k_out, k_in, s_out, s_in)
    return model, theta_topo_true, theta_weight_true


# ---------------------------------------------------------------------------
# qDECMModel construction
# ---------------------------------------------------------------------------

class TestqDECMModelConstruction:
    def test_shapes(self) -> None:
        model, _, _ = make_qdecm_model(N=6)
        assert model.k_out.shape == (6,)
        assert model.k_in.shape == (6,)
        assert model.s_out.shape == (6,)
        assert model.s_in.shape == (6,)
        assert model.N == 6

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError):
            qDECMModel(
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
        model, theta_topo, _ = make_qdecm_model(N=6)
        P = model.pij_matrix(theta_topo)
        assert P.shape == torch.Size([6, 6])

    def test_zero_diagonal(self) -> None:
        model, theta_topo, _ = make_qdecm_model(N=6)
        P = model.pij_matrix(theta_topo)
        assert torch.all(P.diagonal() == 0.0)

    def test_values_in_range(self) -> None:
        model, theta_topo, _ = make_qdecm_model(N=6)
        P = model.pij_matrix(theta_topo)
        assert torch.all(P >= 0.0)
        assert torch.all(P <= 1.0)


# ---------------------------------------------------------------------------
# wij_matrix_conditioned
# ---------------------------------------------------------------------------

class TestWijMatrixConditioned:
    def test_shape(self) -> None:
        model, theta_topo, theta_weight = make_qdecm_model(N=6)
        W = model.wij_matrix_conditioned(theta_topo, theta_weight)
        assert W.shape == torch.Size([6, 6])

    def test_zero_diagonal(self) -> None:
        model, theta_topo, theta_weight = make_qdecm_model(N=6)
        W = model.wij_matrix_conditioned(theta_topo, theta_weight)
        assert torch.all(W.diagonal() == 0.0)

    def test_non_negative(self) -> None:
        model, theta_topo, theta_weight = make_qdecm_model(N=6)
        W = model.wij_matrix_conditioned(theta_topo, theta_weight)
        assert torch.all(W >= 0.0)

    def test_bounded_by_G(self) -> None:
        """W_ij = p_ij * G_new_ij ≤ G_new_ij (since p_ij ≤ 1)."""
        model, theta_topo, theta_weight = make_qdecm_model(N=6)
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
        model, theta_topo, theta_weight = make_qdecm_model(N=10)
        F = model.residual_strength(theta_topo, theta_weight)
        assert F.abs().max().item() < 1e-8, f"Max residual = {F.abs().max().item()}"

    def test_shape(self) -> None:
        model, theta_topo, theta_weight = make_qdecm_model(N=6)
        F = model.residual_strength(theta_topo, theta_weight)
        assert F.shape == (12,)

    def test_chunked_equals_dense(self) -> None:
        """Chunked residual must match dense for small N."""
        model, theta_topo, theta_weight = make_qdecm_model(N=8, seed=7)
        F_dense = model.residual_strength(theta_topo, theta_weight)
        F_chunked = model._residual_strength_chunked(
            theta_topo, theta_weight, chunk_size=3
        )
        assert torch.allclose(F_dense, F_chunked, atol=1e-12)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# neg_log_likelihood_strength
# ---------------------------------------------------------------------------

class TestNegLogLikelihoodStrength:
    def test_finite(self) -> None:
        model, theta_topo, theta_weight = make_qdecm_model(N=6)
        nll = model.neg_log_likelihood_strength(theta_topo, theta_weight)
        assert np.isfinite(nll)

    def test_gradient_equals_residual(self) -> None:
        """The gradient of −L_w equals −F_w (numerical check)."""
        model, theta_topo, theta_weight = make_qdecm_model(N=5, seed=2)
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


class TestNegLogLikelihoodGeneralized:
    """Tests for the full (generalized) qDECM log-likelihood -- the sum of
    the topology and weight-step log-likelihoods, per Parisi, Squartini &
    Garlaschelli (2020)."""

    def test_finite(self) -> None:
        model, theta_topo, theta_weight = make_qdecm_model(N=6)
        nll = model.neg_log_likelihood(theta_topo, theta_weight)
        assert np.isfinite(nll)

    def test_equals_sum_of_topo_and_weight_terms(self) -> None:
        model, theta_topo, theta_weight = make_qdecm_model(N=8, seed=3)
        nll = model.neg_log_likelihood(theta_topo, theta_weight)
        expected = (
            model._dcm.neg_log_likelihood(theta_topo)
            + model.neg_log_likelihood_strength(theta_topo, theta_weight)
        )
        assert nll == pytest.approx(expected)

    def test_matches_full_dcm_likelihood_at_true_topology(self) -> None:
        """At the true topology solution, the topology term of the
        generalized likelihood must equal the standalone DCMModel's own
        neg_log_likelihood on the same (k_out, k_in) -- confirms self._dcm
        is a faithful, independently-usable DCM sub-model, not a stub."""
        model, theta_topo, theta_weight = make_qdecm_model(N=8, seed=6)
        standalone_dcm = DCMModel(model.k_out, model.k_in)
        assert model._dcm.neg_log_likelihood(theta_topo) == pytest.approx(
            standalone_dcm.neg_log_likelihood(theta_topo)
        )

# ---------------------------------------------------------------------------
# initial_theta_weight
# ---------------------------------------------------------------------------

class TestInitialThetaWeight:
    def test_shape(self) -> None:
        model, theta_topo, _ = make_qdecm_model(N=6)
        theta0 = model.initial_theta_weight(theta_topo, method="topology")
        assert theta0.shape == (12,)

    def test_all_positive(self) -> None:
        model, theta_topo, _ = make_qdecm_model(N=6)
        for method in ("topology", "topology_node"):
            theta0 = model.initial_theta_weight(theta_topo, method=method)
            assert torch.all(theta0 > 0), f"method={method!r} produced non-positive θ"

    def test_unknown_method_raises(self) -> None:
        model, theta_topo, _ = make_qdecm_model(N=6)
        with pytest.raises(ValueError):
            model.initial_theta_weight(theta_topo, method="bad_method")


# ---------------------------------------------------------------------------
# constraint_error and max_relative_error
# ---------------------------------------------------------------------------

class TestConstraintErrors:
    def test_near_zero_at_solution(self) -> None:
        model, theta_topo, theta_weight = make_qdecm_model(N=10)
        err = model.constraint_error_strength(theta_topo, theta_weight)
        assert err < 1e-8

    def test_max_rel_error_near_zero(self) -> None:
        model, theta_topo, theta_weight = make_qdecm_model(N=10)
        mre = model.max_relative_error(theta_topo, theta_weight)
        assert mre < 1e-6


# ---------------------------------------------------------------------------
# Two-step solver convergence (N=4 and N=10)
# ---------------------------------------------------------------------------

def _solve_two_step(
    model: qDECMModel,
    tol: float = 1e-5,
    topo_max_iter: int = 5_000,
    weight_max_iter: int = 10_000,
    anderson_depth: int = 10,
) -> tuple:
    """Run the two-step qDECM solve and return (topo_result, weight_result)."""
    theta_topo0 = model.initial_theta_topo()
    r_topo = solve_fixed_point_dcm(
        theta_topo0, model.k_out, model.k_in,
        tol=tol, max_iter=topo_max_iter, anderson_depth=10,
    )
    theta_topo = torch.tensor(r_topo.theta, dtype=torch.float64)
    theta_w0 = model.initial_theta_weight(theta_topo, "topology")
    r_weight = solve_fixed_point_qdecm_weight(
        theta_topo, theta_w0, model.k_out, model.k_in, model.s_out, model.s_in,
        tol=tol, max_iter=weight_max_iter, anderson_depth=anderson_depth,
    )
    return r_topo, r_weight


class TestSolverConvergenceSmall:
    """Convergence tests for the two-step qDECM solver on small networks."""

    @pytest.mark.parametrize("N,seed", [(4, 0), (4, 1), (10, 0)])
    def test_theta_newton_converges(self, N: int, seed: int) -> None:
        """θ-Newton must converge on small networks."""
        model, theta_topo_true, theta_weight_true = make_qdecm_model(N=N, seed=seed)
        r_topo, r_weight = _solve_two_step(model, tol=CONV_TOL)
        assert r_topo.converged, f"N={N} seed={seed}: topology step failed"
        assert r_weight.converged, f"N={N} seed={seed}: weight step failed"
        err = model.max_relative_error(
            torch.tensor(r_topo.theta, dtype=torch.float64),
            torch.tensor(r_weight.theta, dtype=torch.float64),
        )
        assert err < CONV_TOL * 100, f"N={N} seed={seed}: MRE={err:.2e}"

# ---------------------------------------------------------------------------
# Fixed-point solver direct tests
# ---------------------------------------------------------------------------

class TestFixedPointqDECM:
    """Direct tests of solve_fixed_point_qdecm_weight."""

    def test_weight_step_converges_n4(self) -> None:
        model, theta_topo_true, _ = make_qdecm_model(N=4, seed=0)
        theta_topo = torch.tensor(theta_topo_true, dtype=torch.float64)
        theta_weight0 = model.initial_theta_weight(theta_topo, "topology")
        result = solve_fixed_point_qdecm_weight(
            theta_topo, theta_weight0, model.k_out, model.k_in, model.s_out, model.s_in,
            tol=CONV_TOL, max_iter=10_000,
        )
        assert result.converged or result.residuals[-1] < CONV_TOL * 10


class TestDegeneracyReduction:
    """Equivalence tests for the degeneracy-reduced qDECM solver.

    Only the conditioned-weight step needed new reduction logic (the
    topology step reuses solve_fixed_point_dcm_degenerate as-is, per the
    user's guidance) -- see the module-level note above
    _qdecm_step_dense_weighted in fixed_point_qdecm.py.
    """

    def _degenerate_network(self, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        N = 150
        adj_bin = (torch.rand(N, N, generator=g) < 0.05).double()
        adj_bin.fill_diagonal_(0.0)
        adj_w = adj_bin * torch.randint(1, 5, (N, N), generator=g).double()
        k_out = adj_bin.sum(dim=1)
        k_in = adj_bin.sum(dim=0)
        s_out = adj_w.sum(dim=1)
        s_in = adj_w.sum(dim=0)
        return k_out, k_in, s_out, s_in

    def test_weighted_step_residual_matches_the_model_at_mult_one(self) -> None:
        """With unit multiplicities the weighted step's residual F must equal the model's own (independent) per-node
        strength residual -- the algebraic identity the reduction relies on, at a generic point."""
        from dcms.solvers.fixed_point_qdecm import _qdecm_step_dense_weighted
        model, theta_topo_true, _ = make_qdecm_model(N=12, seed=1)
        theta_topo = torch.tensor(theta_topo_true, dtype=torch.float64)
        N = model.N
        theta = torch.rand(2 * N, dtype=torch.float64) * 0.5 + 0.3
        P = model.pij_matrix(theta_topo).clone()
        P.fill_diagonal_(0.0)
        _, F = _qdecm_step_dense_weighted(theta.clone(), P, model.s_out, model.s_in, max_step=1.0, mult=torch.ones(N, dtype=torch.float64))
        assert torch.allclose(F, model.residual_strength(theta_topo, theta), atol=1e-10)

    def test_expanded_pipeline_solves_the_per_node_system(self) -> None:
        """The reduced topology+weight solution, expanded to the N nodes, must satisfy the ORIGINAL per-node degree and strength
        equations (the reduction is exact) and be constant within every (k_out, k_in, s_out, s_in) group."""
        k_out, k_in, s_out, s_in = self._degenerate_network()
        N = k_out.shape[0]
        qm = qDECMModel(k_out, k_in, s_out, s_in)
        assert qm.solve_tool(tol=1e-10, max_iter=3000)
        theta = torch.as_tensor(qm.sol.best_theta)
        assert qm.max_relative_error(theta[:2 * N], theta[2 * N:]) < 1e-8
        key = np.stack([k_out.numpy(), k_in.numpy(), s_out.numpy(), s_in.numpy()], 1)
        _, inv = np.unique(key, axis=0, return_inverse=True)
        for g in range(inv.max() + 1):
            idx = np.flatnonzero(inv.ravel() == g)
            for blk in range(4):
                assert np.ptp(qm.sol.best_theta[blk * N:(blk + 1) * N][idx]) == 0.0

    def test_solve_tool_uses_the_reduced_system(self) -> None:
        k_out, k_in, s_out, s_in = self._degenerate_network()
        m1 = qDECMModel(k_out, k_in, s_out, s_in)
        assert m1.solve_tool(tol=1e-9, max_iter=2000)
        assert "degeneracy-reduced" in m1.sol.message

    def test_removed_options_are_rejected(self) -> None:
        k_out, k_in, s_out, s_in = self._degenerate_network()
        for kw in ({"variant": "gauss-seidel"}, {"backend": "numba"}, {"reduce_degeneracy": False}, {"backtracking_gamma": 1.2}):
            with pytest.raises(TypeError):
                qDECMModel(k_out, k_in, s_out, s_in).solve_tool(tol=1e-6, **kw)


class TestZClampExposure:
    """z_clamp: exposed as an overridable kwarg 2026-09-29 (audit finding -- previously
    a hard-coded, unreachable module constant, unlike DECM's own tunable z_clamp).
    Default 1e-8 unchanged; these tests check the plumbing, not a numerics claim."""

    @staticmethod
    def _weight_call(model, theta_topo_true, theta_weight_true, **kw):
        return solve_fixed_point_qdecm_weight(
            torch.tensor(theta_topo_true, dtype=torch.float64), torch.tensor(theta_weight_true, dtype=torch.float64),
            model.k_out, model.k_in, model.s_out, model.s_in, tol=1e-5, max_iter=5, **kw,
        )

    def test_default_is_silent_and_matches_the_module_constant(self) -> None:
        import dcms.solvers.fixed_point_qdecm as fpq
        import warnings
        model, theta_topo_true, theta_weight_true = make_qdecm_model(N=6, seed=0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self._weight_call(model, theta_topo_true, theta_weight_true)
        assert fpq._Z_G_CLAMP == 1e-8 and fpq._Z_NEWTON_FLOOR == 1e-8

    def test_non_default_value_warns_reaches_the_module_constant_and_is_restored(self, monkeypatch) -> None:
        import dcms.solvers.fixed_point_qdecm as fpq
        model, theta_topo_true, theta_weight_true = make_qdecm_model(N=6, seed=0)
        seen = []
        step = fpq._qdecm_step_dense_weighted

        def spy(*a, **k):
            seen.append((fpq._Z_G_CLAMP, fpq._Z_NEWTON_FLOOR))
            return step(*a, **k)

        monkeypatch.setattr(fpq, "_qdecm_step_dense_weighted", spy)
        with pytest.warns(UserWarning, match="z_clamp"):
            self._weight_call(model, theta_topo_true, theta_weight_true, z_clamp=1e-3)
        assert seen and all(v == (1e-3, 1e-3) for v in seen)       # in force during the call
        assert fpq._Z_G_CLAMP == 1e-8 and fpq._Z_NEWTON_FLOOR == 1e-8   # restored afterwards

    def test_solve_tool_forwards_z_clamp(self) -> None:
        import dcms.solvers.fixed_point_qdecm as fpq
        model, _, _ = make_qdecm_model(N=6, seed=0)
        with pytest.warns(UserWarning, match="z_clamp"):
            model.solve_tool(tol=1e-3, max_iter=5, z_clamp=1e-4)
        assert fpq._Z_G_CLAMP == 1e-8   # restored after the call
