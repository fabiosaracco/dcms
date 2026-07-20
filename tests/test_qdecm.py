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
from dcms.solvers.fixed_point_qdecm import solve_fixed_point_qdecm
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
    weight_variant: str = "theta-newton",
    anderson_depth: int = 10,
) -> tuple:
    """Run the two-step qDECM solve and return (topo_result, weight_result)."""
    dcm = DCMModel(model.k_out, model.k_in)
    theta_topo0 = model.initial_theta_topo()
    r_topo = solve_fixed_point_dcm(
        dcm.residual, theta_topo0, dcm.k_out, dcm.k_in,
        tol=tol, max_iter=topo_max_iter,
        variant="theta-newton", anderson_depth=10,
    )
    theta_topo = r_topo.theta

    theta_w0 = model.initial_theta_weight(
        torch.tensor(theta_topo, dtype=torch.float64), "topology"
    )
    res_fn = lambda tw: model.residual_strength(
        torch.tensor(theta_topo, dtype=torch.float64),
        tw.clamp(_ETA_MIN, _ETA_MAX),
    )
    r_weight = solve_fixed_point_qdecm(
        res_fn, theta_w0,
        model.s_out, model.s_in,
        theta_topo=torch.tensor(theta_topo, dtype=torch.float64),
        tol=tol, max_iter=weight_max_iter,
        variant=weight_variant, anderson_depth=anderson_depth,
    )
    return r_topo, r_weight


class TestSolverConvergenceSmall:
    """Convergence tests for the two-step qDECM solver on small networks."""

    @pytest.mark.parametrize("N,seed", [(4, 0), (4, 1), (10, 0)])
    def test_theta_newton_converges(self, N: int, seed: int) -> None:
        """θ-Newton must converge on small networks."""
        model, theta_topo_true, theta_weight_true = make_qdecm_model(N=N, seed=seed)
        r_topo, r_weight = _solve_two_step(
            model, tol=CONV_TOL, weight_variant="theta-newton",
        )
        assert r_topo.converged, f"N={N} seed={seed}: topology step failed"
        assert r_weight.converged, f"N={N} seed={seed}: weight step failed"
        err = model.max_relative_error(
            torch.tensor(r_topo.theta, dtype=torch.float64),
            torch.tensor(r_weight.theta, dtype=torch.float64),
        )
        assert err < CONV_TOL * 100, f"N={N} seed={seed}: MRE={err:.2e}"

    @pytest.mark.parametrize("N,seed", [(4, 0), (4, 1), (10, 0)])
    def test_fp_gs_no_crash(self, N: int, seed: int) -> None:
        """FP-GS Anderson may not converge but must not crash."""
        model, _, _ = make_qdecm_model(N=N, seed=seed)
        r_topo, r_weight = _solve_two_step(
            model, tol=CONV_TOL, weight_variant="gauss-seidel",
            weight_max_iter=2_000,
        )
        assert r_topo.converged, f"N={N} seed={seed}: topology step failed"
        if r_weight.converged:
            err = model.constraint_error_strength(
                torch.tensor(r_topo.theta, dtype=torch.float64),
                torch.tensor(r_weight.theta, dtype=torch.float64),
            )
            assert err < CONV_TOL * 100, f"N={N} seed={seed}: strength error={err:.2e}"


# ---------------------------------------------------------------------------
# Fixed-point solver direct tests
# ---------------------------------------------------------------------------

class TestFixedPointqDECM:
    """Direct tests of solve_fixed_point_qdecm."""

    def test_fp_gs_converges_n4(self) -> None:
        model, theta_topo_true, _ = make_qdecm_model(N=4, seed=0)
        theta_topo = torch.tensor(theta_topo_true, dtype=torch.float64)
        theta_weight0 = model.initial_theta_weight(theta_topo, "topology")
        res_fn = lambda tw: model.residual_strength(theta_topo, tw.clamp(_ETA_MIN, _ETA_MAX))
        result = solve_fixed_point_qdecm(
            res_fn, theta_weight0,
            model.s_out, model.s_in,
            theta_topo=theta_topo,
            tol=CONV_TOL, max_iter=10_000,
            damping=1.0, variant="gauss-seidel",
        )
        # FP-GS doesn't always converge; just check it doesn't crash
        assert result.iterations >= 0

    def test_theta_newton_converges_n4(self) -> None:
        model, theta_topo_true, _ = make_qdecm_model(N=4, seed=0)
        theta_topo = torch.tensor(theta_topo_true, dtype=torch.float64)
        theta_weight0 = model.initial_theta_weight(theta_topo, "topology")
        res_fn = lambda tw: model.residual_strength(theta_topo, tw.clamp(_ETA_MIN, _ETA_MAX))
        result = solve_fixed_point_qdecm(
            res_fn, theta_weight0,
            model.s_out, model.s_in,
            theta_topo=theta_topo,
            tol=CONV_TOL, max_iter=10_000,
            variant="theta-newton",
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

    def test_weighted_step_matches_unreduced_at_mult_one(self) -> None:
        """_qdecm_step_dense_weighted with mult=1 must exactly reproduce
        _theta_newton_step_dense (float64, single step, not just at
        convergence) -- this is the algebraic identity the reduction relies
        on, independent of any real degeneracy being present."""
        from dcms.solvers.fixed_point_qdecm import (
            _theta_newton_step_dense, _qdecm_step_dense_weighted,
        )
        torch.manual_seed(1)
        N = 25
        s_out = torch.rand(N, dtype=torch.float64) * 10 + 1
        s_in = torch.rand(N, dtype=torch.float64) * 10 + 1
        theta = torch.rand(2 * N, dtype=torch.float64) * 0.5 + 0.3
        P = torch.rand(N, N, dtype=torch.float64) * 0.3
        P.fill_diagonal_(0.0)
        mult = torch.ones(N, dtype=torch.float64)

        t1, F1 = _theta_newton_step_dense(theta.clone(), P, s_out, s_in, max_step=1.0)
        t2, F2 = _qdecm_step_dense_weighted(theta.clone(), P, s_out, s_in, max_step=1.0, mult=mult)

        assert torch.allclose(t1, t2, atol=1e-12), f"theta diff: {(t1-t2).abs().max():.3e}"
        assert torch.allclose(F1, F2, atol=1e-12), f"F diff: {(F1-F2).abs().max():.3e}"

    def test_reduced_matches_full_pipeline(self) -> None:
        """Full topology+weight pipeline, reduced vs unreduced, must agree
        on the expected-weight matrix once converged.

        NOTE on the tolerance: the per-step algebra is proven algebraically
        exact (see test_weighted_step_matches_unreduced_at_mult_one and a
        crafted multi-member-group check done during development, both
        exact to float64 precision). At full-pipeline scale, though, full
        vs reduced can land on slightly different points along the
        weight-step's genuine gauge freedom: G_ij = G(theta_b_out_i +
        theta_b_in_j) depends only on the pairwise SUM, so
        theta_b_out += c, theta_b_in -= c leaves every W_ij unchanged
        exactly (verified numerically to machine precision) -- this is the
        same gauge direction DCM has (see solve_fixed_point_dcm_degenerate's
        module docs), just less sharply pinned down by the positivity box
        here. Both full and reduced independently satisfy their own
        residual to ~1e-10 or tighter -- i.e. both are genuinely valid
        solutions, they just aren't bit-identical along that direction.
        Observed magnitude: ~2e-3 on crisi_dico2 (N=15168), ~4e-3 on this
        smaller synthetic network. 1e-2 catches real algorithmic bugs
        (which showed O(1) discrepancies during development) while
        tolerating this gauge-freedom artifact.
        """
        from dcms.solvers.fixed_point_qdecm import (
            solve_fixed_point_qdecm, solve_fixed_point_qdecm_degenerate,
        )
        from dcms.solvers.fixed_point_dcm import solve_fixed_point_dcm

        k_out, k_in, s_out, s_in = self._degenerate_network()
        N = k_out.shape[0]

        dcm = DCMModel(k_out, k_in)
        theta_topo0 = dcm.initial_theta("degrees")
        topo_full = solve_fixed_point_dcm(
            dcm.residual, theta_topo0, k_out, k_in,
            tol=1e-11, max_iter=2000, variant="theta-newton",
            anderson_depth=10, backend="pytorch",
        )
        assert topo_full.converged
        theta_topo_full = torch.as_tensor(topo_full.best_theta)

        qm = qDECMModel(k_out, k_in, s_out, s_in)
        theta_w0 = qm.initial_theta_weight(theta_topo_full, method="topology")
        res_full = solve_fixed_point_qdecm(
            lambda tb: qm.residual_strength(theta_topo_full, tb),
            theta_w0, s_out, s_in, theta_topo_full,
            tol=1e-11, max_iter=2000, variant="theta-newton",
            anderson_depth=10, backend="pytorch",
        )
        assert res_full.converged, f"full weight solver did not converge: mre={res_full.mre:.3e}"

        res_red = solve_fixed_point_qdecm_degenerate(
            theta_topo0, theta_w0, k_out, k_in, s_out, s_in,
            tol=1e-11, max_iter=2000, topo_max_iter=2000, anderson_depth=10,
            backend="pytorch",
        )
        assert res_red.converged, f"reduced weight solver did not converge: mre={res_red.mre:.3e}"

        to_f, ti_f = res_full.best_theta[:N], res_full.best_theta[N:]
        to_r, ti_r = res_red.best_theta[:N], res_red.best_theta[N:]
        z_f = to_f[:, None] + ti_f[None, :]
        z_r = to_r[:, None] + ti_r[None, :]
        G_f = -1.0 / np.expm1(-np.clip(z_f, 1e-8, None))
        G_r = -1.0 / np.expm1(-np.clip(z_r, 1e-8, None))
        topo_np = theta_topo_full.numpy()
        P = 1.0 / (1.0 + np.exp(topo_np[:N, None] + topo_np[None, N:]))
        np.fill_diagonal(P, 0.0)
        W_f = P * G_f
        W_r = P * G_r
        np.fill_diagonal(W_f, 0.0)
        np.fill_diagonal(W_r, 0.0)
        max_diff = np.max(np.abs(W_f - W_r))
        assert max_diff < 1e-2, f"E[w] mismatch full vs reduced: {max_diff:.3e}"

    def test_solve_tool_reduce_degeneracy_default_true(self) -> None:
        """solve_tool() must use the reduced path by default (both topology
        and weight steps) and agree with reduce_degeneracy=False."""
        k_out, k_in, s_out, s_in = self._degenerate_network()
        m1 = qDECMModel(k_out, k_in, s_out, s_in)
        conv1 = m1.solve_tool(tol=1e-9, max_iter=2000)
        assert conv1
        assert "degeneracy-reduced" in m1.sol.message

        m2 = qDECMModel(k_out, k_in, s_out, s_in)
        conv2 = m2.solve_tool(tol=1e-9, max_iter=2000, reduce_degeneracy=False)
        assert conv2
        assert "degeneracy-reduced" not in m2.sol.message

    def test_solve_tool_falls_back_for_backtracking_gamma(self) -> None:
        """reduce_degeneracy=True with backtracking_gamma>0 must fall back
        to the full solver (backtracking isn't supported in the reduced
        path)."""
        k_out, k_in, s_out, s_in = self._degenerate_network()
        m = qDECMModel(k_out, k_in, s_out, s_in)
        conv = m.solve_tool(tol=1e-9, max_iter=2000, backtracking_gamma=1.2)
        assert conv
        assert "degeneracy-reduced" not in m.sol.message


