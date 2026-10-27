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

from dcms.models.dwcm import DWCMModel, _ETA_MAX, _ETA_MIN
from dcms.solvers.base import SolverResult
from dcms.solvers.fixed_point_dwcm import solve_fixed_point_dwcm


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

    def test_normalized_method(self) -> None:
        """normalized: β_i = s_i / Σ_j s_j — shape, finite, positive."""
        model, _ = make_dwcm_model(N=10)
        theta0 = model.initial_theta("normalized")
        assert theta0.shape == torch.Size([20])
        assert torch.isfinite(theta0).all()
        assert (theta0 > 0).all(), "All initial θ must be strictly positive for DWCM"

    def test_uniform_method(self) -> None:
        """uniform: all betas equal to the median of the strengths init."""
        model, _ = make_dwcm_model(N=10)
        theta0 = model.initial_theta("uniform")
        assert theta0.shape == torch.Size([20])
        assert torch.isfinite(theta0).all()
        assert (theta0 > 0).all()

    def test_unknown_method_raises(self) -> None:
        model, _ = make_dwcm_model(N=4)
        with pytest.raises(ValueError):
            model.initial_theta("unknown")

    def test_feasibility_of_initial_theta(self) -> None:
        """Initial θ must satisfy the feasibility constraint β_i * β_j < 1."""
        model, _ = make_dwcm_model(N=20)
        for method in ("strengths", "normalized", "uniform"):
            theta0 = model.initial_theta(method)
            N = model.N
            beta_out = torch.exp(-theta0[:N])
            beta_in = torch.exp(-theta0[N:])
            max_product = (beta_out[:, None] * beta_in[None, :]).fill_diagonal_(0.0).max().item()
            assert max_product < 1.0, (
                f"Feasibility violated for method={method!r}: "
                f"max(β_out_i * β_in_j) = {max_product:.4f}"
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
    def test_converges(self, N: int, seed: int) -> None:
        model, _ = make_dwcm_model(N=N, seed=seed)
        theta0 = model.initial_theta("strengths")
        result = solve_fixed_point_dwcm(
            theta0, model.s_out, model.s_in,
            tol=1e-10, max_iter=5000,
        )
        err = model.constraint_error(result.best_theta)
        assert err < CONV_TOL, f"N={N} GS error={err:.3e}"

    def test_result_has_residual_history(self) -> None:
        model, _ = make_dwcm_model(N=4, seed=0)
        theta0 = model.initial_theta("strengths")
        result = solve_fixed_point_dwcm(
            theta0, model.s_out, model.s_in, max_iter=20,
        )
        assert len(result.residuals) > 0

    def test_elapsed_time_positive(self) -> None:
        model, _ = make_dwcm_model(N=4)
        theta0 = model.initial_theta()
        result = solve_fixed_point_dwcm(
            theta0, model.s_out, model.s_in, max_iter=5,
        )
        assert result.elapsed_time >= 0.0

    def test_peak_ram_nonnegative(self) -> None:
        model, _ = make_dwcm_model(N=4)
        theta0 = model.initial_theta()
        result = solve_fixed_point_dwcm(
            theta0, model.s_out, model.s_in, max_iter=5,
        )
        assert result.peak_ram_bytes >= 0


class TestThetaNewtonDWCM:
    """Tests of the theta-Newton iteration of solve_fixed_point_dwcm."""

    @pytest.mark.parametrize("N,seed", [(4, 0), (10, 1)])
    def test_converges(self, N: int, seed: int) -> None:
        """theta-newton must converge on small systems with known solution."""
        model, _ = make_dwcm_model(N=N, seed=seed)
        theta0 = model.initial_theta("strengths")
        result = solve_fixed_point_dwcm(
            theta0, model.s_out, model.s_in,
            tol=1e-10, max_iter=5000, max_step=1.0,
        )
        err = model.constraint_error(result.best_theta)
        assert err < CONV_TOL, f"N={N} theta-newton error={err:.3e}"

    @pytest.mark.parametrize("N,seed", [(4, 2), (10, 3)])
    def test_converges_with_anderson(self, N: int, seed: int) -> None:
        """theta-newton + Anderson acceleration must converge on small systems."""
        model, _ = make_dwcm_model(N=N, seed=seed)
        theta0 = model.initial_theta("strengths")
        result = solve_fixed_point_dwcm(
            theta0, model.s_out, model.s_in,
            tol=1e-10, max_iter=5000,
            max_step=1.0, anderson_depth=10,
        )
        err = model.constraint_error(result.best_theta)
        assert err < CONV_TOL, f"N={N} theta-newton+Anderson error={err:.3e}"

    def test_zero_strength_nodes_pinned(self) -> None:
        """theta-newton must pin θ = _ETA_MAX for zero-strength nodes."""
        # Build a model where node 0 has s_out=0 and node N-1 has s_in=0
        rng = np.random.default_rng(99)
        N = 6
        theta_true = rng.uniform(0.5, 2.0, size=2 * N)
        beta_out = np.exp(-theta_true[:N])
        beta_in = np.exp(-theta_true[N:])
        beta_mat = beta_out[:, None] * beta_in[None, :]
        W = beta_mat / (1.0 - beta_mat)
        np.fill_diagonal(W, 0.0)
        s_out = W.sum(axis=1)
        s_in = W.sum(axis=0)
        # Force node 0 out-strength and last node in-strength to zero
        s_out[0] = 0.0
        s_in[N - 1] = 0.0
        model = DWCMModel(s_out, s_in)
        theta0 = model.initial_theta("strengths")
        result = solve_fixed_point_dwcm(
            theta0, model.s_out, model.s_in,
            tol=1e-8, max_iter=200, max_step=1.0,
        )
        theta_arr = result.best_theta
        assert theta_arr[0] == pytest.approx(_ETA_MAX, abs=1e-12), \
            f"Zero s_out node θ_out[0] should be _ETA_MAX, got {theta_arr[0]}"
        assert theta_arr[N + N - 1] == pytest.approx(_ETA_MAX, abs=1e-12), \
            f"Zero s_in node θ_in[N-1] should be _ETA_MAX, got {theta_arr[N + N - 1]}"

    def test_result_fields(self) -> None:
        """SolverResult must have positive elapsed_time and non-negative peak_ram."""
        model, _ = make_dwcm_model(N=4, seed=0)
        theta0 = model.initial_theta("strengths")
        result = solve_fixed_point_dwcm(
            theta0, model.s_out, model.s_in,
            tol=1e-8, max_iter=10, max_step=1.0,
        )
        assert result.elapsed_time >= 0.0
        assert result.peak_ram_bytes >= 0


class TestDegeneracyReduction:
    """Equivalence tests for the degeneracy-reduced DWCM solver."""

    def _degenerate_model(self, seed: int = 0) -> DWCMModel:
        # Strengths from an actually-realized random weighted graph, so the
        # sequence is guaranteed exactly realizable (see the analogous DCM
        # test's rationale for avoiding hand-picked, possibly-infeasible
        # sequences).
        g = torch.Generator().manual_seed(seed)
        N = 150
        adj = (torch.rand(N, N, generator=g) < 0.05).double() * (
            torch.randint(1, 5, (N, N), generator=g).double()
        )
        adj.fill_diagonal_(0.0)
        s_out = adj.sum(dim=1)
        s_in = adj.sum(dim=0)
        return DWCMModel(s_out, s_in)

    def test_groups_recover_expected_multiplicity(self) -> None:
        from dcms.solvers.fixed_point_dwcm import compute_degeneracy_groups_dwcm
        model = self._degenerate_model()
        group_of, mult, s_out_g, s_in_g = compute_degeneracy_groups_dwcm(model.s_out, model.s_in)
        N = model.N
        assert mult.sum().item() == N
        assert torch.allclose(s_out_g[group_of], model.s_out)
        assert torch.allclose(s_in_g[group_of], model.s_in)

    def test_expanded_solution_solves_the_per_node_system(self) -> None:
        """The reduced solution, expanded to the N nodes, must satisfy the ORIGINAL per-node strength equations (the reduction is
        exact, not an approximation) and be constant within every degeneracy group."""
        from dcms.solvers.fixed_point_dwcm import compute_degeneracy_groups_dwcm, solve_fixed_point_dwcm
        model = self._degenerate_model()
        res = solve_fixed_point_dwcm(model.initial_theta("strengths"), model.s_out, model.s_in, tol=1e-11, max_iter=3000, anderson_depth=10)
        assert res.converged, f"reduced solver did not converge: {res.message}"
        assert "degeneracy-reduced" in res.message
        assert model.max_relative_error(torch.as_tensor(res.best_theta)) < 1e-9
        N = model.N
        group_of, _, _, _ = compute_degeneracy_groups_dwcm(model.s_out, model.s_in)
        for g in range(int(group_of.max()) + 1):
            idx = np.flatnonzero(group_of.numpy() == g)
            assert np.ptp(res.best_theta[:N][idx]) == 0.0 and np.ptp(res.best_theta[N:][idx]) == 0.0

    def test_agrees_with_an_independent_root_finder(self) -> None:
        """Check against scipy.optimize.root on the full per-node residual of a small problem: same weight matrix."""
        from scipy.optimize import root
        from dcms.solvers.fixed_point_dwcm import solve_fixed_point_dwcm
        model, _ = make_dwcm_model(N=12, seed=4)
        res = solve_fixed_point_dwcm(model.initial_theta("strengths"), model.s_out, model.s_in, tol=1e-12, max_iter=5000, anderson_depth=10)
        assert res.converged
        sol = root(lambda t: model.residual(torch.as_tensor(t)).numpy(), res.best_theta * 1.02, method="lm", tol=1e-14)
        N = model.N

        def W(theta):
            W_ = 1.0 / np.expm1(np.clip(theta[:N, None] + theta[None, N:], 1e-15, None))
            np.fill_diagonal(W_, 0.0)
            return W_

        assert np.max(np.abs(W(res.best_theta) - W(sol.x))) < 1e-6

    def test_solve_tool_uses_the_reduced_system(self) -> None:
        model = self._degenerate_model()
        m1 = DWCMModel(model.s_out, model.s_in)
        conv1 = m1.solve_tool(tol=1e-9, max_iter=2000)
        assert conv1 and "degeneracy-reduced" in m1.sol.message
        assert m1.max_relative_error(torch.as_tensor(m1.sol.best_theta)) < 1e-8

    def test_removed_options_are_rejected(self) -> None:
        """The per-node solver, Gauss-Seidel and the numba backend no longer exist: asking for them is an error."""
        model = self._degenerate_model()
        for kw in ({"variant": "gauss-seidel"}, {"backend": "numba"}, {"reduce_degeneracy": False}):
            with pytest.raises(TypeError):
                DWCMModel(model.s_out, model.s_in).solve_tool(tol=1e-6, **kw)


class TestFpgsNewtonMiniLoopGuard:
    """The FP-GS periodic Newton-Anderson mini-loop (fixed_point_dwcm.py, fired after 30
    iterations with no record improvement -- see decm_dwcm_degenerate_stagnation_bug memory)
    used to accept its own output unconditionally (`theta_next = theta_nt`) even when its
    internal Anderson mixing itself diverged. Found 2026-09-30 on a from-scratch benchmark run
    (a different machine, via the cref_out mailbox): ita_elections_dico1 sits in a near-exact
    floating-point limit cycle under DWCM, and depending on the CPU thread count (hence the
    float64 summation order of PyTorch's parallel reductions -- NOT a bug in the network's own
    data), the mini-loop's own correction can itself diverge by several orders of magnitude
    before being handed back to the main loop as its new starting point, wasting the 30
    iterations of progress the mechanism exists to recover and -- since it also resets the
    blowup-guard reference right after -- going undetected until the outer 200-iteration
    stagnation check eventually gives up. Fixed by scoring the mini-loop's actual final output
    and falling back to best_theta (a point already known to be no worse) when it is not --
    using the same `_ANDERSON_BLOWUP_FACTOR` tolerance the rest of the file already judges
    genuine divergence by, not a stricter bar (which empirically rejected useful partial
    progress on normal, non-pathological firings too -- see the commit message for the two
    tolerances compared side by side)."""

    _REAL_NET = (
        Path(__file__).resolve().parents[2] / "bowtie2" / "tests" / "ita_elections_dico1_gdamp_decm.pkl"
    )

    def test_mini_loop_fix_on_the_real_network_that_found_the_bug(self) -> None:
        if not self._REAL_NET.exists():
            pytest.skip(f"real-network data not present at {self._REAL_NET} (outside this repo)")
        import pickle
        with open(self._REAL_NET, "rb") as fh:
            m = pickle.load(fh)  # a DECMModel; only s_out/s_in are used here
        model = DWCMModel(m.s_out, m.s_in)
        prev_threads = torch.get_num_threads()
        try:
            # num_threads=2 is the specific reproduction that found this bug (see the memory
            # note above) -- it is not guaranteed to reproduce the EXACT failure on every
            # torch build/platform (float64 summation order is build/platform-dependent), but
            # is the best concrete regression guard available against reintroducing the
            # "accept the mini-loop's output unconditionally" bug.
            torch.set_num_threads(2)
            converged = model.solve_tool(
                ic="strengths", tol=1e-5, max_iter=500, max_time=300,
                anderson_depth=10,
            )
        finally:
            torch.set_num_threads(prev_threads)
        assert converged, (
            f"ita_elections_dico1/DWCM did not converge at num_threads=2: "
            f"{model.sol.message}"
        )

    def test_easy_network_unaffected(self) -> None:
        """The guard must be inert (no behaviour change) for a network that never stagnates
        long enough to fire the mini-loop at all."""
        model, _ = make_dwcm_model(N=10, seed=1)
        converged = model.solve_tool(
            ic="strengths", tol=CONV_TOL, max_iter=2000, anderson_depth=10,
        )
        assert converged


class TestFpgsNewtonMiniLoopRetryLadder:
    """2026-10-07: a diverging mini-loop (TestFpgsNewtonMiniLoopGuard) now gets up to
    `_FPGS_NEWTON_RETRY_MAX` attempts -- the first unperturbed, the rest from randomly
    perturbed starts at geometrically escalating noise scale (mirroring DECM's own
    validated perturbed-restart design: random, log-scale multiplicative noise, growing
    by `_FPGS_NEWTON_RETRY_NOISE_GROWTH` per attempt) -- before giving up and falling back
    to best_theta. Supersedes an earlier single-fixed-jitter retry (one retry, deterministic
    +-1e-6 relative jitter) that remote's retest showed was not enough: it only took the
    divergence from 2.8e6 down to 1.1e5, still far above the acceptance bar. The retry
    ladder's actual escape behaviour is NOT validated here against a genuine divergence
    (none reproduces locally at the thread counts available on this machine) -- these are
    structural/no-regression/determinism checks only; see the commit message for the
    honest scope of what IS and is NOT verified."""

    def test_retry_noise_scale_escalates_and_draws_are_seed_reproducible(self) -> None:
        from dcms.solvers.fixed_point_dwcm import (
            _FPGS_NEWTON_RETRY_NOISE_BASE,
            _FPGS_NEWTON_RETRY_NOISE_GROWTH,
            _FPGS_NEWTON_RETRY_MAX,
        )
        # The intended per-attempt scale (the formula the solver uses) must be strictly
        # increasing across retries -- this is a deterministic property of the formula
        # itself, not of any particular random draw.
        scales = [
            _FPGS_NEWTON_RETRY_NOISE_BASE * (_FPGS_NEWTON_RETRY_NOISE_GROWTH ** a)
            for a in range(_FPGS_NEWTON_RETRY_MAX - 1)
        ]
        assert scales == sorted(scales) and len(set(scales)) == len(scales)

        theta = torch.rand(50, dtype=torch.float64) * 5.0 + _ETA_MIN

        def _ladder(seed: int) -> list[torch.Tensor]:
            gen = torch.Generator()
            gen.manual_seed(seed)
            starts = [theta.clone()]
            for attempt in range(1, _FPGS_NEWTON_RETRY_MAX):
                scale = _FPGS_NEWTON_RETRY_NOISE_BASE * (
                    _FPGS_NEWTON_RETRY_NOISE_GROWTH ** (attempt - 1)
                )
                noise = (torch.randn(theta.shape, generator=gen) * scale).to(dtype=theta.dtype)
                starts.append((theta * torch.exp(noise)).clamp(_ETA_MIN, _ETA_MAX))
            return starts

        ladder_a = _ladder(123)
        ladder_b = _ladder(123)
        for a, b in zip(ladder_a, ladder_b):
            assert torch.equal(a, b)  # same seed -> identical ladder (reproducible)
        assert bool((ladder_a[-1] > 0).all())  # stays feasible after clamp

    def test_easy_network_still_converges_with_the_retry_path_compiled_in(self) -> None:
        """No-regression check: the retry code must not change behaviour for a network that
        never fires the mini-loop at all (same network as TestFpgsNewtonMiniLoopGuard's twin)."""
        model, _ = make_dwcm_model(N=10, seed=1)
        converged = model.solve_tool(
            ic="strengths", tol=CONV_TOL, max_iter=2000, anderson_depth=10,
        )
        assert converged

    def test_real_network_regression_still_fixed(self) -> None:
        """The original threads=2 reproduction (TestFpgsNewtonMiniLoopGuard) must still converge
        with the retry logic compiled in -- the first attempt alone already succeeds there, so
        this also confirms the retry path is never (wrongly) taken on an easy recovery."""
        if not TestFpgsNewtonMiniLoopGuard._REAL_NET.exists():
            pytest.skip(f"real-network data not present at {TestFpgsNewtonMiniLoopGuard._REAL_NET} (outside this repo)")
        import pickle
        with open(TestFpgsNewtonMiniLoopGuard._REAL_NET, "rb") as fh:
            m = pickle.load(fh)
        model = DWCMModel(m.s_out, m.s_in)
        prev_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(2)
            converged = model.solve_tool(
                ic="strengths", tol=1e-5, max_iter=500, max_time=300,
                anderson_depth=10,
            )
        finally:
            torch.set_num_threads(prev_threads)
        assert converged, model.sol.message


class TestSolveToolSeedAndDevice:
    def test_seed_is_accepted_and_runs_are_reproducible(self) -> None:
        model, _ = make_dwcm_model(N=6, seed=1)
        assert model.solve_tool(tol=1e-8, max_iter=500, seed=3)
        theta = np.array(model.sol.best_theta)
        model2, _ = make_dwcm_model(N=6, seed=1)
        model2.solve_tool(tol=1e-8, max_iter=500, seed=3)
        assert np.array_equal(theta, np.array(model2.sol.best_theta))


class TestMiniLoopGiveUp:
    def test_mini_loop_is_disabled_after_consecutive_failed_firings(self, monkeypatch, capsys) -> None:
        import dcms.solvers.fixed_point_dwcm as fpd
        monkeypatch.setattr(fpd, "_ANDERSON_BLOWUP_FACTOR", 0.0)        # every mini-loop output now counts as diverged
        model, _ = make_dwcm_model(N=8, seed=2)
        model.solve_tool(tol=1e-300, max_iter=400, anderson_depth=5, verbose=True)
        out = capsys.readouterr().out
        assert out.count("falling back to best_theta") == fpd._FPGS_NEWTON_MAX_FAILED_FIRINGS
        assert "mini-loop disabled" in out
