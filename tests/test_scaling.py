"""Tests for Phase 3 — DCM scaling and memory-efficient computation.

Covers:
- Correctness: chunked residual and neg_log_likelihood match dense versions.
- Convergence: the reduced θ-Newton solver converges for N=100 and N=1000.
- Auto-switch: residual() calls _residual_chunked when N > _LARGE_N_THRESHOLD
  (verified via monkeypatching the threshold).

All tests are designed to complete in < 30 seconds on a typical CI runner.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dcms.models.dcm import DCMModel, _LARGE_N_THRESHOLD, _DEFAULT_CHUNK
from dcms.solvers.fixed_point_dcm import solve_fixed_point_dcm
from dcms.utils.wng import k_s_generator_pl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CONV_TOL = 1e-5   # acceptance threshold for constraint error


def _make_known_model(N: int, seed: int = 0) -> tuple[DCMModel, np.ndarray]:
    """Build a DCMModel with a known exact solution.

    Generates a random θ_true and computes the corresponding k_out / k_in
    analytically so that the true solution is known.

    Args:
        N:    Number of nodes.
        seed: RNG seed.

    Returns:
        (model, theta_true) where model.residual(theta_true) ≈ 0.
    """
    rng = np.random.default_rng(seed)
    theta_true = rng.uniform(0.5, 2.5, size=2 * N)
    x = np.exp(-theta_true[:N])
    y = np.exp(-theta_true[N:])
    xy = x[:, None] * y[None, :]
    P = xy / (1.0 + xy)
    np.fill_diagonal(P, 0.0)
    k_out = P.sum(axis=1)
    k_in = P.sum(axis=0)
    model = DCMModel(k_out, k_in)
    return model, theta_true


def _make_pl_model(N: int, seed: int = 0) -> DCMModel | None:
    """Build a DCMModel from a power-law network (returns None if infeasible).

    Args:
        N:    Number of nodes.
        seed: Random seed.

    Returns:
        DCMModel or None if the generated degree sequence is infeasible.
    """
    k, _ = k_s_generator_pl(N, rho=0.05, seed=seed)
    k_out = k[:N].numpy().astype(float)
    k_in = k[N:].numpy().astype(float)
    if k_out.max() >= N or k_in.max() >= N:
        return None
    return DCMModel(k_out, k_in)


# ---------------------------------------------------------------------------
# Chunked residual correctness
# ---------------------------------------------------------------------------

class TestChunkedResidualCorrectness:
    """Verify that _residual_chunked produces identical results to dense residual."""

    @pytest.mark.parametrize("N,seed", [(10, 0), (30, 1), (50, 2)])
    def test_chunked_matches_dense(self, N: int, seed: int) -> None:
        """Chunked residual must match the dense computation element-wise."""
        model, theta_true = _make_known_model(N, seed)
        theta = torch.tensor(theta_true, dtype=torch.float64)

        F_dense = model.residual(theta)                         # uses dense path (N ≤ 5000)
        F_chunked = model._residual_chunked(theta, chunk_size=4)  # explicit chunked path

        assert torch.allclose(F_dense, F_chunked, atol=1e-12), (
            f"Max diff: {(F_dense - F_chunked).abs().max():.3e}"
        )

    @pytest.mark.parametrize("chunk_size", [1, 3, 7, 16, 100])
    def test_different_chunk_sizes(self, chunk_size: int) -> None:
        """Results must be identical regardless of chunk_size."""
        model, theta_true = _make_known_model(N=20, seed=5)
        theta = torch.tensor(theta_true, dtype=torch.float64)
        F_ref = model._residual_chunked(theta, chunk_size=1)  # chunk=1 is the most explicit
        F_chunk = model._residual_chunked(theta, chunk_size=chunk_size)
        assert torch.allclose(F_ref, F_chunk, atol=1e-12), (
            f"chunk_size={chunk_size}: max diff {(F_ref - F_chunk).abs().max():.3e}"
        )

    def test_zero_at_true_solution(self) -> None:
        """Chunked residual must be ≈ 0 at the true solution."""
        model, theta_true = _make_known_model(N=15, seed=3)
        theta = torch.tensor(theta_true, dtype=torch.float64)
        F = model._residual_chunked(theta, chunk_size=5)
        assert F.abs().max().item() < 1e-10, (
            f"Max chunked residual at true solution: {F.abs().max().item():.3e}"
        )

    @pytest.mark.parametrize("bad_chunk", [0, -1, -10])
    def test_invalid_chunk_size_raises(self, bad_chunk: int) -> None:
        """_residual_chunked must raise ValueError for chunk_size < 1."""
        model, theta_true = _make_known_model(N=10, seed=0)
        theta = torch.tensor(theta_true, dtype=torch.float64)
        with pytest.raises(ValueError, match="chunk_size"):
            model._residual_chunked(theta, chunk_size=bad_chunk)

    def test_dense_and_chunked_agree(self) -> None:
        """Dense residual() and explicit chunked path must agree element-wise."""
        model, theta_true = _make_known_model(N=20, seed=7)
        theta = torch.tensor(theta_true, dtype=torch.float64)
        # For N < _LARGE_N_THRESHOLD, residual() uses the dense path.
        # Verify it agrees with the explicitly-chunked computation.
        F_auto = model.residual(theta)
        F_chunked = model._residual_chunked(theta, chunk_size=5)
        assert torch.allclose(F_auto, F_chunked, atol=1e-12), (
            f"Dense and chunked paths disagree: max diff "
            f"{(F_auto - F_chunked).abs().max():.3e}"
        )

    def test_auto_switch_for_large_n(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """residual() must call _residual_chunked when N > _LARGE_N_THRESHOLD."""
        import dcms.models.dcm as dcm_module

        model, theta_true = _make_known_model(N=20, seed=7)
        theta = torch.tensor(theta_true, dtype=torch.float64)

        called: list[bool] = []

        original_chunked = model._residual_chunked

        def _spy_chunked(th: torch.Tensor, **kwargs: object) -> torch.Tensor:
            called.append(True)
            return original_chunked(th, **kwargs)

        # Lower the threshold so N=20 triggers the auto-switch
        monkeypatch.setattr(dcm_module, "_LARGE_N_THRESHOLD", 10)
        model._residual_chunked = _spy_chunked  # type: ignore[method-assign]

        model.residual(theta)
        assert called, (
            "residual() did not call _residual_chunked when N > _LARGE_N_THRESHOLD"
        )


# ---------------------------------------------------------------------------
# Chunked neg_log_likelihood correctness
# ---------------------------------------------------------------------------

class TestChunkedNegLogLikelihood:
    """Verify that _neg_log_likelihood_chunked matches the dense version."""

    @pytest.mark.parametrize("N,seed", [(8, 0), (20, 1), (40, 2)])
    def test_chunked_matches_dense(self, N: int, seed: int) -> None:
        model, theta_true = _make_known_model(N, seed)
        theta = torch.tensor(theta_true, dtype=torch.float64)
        nll_dense = model.neg_log_likelihood(theta)
        nll_chunked = model._neg_log_likelihood_chunked(theta, chunk_size=4)
        assert abs(nll_dense - nll_chunked) < 1e-9, (
            f"NLL mismatch: dense={nll_dense:.6f}  chunked={nll_chunked:.6f}"
        )

    @pytest.mark.parametrize("chunk_size", [1, 5, 13, 50])
    def test_different_chunk_sizes(self, chunk_size: int) -> None:
        model, theta_true = _make_known_model(N=25, seed=4)
        theta = torch.tensor(theta_true, dtype=torch.float64)
        nll_ref = model._neg_log_likelihood_chunked(theta, chunk_size=1)
        nll_chunk = model._neg_log_likelihood_chunked(theta, chunk_size=chunk_size)
        assert abs(nll_ref - nll_chunk) < 1e-9, (
            f"chunk={chunk_size}: ref={nll_ref:.6f}  got={nll_chunk:.6f}"
        )

    @pytest.mark.parametrize("bad_chunk", [0, -1, -5])
    def test_invalid_chunk_size_raises(self, bad_chunk: int) -> None:
        """_neg_log_likelihood_chunked must raise ValueError for chunk_size < 1."""
        model, theta_true = _make_known_model(N=10, seed=0)
        theta = torch.tensor(theta_true, dtype=torch.float64)
        with pytest.raises(ValueError, match="chunk_size"):
            model._neg_log_likelihood_chunked(theta, chunk_size=bad_chunk)


# ---------------------------------------------------------------------------
# Convergence at N = 100 and N = 1000
# ---------------------------------------------------------------------------

class TestScalingN1000:
    """Verify that the reduced θ-Newton solver converges for N=100 and N=1000 power-law networks,
    judged by the PER-NODE residual of the expanded solution."""

    @pytest.fixture(scope="class")
    def model_100(self) -> DCMModel | None:
        for seed in range(20):
            m = _make_pl_model(N=100, seed=seed)
            if m is not None:
                return m
        return None

    @pytest.fixture(scope="class")
    def model_1k(self) -> DCMModel | None:
        for seed in range(20):
            m = _make_pl_model(N=1_000, seed=seed)
            if m is not None:
                return m
        return None

    def test_theta_newton_n100(self, model_100: DCMModel | None) -> None:
        if model_100 is None:
            pytest.skip("No feasible N=100 network found")
        result = solve_fixed_point_dcm(
            model_100.initial_theta("degrees"), model_100.k_out, model_100.k_in,
            tol=1e-7, max_iter=5_000, anderson_depth=10,
        )
        assert model_100.constraint_error(result.best_theta) < CONV_TOL

    def test_theta_newton_n1000(self, model_1k: DCMModel | None) -> None:
        if model_1k is None:
            pytest.skip("No feasible N=1000 network found")
        result = solve_fixed_point_dcm(
            model_1k.initial_theta("degrees"), model_1k.k_out, model_1k.k_in,
            tol=1e-6, max_iter=5_000, anderson_depth=10,
        )
        err = model_1k.max_relative_error(result.best_theta)
        assert err < CONV_TOL, (
            f"θ-Newton N=1000: err={err:.3e} conv={result.converged} iters={result.iterations}"
        )

    def test_chunked_residual_n1000(self, model_1k: DCMModel | None) -> None:
        """Chunked residual for N=1000 must produce finite values."""
        if model_1k is None:
            pytest.skip("No feasible N=1000 network found")
        theta0 = model_1k.initial_theta("degrees")
        F = model_1k._residual_chunked(theta0, chunk_size=_DEFAULT_CHUNK)
        assert F.shape == torch.Size([2_000])
        assert torch.isfinite(F).all(), "Chunked residual has NaN/Inf for N=1000"

    def test_chunked_matches_dense_n1000(self, model_1k: DCMModel | None) -> None:
        """Chunked and dense residuals must agree element-wise for N=1000."""
        if model_1k is None:
            pytest.skip("No feasible N=1000 network found")
        theta0 = model_1k.initial_theta("degrees")
        F_dense = model_1k.residual(theta0)          # dense (N=1000 < threshold)
        F_chunk = model_1k._residual_chunked(theta0, chunk_size=_DEFAULT_CHUNK)
        assert torch.allclose(F_dense, F_chunk, atol=1e-10), (
            f"Max diff dense vs chunked for N=1000: {(F_dense - F_chunk).abs().max():.3e}"
        )
