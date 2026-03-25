"""Tests for Phase 3 — DCM scaling and memory-efficient computation.

Covers:
- Correctness: chunked residual and neg_log_likelihood match dense versions.
- Correctness: chunked fixed-point step matches dense step.
- Convergence: fixed-point (Gauss-Seidel) and L-BFGS converge for N=1000.
- Auto-switch: residual() calls _residual_chunked when N > _LARGE_N_THRESHOLD
  (verified via monkeypatching the threshold).
- Scaling: benchmark smoke tests run without error for N=50 and N=200.

All tests are designed to complete in < 30 seconds on a typical CI runner.
Larger-scale benchmarks (N > 1000) are in ``src/benchmarks/dcm_scaling.py``
and are run manually.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.dcm import DCMModel, _LARGE_N_THRESHOLD, _DEFAULT_CHUNK
from src.solvers.fixed_point_dcm import solve_fixed_point_dcm, _fp_step_chunked_dcm
from src.utils.wng import k_s_generator_pl


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
        import src.models.dcm as dcm_module

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
# Chunked fixed-point step correctness
# ---------------------------------------------------------------------------

class TestChunkedFPStep:
    """Verify that _fp_step_chunked produces identical updates as the dense step."""

    def _dense_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        k_out: torch.Tensor,
        k_in: torch.Tensor,
        variant: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reference dense fixed-point step."""
        xy = x[:, None] * y[None, :]
        xy_diag = xy.diagonal()
        D_out = (y[None, :] / (1.0 + xy)).sum(dim=1) - y / (1.0 + xy_diag)
        x_new = torch.where(D_out > 0, k_out / D_out, x)
        x_upd = x_new if variant == "gauss-seidel" else x
        xy2 = x_upd[:, None] * y[None, :]
        xy2_diag = xy2.diagonal()
        D_in = (x_upd[None, :] / (1.0 + xy2.T)).sum(dim=1) - x_upd / (1.0 + xy2_diag)
        y_new = torch.where(D_in > 0, k_in / D_in, y)
        return x_new, y_new

    @pytest.mark.parametrize("variant", ["gauss-seidel"])
    @pytest.mark.parametrize("N,seed", [(6, 0), (15, 1), (30, 2)])
    def test_chunked_matches_dense(self, variant: str, N: int, seed: int) -> None:
        """Chunked step must produce the same x_new and y_new as the dense step."""
        rng = np.random.default_rng(seed)
        x = torch.tensor(rng.uniform(0.1, 1.5, N), dtype=torch.float64)
        y = torch.tensor(rng.uniform(0.1, 1.5, N), dtype=torch.float64)
        k_out = torch.tensor(rng.uniform(1.0, N - 1, N), dtype=torch.float64)
        k_in = torch.tensor(rng.uniform(1.0, N - 1, N), dtype=torch.float64)

        x_dense, y_dense = self._dense_step(x, y, k_out, k_in, variant)
        x_chunk, y_chunk, _ = _fp_step_chunked_dcm(x, y, k_out, k_in, chunk_size=3)

        assert torch.allclose(x_dense, x_chunk, atol=1e-12), (
            f"x_new mismatch ({variant}): max diff {(x_dense - x_chunk).abs().max():.3e}"
        )
        assert torch.allclose(y_dense, y_chunk, atol=1e-12), (
            f"y_new mismatch ({variant}): max diff {(y_dense - y_chunk).abs().max():.3e}"
        )

    @pytest.mark.parametrize("chunk_size", [1, 4, 7, 20])
    def test_chunk_size_invariance(self, chunk_size: int) -> None:
        """Result must be independent of chunk_size."""
        N = 20
        rng = np.random.default_rng(42)
        x = torch.tensor(rng.uniform(0.1, 1.0, N), dtype=torch.float64)
        y = torch.tensor(rng.uniform(0.1, 1.0, N), dtype=torch.float64)
        k_out = torch.tensor(rng.uniform(1.0, 5.0, N), dtype=torch.float64)
        k_in = torch.tensor(rng.uniform(1.0, 5.0, N), dtype=torch.float64)

        x_ref, y_ref, _ = _fp_step_chunked_dcm(x, y, k_out, k_in, chunk_size=1)
        x_c, y_c, _ = _fp_step_chunked_dcm(x, y, k_out, k_in, chunk_size=chunk_size)
        assert torch.allclose(x_ref, x_c, atol=1e-12)
        assert torch.allclose(y_ref, y_c, atol=1e-12)


# ---------------------------------------------------------------------------
# Convergence at N = 100 using chunked computation
# ---------------------------------------------------------------------------

class TestChunkedConvergenceSmall:
    """Verify solver convergence using explicitly forced chunked computation (N=100)."""

    @pytest.fixture(scope="class")
    def model_100(self) -> DCMModel | None:
        """Return a DCMModel for a power-law network with N=100."""
        for seed in range(20):
            m = _make_pl_model(N=100, seed=seed)
            if m is not None:
                return m
        return None

    def test_fixed_point_invalid_chunk_size_raises(self, model_100: DCMModel | None) -> None:
        """solve_fixed_point_dcm must raise ValueError for negative chunk_size."""
        if model_100 is None:
            pytest.skip("No feasible N=100 network found")
        theta0 = model_100.initial_theta("degrees")
        with pytest.raises(ValueError, match="chunk_size"):
            solve_fixed_point_dcm(
                model_100.residual, theta0, model_100.k_out, model_100.k_in,
                chunk_size=-1,
            )

    def test_fixed_point_chunk_size_zero_accepted(self, model_100: DCMModel | None) -> None:
        """solve_fixed_point_dcm must accept chunk_size=0 (auto-select) and converge."""
        if model_100 is None:
            pytest.skip("No feasible N=100 network found")
        theta0 = model_100.initial_theta("degrees")
        result = solve_fixed_point_dcm(
            model_100.residual, theta0, model_100.k_out, model_100.k_in,
            tol=1e-7, max_iter=5_000, variant="gauss-seidel",
            chunk_size=0,  # 0 means auto-select
        )
        err = model_100.constraint_error(result.theta)
        assert err < CONV_TOL, (
            f"Fixed-point GS (chunk_size=0) N=100: err={err:.3e}"
        )

    def test_fixed_point_gs_chunked_convergence(self, model_100: DCMModel | None) -> None:
        """Fixed-point GS must converge for N=100 even with chunk_size=16."""
        if model_100 is None:
            pytest.skip("No feasible N=100 network found")
        theta0 = model_100.initial_theta("degrees")
        result = solve_fixed_point_dcm(
            model_100.residual, theta0, model_100.k_out, model_100.k_in,
            tol=1e-7, max_iter=5_000, variant="gauss-seidel",
            chunk_size=16,
        )
        err = model_100.constraint_error(result.theta)
        assert err < CONV_TOL, (
            f"Fixed-point GS (chunked) N=100: err={err:.3e} conv={result.converged}"
        )


# ---------------------------------------------------------------------------
# Convergence at N = 1000 (validation scale)
# ---------------------------------------------------------------------------

class TestScalingN1000:
    """Verify that the memory-efficient solvers converge for N=1000.

    Uses the auto-switching code paths (residual switches to chunked for
    N > _LARGE_N_THRESHOLD).  Here N=1000 < threshold, so the *dense* path is
    used — but the chunked path is explicitly exercised via chunk_size parameter.
    """

    @pytest.fixture(scope="class")
    def model_1k(self) -> DCMModel | None:
        """Return a DCMModel for a power-law network with N=1000."""
        for seed in range(20):
            m = _make_pl_model(N=1_000, seed=seed)
            if m is not None:
                return m
        return None

    def test_fixed_point_gs_n1000(self, model_1k: DCMModel | None) -> None:
        """Fixed-point Gauss-Seidel must converge for N=1000."""
        if model_1k is None:
            pytest.skip("No feasible N=1000 network found")
        theta0 = model_1k.initial_theta("degrees")
        result = solve_fixed_point_dcm(
            model_1k.residual, theta0, model_1k.k_out, model_1k.k_in,
            tol=1e-6, max_iter=10_000, variant="gauss-seidel",
        )
        err = model_1k.constraint_error(result.theta)
        assert err < CONV_TOL, (
            f"Fixed-point GS N=1000: err={err:.3e} conv={result.converged} "
            f"iters={result.iterations}"
        )

    def test_theta_newton_n1000(self, model_1k: DCMModel | None) -> None:
        """θ-Newton must converge for N=1000."""
        if model_1k is None:
            pytest.skip("No feasible N=1000 network found")
        theta0 = model_1k.initial_theta("degrees")
        result = solve_fixed_point_dcm(
            model_1k.residual, theta0, model_1k.k_out, model_1k.k_in,
            tol=1e-6, max_iter=5_000, variant="theta-newton", anderson_depth=10,
        )
        err = model_1k.constraint_error(result.theta)
        assert err < CONV_TOL, (
            f"θ-Newton N=1000: err={err:.3e} conv={result.converged} "
            f"iters={result.iterations}"
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


# ---------------------------------------------------------------------------
# Benchmark module smoke test
# ---------------------------------------------------------------------------

class TestScalingBenchmarkModule:
    """Smoke-test the dcm_scaling benchmark at a very small N."""

    def test_run_scaling_benchmark_n50(self) -> None:
        """run_scaling_benchmark must complete without error for N=50."""
        from src.benchmarks.dcm_scaling import run_scaling_benchmark
        records = run_scaling_benchmark(N=50, rho=0.3, seed=0, tol=1e-6,
                                        timeout=60.0, verbose=False)
        assert len(records) > 0
        # At least one method must converge
        converged = [r for r in records if r["converged"]]
        assert len(converged) > 0, "No solver converged for N=50 in smoke test"

    def test_run_scaling_benchmark_n200(self) -> None:
        """run_scaling_benchmark must complete without error for N=200."""
        from src.benchmarks.dcm_scaling import run_scaling_benchmark
        records = run_scaling_benchmark(N=200, rho=0.1, seed=0, tol=1e-6,
                                        timeout=60.0, verbose=False)
        assert len(records) > 0
        converged = [r for r in records if r["converged"]]
        assert len(converged) > 0, "No solver converged for N=200 in smoke test"
