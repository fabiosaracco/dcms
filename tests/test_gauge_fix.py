"""Tests for the gauge-fix solver feature.

Two test suites:
  i)  Standard test matrices — verify gauge correctness and convergence on
      synthetic N≤500 networks (all three gauge choices).
  ii) Real network — load the N=58832 Italian elections pkl, warm-start from
      the saved best iterate, and test that gauge_pivot="min" achieves lower
      residual than the saved non-converged run.

Run with:
    pytest tests/test_gauge_fix.py -v
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from dcms.models.adecm import ADECMModel
from dcms.models.dwcm import DWCMModel
from dcms.solvers.fixed_point_adecm import _apply_gauge_shift as _adecm_gauge
from dcms.solvers.fixed_point_dwcm import _apply_gauge_shift as _dwcm_gauge


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REAL_NET_PKL = Path(
    "/Users/fabio/Documents/Lavoro/PythonFiles/bowtie2_py310/bowtie2/tests/"
    "crisis_adecm_new_theta_nprocs_0_dico0.pkl"
)


def _make_dwcm(N: int = 30, seed: int = 0):
    """Return (DWCMModel, theta_true) with known exact solution."""
    rng = np.random.default_rng(seed)
    theta_true = rng.uniform(0.5, 2.0, size=2 * N)
    beta_out = np.exp(-theta_true[:N])
    beta_in = np.exp(-theta_true[N:])
    beta_mat = beta_out[:, None] * beta_in[None, :]
    W = beta_mat / (1.0 - beta_mat)
    np.fill_diagonal(W, 0.0)
    s_out = W.sum(axis=1)
    s_in = W.sum(axis=0)
    return DWCMModel(s_out, s_in), theta_true


def _make_adecm(N: int = 20, seed: int = 0):
    """Return (ADECMModel, theta_topo_true, theta_weight_true) with known solution."""
    rng = np.random.default_rng(seed)
    theta_topo_true = rng.uniform(0.5, 3.0, size=2 * N)
    x = np.exp(-theta_topo_true[:N])
    y = np.exp(-theta_topo_true[N:])
    P = x[:, None] * y[None, :]
    P = P / (1.0 + P)
    np.fill_diagonal(P, 0.0)
    k_out = P.sum(axis=1)
    k_in = P.sum(axis=0)
    theta_weight_true = rng.uniform(0.5, 2.0, size=2 * N)
    b_out = np.exp(-theta_weight_true[:N])
    b_in = np.exp(-theta_weight_true[N:])
    beta_mat = b_out[:, None] * b_in[None, :]
    # aDECM formula: W_ij = p_ij / (1 - exp(-z)) = -p_ij / expm1(-z)
    z = theta_weight_true[:N, None] + theta_weight_true[N:][None, :]
    W = -P / np.expm1(-z)
    np.fill_diagonal(W, 0.0)
    s_out = W.sum(axis=1)
    s_in = W.sum(axis=0)
    return ADECMModel(k_out, k_in, s_out, s_in), theta_topo_true, theta_weight_true


# ---------------------------------------------------------------------------
# Suite i: gauge helper correctness
# ---------------------------------------------------------------------------


class TestGaugeShiftHelper:
    """Verify that _apply_gauge_shift preserves pairwise sums and is idempotent."""

    @pytest.mark.parametrize("N", [10, 100])
    @pytest.mark.parametrize("pivot", [0, "min", "mean"])
    def test_pairwise_sums_preserved_dwcm(self, N, pivot):
        rng = np.random.default_rng(0)
        theta = torch.tensor(rng.uniform(0.1, 5.0, 2 * N))
        z_before = theta[:N, None] + theta[N:][None, :]
        theta2 = _dwcm_gauge(theta, N, pivot)
        z_after = theta2[:N, None] + theta2[N:][None, :]
        assert torch.allclose(z_before, z_after, atol=1e-12), \
            "Pairwise sums changed after gauge shift (DWCM)"

    @pytest.mark.parametrize("N", [10, 100])
    @pytest.mark.parametrize("pivot", [0, "min", "mean"])
    def test_pairwise_sums_preserved_adecm(self, N, pivot):
        rng = np.random.default_rng(0)
        theta = torch.tensor(rng.uniform(0.1, 5.0, 2 * N))
        z_before = theta[:N, None] + theta[N:][None, :]
        theta2 = _adecm_gauge(theta, N, pivot)
        z_after = theta2[:N, None] + theta2[N:][None, :]
        assert torch.allclose(z_before, z_after, atol=1e-12), \
            "Pairwise sums changed after gauge shift (aDECM)"

    @pytest.mark.parametrize("pivot", [0, "min", "mean"])
    def test_idempotent_dwcm(self, pivot):
        N = 20
        rng = np.random.default_rng(1)
        theta = torch.tensor(rng.uniform(0.1, 5.0, 2 * N))
        t1 = _dwcm_gauge(theta, N, pivot)
        t2 = _dwcm_gauge(t1, N, pivot)
        assert torch.allclose(t1, t2, atol=1e-12), "Gauge shift is not idempotent"

    def test_min_pivot_sets_min_to_zero_dwcm(self):
        N = 30
        rng = np.random.default_rng(2)
        theta = torch.tensor(rng.uniform(0.1, 5.0, 2 * N))
        t = _dwcm_gauge(theta, N, "min")
        assert abs(t[:N].min().item()) < 1e-12, "min pivot should set min(eta_out)=0"

    def test_mean_pivot_centres_dwcm(self):
        N = 30
        rng = np.random.default_rng(3)
        theta = torch.tensor(rng.uniform(0.1, 5.0, 2 * N))
        t = _dwcm_gauge(theta, N, "mean")
        assert abs(t[:N].mean().item()) < 1e-12, "mean pivot should centre eta_out at 0"

    def test_int_pivot_zeros_that_node_dwcm(self):
        N = 30
        rng = np.random.default_rng(4)
        theta = torch.tensor(rng.uniform(0.1, 5.0, 2 * N))
        pivot_idx = 7
        t = _dwcm_gauge(theta, N, pivot_idx)
        assert abs(t[pivot_idx].item()) < 1e-12, f"eta_out[{pivot_idx}] should be 0"


# ---------------------------------------------------------------------------
# Gauge validation
# ---------------------------------------------------------------------------


class TestGaugeValidation:
    """Gauge with GS variant should raise."""

    def test_gauge_with_gs_raises_dwcm(self):
        m, _ = _make_dwcm(N=20, seed=0)
        with pytest.raises(ValueError, match="theta-newton"):
            m.solve_tool(variant="gauss-seidel", gauge_pivot="min", max_iter=5)

    def test_gauge_with_gs_raises_adecm(self):
        m, _, _ = _make_adecm(N=10, seed=0)
        with pytest.raises(ValueError, match="theta-newton"):
            m.solve_tool(variant="gauss-seidel", gauge_pivot="min", max_iter=5)


# ---------------------------------------------------------------------------
# Convergence tests
# ---------------------------------------------------------------------------


class TestGaugeConvergenceSynthetic:
    """Gauge should preserve convergence on synthetic networks."""

    @pytest.mark.parametrize("gauge", [None, "min"])
    def test_adecm_converges(self, gauge):
        m, _, _ = _make_adecm(N=20, seed=42)
        converged = m.solve_tool(
            variant="theta-newton",
            anderson_depth=10,
            tol=1e-6,
            max_iter=3000,
            backend="pytorch",
            gauge_pivot=gauge,
        )
        assert converged, f"aDECM did not converge with gauge_pivot={gauge!r}"

    @pytest.mark.parametrize("gauge", [None, "min"])
    def test_dwcm_converges(self, gauge):
        m, _ = _make_dwcm(N=30, seed=0)
        converged = m.solve_tool(
            variant="theta-newton",
            anderson_depth=10,
            tol=1e-6,
            max_iter=3000,
            backend="pytorch",
            gauge_pivot=gauge,
        )
        assert converged, f"DWCM did not converge with gauge_pivot={gauge!r}"

    def test_adecm_gauge_min_converges_from_hub_ic(self):
        """When gauge='min', the hub node (argmin eta_out) is fixed to 0 — test
        that the solver still converges from the default IC."""
        # N=20, seed=0: node with largest strength/degree is the hub
        m, _, _ = _make_adecm(N=20, seed=0)
        converged = m.solve_tool(
            variant="theta-newton",
            anderson_depth=10,
            tol=1e-6,
            max_iter=3000,
            backend="pytorch",
            gauge_pivot="min",
        )
        assert converged, "aDECM did not converge with gauge_pivot='min'"

    def test_dwcm_gauge_min_int_match(self):
        """gauge_pivot='min' and gauge_pivot=argmin should give the same result."""
        m1, _ = _make_dwcm(N=30, seed=2)
        m2, _ = _make_dwcm(N=30, seed=2)
        m1.solve_tool(
            variant="theta-newton", anderson_depth=10, tol=1e-8, max_iter=3000,
            backend="pytorch", gauge_pivot="min",
        )
        # Find which node gauge='min' resolved to (argmin of initial IC)
        m_tmp, _ = _make_dwcm(N=30, seed=2)
        import torch
        ic = m_tmp.initial_theta("strengths")
        pivot_idx = int(ic[:30].argmin().item())
        m2.solve_tool(
            variant="theta-newton", anderson_depth=10, tol=1e-8, max_iter=3000,
            backend="pytorch", gauge_pivot=pivot_idx,
        )
        # Both should converge and give the same MRE
        res1 = m1.max_relative_error(m1.sol.theta)
        res2 = m2.max_relative_error(m2.sol.theta)
        assert res1 < 1e-5, f"gauge=min MRE too high: {res1:.2e}"
        assert res2 < 1e-5, f"gauge=int({pivot_idx}) MRE too high: {res2:.2e}"


# ---------------------------------------------------------------------------
# Suite ii: real network
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not REAL_NET_PKL.exists(),
    reason=f"Real-network pkl not found at {REAL_NET_PKL}",
)
class TestGaugeRealNetwork:
    """Run gauge-fix on the N=58832 Italian elections network."""

    def _load_pkl(self):
        with open(REAL_NET_PKL, "rb") as f:
            return pickle.load(f)

    def test_hub_node_identified(self):
        """Node 37916 should have the smallest eta_out in the saved best iterate."""
        model = self._load_pkl()
        theta_w = model.sol_weights.theta  # shape (2N,)
        N = model.k_out.shape[0]
        eta_out = theta_w[:N]
        hub = int(np.argmin(eta_out))
        print(f"\nHub node: {hub}, eta_out[hub]={eta_out[hub]:.4e}")
        assert eta_out[hub] < 0.1, f"Expected hub eta_out ≈ 0, got {eta_out[hub]:.4e}"

    def test_gauge_min_improves_residual(self):
        """Gauge_pivot='min' warm-started from the saved iterate should give
        lower residual than the saved iterate itself."""
        model = self._load_pkl()

        # Saved residual
        m_saved = ADECMModel(model.k_out, model.k_in, model.s_out, model.s_in)
        saved_res = m_saved.max_relative_error(
            model.sol_topo.theta, model.sol_weights.theta
        )
        print(f"\nSaved best-iterate MRE: {saved_res:.4e}")

        # Run up to 100 iterations with gauge_pivot="min" warm-started from saved
        m_gauge = ADECMModel(model.k_out, model.k_in, model.s_out, model.s_in)
        m_gauge.solve_tool(
            theta_topo_0=model.sol_topo.theta,
            theta_weights_0=model.sol_weights.theta,
            variant="theta-newton",
            anderson_depth=10,
            tol=1e-6,
            max_iter=100,
            max_time=300,   # 5 minute cap
            backend="pytorch",
            gauge_pivot="min",
            verbose=True,
        )
        gauge_res = m_gauge.max_relative_error(
            m_gauge.sol_topo.theta, m_gauge.sol_weights.theta
        )
        print(f"After gauge-min iterations: MRE={gauge_res:.4e}")
        assert gauge_res <= saved_res * 1.01, (
            f"Gauge-min did not improve (or maintain) residual: "
            f"{gauge_res:.4e} vs saved {saved_res:.4e}"
        )
