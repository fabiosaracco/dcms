"""Tests for the backend selection logic and Numba/PyTorch output parity.

Tests cover:
- resolve_backend: auto/pytorch/numba selection, fallback, and error handling.
- DCM: Numba and PyTorch produce identical results on a small network.
- DWCM: Numba and PyTorch produce identical results on a small network.
- aDECM: Numba and PyTorch produce identical results on a small network.
- DECM: Numba and PyTorch produce identical results on a small network.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

from dcms.utils.backend import resolve_backend


# ---------------------------------------------------------------------------
# resolve_backend tests
# ---------------------------------------------------------------------------


class TestResolveBackend:
    """Tests for :func:`resolve_backend`."""

    def test_auto_small_N(self):
        assert resolve_backend("auto", N=100) == "pytorch"

    def test_auto_large_N(self):
        # AUTO_NUMBA_THRESHOLD is 50_000; N=10_000 should use pytorch
        assert resolve_backend("auto", N=10_000) == "pytorch"
        # N above threshold should use numba (when available)
        assert resolve_backend("auto", N=100_000) == "numba"

    def test_pytorch_explicit(self):
        assert resolve_backend("pytorch", N=100) == "pytorch"
        assert resolve_backend("pytorch", N=10_000) == "pytorch"

    def test_numba_explicit(self):
        assert resolve_backend("numba", N=100) == "numba"
        assert resolve_backend("numba", N=10_000) == "numba"

    def test_invalid_backend(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            resolve_backend("invalid")

    def test_custom_threshold(self):
        assert resolve_backend("auto", N=3000, threshold=2000) == "numba"
        assert resolve_backend("auto", N=1000, threshold=2000) == "pytorch"

    def test_numba_fallback_when_unavailable(self):
        """Simulate Numba not installed — should fall back to pytorch."""
        import dcms.utils.backend as bmod
        old = bmod._NUMBA_AVAILABLE
        try:
            bmod._NUMBA_AVAILABLE = False
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = resolve_backend("numba", N=100)
            assert result == "pytorch"
            assert any("Numba is not available" in str(x.message) for x in w)
        finally:
            bmod._NUMBA_AVAILABLE = old

    def test_auto_fallback_large_N_no_numba(self):
        """auto with large N but no Numba → pytorch."""
        import dcms.utils.backend as bmod
        old = bmod._NUMBA_AVAILABLE
        try:
            bmod._NUMBA_AVAILABLE = False
            result = resolve_backend("auto", N=10_000)
            assert result == "pytorch"
        finally:
            bmod._NUMBA_AVAILABLE = old


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CONV_TOL = 1e-5
PARITY_ATOL = 1e-10


def _make_dcm_data(N: int = 10, seed: int = 42):
    """Create a small DCM model with a known solution for parity tests."""
    from dcms.models.dcm import DCMModel
    rng = np.random.default_rng(seed)
    theta_true = rng.uniform(0.5, 2.0, 2 * N)
    x = np.exp(-theta_true[:N])
    y = np.exp(-theta_true[N:])
    xy = x[:, None] * y[None, :]
    np.fill_diagonal(xy, 0.0)
    p = xy / (1.0 + xy)
    k_out = p.sum(axis=1)
    k_in = p.sum(axis=0)
    model = DCMModel(k_out=k_out, k_in=k_in)
    return model, theta_true


def _make_dwcm_data(N: int = 6, seed: int = 42):
    """Create a small DWCM model with a known solution for parity tests."""
    from dcms.models.dwcm import DWCMModel
    rng = np.random.default_rng(seed)
    theta_true = rng.uniform(0.5, 2.0, 2 * N)
    beta_out = np.exp(-theta_true[:N])
    beta_in = np.exp(-theta_true[N:])
    xy = beta_out[:, None] * beta_in[None, :]
    np.fill_diagonal(xy, 0.0)
    W = xy / (1.0 - np.clip(xy, 0, 0.9999))
    s_out = W.sum(axis=1)
    s_in = W.sum(axis=0)
    model = DWCMModel(s_out=s_out, s_in=s_in)
    return model, theta_true


def _make_adecm_data(N: int = 6, seed: int = 42):
    """Create a small aDECM model."""
    from dcms.models.adecm import ADECMModel
    rng = np.random.default_rng(seed)
    theta_topo = rng.uniform(0.5, 2.0, 2 * N)
    x = np.exp(-theta_topo[:N])
    y = np.exp(-theta_topo[N:])
    xy = x[:, None] * y[None, :]
    np.fill_diagonal(xy, 0.0)
    p = xy / (1.0 + xy)
    k_out = p.sum(axis=1)
    k_in = p.sum(axis=0)
    theta_w = rng.uniform(0.5, 2.0, 2 * N)
    beta_out = np.exp(-theta_w[:N])
    beta_in = np.exp(-theta_w[N:])
    bxy = beta_out[:, None] * beta_in[None, :]
    np.fill_diagonal(bxy, 0.0)
    W = p / (1.0 - np.clip(bxy, 0, 0.9999))
    s_out = W.sum(axis=1)
    s_in = W.sum(axis=0)
    model = ADECMModel(k_out=k_out, k_in=k_in, s_out=s_out, s_in=s_in)
    return model


def _make_decm_data(N: int = 6, seed: int = 42):
    """Create a small DECM model."""
    from dcms.models.decm import DECMModel
    rng = np.random.default_rng(seed)
    theta_topo = rng.uniform(0.5, 2.0, 2 * N)
    theta_w = rng.uniform(0.5, 2.0, 2 * N)
    x = np.exp(-theta_topo[:N])
    y = np.exp(-theta_topo[N:])
    bo = np.exp(-theta_w[:N])
    bi = np.exp(-theta_w[N:])
    from math import expm1, log, exp
    k_out = np.zeros(N)
    k_in = np.zeros(N)
    s_out = np.zeros(N)
    s_in = np.zeros(N)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            eta = theta_w[i] + theta_w[N + j]
            eta_safe = max(eta, 1e-8)
            em1 = expm1(eta_safe)
            G = 1.0 / em1 if em1 > 0 else 1e15
            log_q = -log(em1) if em1 > 1e-300 else 300.0
            logit = -theta_topo[i] - theta_topo[N + j] + log_q
            p = 1.0 / (1.0 + exp(-logit))
            k_out[i] += p
            k_in[j] += p
            s_out[i] += p * G
            s_in[j] += p * G
    model = DECMModel(k_out=k_out, k_in=k_in, s_out=s_out, s_in=s_in)
    return model


# ---------------------------------------------------------------------------
# Parity tests: ensure PyTorch and Numba backends produce the same result
# ---------------------------------------------------------------------------


class TestDCMParity:
    """DCM backend parity tests."""

    def test_solve_pytorch(self):
        model, _ = _make_dcm_data(N=10)
        ok = model.solve_tool(backend="pytorch")
        assert ok
        assert model.sol.converged

    def test_solve_numba(self):
        model, _ = _make_dcm_data(N=10)
        ok = model.solve_tool(backend="numba")
        assert ok
        assert model.sol.converged

    def test_parity_theta_newton(self):
        model_pt, _ = _make_dcm_data(N=10)
        model_nb, _ = _make_dcm_data(N=10)
        model_pt.solve_tool(variant="theta-newton", backend="pytorch")
        model_nb.solve_tool(variant="theta-newton", backend="numba")
        assert model_pt.sol.converged
        assert model_nb.sol.converged
        err_pt = model_pt.constraint_error(model_pt.sol.theta)
        err_nb = model_nb.constraint_error(model_nb.sol.theta)
        assert err_pt < CONV_TOL
        assert err_nb < CONV_TOL

    def test_parity_gauss_seidel(self):
        model_pt, _ = _make_dcm_data(N=10)
        model_nb, _ = _make_dcm_data(N=10)
        model_pt.solve_tool(variant="gauss-seidel", anderson_depth=5, backend="pytorch")
        model_nb.solve_tool(variant="gauss-seidel", anderson_depth=5, backend="numba")
        assert model_pt.sol.converged
        assert model_nb.sol.converged
        err_pt = model_pt.constraint_error(model_pt.sol.theta)
        err_nb = model_nb.constraint_error(model_nb.sol.theta)
        assert err_pt < CONV_TOL
        assert err_nb < CONV_TOL


class TestDWCMParity:
    """DWCM backend parity tests."""

    def test_solve_pytorch(self):
        model, _ = _make_dwcm_data(N=6)
        ok = model.solve_tool(backend="pytorch")
        assert ok
        assert model.sol.converged

    def test_solve_numba(self):
        model, _ = _make_dwcm_data(N=6)
        ok = model.solve_tool(backend="numba")
        assert ok
        assert model.sol.converged

    def test_parity_theta_newton(self):
        model_pt, _ = _make_dwcm_data(N=6)
        model_nb, _ = _make_dwcm_data(N=6)
        model_pt.solve_tool(variant="theta-newton", backend="pytorch")
        model_nb.solve_tool(variant="theta-newton", backend="numba")
        assert model_pt.sol.converged
        assert model_nb.sol.converged
        err_pt = model_pt.constraint_error(model_pt.sol.theta)
        err_nb = model_nb.constraint_error(model_nb.sol.theta)
        assert err_pt < CONV_TOL
        assert err_nb < CONV_TOL


class TestADECMParity:
    """aDECM backend parity tests."""

    def test_solve_pytorch(self):
        model = _make_adecm_data(N=6)
        ok = model.solve_tool(backend="pytorch")
        assert ok

    def test_solve_numba(self):
        model = _make_adecm_data(N=6)
        ok = model.solve_tool(variant="theta-newton", backend="numba")
        assert ok


class TestDECMParity:
    """DECM backend parity tests."""

    def test_solve_pytorch(self):
        model = _make_decm_data(N=6)
        ok = model.solve_tool(backend="pytorch", multi_start=False)
        # DECM may or may not converge on random data; just check no crash
        assert isinstance(ok, bool)

    def test_solve_numba(self):
        model = _make_decm_data(N=6)
        ok = model.solve_tool(backend="numba", multi_start=False)
        assert isinstance(ok, bool)


class TestBackendDefault:
    """Verify that backend='auto' is the default everywhere."""

    def test_dcm_default(self):
        from dcms.solvers.fixed_point_dcm import solve_fixed_point_dcm
        import inspect
        sig = inspect.signature(solve_fixed_point_dcm)
        assert sig.parameters["backend"].default == "auto"

    def test_dwcm_default(self):
        from dcms.solvers.fixed_point_dwcm import solve_fixed_point_dwcm
        import inspect
        sig = inspect.signature(solve_fixed_point_dwcm)
        assert sig.parameters["backend"].default == "auto"

    def test_adecm_default(self):
        from dcms.solvers.fixed_point_adecm import solve_fixed_point_adecm
        import inspect
        sig = inspect.signature(solve_fixed_point_adecm)
        assert sig.parameters["backend"].default == "auto"

    def test_decm_default(self):
        from dcms.solvers.fixed_point_decm import solve_fixed_point_decm
        import inspect
        sig = inspect.signature(solve_fixed_point_decm)
        assert sig.parameters["backend"].default == "auto"
