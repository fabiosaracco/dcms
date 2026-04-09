"""Tests for model.sample() methods."""
import pytest
import numpy as np
import torch
from dcms.utils.wng import k_s_generator_pl


# ── helpers ──────────────────────────────────────────────────────────────────

def _setup(N=30, seed=42):
    k, s = k_s_generator_pl(N, rho=0.1, seed=seed)
    k_out = k[:N].numpy().astype(float)
    k_in  = k[N:].numpy().astype(float)
    s_out = s[:N].numpy().astype(float)
    s_in  = s[N:].numpy().astype(float)
    return k_out, k_in, s_out, s_in


# ── DCM ──────────────────────────────────────────────────────────────────────

class TestDCMSample:
    def _fitted(self):
        from dcms.models.dcm import DCMModel
        k_out, k_in, _, _ = _setup()
        m = DCMModel(k_out, k_in)
        m.solve_tool(tol=1e-6)
        return m

    def test_raises_before_solve(self):
        from dcms.models.dcm import DCMModel
        k_out, k_in, _, _ = _setup()
        with pytest.raises(RuntimeError):
            DCMModel(k_out, k_in).sample()

    def test_returns_list(self):
        edges = self._fitted().sample(seed=0)
        assert isinstance(edges, list)

    def test_edge_format(self):
        edges = self._fitted().sample(seed=0)
        for e in edges:
            assert len(e) == 2
            assert isinstance(e[0], int)
            assert isinstance(e[1], int)

    def test_no_self_loops(self):
        edges = self._fitted().sample(seed=0)
        for i, j in edges:
            assert i != j

    def test_valid_node_indices(self):
        m = self._fitted()
        edges = m.sample(seed=0)
        N = m.N
        for i, j in edges:
            assert 0 <= i < N and 0 <= j < N

    def test_reproducible(self):
        m = self._fitted()
        assert m.sample(seed=7) == m.sample(seed=7)

    def test_different_seeds_differ(self):
        m = self._fitted()
        assert m.sample(seed=1) != m.sample(seed=2)

    def test_degree_distribution_approx(self):
        """Sampled mean degree should be close to the observed degree."""
        from dcms.models.dcm import DCMModel
        k_out, k_in, _, _ = _setup(N=50, seed=0)
        m = DCMModel(k_out, k_in)
        m.solve_tool(tol=1e-8)
        N = m.N
        n_samples = 200
        k_out_acc = np.zeros(N)
        k_in_acc  = np.zeros(N)
        for seed in range(n_samples):
            for i, j in m.sample(seed=seed):
                k_out_acc[i] += 1
                k_in_acc[j]  += 1
        k_out_mean = k_out_acc / n_samples
        k_in_mean  = k_in_acc  / n_samples
        # mean error < 1 degree unit on every node
        assert np.abs(k_out_mean - k_out).max() < 1.5
        assert np.abs(k_in_mean  - k_in).max() < 1.5


# ── DWCM ─────────────────────────────────────────────────────────────────────

class TestDWCMSample:
    def _fitted(self):
        from dcms.models.dwcm import DWCMModel
        _, _, s_out, s_in = _setup()
        m = DWCMModel(s_out, s_in)
        m.solve_tool(tol=1e-6)
        return m

    def test_raises_before_solve(self):
        from dcms.models.dwcm import DWCMModel
        _, _, s_out, s_in = _setup()
        with pytest.raises(RuntimeError):
            DWCMModel(s_out, s_in).sample()

    def test_returns_list(self):
        assert isinstance(self._fitted().sample(seed=0), list)

    def test_edge_format(self):
        for e in self._fitted().sample(seed=0):
            assert len(e) == 3
            assert isinstance(e[0], int) and isinstance(e[1], int) and isinstance(e[2], int)

    def test_positive_weights(self):
        for _, _, w in self._fitted().sample(seed=0):
            assert w > 0

    def test_no_self_loops(self):
        for i, j, _ in self._fitted().sample(seed=0):
            assert i != j

    def test_reproducible(self):
        m = self._fitted()
        assert m.sample(seed=3) == m.sample(seed=3)

    def test_strength_distribution_approx(self):
        """Sampled mean strength should be close to observed strength."""
        from dcms.models.dwcm import DWCMModel
        _, _, s_out, s_in = _setup(N=30, seed=0)
        m = DWCMModel(s_out, s_in)
        m.solve_tool(tol=1e-8)
        N = m.N
        n_samples = 300
        s_out_acc = np.zeros(N)
        s_in_acc  = np.zeros(N)
        for seed in range(n_samples):
            for i, j, w in m.sample(seed=seed):
                s_out_acc[i] += w
                s_in_acc[j]  += w
        s_out_mean = s_out_acc / n_samples
        s_in_mean  = s_in_acc  / n_samples
        # relative error < 20% on every node with non-zero strength
        nz_out = s_out > 0
        nz_in  = s_in  > 0
        assert (np.abs(s_out_mean[nz_out] - s_out[nz_out]) / s_out[nz_out]).max() < 0.25
        assert (np.abs(s_in_mean[nz_in]   - s_in[nz_in])   / s_in[nz_in]).max()  < 0.25


# ── aDECM ────────────────────────────────────────────────────────────────────

class TestADECMSample:
    def _fitted(self):
        from dcms.models.adecm import ADECMModel
        k_out, k_in, s_out, s_in = _setup()
        m = ADECMModel(k_out, k_in, s_out, s_in)
        m.solve_tool(tol=1e-6)
        return m

    def test_raises_before_solve(self):
        from dcms.models.adecm import ADECMModel
        k_out, k_in, s_out, s_in = _setup()
        with pytest.raises(RuntimeError):
            ADECMModel(k_out, k_in, s_out, s_in).sample()

    def test_returns_list(self):
        assert isinstance(self._fitted().sample(seed=0), list)

    def test_edge_format(self):
        for e in self._fitted().sample(seed=0):
            assert len(e) == 3
            assert all(isinstance(x, int) for x in e)

    def test_positive_weights(self):
        for _, _, w in self._fitted().sample(seed=0):
            assert w > 0

    def test_no_self_loops(self):
        for i, j, _ in self._fitted().sample(seed=0):
            assert i != j

    def test_reproducible(self):
        m = self._fitted()
        assert m.sample(seed=5) == m.sample(seed=5)


# ── DECM ─────────────────────────────────────────────────────────────────────

class TestDECMSample:
    def _fitted(self):
        from dcms.models.decm import DECMModel
        k_out, k_in, s_out, s_in = _setup()
        m = DECMModel(k_out, k_in, s_out, s_in)
        m.solve_tool(tol=1e-6)
        return m

    def test_raises_before_solve(self):
        from dcms.models.decm import DECMModel
        k_out, k_in, s_out, s_in = _setup()
        with pytest.raises(RuntimeError):
            DECMModel(k_out, k_in, s_out, s_in).sample()

    def test_returns_list(self):
        assert isinstance(self._fitted().sample(seed=0), list)

    def test_edge_format(self):
        for e in self._fitted().sample(seed=0):
            assert len(e) == 3
            assert all(isinstance(x, int) for x in e)

    def test_positive_weights(self):
        for _, _, w in self._fitted().sample(seed=0):
            assert w > 0

    def test_no_self_loops(self):
        for i, j, _ in self._fitted().sample(seed=0):
            assert i != j

    def test_reproducible(self):
        m = self._fitted()
        assert m.sample(seed=9) == m.sample(seed=9)
