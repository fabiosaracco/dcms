"""Tests for model.sample() methods."""
import functools
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

    def test_returns_ndarray(self):
        edges = self._fitted().sample(seed=0)
        assert isinstance(edges, np.ndarray)
        assert edges.ndim == 2 and edges.shape[1] == 2

    def test_edge_format(self):
        edges = self._fitted().sample(seed=0)
        assert np.issubdtype(edges.dtype, np.integer)
        assert edges.shape[1] == 2

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
        assert np.array_equal(m.sample(seed=7), m.sample(seed=7))

    def test_different_seeds_differ(self):
        m = self._fitted()
        assert not np.array_equal(m.sample(seed=1), m.sample(seed=2))

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

    def test_returns_ndarray(self):
        edges = self._fitted().sample(seed=0)
        assert isinstance(edges, np.ndarray)
        assert edges.ndim == 2 and edges.shape[1] == 3

    def test_edge_format(self):
        edges = self._fitted().sample(seed=0)
        assert np.issubdtype(edges.dtype, np.integer)
        assert edges.shape[1] == 3

    def test_positive_weights(self):
        for _, _, w in self._fitted().sample(seed=0):
            assert w > 0

    def test_no_self_loops(self):
        for i, j, _ in self._fitted().sample(seed=0):
            assert i != j

    def test_reproducible(self):
        m = self._fitted()
        assert np.array_equal(m.sample(seed=3), m.sample(seed=3))

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


# ── qDECM ────────────────────────────────────────────────────────────────────

class TestQDECMSample:
    def _fitted(self):
        from dcms.models.qdecm import qDECMModel
        k_out, k_in, s_out, s_in = _setup()
        m = qDECMModel(k_out, k_in, s_out, s_in)
        m.solve_tool(tol=1e-6)
        return m

    def test_raises_before_solve(self):
        from dcms.models.qdecm import qDECMModel
        k_out, k_in, s_out, s_in = _setup()
        with pytest.raises(RuntimeError):
            qDECMModel(k_out, k_in, s_out, s_in).sample()

    def test_returns_ndarray(self):
        edges = self._fitted().sample(seed=0)
        assert isinstance(edges, np.ndarray)
        assert edges.ndim == 2 and edges.shape[1] == 3

    def test_edge_format(self):
        edges = self._fitted().sample(seed=0)
        assert np.issubdtype(edges.dtype, np.integer)
        assert edges.shape[1] == 3

    def test_positive_weights(self):
        assert (self._fitted().sample(seed=0)[:, 2] > 0).all()

    def test_no_self_loops(self):
        edges = self._fitted().sample(seed=0)
        assert np.all(edges[:, 0] != edges[:, 1])

    def test_reproducible(self):
        m = self._fitted()
        assert np.array_equal(m.sample(seed=5), m.sample(seed=5))


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

    def test_returns_ndarray(self):
        edges = self._fitted().sample(seed=0)
        assert isinstance(edges, np.ndarray)
        assert edges.ndim == 2 and edges.shape[1] == 3

    def test_edge_format(self):
        edges = self._fitted().sample(seed=0)
        assert np.issubdtype(edges.dtype, np.integer)
        assert edges.shape[1] == 3

    def test_positive_weights(self):
        assert (self._fitted().sample(seed=0)[:, 2] > 0).all()

    def test_no_self_loops(self):
        edges = self._fitted().sample(seed=0)
        assert np.all(edges[:, 0] != edges[:, 1])

    def test_reproducible(self):
        m = self._fitted()
        assert np.array_equal(m.sample(seed=9), m.sample(seed=9))

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            self._fitted().sample(seed=0, method="nope")

    def test_grouped_format(self):
        m = self._fitted()
        edges = m.sample(seed=0, method="grouped")
        assert isinstance(edges, np.ndarray) and edges.shape[1] == 3
        assert np.issubdtype(edges.dtype, np.integer)
        assert (edges[:, 2] > 0).all()
        assert np.all(edges[:, 0] != edges[:, 1])
        assert len({(a, b) for a, b, _ in edges}) == len(edges)   # distinct ordered pairs
        assert np.array_equal(edges, m.sample(seed=0, method="grouped"))

    def test_grouped_falls_back_when_parameters_not_constant_in_groups(self):
        import copy
        from dcms.utils.grouped_sampling import group_model
        m = self._fitted()
        g = group_model(m)
        grp = int(np.flatnonzero(g["n_g"] > 1)[0])
        node = int(g["members"][g["start"][grp]])
        th = np.array(m.sol.best_theta, dtype=np.float64)
        th[node] += 1e-3
        m.sol = copy.copy(m.sol)
        m.sol.best_theta = th
        with pytest.warns(UserWarning, match="not constant within groups"):
            edges = m.sample(seed=0)
        assert np.array_equal(edges, m.sample(seed=0, method="exact"))

    def test_grouped_matches_pij(self):
        """Empirical link frequencies of the grouped sampler match the DECM link probabilities."""
        m = self._fitted()
        n = 3000
        freq = np.zeros((m.N, m.N))
        for e in m.sample_many(n, seed=1, n_jobs=1, method="grouped"):
            freq[e[:, 0], e[:, 1]] += 1
        freq /= n
        P = np.array(m.pij_matrix(m.sol.best_theta))
        np.fill_diagonal(P, 0.0)
        ok = (P > 0.01) & (P < 0.99)
        z = np.abs(freq - P)[ok] / np.sqrt(P[ok] * (1 - P[ok]) / n)
        assert z.max() < 5.0
        assert freq[np.eye(m.N, dtype=bool)].sum() == 0


# ---------------------------------------------------------------------------
# sample_many tests (all four models)
# ---------------------------------------------------------------------------

import numpy as np


class TestSampleManyDCM:
    @pytest.fixture(autouse=True)
    def _model(self):
        from dcms.models.dcm import DCMModel
        k_out, k_in = _setup()[:2]
        m = DCMModel(k_out, k_in)
        m.solve_tool(tol=1e-6)
        self.m = m

    def test_returns_list_of_arrays(self):
        samples = self.m.sample_many(4, seed=0)
        assert len(samples) == 4
        for s in samples:
            assert isinstance(s, np.ndarray)
            assert s.ndim == 2 and s.shape[1] == 2

    def test_n_jobs_1(self):
        samples = self.m.sample_many(3, seed=0, n_jobs=1)
        assert len(samples) == 3

    def test_no_self_loops(self):
        for arr in self.m.sample_many(3, seed=1):
            assert np.all(arr[:, 0] != arr[:, 1])


class TestSampleManyDWCM:
    @pytest.fixture(autouse=True)
    def _model(self):
        from dcms.models.dwcm import DWCMModel
        _, _, s_out, s_in = _setup()
        m = DWCMModel(s_out, s_in)
        m.solve_tool(tol=1e-6)
        self.m = m

    def test_returns_list_of_arrays(self):
        samples = self.m.sample_many(4, seed=0)
        assert len(samples) == 4
        for s in samples:
            assert isinstance(s, np.ndarray)
            assert s.ndim == 2 and s.shape[1] == 3

    def test_no_self_loops(self):
        for arr in self.m.sample_many(3, seed=1):
            assert np.all(arr[:, 0] != arr[:, 1])


class TestSampleManyQDECM:
    @pytest.fixture(autouse=True)
    def _model(self):
        from dcms.models.qdecm import qDECMModel
        k_out, k_in, s_out, s_in = _setup()
        m = qDECMModel(k_out, k_in, s_out, s_in)
        m.solve_tool(tol=1e-6)
        self.m = m

    def test_returns_list_of_arrays(self):
        samples = self.m.sample_many(4, seed=0)
        assert len(samples) == 4
        for s in samples:
            assert isinstance(s, np.ndarray)
            assert s.ndim == 2 and s.shape[1] == 3

    def test_n_jobs_1(self):
        samples = self.m.sample_many(3, seed=0, n_jobs=1)
        assert len(samples) == 3

    def test_no_self_loops(self):
        for arr in self.m.sample_many(3, seed=1):
            assert np.all(arr[:, 0] != arr[:, 1])


class TestSampleManyDECM:
    @pytest.fixture(autouse=True)
    def _model(self):
        from dcms.models.decm import DECMModel
        k_out, k_in, s_out, s_in = _setup()
        m = DECMModel(k_out, k_in, s_out, s_in)
        m.solve_tool(tol=1e-6)
        self.m = m

    def test_returns_list_of_arrays(self):
        samples = self.m.sample_many(4, seed=0)
        assert len(samples) == 4
        for s in samples:
            assert isinstance(s, np.ndarray)
            assert s.ndim == 2 and s.shape[1] == 3

    def test_no_self_loops(self):
        for arr in self.m.sample_many(3, seed=1):
            assert np.all(arr[:, 0] != arr[:, 1])


class TestDistinctPlacement:
    def test_distinct_values_in_range_including_full_and_singleton_blocks(self):
        from dcms.utils.grouped_sampling import _distinct
        rng = np.random.default_rng(0)
        n_ab = np.array([1, 1, 2, 5, 5, 40, 1000, 7])
        c = np.array([1, 1, 2, 3, 5, 30, 4, 4])
        ids, vals = _distinct(rng, n_ab, c)
        assert np.array_equal(np.bincount(ids, minlength=len(c)), c)
        for k in range(len(c)):
            v = vals[ids == k]
            assert len(set(v.tolist())) == c[k] and v.min() >= 0 and v.max() < n_ab[k]


# ---------------------------------------------------------------------------
# Grouped sampler, all four models
# ---------------------------------------------------------------------------

def _hub_network(N=400, a=1.2, mean_deg=6.0, seed=1):
    """Chung-Lu weighted digraph with heavy-tailed degrees: sparse (mean link probability < 5 %, the regime of the old Poisson
    sampler) but with many node pairs of link probability 0.2-0.95, and with degenerate low-degree nodes."""
    rng = np.random.default_rng(seed)
    w_out = 1 + rng.pareto(a, N)
    w_in = 1 + rng.pareto(a, N)
    w_out = np.minimum(w_out * mean_deg / w_out.mean(), 0.8 * N)
    w_in = np.minimum(w_in * mean_deg / w_in.mean(), 0.8 * N)
    P = np.minimum(np.outer(w_out, w_in) / w_out.sum(), 0.95)
    np.fill_diagonal(P, 0.0)
    A = (rng.random((N, N)) < P).astype(float)
    W = A * rng.geometric(0.5, (N, N))
    keep = np.ones(N, dtype=bool)
    while True:                                 # drop the nodes without out- or in-links (zero constraints)
        idx = np.flatnonzero(keep)
        sub = A[np.ix_(idx, idx)]
        bad = (sub.sum(1) == 0) | (sub.sum(0) == 0)
        if not bad.any():
            break
        keep[idx[bad]] = False
    idx = np.flatnonzero(keep)
    A, W = A[np.ix_(idx, idx)], W[np.ix_(idx, idx)]
    return A.sum(1), A.sum(0), W.sum(1), W.sum(0)


@functools.lru_cache(maxsize=None)
def _fitted(kind):
    """Fitted model and constraints, cached: the tests only read them (the fallback test works on a copy)."""
    from dcms.models.dcm import DCMModel
    from dcms.models.dwcm import DWCMModel
    from dcms.models.qdecm import qDECMModel
    from dcms.models.decm import DECMModel
    k_out, k_in, s_out, s_in = _hub_network()
    if kind == "dcm":
        m = DCMModel(k_out, k_in); m.solve_tool(tol=1e-8, max_iter=3000)
    elif kind == "dwcm":
        m = DWCMModel(s_out, s_in); m.solve_tool(tol=1e-8, max_iter=3000)
    elif kind == "qdecm":
        m = qDECMModel(k_out, k_in, s_out, s_in); m.solve_tool(tol=1e-6, max_iter=3000)
    else:
        m = DECMModel(k_out, k_in, s_out, s_in); m.solve_tool(tol=1e-6, max_iter=20000)
    assert m.sol.converged
    return m, (k_out, k_in, s_out, s_in)


KINDS = ["dcm", "dwcm", "qdecm", "decm"]


@pytest.mark.parametrize("kind", KINDS)
class TestGroupedAllModels:
    def test_format(self, kind):
        m, _ = _fitted(kind)
        cols = 2 if kind == "dcm" else 3
        edges = m.sample(seed=0)
        assert isinstance(edges, np.ndarray) and edges.shape[1] == cols
        assert np.issubdtype(edges.dtype, np.integer)
        assert np.all(edges[:, 0] != edges[:, 1])
        assert len({(a, b) for a, b in edges[:, :2]}) == len(edges)   # distinct ordered pairs
        assert np.array_equal(edges, m.sample(seed=0))
        if cols == 3:
            assert (edges[:, 2] > 0).all()

    def test_invalid_method(self, kind):
        m, _ = _fitted(kind)
        with pytest.raises(ValueError):
            m.sample(seed=0, method="nope")
        with pytest.raises(ValueError):
            m.sample_many(2, method="nope")

    def test_sample_many_methods(self, kind):
        m, _ = _fitted(kind)
        for method in ("grouped", "exact"):
            out = m.sample_many(3, seed=0, n_jobs=1, method=method)
            assert len(out) == 3 and all(isinstance(x, np.ndarray) for x in out)

    def test_falls_back_when_parameters_not_constant_in_groups(self, kind):
        import copy
        from dcms.utils.grouped_sampling import group_model
        m, _ = _fitted(kind)
        g = group_model(m)
        grp = int(np.flatnonzero(g["n_g"] > 1)[0])
        node = int(g["members"][g["start"][grp]])
        th = np.array(m.sol.best_theta, dtype=np.float64)
        th[node] += 1e-3
        m = copy.copy(m)                        # the cached model stays untouched
        m.sol = copy.copy(m.sol)
        m.sol.best_theta = th
        with pytest.warns(UserWarning, match="not constant within groups"):
            edges = m.sample(seed=0)
        assert np.array_equal(edges, m.sample(seed=0, method="exact"))

    @pytest.mark.parametrize("method", ["grouped", "exact"])
    def test_expected_totals(self, kind, method):
        """Mean number of links and total weight of the samples equal the constraints (hubs make p_ij close to 1).

        The Poisson-intensity sampler used before 0.7.0 failed this: P(link) = 1 - exp(-p) lost 4-12 % of the links/weight."""
        m, (k_out, k_in, s_out, s_in) = _fitted(kind)
        n = 1500
        samples = m.sample_many(n, seed=11, n_jobs=1, method=method)
        if kind != "dwcm":                      # the DWCM constrains strengths only
            L = np.array([len(x) for x in samples], dtype=float)
            assert abs(L.mean() - k_out.sum()) < 5 * L.std(ddof=1) / np.sqrt(n)
        if kind != "dcm":
            W = np.array([x[:, 2].sum() for x in samples], dtype=float)
            assert abs(W.mean() - s_out.sum()) < 5 * W.std(ddof=1) / np.sqrt(n)

    def test_grouped_matches_exact_per_node(self, kind):
        """Per-node mean out/in degree (strength for the DWCM) of the grouped sampler matches the constraints."""
        m, (k_out, k_in, s_out, s_in) = _fitted(kind)
        N, n = m.N, 1500
        tgt_out, tgt_in = (s_out, s_in) if kind == "dwcm" else (k_out, k_in)
        outs = np.zeros((n, N)); ins = np.zeros((n, N))
        for r, x in enumerate(m.sample_many(n, seed=5, n_jobs=1)):
            v = x[:, 2] if kind == "dwcm" else np.ones(len(x))
            outs[r] = np.bincount(x[:, 0], weights=v, minlength=N)
            ins[r] = np.bincount(x[:, 1], weights=v, minlength=N)
        for arr, tgt in ((outs, tgt_out), (ins, tgt_in)):
            se = arr.std(0, ddof=1) / np.sqrt(n)
            ok = se > 0
            z = (arr.mean(0)[ok] - tgt[ok]) / se[ok]
            assert np.abs(z).max() < 5.0
