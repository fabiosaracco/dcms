"""Exact degeneracy-grouped samplers (``method="grouped"``, the default of ``sample``/``sample_many`` in all four models).

Nodes with identical constraints have identical fitted parameters, so for two groups a, b all ordered pairs
(i in a, j in b, i != j) are i.i.d. with the same link probability p_ab. The number of links of a group pair is therefore
Binomial(n_ab, p_ab) and, given it, the links sit on distinct uniformly chosen pairs of the block. Same distribution as the
per-pair O(N^2) sampler, at cost O(M^2 + L) with M = number of groups.

* DCM   : p = sigma(-theta_out_a - theta_in_b), no weights.
* DWCM  : p = beta_ab = exp(-theta_out_a - theta_in_b); weight = 1 + Geometric, P(w = k | link) = (1 - beta) beta^(k-1).
* qDECM : p = sigma(-theta_out_a - theta_in_b) (topology); weight geometric from 1 with beta = exp(-eta_out_a - eta_in_b).
* DECM  : p = sigma(-theta_out - theta_in - log(exp(eta_out + eta_in) - 1)); weights as in qDECM.
"""
import numpy as np


# per kind: constraint arrays that define the groups, columns of ``best_theta`` (N each) that hold the parameters, and the two
# parameter columns whose sum gives the geometric weight (None: unweighted)
_KINDS = {
    "dcm":   (("k_out", "k_in"), 2, None),
    "dwcm":  (("s_out", "s_in"), 2, (0, 1)),
    "qdecm": (("k_out", "k_in", "s_out", "s_in"), 4, (2, 3)),
    "decm":  (("k_out", "k_in", "s_out", "s_in"), 4, (2, 3)),
}


def group_model(model):
    """Group the nodes of a fitted model by identical constraints; returns the grouped parameters.

    ``model._SAMPLER_KIND`` selects the model ("dcm", "dwcm", "qdecm", "decm"). ``dev`` is the largest deviation of a node's
    parameters from its group's (0 for a solution of the degeneracy-reduced solver)."""
    kind = model._SAMPLER_KIND
    names, ncol, wcols = _KINDS[kind]
    N = model.N
    key = np.stack([_np(getattr(model, n)) for n in names], 1)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    M = inv.max() + 1
    th = np.asarray(model.sol.best_theta, dtype=np.float64)
    P = np.stack([th[c * N:(c + 1) * N] for c in range(ncol)], 1)
    order = np.argsort(inv, kind='stable')
    n_g = np.bincount(inv, minlength=M)
    start = np.concatenate([[0], np.cumsum(n_g)[:-1]])
    Pg = P[order[start]]
    dev = np.abs(P - Pg[inv]).max()
    return dict(kind=kind, wcols=wcols, M=M, n_g=n_g, start=start, members=order, Pg=Pg, dev=dev)


def _np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def _distinct(rng, n_ab, c):
    """For each block k draw c[k] distinct integers in [0, n_ab[k]); returns (block id, value) arrays."""
    ids, vals = [], []
    full = c == n_ab                       # every pair of the block is a link: no randomness (covers singleton-singleton blocks)
    fk = np.flatnonzero(full)
    if len(fk):
        cf = c[fk]
        ids.append(np.repeat(fk, cf).astype(np.int64))
        vals.append((np.arange(cf.sum()) - np.repeat(np.cumsum(cf) - cf, cf)).astype(np.int64))
    dense = (c * 2 > n_ab) & ~full
    for k in np.flatnonzero(dense):
        v = rng.choice(int(n_ab[k]), size=int(c[k]), replace=False)
        ids.append(np.full(len(v), k, dtype=np.int64)); vals.append(v.astype(np.int64))
    sp = np.flatnonzero(~dense & ~full)
    if len(sp):
        cs, ns = c[sp], n_ab[sp]
        bid = np.repeat(np.arange(len(sp)), cs)
        v = (rng.random(len(bid)) * ns[bid]).astype(np.int64)
        BIG = int(ns.max()) + 1
        while True:
            key = bid * BIG + v
            u, firstpos = np.unique(key, return_index=True)
            if len(u) == len(key):
                break
            bid, v = bid[firstpos], v[firstpos]
            missing = cs - np.bincount(bid, minlength=len(sp))
            nb = np.repeat(np.arange(len(sp)), missing)
            nv = (rng.random(len(nb)) * ns[nb]).astype(np.int64)
            bid, v = np.concatenate([bid, nb]), np.concatenate([v, nv])
        ids.append(sp[bid]); vals.append(v)
    if not ids:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    return np.concatenate(ids), np.concatenate(vals)


def _link_prob(g, a0, a1):
    """Link probabilities of the group pairs (a0:a1, all b), shape (a1 - a0, M)."""
    kind, Pg = g['kind'], g['Pg']
    to, ti = Pg[a0:a1, 0], Pg[:, 1]
    with np.errstate(over='ignore'):
        if kind == "dwcm":
            return np.exp(-(to[:, None] + ti[None, :])).clip(0.0, 1.0 - 1e-12)
        if kind == "decm":
            eta = Pg[a0:a1, 2][:, None] + Pg[:, 3][None, :]
            logq = -np.log(np.expm1(eta).clip(min=1e-300))
            return 1.0 / (1.0 + np.exp(to[:, None] + ti[None, :] - logq))
        return 1.0 / (1.0 + np.exp(to[:, None] + ti[None, :]))      # dcm, qdecm


def sample_grouped(g, rng, chunk_pairs=4_000_000):
    M, n_g, start, members, Pg, wcols = g['M'], g['n_g'], g['start'], g['members'], g['Pg'], g['wcols']
    rows = max(1, chunk_pairs // M)
    out = []
    for a0 in range(0, M, rows):
        a1 = min(a0 + rows, M)
        p = _link_prob(g, a0, a1)
        n_ab = n_g[a0:a1, None].astype(np.int64) * n_g[None, :]
        ar = np.arange(a1 - a0)
        n_ab[ar, a0 + ar] = n_g[a0 + ar] * (n_g[a0 + ar] - 1)        # no self pairs
        c = rng.binomial(n_ab, p)
        ra, rb = np.nonzero(c)
        if len(ra) == 0:
            continue
        cc, nn = c[ra, rb], n_ab[ra, rb]
        bid, idx = _distinct(rng, nn, cc)
        a, b = a0 + ra[bid], rb[bid]
        diag = a == b
        nb_ = n_g[b]
        r = np.where(diag, idx // np.maximum(n_g[a] - 1, 1), idx // nb_)
        col = np.where(diag, idx % np.maximum(n_g[a] - 1, 1), idx % nb_)
        col = col + (diag & (col >= r))
        src = members[start[a] + r]; tgt = members[start[b] + col]
        if wcols is None:
            out.append(np.column_stack([src, tgt]))
            continue
        beta = np.exp(-(Pg[a, wcols[0]] + Pg[b, wcols[1]])).clip(0.0, 1.0 - 1e-12)
        w = rng.geometric(1.0 - beta)
        out.append(np.column_stack([src, tgt, w]))
    if out:
        return np.vstack(out)
    return np.empty((0, 2 if wcols is None else 3), dtype=int)


class GroupedSamplingMixin:
    """``sample``/``sample_many`` plumbing shared by the four models.

    The model defines ``_SAMPLER_KIND`` (a key of ``_KINDS``) and ``_sample_raw`` (the exact per-pair O(N^2) sampler); the
    mixin adds the grouped sampler, its per-solution group cache and the parallel ``sample_many``."""

    def _grouped_sampler_state(self):
        """Group the nodes once per fitted solution; the group dict is ``None`` if the parameters are not constant within groups."""
        if self.sol is None:
            raise RuntimeError("Call solve_tool() first.")
        cache = getattr(self, "_grouped_cache", None)
        if cache is None or cache[0] is not self.sol:
            g = group_model(self)
            cache = (self.sol, g if g["dev"] <= 1e-8 else None, g["dev"])
            self._grouped_cache = cache
        return cache

    def _sample_grouped(self, seed=None):
        """Exact degeneracy-grouped sampler.

        Falls back to the per-pair sampler, with a warning, when the fitted parameters are not constant within
        groups of identical constraints (e.g. a ``sol`` assigned by hand or loaded from an older, unreduced solve)."""
        import warnings
        _, g, dev = self._grouped_sampler_state()
        if g is None:
            warnings.warn(
                f"Parameters are not constant within groups of identical constraints (max deviation {dev:.1e}); "
                "falling back to method='exact'.", stacklevel=3)
            return self._sample_raw(seed=seed)
        return sample_grouped(g, np.random.default_rng(seed))

    def _sample(self, seed, chunk_size, method):
        if method == "grouped":
            return self._sample_grouped(seed=seed)
        if method != "exact":
            raise ValueError(f"method must be 'exact' or 'grouped', got {method!r}")
        return self._sample_raw(seed=seed, chunk_size=chunk_size)

    def sample_many(self, n: int, seed: int | None = None, n_jobs: int = -1, method: str = "grouped") -> list:
        """Generate ``n`` independent samples in parallel.

        Each sample is a ``np.ndarray`` of shape ``(L_k, 2)`` (DCM: ``[source, target]``) or ``(L_k, 3)`` (weighted models:
        ``[source, target, weight]``). Iterating ``for i, j, w in sample`` works the same as with the array returned by
        :meth:`sample`.

        Uses :class:`~concurrent.futures.ThreadPoolExecutor`; NumPy releases the GIL during random-number generation and array
        operations, giving near-linear speedup up to the number of physical cores.

        Args:
            n: Number of independent samples to draw.
            seed: Master seed from which per-sample seeds are derived.
            n_jobs: Number of worker threads. ``-1`` (default) uses all CPUs.
            method: ``"grouped"`` (default) or ``"exact"``, see :meth:`sample`.

        Returns:
            List of ``n`` NumPy arrays, one per sample.

        Raises:
            RuntimeError: if :meth:`solve_tool` has not been called yet.
            ValueError: if ``method`` is not ``"exact"`` or ``"grouped"``.
        """
        from concurrent.futures import ThreadPoolExecutor
        if method not in ("exact", "grouped"):
            raise ValueError(f"method must be 'exact' or 'grouped', got {method!r}")
        draw = self._sample_grouped if method == "grouped" else self._sample_raw
        if method == "grouped":
            self._grouped_sampler_state()   # build the group cache before the threads start
        seeds = np.random.default_rng(seed).integers(0, 2**31, n).tolist()
        if n_jobs == 1:
            return [draw(seed=s) for s in seeds]
        max_workers = None if n_jobs < 0 else n_jobs
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            return list(ex.map(lambda s: draw(seed=s), seeds))
