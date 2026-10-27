"""Exact DECM sampler exploiting degeneracy (``DECMModel.sample(method="grouped")``).

Nodes with identical constraints (k_out, k_in, s_out, s_in) have identical fitted parameters, so for two groups
a, b all ordered pairs (i in a, j in b, i != j) are i.i.d. Bernoulli(p_ab). The number of links of a group pair is
therefore Binomial(n_ab, p_ab), and, given it, the links sit on distinct uniformly chosen pairs. Weights are geometric
as in the plain sampler. Same distribution as the O(N^2) sampler, at cost O(M^2 + L) with M = number of groups.
"""
import numpy as np


def group_model(model):
    """Group the nodes of a fitted DECM by identical constraints; returns the grouped parameters."""
    N = model.N
    key = np.stack([_np(model.k_out), _np(model.k_in), _np(model.s_out), _np(model.s_in)], 1)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    M = inv.max() + 1
    th = np.asarray(model.sol.best_theta, dtype=np.float64)
    P = np.stack([th[:N], th[N:2 * N], th[2 * N:3 * N], th[3 * N:]], 1)
    order = np.argsort(inv, kind='stable')
    n_g = np.bincount(inv, minlength=M)
    start = np.concatenate([[0], np.cumsum(n_g)[:-1]])
    Pg = P[order[start]]
    dev = np.abs(P - Pg[inv]).max()
    return dict(M=M, n_g=n_g, start=start, members=order, Pg=Pg, dev=dev)


def _np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def _distinct(rng, n_ab, c):
    """For each block k draw c[k] distinct integers in [0, n_ab[k]); returns (block id, value) arrays."""
    K = len(c)
    ids, vals = [], []
    full = c == n_ab                       # every pair of the block is a link: no randomness (covers singleton-singleton blocks)
    for k in np.flatnonzero(full):
        ids.append(np.full(int(c[k]), k, dtype=np.int64)); vals.append(np.arange(int(c[k]), dtype=np.int64))
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


def sample_grouped(g, rng, chunk_pairs=4_000_000):
    M, n_g, start, members, Pg = g['M'], g['n_g'], g['start'], g['members'], g['Pg']
    to, ti, eo, ei = Pg[:, 0], Pg[:, 1], Pg[:, 2], Pg[:, 3]
    rows = max(1, chunk_pairs // M)
    out = []
    for a0 in range(0, M, rows):
        a1 = min(a0 + rows, M)
        eta = eo[a0:a1, None] + ei[None, :]
        logq = -np.log(np.expm1(eta).clip(min=1e-300))
        p = 1.0 / (1.0 + np.exp(to[a0:a1, None] + ti[None, :] - logq))
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
        beta = np.exp(-(eo[a] + ei[b])).clip(0.0, 1.0 - 1e-12)
        w = rng.geometric(1.0 - beta)
        out.append(np.column_stack([src, tgt, w]))
    return np.vstack(out) if out else np.empty((0, 3), dtype=int)
