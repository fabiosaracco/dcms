"""The degeneracy reduction is an identity, not an approximation: for ANY parameters that are constant within groups of identical
constraints, the residual of the reduced (group-level) system equals the residual of the original per-node system at the nodes of
each group. This is checked here at random points (not only at the solution) for the four models, with groups of multiplicity > 1.

Together with the "expanded solution solves the per-node system" tests of each model, this replaces the old comparison against
the per-node solver (removed): the per-node RESIDUALS (model.residual, independent code) remain the reference.
"""
import numpy as np
import pytest
import torch

from dcms.models.dcm import DCMModel
from dcms.models.decm import DECMModel
from dcms.models.dwcm import DWCMModel
from dcms.models.qdecm import qDECMModel


def _grouped_targets(M=7, seed=0):
    """Targets of N nodes made of M distinct (k_out, k_in, s_out, s_in) tuples repeated with multiplicity 1..4."""
    g = np.random.default_rng(seed)
    mult = g.integers(1, 5, size=M)
    mult[0] = 4                                          # make sure some group has several members
    base = np.stack([g.uniform(1.5, 6, M), g.uniform(1.5, 6, M), g.uniform(3, 20, M), g.uniform(3, 20, M)], 1)
    rows = np.repeat(base, mult, axis=0)
    perm = g.permutation(len(rows))                      # nodes of a group need not be contiguous
    rows = rows[perm]
    group_of = np.repeat(np.arange(M), mult)[perm]
    return rows, group_of, mult.astype(float)


def _expand(theta_g, group_of, blocks):
    M = theta_g.shape[0] // blocks
    return torch.cat([theta_g[b * M:(b + 1) * M][torch.as_tensor(group_of)] for b in range(blocks)])


def _check(F_group, F_node, group_of, M, blocks):
    for b in range(blocks):
        Fg = F_group[b * M:(b + 1) * M]
        Fn = F_node[b * len(group_of):(b + 1) * len(group_of)]
        assert torch.allclose(Fn, Fg[torch.as_tensor(group_of)], atol=1e-10, rtol=1e-10), f"block {b}"


def test_dcm_reduced_residual_equals_per_node_residual():
    from dcms.solvers.fixed_point_dcm import _dcm_step_dense_weighted
    rows, group_of, mult = _grouped_targets()
    M = len(mult)
    model = DCMModel(rows[:, 0], rows[:, 1])
    base = np.unique(np.stack([group_of, rows[:, 0], rows[:, 1]], 1), axis=0)          # one (k_out, k_in) per group
    k_out_g = torch.as_tensor(base[np.argsort(base[:, 0]), 1]); k_in_g = torch.as_tensor(base[np.argsort(base[:, 0]), 2])
    theta_g = torch.rand(2 * M, dtype=torch.float64, generator=torch.Generator().manual_seed(1)) * 2 - 0.5
    _, F_g = _dcm_step_dense_weighted(theta_g, k_out_g, k_in_g, 1.0, torch.as_tensor(mult))
    _check(F_g, model.residual(_expand(theta_g, group_of, 2)), group_of, M, 2)


def test_dwcm_reduced_residual_equals_per_node_residual():
    from dcms.solvers.fixed_point_dwcm import _dwcm_step_dense_weighted
    rows, group_of, mult = _grouped_targets(seed=1)
    M = len(mult)
    model = DWCMModel(rows[:, 2], rows[:, 3])
    s_out_g = torch.zeros(M, dtype=torch.float64); s_in_g = torch.zeros(M, dtype=torch.float64)
    s_out_g[torch.as_tensor(group_of)] = torch.as_tensor(rows[:, 2]); s_in_g[torch.as_tensor(group_of)] = torch.as_tensor(rows[:, 3])
    theta_g = torch.rand(2 * M, dtype=torch.float64, generator=torch.Generator().manual_seed(2)) * 1.5 + 0.3
    _, F_g = _dwcm_step_dense_weighted(theta_g, s_out_g, s_in_g, 1.0, torch.as_tensor(mult), eta_lo=1e-10)
    _check(F_g, model.residual(_expand(theta_g, group_of, 2)), group_of, M, 2)


def test_decm_reduced_residual_equals_per_node_residual():
    from dcms.solvers.fixed_point_decm import _decm_step_dense_weighted
    rows, group_of, mult = _grouped_targets(seed=2)
    M = len(mult)
    model = DECMModel(rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3])
    gi = torch.as_tensor(group_of)
    tg = []
    for c in range(4):
        v = torch.zeros(M, dtype=torch.float64); v[gi] = torch.as_tensor(rows[:, c]); tg.append(v)
    z = [t == 0 for t in tg]
    g = torch.Generator().manual_seed(3)
    theta_g = torch.cat([torch.randn(2 * M, generator=g) * 0.5, torch.rand(2 * M, generator=g) * 2 + 0.3]).double()
    _, F_g = _decm_step_dense_weighted(theta_g, *tg, *z, 1.0, torch.as_tensor(mult))
    _check(F_g, model.residual(_expand(theta_g, group_of, 4)), group_of, M, 4)


def test_qdecm_reduced_residual_equals_per_node_residual():
    """This is the test that exposed the dropped within-group pairs (2026-10-27): the group-level topology matrix must keep its
    DIAGONAL (the probability between two distinct members of the same group)."""
    from dcms.solvers.fixed_point_qdecm import _qdecm_step_dense_weighted
    rows, group_of, mult = _grouped_targets(seed=3)
    M = len(mult)
    model = qDECMModel(rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3])
    gi = torch.as_tensor(group_of)
    s_out_g = torch.zeros(M, dtype=torch.float64); s_in_g = torch.zeros(M, dtype=torch.float64)
    s_out_g[gi] = torch.as_tensor(rows[:, 2]); s_in_g[gi] = torch.as_tensor(rows[:, 3])
    g = torch.Generator().manual_seed(4)
    theta_topo_g = torch.randn(2 * M, generator=g, dtype=torch.float64) * 0.7
    theta_w_g = torch.rand(2 * M, generator=g, dtype=torch.float64) * 1.5 + 0.3
    P_g = torch.sigmoid(-theta_topo_g[:M, None] - theta_topo_g[None, M:])                # real diagonal, as in the solver
    _, F_g = _qdecm_step_dense_weighted(theta_w_g, P_g, s_out_g, s_in_g, 1.0, torch.as_tensor(mult))
    F_n = model.residual_strength(_expand(theta_topo_g, group_of, 2), _expand(theta_w_g, group_of, 2))
    _check(F_g, F_n, group_of, M, 2)
