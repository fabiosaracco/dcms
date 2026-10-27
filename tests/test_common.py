"""Shared helpers of the reduced solvers (dcms/solvers/_common.py)."""
import numpy as np
import torch

from dcms.solvers._common import anderson_mixing, compute_groups, expand_result, first_member_index, reduce_blocks
from dcms.solvers.base import SolverResult


def _keys(seed=0, N=40, M=7):
    g = np.random.default_rng(seed)
    base = g.integers(0, 5, size=(M, 2)).astype(float)
    pick = g.integers(0, M, size=N)
    return torch.as_tensor(base[pick, 0]), torch.as_tensor(base[pick, 1])


def test_compute_groups_partitions_the_nodes():
    a, b = _keys()
    group_of, mult, a_g, b_g = compute_groups(a, b)
    assert mult.sum().item() == a.shape[0]
    assert torch.equal(a_g[group_of], a) and torch.equal(b_g[group_of], b)
    pairs = set(zip(a_g.tolist(), b_g.tolist()))
    assert len(pairs) == a_g.shape[0]                       # groups are distinct


def test_first_member_index_is_the_first_node_of_each_group():
    a, b = _keys(seed=3)
    group_of, mult, _, _ = compute_groups(a, b)
    M = mult.shape[0]
    expected = [int(np.flatnonzero(group_of.numpy() == g)[0]) for g in range(M)]
    assert first_member_index(group_of, M).tolist() == expected


def test_reduce_then_expand_round_trip_on_group_constant_vectors():
    a, b = _keys(seed=5)
    group_of, mult, _, _ = compute_groups(a, b)
    N, M = a.shape[0], mult.shape[0]
    theta_g = torch.arange(3 * M, dtype=torch.float64)
    theta = torch.cat([theta_g[k * M:(k + 1) * M][group_of] for k in range(3)])       # constant within groups
    back = reduce_blocks(theta, first_member_index(group_of, M), N, 3)
    assert torch.equal(back, theta_g)
    res = SolverResult(theta=theta_g.numpy(), best_theta=theta_g.numpy(), converged=True, iterations=1, residuals=[1.0],
                       elapsed_time=0.0, peak_ram_bytes=0, message="ok")
    out = expand_result(res, group_of, N, M, 3)
    assert np.array_equal(out.best_theta, theta.numpy()) and "N=40 -> M=" in out.message


def test_anderson_mixing_returns_the_only_iterate_and_an_affine_combination():
    g = torch.Generator().manual_seed(0)
    outs = [torch.randn(6, generator=g, dtype=torch.float64) for _ in range(3)]
    thetas = [torch.randn(6, generator=g, dtype=torch.float64) for _ in range(3)]
    res = [o - t for o, t in zip(outs, thetas)]
    assert torch.equal(anderson_mixing(outs[:1], res[:1]), outs[0])
    mixed = anderson_mixing(outs, res)
    c = torch.linalg.lstsq(torch.stack(outs, 1), mixed[:, None]).solution.squeeze()
    assert abs(c.sum().item() - 1.0) < 1e-8                 # coefficients of the mix sum to one
