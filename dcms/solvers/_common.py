"""Pieces shared by the four reduced solvers: Anderson mixing and the node <-> group bookkeeping of the degeneracy reduction."""
from __future__ import annotations

import math

import numpy as np
import torch

from dcms.solvers.base import SolverResult


def anderson_mixing(
    fp_outputs: list[torch.Tensor],
    residuals_hist: list[torch.Tensor],
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Anderson mixing with per-component row-weighting (PR#12).

    Solves the *per-component-weighted* constrained least-squares problem:

        min  ‖W⁻¹ Σ_i c_i r_i‖²    s.t.  Σ_i c_i = 1

    where w_j = max_i |r_i[j]| prevents hub nodes from dominating the mixing.

    Args:
        fp_outputs:     List of g(θ_k) values (FP outputs), each shape (2N,).
        residuals_hist: List of r_k = g(θ_k) − θ_k values, each shape (2N,).
        weights:        Optional per-row (per-component) multiplicity
                        weights, shape (2N,). Used by the degeneracy-reduced
                        (``mult``) path: in the unreduced system, a
                        degeneracy class of ``mult[g]`` identical nodes
                        contributes ``mult[g]`` identical rows to the
                        least-squares fit below; in the reduced system that
                        class is a single row, so without this weight the
                        fit implicitly under-counts high-multiplicity
                        classes relative to the unreduced system. ``None``
                        (default) reproduces the old unweighted behaviour
                        exactly, and passing all-ones is mathematically
                        identical to ``None``.

    Returns:
        Anderson-mixed next iterate θ_{k+1}, shape (2N,).
    """
    m = len(fp_outputs)
    if m == 1:
        return fp_outputs[0]

    R = torch.stack(residuals_hist, dim=1)  # (2N, m)
    G = torch.stack(fp_outputs, dim=1)      # (2N, m)

    w = R.abs().max(dim=1, keepdim=True).values.clamp(min=1e-15)
    R_w = R / w

    if weights is not None:
        RtR = R_w.T @ (weights[:, None] * R_w)
    else:
        RtR = R_w.T @ R_w
    RtR = RtR + 1e-10 * torch.eye(m, dtype=RtR.dtype, device=RtR.device)
    ones = torch.ones(m, dtype=RtR.dtype, device=RtR.device)
    try:
        c = torch.linalg.solve(RtR, ones)
        c_sum = c.sum().item()
        if abs(c_sum) < 1e-14 or not math.isfinite(c_sum):
            raise RuntimeError("Degenerate Anderson weights.")
        c = c / c_sum
        c = c.clamp(-10.0, 10.0)
        c_sum_clamped = c.sum().item()
        if abs(c_sum_clamped) < 1e-14 or not math.isfinite(c_sum_clamped):
            c = ones / m
        else:
            c = c / c_sum_clamped
    except RuntimeError:
        c = ones / m

    return G @ c



def compute_groups(*keys: torch.Tensor):
    """Group nodes by identical values of all ``keys`` (sequences of shape (N,)).

    Returns ``(group_of, mult, *keys_g)``: ``group_of`` (N,) long tensor mapping each node to its group in [0, M)
    (groups ordered by the lexicographic order of the key tuple), ``mult`` (M,) float64 group sizes, and the shared
    value of each key in each group, shape (M,).
    """
    N = keys[0].shape[0]
    uniq, group_of = torch.unique(torch.stack(list(keys), dim=1), dim=0, return_inverse=True)
    M = uniq.shape[0]
    mult = torch.zeros(M, dtype=torch.float64)
    mult.scatter_add_(0, group_of, torch.ones(N, dtype=torch.float64))
    return (group_of, mult, *[uniq[:, c] for c in range(len(keys))])


def first_member_index(group_of: torch.Tensor, M: int) -> torch.Tensor:
    """Index of the first node (in node order) of every group: the representative whose value stands for the group."""
    N = group_of.shape[0]
    first = torch.full((M,), N, dtype=torch.long)
    return first.scatter_reduce(0, group_of, torch.arange(N), reduce="amin", include_self=True)


def reduce_blocks(theta: torch.Tensor, first_idx: torch.Tensor, N: int, blocks: int) -> torch.Tensor:
    """Per-node parameter vector of ``blocks`` blocks of length N -> group-level vector (one representative per group)."""
    return torch.cat([theta[b * N:(b + 1) * N][first_idx] for b in range(blocks)])


def expand_result(result_g: SolverResult, group_of: torch.Tensor, N: int, M: int, blocks: int) -> SolverResult:
    """Group-level ``SolverResult`` -> the same result with ``theta``/``best_theta`` expanded to the N nodes."""
    g = group_of.numpy()

    def _expand(theta_m):
        return np.concatenate([theta_m[b * M:(b + 1) * M][g] for b in range(blocks)])

    return SolverResult(
        theta=_expand(result_g.theta),
        best_theta=_expand(result_g.best_theta),
        converged=result_g.converged,
        iterations=result_g.iterations,
        residuals=result_g.residuals,
        elapsed_time=result_g.elapsed_time,
        peak_ram_bytes=result_g.peak_ram_bytes,
        message=result_g.message + f" [degeneracy-reduced: N={N} -> M={M}]",
        best_mre=result_g.best_mre,
    )
