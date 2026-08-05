"""Fixed-point iteration solver for the DECM (Directed Enhanced Configuration Model).

The DECM has 4N unknowns θ = [θ_out | θ_in | η_out | η_in] and 4N coupled
equations:

    F_k_out_i = Σ_{j≠i} p_ij − k_out_i = 0
    F_k_in_j  = Σ_{i≠j} p_ij − k_in_j  = 0
    F_s_out_i = Σ_{j≠i} p_ij · G_ij − s_out_i = 0
    F_s_in_j  = Σ_{i≠j} p_ij · G_ij − s_in_j  = 0

where:
    η_ij     = η_out_i + η_in_j
    G_ij     = −1 / expm1(−η_ij)           (weight factor)
    log_q_ij = −log(expm1(η_ij))
    logit_p_ij = −θ_out_i − θ_in_j + log_q_ij
    p_ij     = sigmoid(logit_p_ij)          (coupled connection probability)

**Solver strategy — alternating Gauss-Seidel Newton:**

Each iteration consists of two passes:

* **Pass 1** (out-group): compute all sums at the current (θ,η); update
  θ_out via a Newton step targeting F_k_out = 0, and η_out via a Newton
  step targeting F_s_out = 0, with a per-node z-floor line-search that
  keeps η_out_i + η_in_j ≥ z_floor for all significant pairs j.

* **Pass 2** (in-group): recompute col sums using the updated (θ_out, η_out);
  update θ_in via a Newton step targeting F_k_in = 0, and η_in via a
  Newton step targeting F_s_in = 0 with the same z-floor mechanism.

**Anderson acceleration** (depth 10 by default) is applied to the full 4N
vector.  An infeasibility guard rejects any Anderson mix that would push
min(η_out) + min(η_in) below the numerical floor.

For N > ``_LARGE_N_THRESHOLD`` (2000) the N×N matrices are never
materialised; row chunks of size ``_DEFAULT_CHUNK`` are used.
"""
from __future__ import annotations

import datetime
import math
import sys
import time
from typing import Callable

import torch

from dcms.models.parameters import qDECM_LARGE_N_THRESHOLD as _LARGE_N_THRESHOLD
from dcms.models.parameters import _DEFAULT_CHUNK, _ETA_MAX, _ETA_MIN
from dcms.solvers.base import SolverResult
from dcms.utils.profiling import _PeakRAMMonitor

# -------------------------------------------------------------------------
# Numerical constants (mirrors fixed_point_qdecm.py)
# -------------------------------------------------------------------------
_ANDERSON_MAX_NORM: float = 1e6
_ANDERSON_BLOWUP_FACTOR: float = 50.0
_Q_MAX: float = 0.9999

_Z_G_CLAMP: float = 1e-8
_Z_NEWTON_FLOOR: float = _Z_G_CLAMP
_Z_NEWTON_FRAC: float = 0.5

_THETA_MAX: float = 50.0
_ANDERSON_THETA_FLOOR: float = 0.1


# -------------------------------------------------------------------------
# Anderson mixing (identical to the version in fixed_point_qdecm.py)
# -------------------------------------------------------------------------

def _anderson_mixing(
    fp_outputs: list[torch.Tensor],
    residuals_hist: list[torch.Tensor],
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Anderson mixing: compute the next iterate from FP history.

    Uses per-component-weighted least-squares to compute mixing coefficients,
    preventing high-magnitude components from dominating the mix.

    Args:
        fp_outputs:     List of g(θ_k) values, each shape (M,).
        residuals_hist: List of r_k = g(θ_k) − θ_k values, each shape (M,).
        weights:        Optional per-component weight, shape (M,), used only
                        in the degeneracy-reduced path (see
                        :func:`solve_fixed_point_decm_degenerate`). The
                        mixing coefficients are found by minimising a
                        WEIGHTED norm of the combined residual instead of
                        the plain L2 norm -- needed because in the reduced
                        representation each group entry stands in for
                        ``mult[g]`` original nodes; without this weighting,
                        a group of 500 identical nodes and a singleton group
                        would count equally in the least-squares fit, unlike
                        in the unreduced (per-node) system where the group
                        of 500 implicitly contributes 500 identical rows.
                        None (default) reproduces the original unweighted
                        behaviour exactly.

    Returns:
        Anderson-mixed next iterate, shape (M,).
    """
    m = len(fp_outputs)
    if m == 1:
        return fp_outputs[0]

    R = torch.stack(residuals_hist, dim=1)   # (M, m)
    G = torch.stack(fp_outputs, dim=1)        # (M, m)

    w = R.abs().max(dim=1, keepdim=True).values.clamp(min=1e-15)
    R_w = R / w

    if weights is not None:
        RtR = R_w.T @ (weights[:, None] * R_w)
    else:
        RtR = R_w.T @ R_w
    RtR = RtR + 1e-10 * torch.eye(m, dtype=RtR.dtype)
    ones = torch.ones(m, dtype=RtR.dtype)
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


# -------------------------------------------------------------------------
# Dense DECM step (for N ≤ _LARGE_N_THRESHOLD)
# -------------------------------------------------------------------------

def _decm_step_dense(
    theta: torch.Tensor,
    k_out: torch.Tensor,
    k_in: torch.Tensor,
    s_out: torch.Tensor,
    s_in: torch.Tensor,
    zero_k_out: torch.Tensor,
    zero_k_in: torch.Tensor,
    zero_s_out: torch.Tensor,
    zero_s_in: torch.Tensor,
    max_step: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One alternating GS-Newton step for the DECM (dense N×N computation).

    Performs two passes:
    1. Update θ_out and η_out using row sums at the current state.
    2. Update θ_in and η_in using col sums at the updated (θ_out, η_out).

    The residual returned is evaluated at the *input* theta (before update).

    Args:
        theta:      Current 4N parameter vector [θ_out|θ_in|η_out|η_in].
        k_out:      Observed out-degree sequence, shape (N,).
        k_in:       Observed in-degree sequence, shape (N,).
        s_out:      Observed out-strength sequence, shape (N,).
        s_in:       Observed in-strength sequence, shape (N,).
        zero_k_out: Boolean mask of zero-out-degree nodes.
        zero_k_in:  Boolean mask of zero-in-degree nodes.
        zero_s_out: Boolean mask of zero-out-strength nodes.
        zero_s_in:  Boolean mask of zero-in-strength nodes.
        max_step:   Maximum |Δθ| per node per step.

    Returns:
        ``(theta_new, F_current)`` where F_current is the 4N residual at the
        input state.
    """
    N = k_out.shape[0]
    theta_out = theta[:N]
    theta_in = theta[N : 2 * N]
    eta_out = theta[2 * N : 3 * N]
    eta_in = theta[3 * N :]

    # ------- Pass 1: compute all sums at current state -------
    eta = eta_out[:, None] + eta_in[None, :]       # (N, N)
    eta_safe = eta.clamp(min=_Z_G_CLAMP)
    G = -1.0 / torch.expm1(-eta_safe)              # G = 1/(1-z)
    log_q = -torch.log(torch.expm1(eta_safe))
    logit_p = -theta_out[:, None] - theta_in[None, :] + log_q
    P = torch.sigmoid(logit_p)

    P.fill_diagonal_(0.0)
    G.fill_diagonal_(0.0)

    W = P * G
    pq = P * (1.0 - P)
    pq.fill_diagonal_(0.0)
    PGG1 = P * G * (G - 1.0)           # p·G²·z (= p·G·(G−1))
    PGG1.fill_diagonal_(0.0)
    CORR = pq * G.pow(2)
    CORR.fill_diagonal_(0.0)

    k_out_hat = P.sum(1)
    k_in_hat = P.sum(0)
    s_out_hat = W.sum(1)
    s_in_hat = W.sum(0)

    H_k_out = pq.sum(1).clamp(min=1e-15)
    H_s_out = (PGG1 + CORR).sum(1).clamp(min=1e-15)

    F_current = torch.cat(
        [k_out_hat - k_out, k_in_hat - k_in,
         s_out_hat - s_out, s_in_hat - s_in]
    )

    # ------- Update θ_out (Newton step on F_k_out) -------
    delta_theta_out = ((k_out_hat - k_out) / H_k_out).clamp(-max_step, max_step)
    theta_out_new = (theta_out + delta_theta_out).clamp(-_THETA_MAX, _THETA_MAX)
    theta_out_new = torch.where(zero_k_out, torch.full_like(theta_out_new, _THETA_MAX), theta_out_new)

    # ------- Update η_out (Newton step on F_s_out, with z-floor) -------
    delta_eta_out = ((s_out_hat - s_out) / H_s_out).clamp(-max_step, max_step)

    # z_min_out[i] = min over j of eta_ij (excluding diagonal)
    eta_for_min = eta_safe.clone()
    eta_for_min.fill_diagonal_(float("inf"))
    z_min_out = eta_for_min.min(1).values.clamp(min=_Z_G_CLAMP)
    # Global guard: include all non-zero-strength in-nodes
    nz_in = s_in > 0
    if nz_in.any():
        z_min_out = torch.minimum(z_min_out, eta_out + eta_in[nz_in].min())

    z_floor_out = (z_min_out * _Z_NEWTON_FRAC).clamp(min=_Z_NEWTON_FLOOR)
    available_out = (z_min_out - z_floor_out).clamp(min=0.0)
    alpha_out = torch.where(
        delta_eta_out < 0,
        (available_out / delta_eta_out.abs().clamp(min=1e-30)).clamp(max=1.0),
        torch.ones(N, dtype=torch.float64),
    )
    eta_out_new = (eta_out + alpha_out * delta_eta_out).clamp(_ETA_MIN, _ETA_MAX)
    eta_out_new = torch.where(zero_s_out, torch.full_like(eta_out_new, _ETA_MAX), eta_out_new)

    # ------- Pass 2: recompute col sums with updated θ_out, η_out -------
    eta2 = eta_out_new[:, None] + eta_in[None, :]
    eta2_safe = eta2.clamp(min=_Z_G_CLAMP)
    G2 = -1.0 / torch.expm1(-eta2_safe)
    log_q2 = -torch.log(torch.expm1(eta2_safe))
    logit_p2 = -theta_out_new[:, None] - theta_in[None, :] + log_q2
    P2 = torch.sigmoid(logit_p2)

    P2.fill_diagonal_(0.0)
    G2.fill_diagonal_(0.0)

    W2 = P2 * G2
    pq2 = P2 * (1.0 - P2)
    pq2.fill_diagonal_(0.0)
    PGG1_2 = P2 * G2 * (G2 - 1.0)
    PGG1_2.fill_diagonal_(0.0)
    CORR_2 = pq2 * G2.pow(2)
    CORR_2.fill_diagonal_(0.0)

    k_in_hat2 = P2.sum(0)
    s_in_hat2 = W2.sum(0)
    H_k_in2 = pq2.sum(0).clamp(min=1e-15)
    H_s_in2 = (PGG1_2 + CORR_2).sum(0).clamp(min=1e-15)

    # ------- Update θ_in (Newton step on F_k_in) -------
    delta_theta_in = ((k_in_hat2 - k_in) / H_k_in2).clamp(-max_step, max_step)
    theta_in_new = (theta_in + delta_theta_in).clamp(-_THETA_MAX, _THETA_MAX)
    theta_in_new = torch.where(zero_k_in, torch.full_like(theta_in_new, _THETA_MAX), theta_in_new)

    # ------- Update η_in (Newton step on F_s_in, with z-floor) -------
    delta_eta_in = ((s_in_hat2 - s_in) / H_s_in2).clamp(-max_step, max_step)

    eta2_for_min = eta2_safe.clone()
    eta2_for_min.fill_diagonal_(float("inf"))
    z_min_in = eta2_for_min.min(0).values.clamp(min=_Z_G_CLAMP)
    nz_out = s_out > 0
    if nz_out.any():
        z_min_in = torch.minimum(z_min_in, eta_out_new[nz_out].min() + eta_in)

    z_floor_in = (z_min_in * _Z_NEWTON_FRAC).clamp(min=_Z_NEWTON_FLOOR)
    available_in = (z_min_in - z_floor_in).clamp(min=0.0)
    alpha_in = torch.where(
        delta_eta_in < 0,
        (available_in / delta_eta_in.abs().clamp(min=1e-30)).clamp(max=1.0),
        torch.ones(N, dtype=torch.float64),
    )
    eta_in_new = (eta_in + alpha_in * delta_eta_in).clamp(_ETA_MIN, _ETA_MAX)
    eta_in_new = torch.where(zero_s_in, torch.full_like(eta_in_new, _ETA_MAX), eta_in_new)

    theta_new = torch.cat([theta_out_new, theta_in_new, eta_out_new, eta_in_new])
    return theta_new, F_current


# -------------------------------------------------------------------------
# Degenerate-multiplier reduction (Vallarano et al., Sci. Rep. 11:15227, 2021)
# -------------------------------------------------------------------------
# Nodes that share the exact same (k_out, s_out, k_in, s_in) 4-tuple have
# identical (theta_out, eta_out, theta_in, eta_in) at the true DECM
# solution. Proof sketch: the network log-likelihood is exactly invariant
# under simultaneously swapping two such nodes' full identities (in AND out
# roles together), and since the likelihood is strictly concave the unique
# maximizer must respect that symmetry.
#
# IMPORTANT: this requires the FULL 4-tuple to match. Matching only
# (k_out, s_out) or only (k_in, s_in) is NOT sufficient -- each node's own
# equation structurally excludes its own self-loop term, and that excluded
# term depends on the node's own (k_in, s_in) identity (for the out
# equations) or (k_out, s_out) identity (for the in equations), breaking
# exact degeneracy on either 2-tuple alone. Verified analytically (a
# label-swap argument) and confirmed with the user 2026-07-16.
#
# Reducing the N per-node unknowns to M := #unique-4-tuples group-level
# unknowns can shrink the problem substantially for real degree/strength
# distributions with a heavy low-degree bulk (e.g. one real network in
# production use: N=15168 -> M=3003, ~5x fewer unknowns and ~25x fewer
# pairwise entries per iteration).

def compute_degeneracy_groups(
    k_out: torch.Tensor,
    k_in: torch.Tensor,
    s_out: torch.Tensor,
    s_in: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group nodes by identical (k_out, k_in, s_out, s_in).

    Args:
        k_out, k_in, s_out, s_in: Observed sequences, each shape (N,).

    Returns:
        group_of:  (N,) long tensor mapping each original node to its group
            index in [0, M).
        mult:      (M,) float tensor, multiplicity (node count) of each
            group.
        k_out_g, k_in_g, s_out_g, s_in_g: (M,) float tensors, the shared
            target values for each group.
    """
    N = k_out.shape[0]
    key = torch.stack([k_out, k_in, s_out, s_in], dim=1)
    uniq, group_of = torch.unique(key, dim=0, return_inverse=True)
    M = uniq.shape[0]
    mult = torch.zeros(M, dtype=torch.float64)
    mult.scatter_add_(0, group_of, torch.ones(N, dtype=torch.float64))
    k_out_g, k_in_g, s_out_g, s_in_g = uniq[:, 0], uniq[:, 1], uniq[:, 2], uniq[:, 3]
    return group_of, mult, k_out_g, k_in_g, s_out_g, s_in_g


def _decm_step_dense_weighted(
    theta: torch.Tensor,
    k_out: torch.Tensor,
    k_in: torch.Tensor,
    s_out: torch.Tensor,
    s_in: torch.Tensor,
    zero_k_out: torch.Tensor,
    zero_k_in: torch.Tensor,
    zero_s_out: torch.Tensor,
    zero_s_in: torch.Tensor,
    max_step: float,
    mult: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One alternating GS-Newton step for the degeneracy-reduced DECM.

    Identical algebra to :func:`_decm_step_dense`, generalised so each
    group ``g`` stands in for ``mult[g]`` original nodes. The "no
    self-loop" exclusion (``P.fill_diagonal_(0.0)`` in the per-node
    version) becomes "subtract exactly one diagonal term" here, which is
    the exact generalisation: a group of ``mult[g]`` identical nodes
    contributes ``mult[g] * mult[h]`` ordered pairs to group pair ``(g,
    h)`` when ``g != h``, but only ``mult[g] * (mult[g] - 1)`` when ``g ==
    h`` (self-pairs excluded). Dividing by ``mult[g]`` to get the
    per-member equation gives ``sum_h mult[h] * P_gh - P_gg`` -- i.e. the
    full weighted row sum (diagonal included) minus one diagonal term.
    With ``mult`` all-ones (M == N, one node per group) this reduces
    exactly to :func:`_decm_step_dense`'s per-node formula.

    Args:
        theta:  Current 4M parameter vector [th_out|th_in|eta_out|eta_in],
            one entry per group.
        k_out, k_in, s_out, s_in: Group-level target sequences, shape (M,).
        zero_k_out, zero_k_in, zero_s_out, zero_s_in: Boolean masks, shape
            (M,).
        max_step: Maximum |Delta theta| per group per step.
        mult: Group multiplicities (node count per group), shape (M,).

    Returns:
        ``(theta_new, F_current)``, both shape (4M,)/(4M,); ``F_current``
        is the exact per-member residual (identical for every original
        node in a group), evaluated at the *input* theta.
    """
    M = k_out.shape[0]
    theta_out = theta[:M]
    theta_in = theta[M : 2 * M]
    eta_out = theta[2 * M : 3 * M]
    eta_in = theta[3 * M :]

    # ------- Pass 1: compute all sums at current state -------
    eta = eta_out[:, None] + eta_in[None, :]       # (M, M)
    eta_safe = eta.clamp(min=_Z_G_CLAMP)
    G = -1.0 / torch.expm1(-eta_safe)
    log_q = -torch.log(torch.expm1(eta_safe))
    logit_p = -theta_out[:, None] - theta_in[None, :] + log_q
    P = torch.sigmoid(logit_p)

    # NOTE: no fill_diagonal_ here -- P[g,g] is a real, used quantity (the
    # "would-be self-interaction" of group g), handled via the one-term
    # diagonal subtraction below rather than blanket zeroing.
    W = P * G
    pq = P * (1.0 - P)
    PGG1 = P * G * (G - 1.0)
    CORR = pq * G.pow(2)

    P_diag = P.diagonal()
    W_diag = W.diagonal()
    pq_diag = pq.diagonal()
    H_s_diag = (PGG1 + CORR).diagonal()

    k_out_hat = (P * mult[None, :]).sum(1) - P_diag
    k_in_hat = (P * mult[:, None]).sum(0) - P_diag
    s_out_hat = (W * mult[None, :]).sum(1) - W_diag
    s_in_hat = (W * mult[:, None]).sum(0) - W_diag

    H_k_out = ((pq * mult[None, :]).sum(1) - pq_diag).clamp(min=1e-15)
    H_s_out = (((PGG1 + CORR) * mult[None, :]).sum(1) - H_s_diag).clamp(min=1e-15)

    F_current = torch.cat(
        [k_out_hat - k_out, k_in_hat - k_in,
         s_out_hat - s_out, s_in_hat - s_in]
    )

    # ------- Update θ_out (Newton step on F_k_out) -------
    delta_theta_out = ((k_out_hat - k_out) / H_k_out).clamp(-max_step, max_step)
    theta_out_new = (theta_out + delta_theta_out).clamp(-_THETA_MAX, _THETA_MAX)
    theta_out_new = torch.where(zero_k_out, torch.full_like(theta_out_new, _THETA_MAX), theta_out_new)

    # ------- Update η_out (Newton step on F_s_out, with z-floor) -------
    delta_eta_out = ((s_out_hat - s_out) / H_s_out).clamp(-max_step, max_step)

    # z_min_out[g] = min over h of eta_gh, excluding the diagonal only for
    # singleton groups (mult==1: no real edge behind that entry). For
    # mult>1, the diagonal represents real edges between distinct members
    # of the same group and must stay eligible for the z-floor guard.
    eta_for_min = eta_safe.clone()
    singleton = mult <= 1.0
    if singleton.any():
        idx = singleton.nonzero(as_tuple=True)[0]
        eta_for_min[idx, idx] = float("inf")
    z_min_out = eta_for_min.min(1).values.clamp(min=_Z_G_CLAMP)
    nz_in = s_in > 0
    if nz_in.any():
        z_min_out = torch.minimum(z_min_out, eta_out + eta_in[nz_in].min())

    z_floor_out = (z_min_out * _Z_NEWTON_FRAC).clamp(min=_Z_NEWTON_FLOOR)
    available_out = (z_min_out - z_floor_out).clamp(min=0.0)
    alpha_out = torch.where(
        delta_eta_out < 0,
        (available_out / delta_eta_out.abs().clamp(min=1e-30)).clamp(max=1.0),
        torch.ones(M, dtype=torch.float64),
    )
    eta_out_new = (eta_out + alpha_out * delta_eta_out).clamp(_ETA_MIN, _ETA_MAX)
    eta_out_new = torch.where(zero_s_out, torch.full_like(eta_out_new, _ETA_MAX), eta_out_new)

    # ------- Pass 2: recompute col sums with updated θ_out, η_out -------
    eta2 = eta_out_new[:, None] + eta_in[None, :]
    eta2_safe = eta2.clamp(min=_Z_G_CLAMP)
    G2 = -1.0 / torch.expm1(-eta2_safe)
    log_q2 = -torch.log(torch.expm1(eta2_safe))
    logit_p2 = -theta_out_new[:, None] - theta_in[None, :] + log_q2
    P2 = torch.sigmoid(logit_p2)

    W2 = P2 * G2
    pq2 = P2 * (1.0 - P2)
    PGG1_2 = P2 * G2 * (G2 - 1.0)
    CORR_2 = pq2 * G2.pow(2)

    P2_diag = P2.diagonal()
    W2_diag = W2.diagonal()
    pq2_diag = pq2.diagonal()
    H_s_in2_diag = (PGG1_2 + CORR_2).diagonal()

    k_in_hat2 = (P2 * mult[:, None]).sum(0) - P2_diag
    s_in_hat2 = (W2 * mult[:, None]).sum(0) - W2_diag
    H_k_in2 = ((pq2 * mult[:, None]).sum(0) - pq2_diag).clamp(min=1e-15)
    H_s_in2 = (((PGG1_2 + CORR_2) * mult[:, None]).sum(0) - H_s_in2_diag).clamp(min=1e-15)

    # ------- Update θ_in (Newton step on F_k_in) -------
    delta_theta_in = ((k_in_hat2 - k_in) / H_k_in2).clamp(-max_step, max_step)
    theta_in_new = (theta_in + delta_theta_in).clamp(-_THETA_MAX, _THETA_MAX)
    theta_in_new = torch.where(zero_k_in, torch.full_like(theta_in_new, _THETA_MAX), theta_in_new)

    # ------- Update η_in (Newton step on F_s_in, with z-floor) -------
    delta_eta_in = ((s_in_hat2 - s_in) / H_s_in2).clamp(-max_step, max_step)

    eta2_for_min = eta2_safe.clone()
    if singleton.any():
        idx = singleton.nonzero(as_tuple=True)[0]
        eta2_for_min[idx, idx] = float("inf")
    z_min_in = eta2_for_min.min(0).values.clamp(min=_Z_G_CLAMP)
    nz_out = s_out > 0
    if nz_out.any():
        z_min_in = torch.minimum(z_min_in, eta_out_new[nz_out].min() + eta_in)

    z_floor_in = (z_min_in * _Z_NEWTON_FRAC).clamp(min=_Z_NEWTON_FLOOR)
    available_in = (z_min_in - z_floor_in).clamp(min=0.0)
    alpha_in = torch.where(
        delta_eta_in < 0,
        (available_in / delta_eta_in.abs().clamp(min=1e-30)).clamp(max=1.0),
        torch.ones(M, dtype=torch.float64),
    )
    eta_in_new = (eta_in + alpha_in * delta_eta_in).clamp(_ETA_MIN, _ETA_MAX)
    eta_in_new = torch.where(zero_s_in, torch.full_like(eta_in_new, _ETA_MAX), eta_in_new)

    theta_new = torch.cat([theta_out_new, theta_in_new, eta_out_new, eta_in_new])
    return theta_new, F_current


# -------------------------------------------------------------------------
# Chunked DECM step (for N > _LARGE_N_THRESHOLD)
# -------------------------------------------------------------------------

def _decm_step_chunked(
    theta: torch.Tensor,
    k_out: torch.Tensor,
    k_in: torch.Tensor,
    s_out: torch.Tensor,
    s_in: torch.Tensor,
    zero_k_out: torch.Tensor,
    zero_k_in: torch.Tensor,
    zero_s_out: torch.Tensor,
    zero_s_in: torch.Tensor,
    chunk_size: int,
    max_step: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunked alternating GS-Newton step for the DECM.

    Mirrors :func:`_decm_step_dense` but processes the N×N matrices in
    row chunks to keep memory usage at O(chunk_size × N).

    Pass 1 accumulates row sums (F_k_out, F_s_out, H_k_out, H_s_out) and
    preliminary col sums for the current-state residual.  Pass 2 recomputes
    col sums (F_k_in, F_s_in, H_k_in, H_s_in) using the updated θ_out and
    η_out.

    Args:
        theta:      Current 4N parameter vector.
        k_out:      Observed out-degree sequence, shape (N,).
        k_in:       Observed in-degree sequence, shape (N,).
        s_out:      Observed out-strength sequence, shape (N,).
        s_in:       Observed in-strength sequence, shape (N,).
        zero_k_out: Boolean mask of zero-out-degree nodes.
        zero_k_in:  Boolean mask of zero-in-degree nodes.
        zero_s_out: Boolean mask of zero-out-strength nodes.
        zero_s_in:  Boolean mask of zero-in-strength nodes.
        chunk_size: Rows per processing chunk.
        max_step:   Maximum |Δθ| per node per step.

    Returns:
        ``(theta_new, F_current)`` where F_current is evaluated at the input
        state.
    """
    N = k_out.shape[0]
    theta_out = theta[:N]
    theta_in = theta[N : 2 * N]
    eta_out = theta[2 * N : 3 * N]
    eta_in = theta[3 * N :]

    sig_log_thresh: float = math.log(0.5 / N) - math.log(1.0 - 0.5 / N)

    # ------------------------------------------------------------------
    # Pass 1: accumulate row sums + preliminary col sums
    # ------------------------------------------------------------------
    k_out_hat = torch.zeros(N, dtype=torch.float64)
    k_in_hat = torch.zeros(N, dtype=torch.float64)
    s_out_hat = torch.zeros(N, dtype=torch.float64)
    s_in_hat = torch.zeros(N, dtype=torch.float64)
    H_k_out = torch.zeros(N, dtype=torch.float64)
    H_s_out = torch.zeros(N, dtype=torch.float64)
    z_min_out = torch.full((N,), float("inf"), dtype=torch.float64)

    for i_start in range(0, N, chunk_size):
        i_end = min(i_start + chunk_size, N)
        chunk_len = i_end - i_start
        local_i = torch.arange(chunk_len, dtype=torch.long)
        global_j = torch.arange(i_start, i_end, dtype=torch.long)

        eta_chunk = eta_out[i_start:i_end, None] + eta_in[None, :]  # (chunk, N)
        eta_safe = eta_chunk.clamp(min=_Z_G_CLAMP)
        G_chunk = -1.0 / torch.expm1(-eta_safe)
        log_q_chunk = -torch.log(torch.expm1(eta_safe))
        logit_p_chunk = (
            -theta_out[i_start:i_end, None] - theta_in[None, :] + log_q_chunk
        )
        p_chunk = torch.sigmoid(logit_p_chunk)
        w_chunk = p_chunk * G_chunk

        p_chunk[local_i, global_j] = 0.0
        w_chunk[local_i, global_j] = 0.0
        G_chunk[local_i, global_j] = 0.0

        pq_chunk = p_chunk * (1.0 - p_chunk)
        PGG1_chunk = p_chunk * G_chunk * (G_chunk - 1.0)
        CORR_chunk = pq_chunk * G_chunk.pow(2)

        k_out_hat[i_start:i_end] = p_chunk.sum(1)
        k_in_hat += p_chunk.sum(0)
        s_out_hat[i_start:i_end] = w_chunk.sum(1)
        s_in_hat += w_chunk.sum(0)

        H_k_out[i_start:i_end] = pq_chunk.sum(1)
        H_s_out[i_start:i_end] = (PGG1_chunk + CORR_chunk).sum(1)

        # Track z_min_out[i] for line-search (significant pairs only)
        sig_mask = logit_p_chunk > sig_log_thresh
        z_sig = torch.where(sig_mask, eta_chunk, torch.full_like(eta_chunk, float("inf")))
        z_min_out[i_start:i_end] = torch.minimum(
            z_min_out[i_start:i_end], z_sig.min(1).values
        )

    # Global z-floor guard for out-direction
    nz_in = s_in > 0
    if nz_in.any():
        min_eta_in_nz = eta_in[nz_in].min()
        z_min_out = torch.minimum(z_min_out, eta_out + min_eta_in_nz)

    F_current = torch.cat(
        [k_out_hat - k_out, k_in_hat - k_in,
         s_out_hat - s_out, s_in_hat - s_in]
    )

    H_k_out = H_k_out.clamp(min=1e-15)
    H_s_out = H_s_out.clamp(min=1e-15)

    # ------- Update θ_out -------
    delta_theta_out = ((k_out_hat - k_out) / H_k_out).clamp(-max_step, max_step)
    theta_out_new = (theta_out + delta_theta_out).clamp(-_THETA_MAX, _THETA_MAX)
    theta_out_new = torch.where(
        zero_k_out, torch.full_like(theta_out_new, _THETA_MAX), theta_out_new
    )

    # ------- Update η_out (with z-floor line-search) -------
    delta_eta_out = ((s_out_hat - s_out) / H_s_out).clamp(-max_step, max_step)
    z_floor_out = (z_min_out * _Z_NEWTON_FRAC).clamp(min=_Z_NEWTON_FLOOR)
    available_out = (z_min_out - z_floor_out).clamp(min=0.0)
    alpha_out = torch.where(
        delta_eta_out < 0,
        (available_out / delta_eta_out.abs().clamp(min=1e-30)).clamp(max=1.0),
        torch.ones(N, dtype=torch.float64),
    )
    eta_out_new = (eta_out + alpha_out * delta_eta_out).clamp(_ETA_MIN, _ETA_MAX)
    eta_out_new = torch.where(
        zero_s_out, torch.full_like(eta_out_new, _ETA_MAX), eta_out_new
    )

    # ------------------------------------------------------------------
    # Pass 2: accumulate col sums using updated (θ_out_new, η_out_new)
    # ------------------------------------------------------------------
    k_in_hat2 = torch.zeros(N, dtype=torch.float64)
    s_in_hat2 = torch.zeros(N, dtype=torch.float64)
    H_k_in2 = torch.zeros(N, dtype=torch.float64)
    H_s_in2 = torch.zeros(N, dtype=torch.float64)
    z_min_in = torch.full((N,), float("inf"), dtype=torch.float64)

    for i_start in range(0, N, chunk_size):
        i_end = min(i_start + chunk_size, N)
        chunk_len = i_end - i_start
        local_i = torch.arange(chunk_len, dtype=torch.long)
        global_j = torch.arange(i_start, i_end, dtype=torch.long)

        eta2_chunk = eta_out_new[i_start:i_end, None] + eta_in[None, :]
        eta2_safe = eta2_chunk.clamp(min=_Z_G_CLAMP)
        G2_chunk = -1.0 / torch.expm1(-eta2_safe)
        log_q2_chunk = -torch.log(torch.expm1(eta2_safe))
        logit_p2_chunk = (
            -theta_out_new[i_start:i_end, None] - theta_in[None, :] + log_q2_chunk
        )
        p2_chunk = torch.sigmoid(logit_p2_chunk)
        w2_chunk = p2_chunk * G2_chunk

        p2_chunk[local_i, global_j] = 0.0
        w2_chunk[local_i, global_j] = 0.0
        G2_chunk[local_i, global_j] = 0.0

        pq2_chunk = p2_chunk * (1.0 - p2_chunk)
        PGG1_2_chunk = p2_chunk * G2_chunk * (G2_chunk - 1.0)
        CORR_2_chunk = pq2_chunk * G2_chunk.pow(2)

        k_in_hat2 += p2_chunk.sum(0)
        s_in_hat2 += w2_chunk.sum(0)
        H_k_in2 += pq2_chunk.sum(0)
        H_s_in2 += (PGG1_2_chunk + CORR_2_chunk).sum(0)

        # Track z_min_in[j] for line-search
        sig_mask2 = logit_p2_chunk > sig_log_thresh
        z2_sig = torch.where(sig_mask2, eta2_chunk, torch.full_like(eta2_chunk, float("inf")))
        z_min_in = torch.minimum(z_min_in, z2_sig.min(0).values)

    # Global z-floor guard for in-direction
    nz_out = s_out > 0
    if nz_out.any():
        min_eta_out_new_nz = eta_out_new[nz_out].min()
        z_min_in = torch.minimum(z_min_in, min_eta_out_new_nz + eta_in)

    H_k_in2 = H_k_in2.clamp(min=1e-15)
    H_s_in2 = H_s_in2.clamp(min=1e-15)

    # ------- Update θ_in -------
    delta_theta_in = ((k_in_hat2 - k_in) / H_k_in2).clamp(-max_step, max_step)
    theta_in_new = (theta_in + delta_theta_in).clamp(-_THETA_MAX, _THETA_MAX)
    theta_in_new = torch.where(
        zero_k_in, torch.full_like(theta_in_new, _THETA_MAX), theta_in_new
    )

    # ------- Update η_in (with z-floor line-search) -------
    delta_eta_in = ((s_in_hat2 - s_in) / H_s_in2).clamp(-max_step, max_step)
    z_floor_in = (z_min_in * _Z_NEWTON_FRAC).clamp(min=_Z_NEWTON_FLOOR)
    available_in = (z_min_in - z_floor_in).clamp(min=0.0)
    alpha_in = torch.where(
        delta_eta_in < 0,
        (available_in / delta_eta_in.abs().clamp(min=1e-30)).clamp(max=1.0),
        torch.ones(N, dtype=torch.float64),
    )
    eta_in_new = (eta_in + alpha_in * delta_eta_in).clamp(_ETA_MIN, _ETA_MAX)
    eta_in_new = torch.where(
        zero_s_in, torch.full_like(eta_in_new, _ETA_MAX), eta_in_new
    )

    theta_new = torch.cat([theta_out_new, theta_in_new, eta_out_new, eta_in_new])
    return theta_new, F_current


# -------------------------------------------------------------------------
# Hub bisection helper (DECM)
# -------------------------------------------------------------------------

def _bisect_hub_eta_decm(
    theta_self: float,
    theta_other: "np.ndarray",
    eta_other: "np.ndarray",
    s_target: float,
    self_idx: int,
    n_bisect: int = 60,
    mult: "np.ndarray | None" = None,
) -> float:
    """Find η such that Σ_{j≠i} p_ij(η) · G_ij(η) = s_target via bisection.

    For a hub node i (out or in) with high s/k the global Newton step tends to
    produce very small η, causing numerical instability.  This function solves
    the 1D strength equation exactly while keeping all other parameters fixed.

    The function

        f(η_i) = Σ_{j≠i} p_ij · G_ij

    where z_j = η_i + η_j^{other},  G_j = -1/expm1(-z_j),
    logit(p_ij) = -θ_i - θ_j^{other} - log(expm1(z_j))

    is strictly *decreasing* in η_i: as η_i ↑, z_j ↑, G_j → 1, p_ij → 0.
    The bracket is [_ETA_MIN, _ETA_MAX]; f → +∞ as η_i → 0⁺, f → 0 as η_i → ∞.

    Args:
        theta_self:  Current θ parameter of hub node (θ_out_i for out-hub).
        theta_other: θ parameters of the opposing side (θ_in for out-hub),
                     length N.
        eta_other:   η parameters of the opposing side (η_in for out-hub),
                     length N.
        s_target:    Observed strength to match (must be positive).
        self_idx:    Index of the hub node (to zero out the diagonal term).
        n_bisect:    Number of bisection steps.
        mult:        Optional group multiplicities, length N (degeneracy-
                     reduced path only). When given, each "other" index j
                     stands in for mult[j] original nodes, and the sum is
                     weighted accordingly with a single diagonal term
                     subtracted for self-exclusion (same "weighted sum
                     minus one diagonal entry" pattern as
                     :func:`_decm_step_dense_weighted`). None (default)
                     reproduces the original per-node behaviour exactly.

    Returns:
        Scalar η* such that f(η*) ≈ s_target.
    """
    import math as _math
    import numpy as _np

    N = len(theta_other)

    def _f(eta_val: float) -> float:
        z = eta_val + eta_other                     # (N,)
        z = _np.maximum(z, _Z_G_CLAMP)
        G = -1.0 / _np.expm1(-z)                   # G = 1/(1-e^{-z})
        log_q = -_np.log(_np.expm1(z))             # log_q = -log(e^z - 1)
        logit_p = -theta_self - theta_other + log_q
        p = 1.0 / (1.0 + _np.exp(-logit_p))        # sigmoid
        W = p * G
        if mult is not None:
            total = float((W * mult).sum()) - float(W[self_idx])
        else:
            W[self_idx] = 0.0                        # exclude diagonal
            total = float(W.sum())
        return total - s_target

    # Bracket: [_ETA_MIN, _ETA_MAX]; f is decreasing so f(lo) > 0 > f(hi)
    eta_lo = _ETA_MIN
    eta_hi = _ETA_MAX

    # If even the minimum gives f < 0, the target is unreachable → return lo
    f_lo = _f(eta_lo)
    if f_lo <= 0.0:
        return eta_lo

    for _ in range(n_bisect):
        eta_mid = 0.5 * (eta_lo + eta_hi)
        if _math.isinf(eta_mid) or _math.isnan(eta_mid):
            break
        if _f(eta_mid) > 0.0:
            eta_lo = eta_mid
        else:
            eta_hi = eta_mid

    return 0.5 * (eta_lo + eta_hi)


def _bisect_leaf_theta_decm(
    theta_other: "np.ndarray",
    eta_self: float,
    eta_other: "np.ndarray",
    k_target: float,
    self_idx: int,
    n_bisect: int = 60,
    mult: "np.ndarray | None" = None,
) -> float:
    """Find θ such that Σ_{j≠i} p_ij(θ) = k_target via bisection.

    For a leaf node i (out or in) with low k, the *relative* tolerance
    (MRE = |residual| / k_target) demands very tight *absolute* precision on
    the degree equation precisely because k_target itself is small -- e.g.
    k_target=1 needs an absolute residual below tol itself, while a hub with
    k_target=1000 tolerates an absolute residual 1000x larger for the same
    relative tolerance. The global Newton/Anderson step, which sums O(N)
    small per-edge contributions in floating point, can struggle to land
    inside that tight an absolute window. This function solves the 1D degree
    equation exactly, keeping all other parameters (including this node's
    own η) fixed -- the θ-side analogue of :func:`_bisect_hub_eta_decm`.

    The function

        f(θ_i) = Σ_{j≠i} p_ij

    where z_j = η_i + η_j^{other},  log_q_j = -log(expm1(z_j)),
    logit(p_ij) = -θ_i - θ_j^{other} + log_q_j

    is strictly *decreasing* in θ_i: as θ_i ↑, logit ↓, p_ij → 0.
    The bracket is [-_THETA_MAX, _THETA_MAX]; f → (N-1) as θ_i → -∞,
    f → 0 as θ_i → +∞.

    Args:
        theta_other: θ parameters of the opposing side (θ_in for an
                     out-leaf), length N.
        eta_self:    Current η parameter of the leaf node (η_out_i for an
                     out-leaf) -- fixed, not the bisection variable.
        eta_other:   η parameters of the opposing side (η_in for an
                     out-leaf), length N.
        k_target:    Observed degree to match (must be positive).
        self_idx:    Index of the leaf node (to zero out the diagonal term).
        n_bisect:    Number of bisection steps.
        mult:        Optional group multiplicities, length N (degeneracy-
                     reduced path only). Same "weighted sum minus one
                     diagonal entry" convention as
                     :func:`_bisect_hub_eta_decm`. None (default) reproduces
                     the original per-node behaviour exactly.

    Returns:
        Scalar θ* such that f(θ*) ≈ k_target.
    """
    import math as _math
    import numpy as _np

    z = eta_self + eta_other
    z = _np.maximum(z, _Z_G_CLAMP)
    log_q = -_np.log(_np.expm1(z))  # constant w.r.t. theta_self

    def _f(theta_val: float) -> float:
        logit_p = -theta_val - theta_other + log_q
        p = 1.0 / (1.0 + _np.exp(-logit_p))  # sigmoid
        if mult is not None:
            total = float((p * mult).sum()) - float(p[self_idx])
        else:
            p = p.copy()
            p[self_idx] = 0.0
            total = float(p.sum())
        return total - k_target

    # Bracket: [-_THETA_MAX, _THETA_MAX]; f is decreasing so f(lo) > 0 > f(hi)
    theta_lo = -_THETA_MAX
    theta_hi = _THETA_MAX

    # If even the maximum θ (most restrictive) still overshoots the target,
    # or the minimum θ can't reach it, the target is unreachable -> clamp.
    f_lo = _f(theta_lo)
    if f_lo <= 0.0:
        return theta_lo
    f_hi = _f(theta_hi)
    if f_hi >= 0.0:
        return theta_hi

    for _ in range(n_bisect):
        theta_mid = 0.5 * (theta_lo + theta_hi)
        if _math.isinf(theta_mid) or _math.isnan(theta_mid):
            break
        if _f(theta_mid) > 0.0:
            theta_lo = theta_mid
        else:
            theta_hi = theta_mid

    return 0.5 * (theta_lo + theta_hi)


def _bisect_hub_eta_decm_batch(
    hub_indices: "list[int]",
    theta_self_all: "np.ndarray",
    theta_other: "np.ndarray",
    eta_other: "np.ndarray",
    s_target_all: "np.ndarray",
    n_bisect: int = 60,
    mult: "np.ndarray | None" = None,
) -> "np.ndarray":
    """Vectorized batch version of :func:`_bisect_hub_eta_decm`: solves η for
    every node in ``hub_indices`` simultaneously (each bisection step is one
    ``(k, N)`` vectorized array op instead of ``k`` separate per-node Python
    calls), against the *same* snapshot of ``theta_other``/``eta_other``.

    Same equation, same monotonicity, same bracket as the per-node version;
    see its docstring for the derivation. Only the "unreachable" edge case
    at the low end (``f(eta_lo) <= 0``) is checked, matching the per-node
    version -- ``f(eta_hi) < 0`` always holds for a positive ``s_target``.

    Args:
        hub_indices:    Indices (into the full length-N arrays) of the hub
                         nodes to solve for, length k.
        theta_self_all: Full θ array for this side (θ_out for an out-hub
                         batch), length N -- indexed internally by
                         ``hub_indices``.
        theta_other:    θ parameters of the opposing side, length N.
        eta_other:      η parameters of the opposing side, length N.
        s_target_all:   Full strength-target array for this side, length N
                         -- indexed internally by ``hub_indices``.
        n_bisect:       Number of bisection steps.
        mult:           Optional group multiplicities, length N (see
                         :func:`_bisect_hub_eta_decm`).

    Returns:
        Array of shape ``(k,)``: η* for each node in ``hub_indices``, in the
        same order.
    """
    import numpy as _np

    idx = _np.asarray(hub_indices, dtype=_np.int64)
    k = idx.shape[0]
    rows = _np.arange(k)
    theta_self = theta_self_all[idx]        # (k,)
    s_target = s_target_all[idx]            # (k,)
    theta_other_row = theta_other[None, :]  # (1, N), broadcasts over rows

    def _f(eta_val: "_np.ndarray") -> "_np.ndarray":
        z = eta_val[:, None] + eta_other[None, :]       # (k, N)
        z = _np.maximum(z, _Z_G_CLAMP)
        G = -1.0 / _np.expm1(-z)
        log_q = -_np.log(_np.expm1(z))
        logit_p = -theta_self[:, None] - theta_other_row + log_q
        p = 1.0 / (1.0 + _np.exp(-logit_p))
        W = p * G
        if mult is not None:
            total = (W * mult[None, :]).sum(axis=1) - W[rows, idx]
        else:
            W = W.copy()
            W[rows, idx] = 0.0
            total = W.sum(axis=1)
        return total - s_target

    eta_lo = _np.full(k, _ETA_MIN)
    eta_hi = _np.full(k, _ETA_MAX)
    unreachable = _f(eta_lo) <= 0.0

    for _ in range(n_bisect):
        eta_mid = 0.5 * (eta_lo + eta_hi)
        go_lo = _f(eta_mid) > 0.0
        eta_lo = _np.where(go_lo, eta_mid, eta_lo)
        eta_hi = _np.where(go_lo, eta_hi, eta_mid)

    result = 0.5 * (eta_lo + eta_hi)
    return _np.where(unreachable, _ETA_MIN, result)


def _bisect_leaf_theta_decm_batch(
    leaf_indices: "list[int]",
    theta_other: "np.ndarray",
    eta_self_all: "np.ndarray",
    eta_other: "np.ndarray",
    k_target_all: "np.ndarray",
    n_bisect: int = 60,
    mult: "np.ndarray | None" = None,
) -> "np.ndarray":
    """Vectorized batch version of :func:`_bisect_leaf_theta_decm`: solves θ
    for every node in ``leaf_indices`` simultaneously. See
    :func:`_bisect_hub_eta_decm_batch`'s docstring for the general shape of
    this transformation; see :func:`_bisect_leaf_theta_decm` for the
    per-node equation being solved.

    Args:
        leaf_indices:   Indices (into the full length-N arrays) of the leaf
                         nodes to solve for, length k.
        theta_other:    θ parameters of the opposing side, length N.
        eta_self_all:   Full η array for this side (η_out for an out-leaf
                         batch), length N -- indexed internally by
                         ``leaf_indices``.
        eta_other:      η parameters of the opposing side, length N.
        k_target_all:   Full degree-target array for this side, length N --
                         indexed internally by ``leaf_indices``.
        n_bisect:       Number of bisection steps.
        mult:           Optional group multiplicities, length N (see
                         :func:`_bisect_leaf_theta_decm`).

    Returns:
        Array of shape ``(k,)``: θ* for each node in ``leaf_indices``, in
        the same order.
    """
    import numpy as _np

    idx = _np.asarray(leaf_indices, dtype=_np.int64)
    k = idx.shape[0]
    rows = _np.arange(k)
    eta_self = eta_self_all[idx]            # (k,)
    k_target = k_target_all[idx]            # (k,)

    z = eta_self[:, None] + eta_other[None, :]  # (k, N)
    z = _np.maximum(z, _Z_G_CLAMP)
    log_q = -_np.log(_np.expm1(z))              # (k, N), constant w.r.t. θ
    theta_other_row = theta_other[None, :]

    def _f(theta_val: "_np.ndarray") -> "_np.ndarray":
        logit_p = -theta_val[:, None] - theta_other_row + log_q
        p = 1.0 / (1.0 + _np.exp(-logit_p))
        if mult is not None:
            total = (p * mult[None, :]).sum(axis=1) - p[rows, idx]
        else:
            p = p.copy()
            p[rows, idx] = 0.0
            total = p.sum(axis=1)
        return total - k_target

    theta_lo = _np.full(k, -_THETA_MAX)
    theta_hi = _np.full(k, _THETA_MAX)
    unreachable_lo = _f(theta_lo) <= 0.0
    unreachable_hi = _f(theta_hi) >= 0.0

    for _ in range(n_bisect):
        theta_mid = 0.5 * (theta_lo + theta_hi)
        go_lo = _f(theta_mid) > 0.0
        theta_lo = _np.where(go_lo, theta_mid, theta_lo)
        theta_hi = _np.where(go_lo, theta_hi, theta_mid)

    result = 0.5 * (theta_lo + theta_hi)
    result = _np.where(unreachable_lo, -_THETA_MAX, result)
    return _np.where(unreachable_hi, _THETA_MAX, result)


# -------------------------------------------------------------------------
# Main solver
# -------------------------------------------------------------------------

def solve_fixed_point_decm(
    residual_fn: Callable[[torch.Tensor], torch.Tensor],
    theta0: torch.Tensor,
    k_out: torch.Tensor,
    k_in: torch.Tensor,
    s_out: torch.Tensor,
    s_in: torch.Tensor,
    tol: float = 1e-8,
    max_iter: int = 5000,
    variant: str = "theta-newton",
    chunk_size: int = 0,
    anderson_depth: int = 10,
    max_step: float = 0.5,
    max_time: float = 0.0,
    backend: str = "auto",
    num_threads: int = 0,
    verbose: bool = False,
    monitor: bool = False,
    topo_weig: bool = False,
    hub_sk_threshold: float = 0.0,
    leaf_k_threshold: float = 0.0,
    backtracking_gamma: float = 0.0,
    z_clamp: float = 1e-8,
    mult: torch.Tensor | None = None,
    weight_anderson: bool = True,
    init_best_theta: torch.Tensor | None = None,
    init_best_res: float = float("inf"),
    blowup_factor: float | None = None,
    patience: int = 750,
    noise_base: float = 1e-2,
    noise_cap_mult: float = 16.0,
    noise_growth: float = 2.0,
    max_stalls: int = 5,
    seed: int | None = None,
) -> SolverResult:
    """Alternating GS-Newton fixed-point solver for the DECM.

    Each iteration applies a 2-pass GS-Newton step: pass 1 updates
    (θ_out, η_out) from row sums; pass 2 updates (θ_in, η_in) from col
    sums using the freshly updated values.

    Anderson acceleration (depth ``anderson_depth``) is applied to the full
    4N vector with a feasibility guard that rejects infeasible mixes.

    Args:
        residual_fn:    Function F(θ) → 4N residual tensor. Used for
                        convergence bookkeeping only (residual is computed
                        inside the step function for efficiency).
        theta0:         Initial guess [θ_out|θ_in|η_out|η_in], shape (4N,).
        k_out:          Observed out-degree sequence, shape (N,).
        k_in:           Observed in-degree sequence, shape (N,).
        s_out:          Observed out-strength sequence, shape (N,).
        s_in:           Observed in-strength sequence, shape (N,).
        tol:            Convergence tolerance on the ℓ∞ relative residual (MRE = max|F_i|/target_i).
        max_iter:       Maximum number of iterations.
        variant:        Solver variant: only ``"theta-newton"`` is supported.
        chunk_size:     If > 0, use chunked computation with this row-chunk
                        size.  If 0, auto-select: dense for
                        N ≤ ``_LARGE_N_THRESHOLD``, chunked otherwise.
        anderson_depth: Anderson acceleration depth (0 = plain Newton).
        max_step:       Maximum |Δ| per node per Newton step.
        max_time:       Wall-clock time limit in seconds (0 = no limit).
        backend:        Compute backend: ``"auto"`` (default), ``"pytorch"``,
                        or ``"numba"``.  ``"auto"`` uses PyTorch for N ≤ 5 000
                        and Numba for larger networks.  Falls back automatically
                        with a warning if the requested backend is unavailable.
        num_threads:    Number of Numba parallel threads.  0 (default) leaves
                        the global Numba thread count unchanged.  Only takes
                        effect when ``backend="numba"`` (or ``"auto"`` at large N).
        verbose:        If ``True``, print a progress line at every iteration
                        showing timestamp, iteration count, elapsed time, and MRE.
        monitor:        If ``True`` (and ``verbose=True``), overwrite the same
                        terminal line at each iteration (``end='\\r'``) so only
                        the latest status is visible.  Default=False.
        topo_weig:      If ``True`` (and ``verbose=True``), the per-iteration
                        progress line splits the MRE into ``MRE_topo``
                        (degree part) and ``MRE_weights`` (strength part).
                        If ``False`` (default), it prints a single global
                        ``MRE`` (the same ℓ∞ relative residual used for the
                        ``tol`` check) instead.
        hub_sk_threshold: When > 0, nodes whose observed strength-to-degree
                        ratio ``s_i / k_hat_i`` exceeds this value are treated
                        as *hub nodes* and solved exactly each iteration via 1D
                        bisection over η rather than the global Newton step.
                        Use this for networks with a few nodes that have very
                        high ``s/k`` ratios (e.g. ``s/k > 5``) which cause the
                        global solver to stagnate.  Default=0.0 (disabled).
        leaf_k_threshold: When > 0, nodes whose observed degree ``k_out_i``
                        (resp. ``k_in_i``) is ``<= leaf_k_threshold`` are
                        treated as *leaf nodes* and have ``θ_out_i`` (resp.
                        ``θ_in_i``) solved exactly each iteration via 1D
                        bisection over the degree equation, instead of the
                        global Newton/Anderson step -- the θ-side analogue of
                        ``hub_sk_threshold``. Low-degree nodes need very tight
                        *absolute* precision to satisfy the same *relative*
                        tolerance (MRE = |residual|/k_target, and k_target is
                        small), which the global step's floating-point
                        summation can struggle to hit reliably. Use this for
                        networks that stagnate just above ``tol`` with the
                        worst offenders concentrated at low-degree nodes
                        (e.g. ``k=1``). Default=0.0 (disabled). See
                        :func:`_bisect_leaf_theta_decm` for the exact
                        equation solved.
                        Bisection is applied once per iteration, to the
                        *final* ``theta_next`` (after Anderson mixing/
                        restart/rollback have all settled it), not to the
                        pre-mixing proposal -- this is what makes it
                        self-consistent: the exact fit is against the
                        opposing values that actually carry into the next
                        iteration, not a snapshot that then drifts during
                        mixing. (An earlier pre-mixing + "freeze-restore"
                        design, mirroring how ``hub_sk_threshold`` alone
                        still works internally, was found to sometimes
                        prevent convergence outright for leaf nodes -- large
                        hub targets make the same imprecision invisible
                        there, but small leaf targets don't.) Vectorized
                        across all leaf nodes at once (see
                        :func:`_bisect_leaf_theta_decm_batch`); still has
                        some per-iteration overhead relative to the general
                        step, roughly proportional to the number of leaf
                        nodes/groups, so it's worth checking on a real
                        instance before assuming it's free.
        backtracking_gamma (float): When > 0, after each Newton step a
                        backtracking line search is applied: the residual at
                        the proposed ``theta_fp`` is evaluated, and if it
                        exceeds ``backtracking_gamma × res_norm`` (residual at
                        the current iterate), the step is halved up to 5 times
                        until the condition is met.  Typical value: ``2.0``.
                        Default 0 (disabled).  Anderson history is cleared
                        whenever a dampened step is accepted.
        z_clamp:        Floor applied to the raw z=eta_out+eta_in sum in two
                        places, both of which govern the same trade-off
                        between damping near-degenerate-hub curvature
                        blowups and preserving the solver's gradual
                        z-approach self-escape: (1) it floors ``eta`` before
                        computing ``G = -1/expm1(-z)`` in every objective/
                        step evaluation, which damps the curvature blowup
                        as z -> 0 for near-degenerate hub pairs; (2) it is
                        also the hard floor of the per-step trust-region
                        limiter (``z_floor = max(z_min * 0.5, z_clamp)``)
                        that governs how far eta can approach z=0 in ONE
                        iteration -- raising it too far can wall off pairs
                        whose true z* sits below the new floor, creating an
                        artificial plateau (confirmed via a controlled A/B
                        on ita_election_dico3: decoupling this into two
                        independent module constants was tried and made
                        things *worse*, not better, so both roles are kept
                        tied to a single value here by design). Default
                        1e-8 (the long-standing value, originally tuned for
                        a different solver's z->0 deadlock and reused here
                        without re-derivation) leaves existing behaviour
                        unchanged. Raising it (e.g. to 1e-6) resolved a
                        real stagnation on a hub-heavy N=28156 network
                        (ita_election_dico3) that would not converge at the
                        default; there is currently no general rule for
                        picking a value beyond "try raising it if the
                        solver stagnates on a network with extreme s/k
                        hubs" -- see the DECM z-clamp investigation notes
                        for the full experimental history.
        mult:           Internal use by :func:`solve_fixed_point_decm_degenerate`.
                        When provided (shape (M,)), ``k_out``/``k_in``/``s_out``/
                        ``s_in``/``theta0`` are interpreted as *group-level*
                        quantities (one entry per unique (k_out,s_out,k_in,s_in)
                        4-tuple) and the step uses :func:`_decm_step_dense_weighted`
                        instead of the per-node dense/chunked/Numba paths.
                        ``hub_sk_threshold`` and ``leaf_k_threshold`` operate
                        at group granularity when ``mult`` is given (both
                        bisection helpers accept it). Not compatible with
                        ``backtracking_gamma`` (not yet implemented for the
                        reduced case) or ``backend="numba"``.  Default None
                        (standard per-node behaviour, unaffected).
        init_best_theta: Optional externally-supplied "best theta ever seen"
                        (same shape as ``theta0``), used to seed this call's
                        internal best-tracking and Anderson-blowup reference
                        instead of starting from scratch. Needed for
                        checkpointed multi-chunk runs: each chunk is normally
                        a *fresh* call with no memory of the record set by an
                        earlier chunk, so its own in-call blowup-rollback can
                        only ever fall back to a chunk-local best -- which
                        may itself already be far worse than the true
                        historical record. Passing the true record here lets
                        the in-call rollback (and the returned
                        ``best_theta``/``mre``) stay anchored to it across
                        chunk boundaries. Default None (old behaviour:
                        start from ``theta0`` with no prior record).
        init_best_res:  The residual (MRE) achieved by ``init_best_theta``.
                        Required (and meaningful) only together with
                        ``init_best_theta``. Default ``inf``.
        blowup_factor:  Multiplicative threshold for the Anderson blowup
                        guard: a rollback triggers when the current
                        iteration's residual exceeds ``blowup_factor`` times
                        the best residual ever seen this call (a *running
                        minimum*, so this also catches a slow multi-hundred-
                        iteration drift away from the record, not just a
                        sudden single-step spike). ``None`` (default) uses
                        the built-in scale-adaptive formula
                        ``max(200, min(5000, 200_000/N))`` -- generous for
                        small N, strict for large N. Lower this (e.g. 20-50)
                        on instances that visibly wander far from their best
                        point over many iterations without any single jump
                        large enough to trip the default floor; this trades
                        more frequent, cheap (noise-free) rollbacks for less
                        wasted exploration of bad regions. A float here
                        replaces the N-adaptive formula outright, applied as
                        given regardless of N.
        patience:       Stagnation-recovery trigger, in iterations. If
                        ``best_theta_res`` has not improved for ``patience``
                        consecutive iterations, the solver no longer just
                        continues from the (stuck) current iterate. The
                        FIRST such stall since the last record gets the same
                        cheap, noise-free fix as an isolated blowup (clear
                        Anderson history, plain Newton step from
                        ``best_theta``) -- a stagnant Anderson mix can look
                        identical to a genuine attracting fixed point, and
                        this is free to try before assuming the latter. Only
                        if that ALSO fails to beat the record within another
                        ``patience`` window does the solver escalate to
                        restarting from ``best_theta`` plus Gaussian noise --
                        this is what actually resolves the quasi-fixed-point
                        traps that plain Anderson/Newton alone cannot escape
                        on hard hub-heavy instances. The same escalation path
                        is shared with the Anderson blowup guard (see
                        ``blowup_factor`` above): it also starts with the
                        cheap no-noise rollback, and only escalates to a
                        perturbed restart if the blowup trips *repeatedly*
                        with no record improvement in between. Set
                        ``patience <= 0`` to disable and restore the old
                        behaviour (stop immediately instead of retrying).
                        Default 750, the value validated end-to-end on a hard
                        N=15168 instance (see README "Checkpointed
                        multi-chunk runs") -- validated before the soft-reset
                        first attempt was added, so a stall may now resolve
                        without ever reaching a perturbed restart at all.
        noise_base:     Scale of the *multiplicative* (log-scale) Gaussian
                        noise applied to ``best_theta`` on the *first*
                        perturbed restart: every component of
                        [θ_out|θ_in|η_out|η_in] is scaled as
                        ``x_i *= exp(N(0, noise_base))``, so the perturbation
                        is relative to each parameter's own current
                        magnitude rather than one fixed absolute scale. This
                        matters on hub-heavy networks in both halves of
                        theta: a topological hub's |θ_out_i| can sit near
                        the ±``_THETA_MAX`` boundary, where fixed-scale
                        additive noise is too weak to matter, while a
                        strength hub's η_i can sit near ``_ETA_MIN``, where
                        the model is extremely sensitive to *absolute*
                        changes (``G ~= 1/η``, ``log_q ~= -log(η)``) and the
                        same fixed-scale additive noise is catastrophic --
                        observed on crisi_dico2/ita_election_dico3
                        (2026-07-24): MRE_weights jumping to ~1e5 on the very
                        first post-restart step under additive noise.
                        Default 1e-2.
        noise_cap_mult: Each consecutive restart that fails to improve the
                        record grows the noise scale by ``noise_growth``
                        (``noise_base * min(noise_growth**(restarts-1),
                        noise_cap_mult)``), so repeated failures try
                        progressively bigger kicks instead of looping on the
                        same (ineffective) perturbation size -- capped at
                        ``noise_base * noise_cap_mult`` so it never
                        degenerates into an unbounded random restart. The
                        escalation counter resets to 0 only on a genuine
                        record improvement. Default 16.0.
        noise_growth:   Per-restart multiplicative growth rate of the noise
                        scale (see ``noise_cap_mult``). Default 2.0
                        (doubling). Lower this (e.g. 1.2-1.5) on instances
                        where escalating noise makes each successive
                        recovery attempt land *further* from the record
                        instead of closer -- a sign the trap's basin is
                        being overshot rather than escaped, observed on
                        crisi_dico2 (2026-07-24): with the default doubling,
                        every restart's post-recovery residual was worse
                        than the last (3.6e-4 -> 4.0e-4 -> ... -> 1.4e-3),
                        never beating the pre-restart record.
        max_stalls:     Give up (stop, ``converged=False``) after this many
                        restarts *at the noise cap* in a row without
                        improving the record -- i.e. only once the strongest
                        available kick has been tried ``max_stalls`` times
                        and still hasn't helped. Restarts made while still
                        escalating (noise below the cap) don't count toward
                        this limit. Default 5.
        seed:           Seed for the perturbation RNG (a private
                        ``numpy.random.default_rng``, does not touch global
                        RNG state). ``None`` (default) means unseeded/
                        non-reproducible -- only matters for instances that
                        actually hit a perturbed restart; well-behaved
                        instances that never stagnate or blow up are
                        unaffected and fully deterministic regardless.

    Returns:
        :class:`~src.solvers.base.SolverResult` with the best iterate found.
    """
    if variant not in ("theta-newton",):
        raise ValueError(
            f"Unknown variant {variant!r}. "
            "Supported: 'theta-newton'."
        )
    if chunk_size < 0:
        raise ValueError(f"chunk_size must be ≥ 0 (0 = auto), got {chunk_size}")

    global _Z_G_CLAMP, _Z_NEWTON_FLOOR
    _Z_G_CLAMP = z_clamp
    _Z_NEWTON_FLOOR = z_clamp

    if mult is not None:
        if backtracking_gamma > 0.0:
            raise NotImplementedError(
                "backtracking_gamma is not yet supported together with the "
                "degeneracy-reduced (mult) path."
            )
        if backend == "numba":
            raise NotImplementedError(
                "backend='numba' is not yet supported together with the "
                "degeneracy-reduced (mult) path; use 'pytorch' or 'auto'."
            )

    # Convert inputs
    def _t(x):
        if isinstance(x, torch.Tensor):
            return x.to(dtype=torch.float64)
        return torch.tensor(x, dtype=torch.float64)

    k_out = _t(k_out)
    k_in = _t(k_in)
    s_out = _t(s_out)
    s_in = _t(s_in)
    theta = _t(theta0).clone()

    N = k_out.shape[0]

    zero_k_out = k_out == 0
    zero_k_in = k_in == 0
    zero_s_out = s_out == 0
    zero_s_in = s_in == 0

    # Restore fixed nodes in the initial theta
    _tmax = torch.full((N,), _THETA_MAX, dtype=torch.float64)
    _emax = torch.full((N,), _ETA_MAX, dtype=torch.float64)
    theta[:N] = torch.where(zero_k_out, _tmax, theta[:N])
    theta[N:2*N] = torch.where(zero_k_in, _tmax, theta[N:2*N])
    theta[2*N:3*N] = torch.where(zero_s_out, _emax, theta[2*N:3*N])
    theta[3*N:] = torch.where(zero_s_in, _emax, theta[3*N:])

    # Clamp to valid range
    theta[:2*N] = theta[:2*N].clamp(-_THETA_MAX, _THETA_MAX)
    theta[2*N:] = theta[2*N:].clamp(_ETA_MIN, _ETA_MAX)

    # Resolve compute backend
    from dcms.utils.backend import resolve_backend
    _backend = resolve_backend(backend, N)
    _use_numba = (_backend == "numba")
    _prev_numba_threads: int | None = None
    if _use_numba:
        import numpy as np
        from dcms.solvers._numba_kernels import _decm_step_numba
        from dcms.utils.backend import resolve_num_threads as _rnt
        _safe_threads = _rnt(num_threads)
        import numba as _numba_mod
        _prev_numba_threads = _numba_mod.get_num_threads()
        _numba_mod.set_num_threads(_safe_threads)

    # Decide chunked vs dense (PyTorch path only)
    if chunk_size == 0:
        effective_chunk = 0 if N <= _LARGE_N_THRESHOLD else _DEFAULT_CHUNK
    else:
        effective_chunk = chunk_size

    # Step function with bound arguments
    if mult is not None:
        def _step(th):
            return _decm_step_dense_weighted(
                th, k_out, k_in, s_out, s_in,
                zero_k_out, zero_k_in, zero_s_out, zero_s_in,
                max_step, mult,
            )
    elif _use_numba and variant == "theta-newton":
        def _step(th):
            to_ = th[:N].numpy()
            ti_ = th[N:2*N].numpy()
            eo_ = th[2*N:3*N].numpy()
            ei_ = th[3*N:].numpy()
            result = _decm_step_numba(
                to_, ti_, eo_, ei_,
                k_out.numpy(), k_in.numpy(),
                s_out.numpy(), s_in.numpy(),
                zero_k_out.numpy(), zero_k_in.numpy(),
                zero_s_out.numpy(), zero_s_in.numpy(),
                max_step, _THETA_MAX, _ETA_MAX,
                _Z_G_CLAMP, _Z_NEWTON_FLOOR, _Z_NEWTON_FRAC,
            )
            theta_new = torch.from_numpy(
                np.concatenate([result[0], result[1], result[2], result[3]])
            )
            F_current = torch.from_numpy(
                np.concatenate([result[4], result[5], result[6], result[7]])
            )
            return theta_new, F_current
    if mult is not None:
        pass  # _step already bound to _decm_step_dense_weighted above
    elif effective_chunk > 0:
        def _step(th):
            return _decm_step_chunked(
                th, k_out, k_in, s_out, s_in,
                zero_k_out, zero_k_in, zero_s_out, zero_s_in,
                effective_chunk, max_step,
            )
    else:
        def _step(th):
            return _decm_step_dense(
                th, k_out, k_in, s_out, s_in,
                zero_k_out, zero_k_in, zero_s_out, zero_s_in,
                max_step,
            )

    # Anderson mixing needs per-component weights in the degeneracy-reduced
    # path: each of the 4 blocks [theta_out|theta_in|eta_out|eta_in] has one
    # entry per group, and a group of multiplicity mult[g] should count
    # mult[g] times in the mixing least-squares fit -- exactly like it would
    # if those mult[g] identical nodes were each their own row in the
    # unreduced (per-node) system (see _anderson_mixing's docstring).
    _anderson_weights = (
        torch.cat([mult, mult, mult, mult]) if (mult is not None and weight_anderson) else None
    )

    _peak_ram_monitor = _PeakRAMMonitor()
    _peak_ram_monitor.__enter__()
    t0 = time.perf_counter()

    # Scale-adaptive Anderson blowup: generous for tiny networks (where wild
    # Anderson excursions can find good iterates), strict for large ones
    # (where residual cascades corrupt history for hundreds of iterations).
    # EXPERIMENT (crisi_dico2 quasi-fixed-point trap, 2026-07-16): raising
    # the floor 50 -> 200 lets instabilities run longer before rollback,
    # giving the trajectory more room to escape a quasi-fixed point instead
    # of always being pulled back to the same one. Real but modest effect
    # (~1.8x faster net progress, one genuine escape to a better plateau
    # observed at N=15168) -- best of the options tried so far (a
    # recent-anchor rollback, tried as an alternative, was worse: see git
    # history / RESUME.md). Kept at 200 pending a better idea.
    # Overridable via `blowup_factor` (see its docstring) for instances that
    # drift far from their best point without ever tripping this default.
    eff_blowup = (
        blowup_factor if blowup_factor is not None
        else max(200.0, min(5000.0, 200_000.0 / N))
    )

    n_iter = 0
    residuals: list[float] = []
    converged = False
    message = "Maximum iterations reached without convergence."

    if init_best_theta is not None:
        best_theta: torch.Tensor = _t(init_best_theta).clone()
        best_theta_res: float = float(init_best_res)
    else:
        best_theta = theta.clone()
        best_theta_res = float("inf")

    # ------------------------------------------------------------------
    # Stagnation/blowup perturbed-restart state (see `patience` docstring).
    # `restarts`/`stalls_at_cap` are shared between the patience-triggered
    # path and the repeated-blowup path -- this is deliberate: an earlier,
    # external version of this mechanism kept them separate (one fixed,
    # non-escalating noise scale for blowup-retries, one escalating counter
    # for stagnation-retries) and that combination produced a genuine
    # infinite loop (59 consecutive chunks converging on the exact same
    # bit-identical stuck point) on the crisi_dico2 validation run. A single
    # shared, monotonically-escalating counter that resets only on a real
    # record improvement is what actually fixed it.
    # ------------------------------------------------------------------
    import numpy as _np

    _restart_rng = _np.random.default_rng(seed)
    restarts = 0
    stalls_at_cap = 0
    iters_since_improve = 0
    # True until a patience stall has already been given one no-noise
    # Anderson-reset attempt since the last genuine improvement. Mirrors
    # the isolated-vs-repeated blowup escalation: the first stall gets the
    # cheap fix (clear Anderson history, plain Newton step from
    # best_theta, no noise); only if that ALSO fails to beat the record
    # within another `patience` window do we escalate to a perturbed
    # (noisy) restart.
    _soft_stall_reset_tried = False
    # True until the first blowup/stagnation intervention; reset back to
    # True by a genuine record improvement *or* by a restart that actually
    # fires (a restart is itself a fresh start -- see the two call sites of
    # _perturbed_restart() below). Only a blowup encountered while this is
    # already False (i.e. no record beaten and no restart fired since the
    # last time it went False) counts as "repeated" and escalates.
    progressed_since_restart = True
    # Set True whenever theta_next comes from _perturbed_restart(); consumed
    # on the *next* iteration to (a) skip the blowup check against the
    # deliberately-large post-restart residual, and (b) re-anchor
    # `_best_res_for_anderson` to that residual instead of the pre-restart
    # historical record. Without this, the blowup guard (calibrated for
    # small steady-state Anderson excursions, not intentional exploration)
    # immediately re-fires on the noisy restart point itself -- observed on
    # crisi_dico2, 2026-07-24: two real Newton steps brought the residual
    # down from 0.397 to 0.060 post-restart (genuine recovery in progress),
    # but the guard still judged 0.060 against the ancient 3.4e-4 record and
    # killed the recovery, escalating to a bigger-noise restart instead.
    _post_restart_reset = False

    def _perturbed_restart() -> tuple[torch.Tensor, bool]:
        """Build a noisy restart around `best_theta`; escalate + maybe give up."""
        nonlocal restarts, stalls_at_cap
        restarts += 1
        noise_mult = min(noise_growth ** (restarts - 1), noise_cap_mult)
        give_up = False
        if noise_mult >= noise_cap_mult:
            stalls_at_cap += 1
            give_up = stalls_at_cap >= max_stalls
        noise_scale = noise_base * noise_mult
        noise = torch.from_numpy(
            _restart_rng.normal(scale=noise_scale, size=tuple(best_theta.shape))
        )
        # Both theta and eta get *multiplicative* (log-scale) noise --
        # theta_i *= exp(noise_i) -- rather than fixed-scale additive noise.
        # Additive noise of one absolute scale is a poor fit for either
        # block on hub-heavy networks: a topological hub's |theta_out_i| can
        # sit near the +-_THETA_MAX boundary, where a small additive kick is
        # too weak to do anything useful, while a strength hub's eta_i can
        # sit near _ETA_MIN, where the same small additive kick is
        # catastrophic -- G = -1/expm1(-eta) ~= 1/eta and log_q ~=
        # -log(eta) are both wildly sensitive to *absolute* changes when eta
        # is small (observed on crisi_dico2/ita_election_dico3, 2026-07-24:
        # MRE_weights jumping to 1e5 on the very first post-restart step).
        # Scaling the noise by each component's own current magnitude keeps
        # the *relative* perturbation comparable across boundary and
        # interior nodes alike, for both theta and eta.
        theta_restart = best_theta * torch.exp(noise)
        theta_restart[: 2 * N] = theta_restart[: 2 * N].clamp(-_THETA_MAX, _THETA_MAX)
        theta_restart[2 * N :] = theta_restart[2 * N :].clamp(_ETA_MIN, _ETA_MAX)
        if verbose:
            print(
                f"[perturbed-restart] restart #{restarts} (noise_scale={noise_scale:.1e}) "
                f"around best={best_theta_res:.3e}."
            )
        return theta_restart, give_up

    _and_g: list[torch.Tensor] = []
    _and_r: list[torch.Tensor] = []
    # Seeded from the same external record (when given) so the Anderson
    # blowup guard compares against the true cross-chunk best, not just
    # whatever this chunk has seen so far (see init_best_theta docstring).
    _best_res_for_anderson: float = best_theta_res

    # EXPERIMENT (crisi_dico2 quasi-fixed-point trap, option 2, 2026-07-16):
    # tried rolling back to a "recent" anchor (theta from _ROLLBACK_LOOKBACK
    # iterations ago) instead of the global best_theta, hoping to avoid
    # always snapping back to the same unstable fixed point. Result was
    # WORSE: a 15-iteration lookback isn't long enough to guarantee a clean
    # pre-blowup anchor when the instability itself takes ~14+ iterations to
    # build, so the rollback landed on an already-corrupted point and
    # settled into a new, worse quasi-fixed point. Reverted to best_theta.

    # Precompute verbose targets once (split into topo and weights parts)
    _v_targets = torch.cat([k_out, k_in, s_out, s_in])
    _v_nonzero = _v_targets > 0

    # ------------------------------------------------------------------
    # Hub precomputation (only when hub_sk_threshold > 0)
    # ------------------------------------------------------------------
    import numpy as _np_hub

    _hub_active = hub_sk_threshold > 0.0 and not _use_numba
    _hub_out_mask: torch.Tensor | None = None   # shape (N,) bool
    _hub_in_mask: torch.Tensor | None = None
    # Group multiplicities as numpy, for the weighted hub/leaf-bisection sum
    # (degeneracy-reduced path only; None reproduces the per-node behaviour).
    _mult_np = mult.numpy() if mult is not None else None
    # Precomputed once: target arrays as numpy, reused by every bisection
    # call each iteration (both hub and leaf).
    _k_out_np = k_out.numpy()
    _k_in_np = k_in.numpy()
    _s_out_np = s_out.numpy()
    _s_in_np = s_in.numpy()

    if _hub_active:
        # k_hat ≈ k_out (use observed degrees as proxy; also, a node with
        # zero strength is excluded from hub treatment regardless)
        k_hat_out = k_out.clamp(min=1.0)
        k_hat_in = k_in.clamp(min=1.0)
        sk_out = s_out / k_hat_out
        sk_in = s_in / k_hat_in
        _hub_out_mask = (sk_out > hub_sk_threshold) & (s_out > 0) & (~zero_k_out)
        _hub_in_mask = (sk_in > hub_sk_threshold) & (s_in > 0) & (~zero_k_in)

        _hub_out_indices = _hub_out_mask.nonzero(as_tuple=True)[0].tolist()
        _hub_in_indices = _hub_in_mask.nonzero(as_tuple=True)[0].tolist()
        _hub_out_idx_arr = _np_hub.array(_hub_out_indices, dtype=_np_hub.int64)
        _hub_in_idx_arr = _np_hub.array(_hub_in_indices, dtype=_np_hub.int64)

        # Build hub masks for the 4N Anderson exclusion vector
        _hub_4N_mask = torch.zeros(4 * N, dtype=torch.bool)
        for _idx in _hub_out_indices:
            _hub_4N_mask[2 * N + _idx] = True          # η_out part
        for _idx in _hub_in_indices:
            _hub_4N_mask[3 * N + _idx] = True          # η_in part

        if verbose:
            print(
                f"[hub-bisect] threshold={hub_sk_threshold}: "
                f"{len(_hub_out_indices)} out-hubs, "
                f"{len(_hub_in_indices)} in-hubs identified."
            )
    else:
        _hub_out_indices = []
        _hub_in_indices = []
        _hub_4N_mask = torch.zeros(4 * N, dtype=torch.bool)

    def _apply_hub_bisection(theta_fp: torch.Tensor) -> torch.Tensor:
        """Exact 1D-bisection correction of hub eta components, vectorized
        across all hub nodes at once (see :func:`_bisect_hub_eta_decm_batch`).
        Applied to the *final* ``theta_next`` for this iteration (after
        Anderson mixing, restarts, or rollback have all already settled on
        it -- see the single call site near the end of the main loop) so
        the bisection is always against the opposing values that actually
        carry forward, not a pre-mixing snapshot that then drifts."""
        if not (_hub_active and (_hub_out_indices or _hub_in_indices)):
            return theta_fp
        _th_fp_np = theta_fp.numpy()
        for _sweep in range(3):
            if _hub_out_indices:
                _new_eta_out = _bisect_hub_eta_decm_batch(
                    hub_indices=_hub_out_indices,
                    theta_self_all=_th_fp_np[:N],
                    theta_other=_th_fp_np[N:2 * N],
                    eta_other=_th_fp_np[3 * N:],
                    s_target_all=_s_out_np,
                    mult=_mult_np,
                )
                _th_fp_np[2 * N + _hub_out_idx_arr] = _new_eta_out
            if _hub_in_indices:
                _new_eta_in = _bisect_hub_eta_decm_batch(
                    hub_indices=_hub_in_indices,
                    theta_self_all=_th_fp_np[N:2 * N],
                    theta_other=_th_fp_np[:N],
                    eta_other=_th_fp_np[2 * N:3 * N],
                    s_target_all=_s_in_np,
                    mult=_mult_np,
                )
                _th_fp_np[3 * N + _hub_in_idx_arr] = _new_eta_in
        return torch.from_numpy(_th_fp_np)

    # ------------------------------------------------------------------
    # Leaf precomputation (only when leaf_k_threshold > 0)
    # ------------------------------------------------------------------
    _leaf_active = leaf_k_threshold > 0.0 and not _use_numba
    _leaf_out_mask: torch.Tensor | None = None   # shape (N,) bool
    _leaf_in_mask: torch.Tensor | None = None

    if _leaf_active:
        _leaf_out_mask = (k_out <= leaf_k_threshold) & (~zero_k_out)
        _leaf_in_mask = (k_in <= leaf_k_threshold) & (~zero_k_in)

        _leaf_out_indices = _leaf_out_mask.nonzero(as_tuple=True)[0].tolist()
        _leaf_in_indices = _leaf_in_mask.nonzero(as_tuple=True)[0].tolist()
        _leaf_out_idx_arr = _np_hub.array(_leaf_out_indices, dtype=_np_hub.int64)
        _leaf_in_idx_arr = _np_hub.array(_leaf_in_indices, dtype=_np_hub.int64)

        # Build leaf masks for the 4N Anderson exclusion vector
        _leaf_4N_mask = torch.zeros(4 * N, dtype=torch.bool)
        for _idx in _leaf_out_indices:
            _leaf_4N_mask[_idx] = True                 # θ_out part
        for _idx in _leaf_in_indices:
            _leaf_4N_mask[N + _idx] = True              # θ_in part

        if verbose:
            print(
                f"[leaf-bisect] threshold={leaf_k_threshold}: "
                f"{len(_leaf_out_indices)} out-leaves, "
                f"{len(_leaf_in_indices)} in-leaves identified."
            )
    else:
        _leaf_out_indices = []
        _leaf_in_indices = []
        _leaf_4N_mask = torch.zeros(4 * N, dtype=torch.bool)

    # Combined hub+leaf Anderson exclusion mask: both kinds of node get a
    # component solved exactly by bisection *after* mixing (see the main
    # loop), so their raw (unstable, unbisected) Newton-step contribution
    # must not feed into the least-squares mixing fit used for everyone
    # else -- excluded here, corrected properly by bisection afterward.
    _exclude_4N_mask = _hub_4N_mask | _leaf_4N_mask

    def _apply_leaf_bisection(theta_fp: torch.Tensor) -> torch.Tensor:
        """Exact 1D-bisection correction of leaf theta components -- the
        θ-side analogue of :func:`_apply_hub_bisection`, vectorized across
        all leaf nodes at once (see :func:`_bisect_leaf_theta_decm_batch`).
        Same single-call-site-on-theta_next placement as
        `_apply_hub_bisection`; see its docstring for why."""
        if not (_leaf_active and (_leaf_out_indices or _leaf_in_indices)):
            return theta_fp
        _th_fp_np = theta_fp.numpy()
        for _sweep in range(3):
            if _leaf_out_indices:
                _new_theta_out = _bisect_leaf_theta_decm_batch(
                    leaf_indices=_leaf_out_indices,
                    theta_other=_th_fp_np[N:2 * N],
                    eta_self_all=_th_fp_np[2 * N:3 * N],
                    eta_other=_th_fp_np[3 * N:],
                    k_target_all=_k_out_np,
                    mult=_mult_np,
                )
                _th_fp_np[_leaf_out_idx_arr] = _new_theta_out
            if _leaf_in_indices:
                _new_theta_in = _bisect_leaf_theta_decm_batch(
                    leaf_indices=_leaf_in_indices,
                    theta_other=_th_fp_np[:N],
                    eta_self_all=_th_fp_np[3 * N:],
                    eta_other=_th_fp_np[2 * N:3 * N],
                    k_target_all=_k_in_np,
                    mult=_mult_np,
                )
                _th_fp_np[N + _leaf_in_idx_arr] = _new_theta_in
        return torch.from_numpy(_th_fp_np)

    try:
        for _ in range(max_iter):
            if max_time > 0 and (time.perf_counter() - t0) > max_time:
                message = f"Time limit ({max_time:.0f}s) reached at iteration {n_iter}."
                break

            theta_fp, F_current = _step(theta)

            # Single-step floor: don't let η drop faster than _ANDERSON_THETA_FLOOR
            # fraction of its current value in one step (apply only to η part).
            eta_part_old = theta[2 * N :]
            eta_part_new = theta_fp[2 * N :]
            _eta_floor = (eta_part_old * _ANDERSON_THETA_FLOOR).clamp(min=_ETA_MIN)
            eta_part_new = torch.where(
                eta_part_new > 0,
                torch.maximum(eta_part_new, _eta_floor),
                eta_part_new,
            )
            theta_fp = torch.cat([theta_fp[:2 * N], eta_part_new])

            # Hub/leaf bisection is applied once, at the very end of the loop
            # (to theta_next, after Anderson mixing/restart/rollback have
            # settled it) -- not here. F_current below is therefore the
            # genuine residual of the *input* theta, honest bookkeeping: it
            # already reflects last iteration's bisection (theta was bisected
            # before becoming this iteration's input), with no artificial
            # zeroing. See the single bisection call site near the end of
            # this loop for why (moving-target self-consistency fix).
            res_norm = (
                (F_current.abs()[_v_nonzero] / _v_targets[_v_nonzero]).max().item()
                if _v_nonzero.any() else 0.0
            )

            if not math.isfinite(res_norm):
                message = f"NaN/Inf detected at iteration {n_iter}."
                break

            # --- Backtracking line search (PyTorch path only) ---
            # Evaluate residual at the proposed theta_fp; if it exceeds
            # backtracking_gamma × res_norm, halve the step up to 5 times.
            # Anderson history is cleared whenever a dampened step is accepted.
            if (
                backtracking_gamma > 0.0
                and res_norm > 0.0
                and not _use_numba
            ):
                _F_fp = residual_fn(theta_fp)
                _res_fp = (
                    (_F_fp.abs()[_v_nonzero] / _v_targets[_v_nonzero]).max().item()
                    if _v_nonzero.any() else 0.0
                )
                if math.isfinite(_res_fp) and _res_fp > backtracking_gamma * res_norm:
                    _bt_best_res = _res_fp
                    _bt_best_theta = theta_fp.clone()
                    _bt_alpha = 0.5
                    for _ in range(5):  # min alpha = 1/32
                        _theta_bt = theta + _bt_alpha * (theta_fp - theta)
                        _theta_bt[:2 * N] = _theta_bt[:2 * N].clamp(-_THETA_MAX, _THETA_MAX)
                        _theta_bt[2 * N:] = _theta_bt[2 * N:].clamp(min=_ETA_MIN, max=_ETA_MAX)
                        _F_bt = residual_fn(_theta_bt)
                        _res_bt = (
                            (_F_bt.abs()[_v_nonzero] / _v_targets[_v_nonzero]).max().item()
                            if _v_nonzero.any() else 0.0
                        )
                        if math.isfinite(_res_bt) and _res_bt < _bt_best_res:
                            _bt_best_res = _res_bt
                            _bt_best_theta = _theta_bt.clone()
                        if math.isfinite(_res_bt) and _res_bt <= backtracking_gamma * res_norm:
                            break
                        _bt_alpha *= 0.5
                    if _bt_best_res < _res_fp:
                        theta_fp = _bt_best_theta
                        _and_g.clear()
                        _and_r.clear()
                        _best_res_for_anderson = float("inf")

            n_iter += 1
            residuals.append(res_norm)

            if verbose:
                _elapsed = time.perf_counter() - t0
                if topo_weig:
                    _N = len(k_out)
                    _topo_targets = _v_targets[:2 * _N]
                    _topo_nz = _topo_targets > 0
                    _weights_targets = _v_targets[2 * _N:]
                    _weights_nz = _weights_targets > 0
                    _mre_topo = (
                        (F_current[:2 * _N].abs()[_topo_nz] / _topo_targets[_topo_nz]).max().item()
                        if _topo_nz.any() else float("nan")
                    )
                    _mre_weights = (
                        (F_current[2 * _N:].abs()[_weights_nz] / _weights_targets[_weights_nz]).max().item()
                        if _weights_nz.any() else float("nan")
                    )
                    print(
                        f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] "
                        f"iteration={n_iter:5d}, "
                        f"elapsed time={int(_elapsed // 3600):4d}:{int((_elapsed % 3600) // 60):02d}:{int(_elapsed % 60):02d}, "
                        f"MRE_topo={_mre_topo:.5e}, MRE_weights={_mre_weights:.5e}",
                        end="\r" if monitor else "\n",
                    )
                else:
                    print(
                        f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] "
                        f"iteration={n_iter:5d}, "
                        f"elapsed time={int(_elapsed // 3600):4d}:{int((_elapsed % 3600) // 60):02d}:{int(_elapsed % 60):02d}, "
                        f"MRE={res_norm:.5e}",
                        end="\r" if monitor else "\n",
                    )
                sys.stdout.flush()

            if res_norm < best_theta_res:
                best_theta_res = res_norm
                best_theta = theta.clone()
                restarts = 0
                stalls_at_cap = 0
                iters_since_improve = 0
                progressed_since_restart = True
                _soft_stall_reset_tried = False
            else:
                iters_since_improve += 1

            if res_norm < tol:
                converged = True
                message = f"Converged in {n_iter} iteration(s)."
                break

            # Stagnation detection: no record improvement for `patience`
            # iterations straight. Mirrors the isolated-vs-repeated blowup
            # escalation below: the FIRST stall since the last improvement
            # gets the same cheap fix as an isolated blowup (clear Anderson
            # history, plain Newton step from best_theta, no noise) instead
            # of jumping straight to a perturbed restart -- a stagnant
            # Anderson mix (ill-conditioned mixing weights reproducing the
            # same near-best point) can look identical to a genuine
            # attracting fixed point, and the cheap fix is free to try
            # first. Only if that ALSO fails to beat the record within
            # another `patience` window do we escalate to a perturbed
            # (noisy) restart (see `patience` docstring, and the shared
            # escalation counter with the repeated-blowup path below).
            _patience_restart_theta: torch.Tensor | None = None
            if patience > 0 and iters_since_improve >= patience:
                if not _soft_stall_reset_tried:
                    if verbose:
                        print(
                            f"[patience] stalled {patience} iters with no "
                            f"improvement -- soft reset (Anderson clear, no "
                            f"noise) at iter {n_iter}."
                        )
                    theta_rb, _ = _step(best_theta)
                    eta_old_rb = best_theta[2 * N :]
                    eta_new_rb = theta_rb[2 * N :]
                    _eta_floor_rb = (eta_old_rb * _ANDERSON_THETA_FLOOR).clamp(min=_ETA_MIN)
                    eta_new_rb = torch.where(
                        eta_new_rb > 0,
                        torch.maximum(eta_new_rb, _eta_floor_rb),
                        eta_new_rb,
                    )
                    theta_rb = torch.cat([theta_rb[:2 * N], eta_new_rb])
                    _patience_restart_theta = theta_rb
                    _soft_stall_reset_tried = True
                    iters_since_improve = 0
                    _and_g.clear()
                    _and_r.clear()
                else:
                    _patience_restart_theta, _give_up = _perturbed_restart()
                    iters_since_improve = 0
                    # A restart is itself a fresh start: give the resulting
                    # trajectory its own first-blowup grace (see
                    # `progressed_since_restart`'s declaration above) instead of
                    # inheriting "no progress" from whatever triggered this
                    # restart. The escalation counter (`restarts`, inside
                    # `_perturbed_restart`) still only resets on a genuine
                    # record improvement, so this doesn't weaken the
                    # loop-prevention guarantee -- it just stops every single
                    # post-restart blowup, however far downstream and however
                    # much clean progress happened first, from being misread as
                    # "still the same trap" and escalating immediately.
                    progressed_since_restart = True
                    # Stale pre-stagnation Anderson history is irrelevant (and
                    # potentially harmful) once we jump to a discontinuous new
                    # point; also arm the post-restart blowup-guard grace (see
                    # `_post_restart_reset` above).
                    _and_g.clear()
                    _and_r.clear()
                    _post_restart_reset = True
                    _soft_stall_reset_tried = False
                    if _give_up:
                        message = (
                            f"Stagnation recovery exhausted: {max_stalls} restarts at "
                            f"max noise (scale={noise_base * noise_cap_mult:.1e}) without "
                            f"improving best={best_theta_res:.3e}. Stopping at iter {n_iter}."
                        )
                        break

            # Anderson acceleration
            if _patience_restart_theta is not None:
                # Already decided by the patience check above -- skip the
                # normal Anderson/blowup handling for this iteration.
                theta_next = _patience_restart_theta
            elif anderson_depth > 1:
                # Blowup guard -- skipped for exactly one iteration right
                # after a perturbed restart (see `_post_restart_reset`
                # above): the restart's own residual is deliberately large
                # by design and must not be judged as a "blowup" against the
                # pre-restart record.
                _blowup_recovered = False
                if (
                    len(_and_g) >= 2
                    and math.isfinite(res_norm)
                    and not _post_restart_reset
                    and res_norm > eff_blowup * _best_res_for_anderson
                ):
                    _and_g.clear()
                    _and_r.clear()
                    _blowup_recovered = True
                    if verbose:
                        print(
                            f"[blowup] res_norm={res_norm:.3e} > "
                            f"{eff_blowup:.0f}x best={_best_res_for_anderson:.3e} "
                            f"at iter {n_iter}."
                        )

                if _post_restart_reset:
                    # Re-anchor the blowup reference to the restart's own
                    # residual instead of the (possibly far-off) historical
                    # record, so subsequent iterations are judged on
                    # progress *since the restart*, not distance from the
                    # pre-restart optimum.
                    _best_res_for_anderson = res_norm
                    _post_restart_reset = False
                else:
                    _best_res_for_anderson = min(_best_res_for_anderson, res_norm)

                if _blowup_recovered and not progressed_since_restart:
                    # This blowup recurred with no record improvement since
                    # the last restart (of either kind) -- a plain rollback
                    # already failed to escape this trap once, so escalate
                    # to the same perturbed-restart mechanism as stagnation
                    # instead of repeating the same ineffective rollback.
                    if verbose:
                        print(
                            f"[blowup] repeated with no improvement since the "
                            f"last restart -- escalating to perturbed restart."
                        )
                    theta_next, _give_up = _perturbed_restart()
                    _post_restart_reset = True
                    # See the patience-triggered restart above: a restart is
                    # a fresh start, so the resulting trajectory gets its
                    # own first-blowup grace rather than inheriting "no
                    # progress" from the trap that triggered this restart.
                    progressed_since_restart = True
                    if _give_up:
                        message = (
                            f"Stagnation recovery exhausted: {max_stalls} restarts at "
                            f"max noise (scale={noise_base * noise_cap_mult:.1e}) without "
                            f"improving best={best_theta_res:.3e}. Stopping at iter {n_iter}."
                        )
                        break
                elif _blowup_recovered:
                    # Isolated/first blowup since the last improvement: leave
                    # the existing, cheap recovery alone (no noise) -- on
                    # hub-heavy networks this fires often and normally
                    # self-resolves within a few iterations, see eff_blowup.
                    # Don't continue from the corrupted current theta: a plain
                    # Newton step from there still has to crawl back across
                    # whatever distance the bad Anderson mix introduced, which
                    # empirically costs 100+ iterations. Restart the plain
                    # Newton step from the best iterate seen so far instead.
                    if verbose:
                        print(
                            f"[blowup] isolated, rolling back to best_theta "
                            f"(best={best_theta_res:.3e}, no noise) at "
                            f"iter {n_iter}."
                        )
                    theta_rb, _ = _step(best_theta)
                    eta_old_rb = best_theta[2 * N :]
                    eta_new_rb = theta_rb[2 * N :]
                    _eta_floor_rb = (eta_old_rb * _ANDERSON_THETA_FLOOR).clamp(min=_ETA_MIN)
                    eta_new_rb = torch.where(
                        eta_new_rb > 0,
                        torch.maximum(eta_new_rb, _eta_floor_rb),
                        eta_new_rb,
                    )
                    theta_rb = torch.cat([theta_rb[:2 * N], eta_new_rb])
                    theta_next = theta_rb
                    # Mark the baseline so a blowup that recurs before any
                    # further improvement is treated as "repeated" above.
                    progressed_since_restart = False
                else:
                    r_k = theta_fp - theta
                    # Hub/leaf exclusion: zero their components in r_k so
                    # Anderson mixing weights focus on convergence of the
                    # rest of the network.
                    if _hub_active or _leaf_active:
                        r_k[_exclude_4N_mask] = 0.0
                    r_k_norm = r_k.abs().max().item()
                    if math.isfinite(r_k_norm) and r_k_norm < _ANDERSON_MAX_NORM:
                        _and_g.append(theta_fp.clone())
                        _and_r.append(r_k.clone())

                    if len(_and_g) > anderson_depth:
                        _and_g.pop(0)
                        _and_r.pop(0)

                    if len(_and_g) >= 2:
                        theta_next = _anderson_mixing(_and_g, _and_r, weights=_anderson_weights)

                        # Enforce η ≥ _ETA_MIN after mixing
                        theta_next[2 * N :] = theta_next[2 * N :].clamp(min=_ETA_MIN)
                        theta_next = theta_next.clamp(-_THETA_MAX, _THETA_MAX)
                        # Restore η part clamp
                        theta_next[2 * N :] = theta_next[2 * N :].clamp(min=_ETA_MIN, max=_ETA_MAX)

                        # Geometric floor on η: don't go below fraction of theta_fp's η
                        eta_floor_mix = (theta_fp[2 * N :] * _ANDERSON_THETA_FLOOR).clamp(min=_ETA_MIN)
                        theta_next_eta = torch.maximum(theta_next[2 * N :], eta_floor_mix)
                        theta_next = torch.cat([theta_next[:2 * N], theta_next_eta])

                        # Feasibility guard: if min(η_out) + min(η_in) < floor → reject mix
                        eta_out_mix = theta_next[2 * N : 3 * N]
                        eta_in_mix = theta_next[3 * N :]
                        z_min_and = (eta_out_mix.min() + eta_in_mix.min()).item()
                        if z_min_and < _Z_NEWTON_FLOOR:
                            theta_next = theta_fp
                            _and_g.clear()
                            _and_r.clear()
                    else:
                        theta_next = theta_fp
            else:
                theta_next = theta_fp

            # Hub/leaf bisection: applied once here, to the *final*
            # theta_next for this iteration -- whichever branch produced it
            # (normal Anderson mix, no-history-yet, isolated-blowup
            # rollback, or a patience/blowup-escalation perturbed restart).
            # Solving against theta_next's own opposing values (not a
            # pre-mixing snapshot) is what makes this self-consistent: the
            # exact fit is against the values that actually carry forward
            # into the next iteration, not ones that then drift during
            # mixing. See _apply_hub_bisection/_apply_leaf_bisection.
            if _hub_active or _leaf_active:
                theta_next = _apply_hub_bisection(theta_next)
                theta_next = _apply_leaf_bisection(theta_next)

            theta = theta_next

    finally:
        elapsed = time.perf_counter() - t0
        _peak_ram_monitor.__exit__(None, None, None)
        peak_ram = _peak_ram_monitor.peak_bytes
        if _prev_numba_threads is not None:
            import numba as _numba_mod
            _numba_mod.set_num_threads(_prev_numba_threads)

    return SolverResult(
        theta=theta.detach().numpy(),
        best_theta=best_theta.detach().numpy(),
        converged=converged,
        iterations=n_iter,
        residuals=residuals,
        elapsed_time=elapsed,
        peak_ram_bytes=peak_ram,
        message=message,
        best_mre=best_theta_res,
    )


def solve_fixed_point_decm_degenerate(
    theta0: torch.Tensor,
    k_out: torch.Tensor,
    k_in: torch.Tensor,
    s_out: torch.Tensor,
    s_in: torch.Tensor,
    tol: float = 1e-8,
    max_iter: int = 5000,
    anderson_depth: int = 10,
    max_step: float = 0.5,
    max_time: float = 0.0,
    backend: str = "auto",
    num_threads: int = 0,
    verbose: bool = False,
    monitor: bool = False,
    topo_weig: bool = False,
    weight_anderson: bool = True,
    hub_sk_threshold: float = 0.0,
    leaf_k_threshold: float = 0.0,
    z_clamp: float = 1e-8,
    init_best_theta: torch.Tensor | None = None,
    init_best_res: float = float("inf"),
    blowup_factor: float | None = None,
    patience: int = 750,
    noise_base: float = 1e-2,
    noise_cap_mult: float = 16.0,
    noise_growth: float = 2.0,
    max_stalls: int = 5,
    seed: int | None = None,
) -> SolverResult:
    """Degeneracy-reduced alternating GS-Newton solver for the DECM.

    Groups nodes that share the exact same (k_out, s_out, k_in, s_in)
    4-tuple -- which are proven to share identical (theta_out, eta_out,
    theta_in, eta_in) at the true DECM solution, see the module-level note
    above :func:`compute_degeneracy_groups` -- and solves the resulting
    M-unknown reduced system (M = number of unique 4-tuples <= N) instead
    of the full 4N-unknown system. The returned ``SolverResult`` is
    expanded back to the original per-node shape (4N,) so it is a drop-in
    replacement for :func:`solve_fixed_point_decm` at the call site.

    ``hub_sk_threshold`` and ``leaf_k_threshold`` ARE supported: hub/leaf
    identification and their exact 1D bisections both operate at group
    granularity, with the bisection sums weighted by group multiplicity
    (see :func:`_bisect_hub_eta_decm`'s and :func:`_bisect_leaf_theta_decm`'s
    ``mult`` parameter). A group's degree/strength is shared by every member
    node by construction, so hub/leaf membership at group granularity is
    exactly equivalent to per-node membership. Does not (yet) support
    ``backtracking_gamma`` (unavailable in this reduced path -- see
    :func:`solve_fixed_point_decm`'s ``mult`` parameter docs) or
    ``backend="numba"``.

    Args:
        theta0: Initial guess [theta_out|theta_in|eta_out|eta_in], shape
            (4N,). Nodes sharing a 4-tuple should already carry identical
            values here (true for the standard "degrees"-based initial
            guess); if not, one representative member's value is used for
            the whole group.
        k_out, k_in, s_out, s_in: Observed sequences, each shape (N,).
        z_clamp: Forwarded to :func:`solve_fixed_point_decm` unchanged
            (operates at group/M granularity here, since the reduced eta
            values are already one entry per degeneracy group). See its
            docstring for the full explanation.
        (all other args): see :func:`solve_fixed_point_decm`.
        init_best_theta: Optional externally-supplied record theta, same
            expanded per-node shape (4N,) as ``theta0``/``best_theta``
            (i.e. *not* pre-reduced) -- reduced internally the same way as
            ``theta0``. Since a genuine cross-chunk record from this same
            reduced solver is always exactly uniform within each degeneracy
            group already, this reduction is exact (not an approximation,
            unlike the analogous step for an arbitrary ``theta0``). See
            :func:`solve_fixed_point_decm`'s ``init_best_theta`` docs for why
            this matters.
        init_best_res: The residual (MRE) achieved by ``init_best_theta``.
        blowup_factor, patience, noise_base, noise_cap_mult, max_stalls, seed:
            Blowup-guard threshold and perturbed restart on stagnation/
            repeated-blowup, operating at group (M-length) granularity --
            see :func:`solve_fixed_point_decm`'s docstring for the full
            explanation. Forwarded unchanged.

    Returns:
        :class:`~dcms.solvers.base.SolverResult` with ``theta``/``best_theta``
        expanded back to shape (4N,).
    """
    k_out = k_out if isinstance(k_out, torch.Tensor) else torch.tensor(k_out, dtype=torch.float64)
    k_in = k_in if isinstance(k_in, torch.Tensor) else torch.tensor(k_in, dtype=torch.float64)
    s_out = s_out if isinstance(s_out, torch.Tensor) else torch.tensor(s_out, dtype=torch.float64)
    s_in = s_in if isinstance(s_in, torch.Tensor) else torch.tensor(s_in, dtype=torch.float64)
    k_out = k_out.to(dtype=torch.float64)
    k_in = k_in.to(dtype=torch.float64)
    s_out = s_out.to(dtype=torch.float64)
    s_in = s_in.to(dtype=torch.float64)
    theta0 = (
        theta0 if isinstance(theta0, torch.Tensor) else torch.tensor(theta0, dtype=torch.float64)
    ).to(dtype=torch.float64)

    N = k_out.shape[0]
    group_of, mult, k_out_g, k_in_g, s_out_g, s_in_g = compute_degeneracy_groups(
        k_out, k_in, s_out, s_in
    )
    M = k_out_g.shape[0]

    # Reduce theta0: take one representative member per group (the first
    # node index encountered). Degenerate nodes are expected to already
    # carry identical theta0 values (true for "degrees"-based initial
    # guesses, which are themselves a function of (k, s) alone); this is
    # only an approximation for initial guesses that break that pattern,
    # and only affects the *starting point*, not the converged solution.
    _first_idx = torch.zeros(M, dtype=torch.long)
    _seen = torch.zeros(M, dtype=torch.bool)
    _group_of_list = group_of.tolist()
    for _i, _g in enumerate(_group_of_list):
        if not _seen[_g]:
            _seen[_g] = True
            _first_idx[_g] = _i
    theta0_g = torch.cat(
        [
            theta0[:N][_first_idx],
            theta0[N : 2 * N][_first_idx],
            theta0[2 * N : 3 * N][_first_idx],
            theta0[3 * N :][_first_idx],
        ]
    )

    init_best_theta_g = None
    if init_best_theta is not None:
        _ibt = (
            init_best_theta if isinstance(init_best_theta, torch.Tensor)
            else torch.tensor(init_best_theta, dtype=torch.float64)
        ).to(dtype=torch.float64)
        init_best_theta_g = torch.cat(
            [
                _ibt[:N][_first_idx],
                _ibt[N : 2 * N][_first_idx],
                _ibt[2 * N : 3 * N][_first_idx],
                _ibt[3 * N :][_first_idx],
            ]
        )

    def _residual_fn_g(theta_m: torch.Tensor) -> torch.Tensor:
        zero_k_out_g = k_out_g == 0
        zero_k_in_g = k_in_g == 0
        zero_s_out_g = s_out_g == 0
        zero_s_in_g = s_in_g == 0
        return _decm_step_dense_weighted(
            theta_m, k_out_g, k_in_g, s_out_g, s_in_g,
            zero_k_out_g, zero_k_in_g, zero_s_out_g, zero_s_in_g,
            max_step, mult,
        )[1]

    result_g = solve_fixed_point_decm(
        _residual_fn_g,
        theta0_g,
        k_out_g, k_in_g, s_out_g, s_in_g,
        tol=tol,
        max_iter=max_iter,
        variant="theta-newton",
        anderson_depth=anderson_depth,
        max_step=max_step,
        max_time=max_time,
        backend=backend,
        num_threads=num_threads,
        verbose=verbose,
        monitor=monitor,
        topo_weig=topo_weig,
        mult=mult,
        weight_anderson=weight_anderson,
        hub_sk_threshold=hub_sk_threshold,
        leaf_k_threshold=leaf_k_threshold,
        z_clamp=z_clamp,
        init_best_theta=init_best_theta_g,
        init_best_res=init_best_res,
        blowup_factor=blowup_factor,
        patience=patience,
        noise_base=noise_base,
        noise_cap_mult=noise_cap_mult,
        noise_growth=noise_growth,
        max_stalls=max_stalls,
        seed=seed,
    )

    def _expand(theta_m):
        import numpy as np
        g = group_of.numpy()
        to_ = theta_m[:M][g]
        ti_ = theta_m[M : 2 * M][g]
        eo_ = theta_m[2 * M : 3 * M][g]
        ei_ = theta_m[3 * M :][g]
        return np.concatenate([to_, ti_, eo_, ei_])

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
