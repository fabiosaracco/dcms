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
import warnings
from typing import Callable

import torch

from dcms.models.parameters import DECM_LARGE_N_THRESHOLD as _LARGE_N_THRESHOLD
from dcms.models.parameters import _DEFAULT_CHUNK, _ETA_MAX, _ETA_MIN
from dcms.solvers.base import SolverResult
from dcms.utils.profiling import _PeakRAMMonitor

# -------------------------------------------------------------------------
# Numerical constants (mirrors fixed_point_qdecm.py)
# -------------------------------------------------------------------------
_ANDERSON_MAX_NORM: float = 1e6

_Z_G_CLAMP: float = 1e-6      # project-wide default (2026-09-25): solver, model residual and targeted fix all use it
_Z_NEWTON_FLOOR: float = _Z_G_CLAMP
_Z_NEWTON_FRAC: float = 0.5

# Max relative growth of G(z) per call allowed for a hub node using the
# negative-eta relaxation (decm-negative-eta-gauge / decm-damped-negative-eta
# branches) -- see _damp_hub_step. Only ever active for a node already
# flagged unreachable under the old eta>=0-only rule.
_G_DAMP_FRAC: float = 0.05
# Adaptive hub-bisection sweeps: after the first 3, a hub is re-solved only while its eta change exceeds _HUB_SWEEP_TOL * z_min.
_HUB_SWEEP_TOL: float = 1e-9   # relative to the hub's tightest pair sum z: dG/G ~ d_eta / z

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
                        :func:`solve_fixed_point_decm`). The
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


# -------------------------------------------------------------------------
# Dense DECM step (for N ≤ _LARGE_N_THRESHOLD)
# -------------------------------------------------------------------------

def _min_pair_z(
    eta_out: torch.Tensor, eta_in: torch.Tensor, mult: "torch.Tensor | None" = None
) -> float:
    """Smallest pair sum z_ij = eta_out_i + eta_in_j over the pairs that EXIST.

    The model only requires z_ij > 0 pairwise (individual eta may be negative,
    see the negative-eta relaxation of the hub bisection). ``min(eta_out) +
    min(eta_in)`` is a valid lower bound only when the two arg-mins are
    different groups: the diagonal pair (i, i) does not exist for a
    single-node group (mult <= 1), so when both minima sit on such a group the
    binding pair is the next-smallest combination. (Using the plain sum made
    the Anderson feasibility guard reject EVERY mix as soon as one relaxed hub
    had a negative eta on both sides -- q4 group 1403, 2026-09-25.)
    """
    n = eta_out.shape[0]
    if n == 1:
        return float("inf") if (mult is None or float(mult[0]) <= 1) else float((eta_out + eta_in)[0])
    vo, io = torch.topk(eta_out, 2, largest=False)
    vi, ii = torch.topk(eta_in, 2, largest=False)
    z = float(vo[0] + vi[0])
    if int(io[0]) == int(ii[0]) and (mult is None or float(mult[int(io[0])]) <= 1):
        z = min(float(vo[0] + vi[1]), float(vo[1] + vi[0]))
    return z


def _partner_min_excl_self(other: torch.Tensor, mult: "torch.Tensor | None") -> torch.Tensor:
    """For every index i: smallest entry of ``other`` over the partners of i. The own position (i, i) is a real pair only
    when the group has more than one node (mult > 1), otherwise it is excluded."""
    n = other.shape[0]
    if n == 1:
        own = other.clone()
        return own if (mult is not None and float(mult[0]) > 1) else torch.full_like(other, float("inf"))
    v, ix = torch.topk(other, 2, largest=False)
    out = torch.where(torch.arange(n, device=other.device) == ix[0], v[1].expand(n), v[0].expand(n))
    if mult is not None:
        out = torch.where(mult > 1, torch.minimum(out, other), out)
    return out


def _project_pair_floor(
    eta_out: torch.Tensor,
    eta_in: torch.Tensor,
    mult: "torch.Tensor | None",
    floor: float,
    frozen_out: "torch.Tensor | None" = None,
    frozen_in: "torch.Tensor | None" = None,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Raise eta so that every existing pair sum z_ij = eta_out_i + eta_in_j is >= ``floor``.

    The model only requires z_ij > 0 pairwise (individual eta may be negative). Instead of letting the equations silently
    clamp a violating pair to the floor (which evaluates it with a huge weight G ~ 1/floor), the offending eta is moved to
    the smallest feasible value: eta_out_i >= floor - min_j eta_in_j, then eta_in_j >= floor - min_i eta_out_i (with the
    already-raised eta_out), which makes every pair feasible. Entries flagged in ``frozen_*`` (hub-owned eta, which have
    their own feasible bracket in the hub bisection) are never modified. A feasible input is returned unchanged.
    """
    need_out = floor - _partner_min_excl_self(eta_in, mult)
    new_out = torch.maximum(eta_out, need_out)
    if frozen_out is not None:
        new_out = torch.where(frozen_out, eta_out, new_out)
    need_in = floor - _partner_min_excl_self(new_out, mult)
    new_in = torch.maximum(eta_in, need_in)
    if frozen_in is not None:
        new_in = torch.where(frozen_in, eta_in, new_in)
    return new_out, new_in


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


def _block_newton_deltas(
    F_k: torch.Tensor,
    F_s: torch.Tensor,
    A: torch.Tensor,
    Bp: torch.Tensor,
    Cp: torch.Tensor,
    E: torch.Tensor,
    C: torch.Tensor,
    gate: float,
    max_step: float,
    delta_theta: torch.Tensor,
    delta_eta: torch.Tensor,
    exclude: "torch.Tensor | None" = None,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    """Joint 2x2 Newton step for (theta_i, eta_i) of the rows whose scalar steps are ill-conditioned.

    The plain step solves theta_i from the k-equation alone (Delta = F_k / A) and eta_i from the s-equation alone (Delta = F_s / C),
    which is block-Jacobi with the cross-derivative B ignored. The exact per-row Hessian block is [[A, B], [B, C]] with
    A = sum m pq, B = sum m pq G, C = sum m (pq G^2 + p G (G-1)). For a nearly saturated row (s ~ k, G ~ 1) the two equations are
    almost the same function: det/(A C) = 1 - rho^2 ~ 1e-5 (q4) .. 1e-4 (e1) and the scalar iteration contracts by only rho ~ 1 - 1e-5
    per step (tens of thousands of iterations per e-fold -- the "crawl to infinity" plateau). The joint step removes that intra-row
    ill-conditioning exactly.

    Written with g = G - 1 (small for saturated rows) to avoid the catastrophic cancellation of A*C - B^2:
        Bp = sum m pq g,  Cp = sum m pq g^2,  E = sum m p G g
        B = A + Bp,   det = A*E + (A*Cp - Bp^2)   (second term >= 0 by Cauchy-Schwarz).
    Rows with det/(A*C) < ``gate`` (and not in ``exclude``) get the joint step, scaled as one vector so that
    max(|d_theta|, |d_eta|) <= max_step (clipping one component alone would break the theta/eta cancellation the soft mode needs).

    Returns ``(delta_theta, delta_eta, gated)`` where the deltas equal the inputs for the non-gated rows.
    """
    B = A + Bp
    det = A * E + (A * Cp - Bp * Bp).clamp(min=0.0)
    rel = det / (A * C).clamp(min=1e-300)
    gated = (rel < gate) & (A > 1e-15) & (C > 1e-15) & (det > 0.0)
    if exclude is not None:
        gated = gated & ~exclude
    det_s = det.clamp(min=1e-300)
    d_th = (C * F_k - B * F_s) / det_s
    d_et = (A * F_s - B * F_k) / det_s
    scale = (max_step / torch.maximum(d_th.abs(), d_et.abs()).clamp(min=1e-300)).clamp(max=1.0)
    d_th = d_th * scale
    d_et = d_et * scale
    return torch.where(gated, d_th, delta_theta), torch.where(gated, d_et, delta_eta), gated


def _block_exclude(hub_mask, F_k, F_s, k_t, s_t, min_rel):
    """Rows the joint 2x2 step must NOT touch: hub-owned rows plus rows already below ``min_rel`` in relative residual.

    A nearly saturated row with s == k exactly has its true solution at infinity (residual ~ e^-c along the soft direction theta-c,
    eta+c), so the joint Newton step would march it towards |theta|, eta ~ 30+ at max_step per iteration long after the row satisfies
    its equations (e1: theta -31, eta +37, then MRE > 1). Rows already within ``min_rel`` (default 0.1*tol) are left to the scalar step."""
    ex = hub_mask
    if min_rel > 0.0:
        rel = torch.maximum(F_k.abs() / k_t.clamp(min=1e-300), F_s.abs() / s_t.clamp(min=1e-300))
        small = rel < min_rel
        ex = small if ex is None else (ex | small)
    return ex


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
    hub_out_mask: "torch.Tensor | None" = None,
    hub_in_mask: "torch.Tensor | None" = None,
    block_newton_gate: float = 0.0,
    block_newton_min_rel: float = 0.0,
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
        block_newton_gate: 0 (default) = scalar per-parameter Newton steps (previous behaviour, bit-identical). > 0: rows whose 2x2
            (theta, eta) Hessian block has det/(A*C) below this value are updated with the joint 2x2 Newton step (see
            :func:`_block_newton_deltas`); non-hub rows only on the out side.
        block_newton_min_rel: rows whose relative residual is already below this value keep the scalar step (see :func:`_block_exclude`).

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

    # ------- Newton deltas of θ_out (k-equation) and η_out (s-equation) -------
    delta_theta_out = ((k_out_hat - k_out) / H_k_out).clamp(-max_step, max_step)
    delta_eta_out = ((s_out_hat - s_out) / H_s_out).clamp(-max_step, max_step)
    gated_out = None
    if block_newton_gate > 0.0:
        # Joint 2x2 step for the ill-conditioned (nearly saturated) rows -- see _block_newton_deltas.
        g_ = G - 1.0
        mm_ = mult[None, :]
        Bp_out = ((pq * g_) * mm_).sum(1) - (pq * g_).diagonal()
        Cp_out = ((pq * g_ * g_) * mm_).sum(1) - (pq * g_ * g_).diagonal()
        E_out = ((P * G * g_) * mm_).sum(1) - (P * G * g_).diagonal()
        delta_theta_out, delta_eta_out, gated_out = _block_newton_deltas(
            k_out_hat - k_out, s_out_hat - s_out, H_k_out, Bp_out, Cp_out, E_out, H_s_out,
            block_newton_gate, max_step, delta_theta_out, delta_eta_out,
            exclude=_block_exclude(hub_out_mask, k_out_hat - k_out, s_out_hat - s_out, k_out, s_out, block_newton_min_rel),
        )

    # ------- Update η_out (with z-floor) -------

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
        torch.ones(M, dtype=delta_eta_out.dtype, device=delta_eta_out.device),
    )
    eta_out_new = (eta_out + alpha_out * delta_eta_out).clamp(_ETA_MIN, _ETA_MAX)
    eta_out_new = torch.where(zero_s_out, torch.full_like(eta_out_new, _ETA_MAX), eta_out_new)
    # ------- Update θ_out (a jointly-stepped row shares the z-floor scaling of its η step) -------
    theta_out_new = (theta_out + (delta_theta_out if gated_out is None else torch.where(gated_out, alpha_out, torch.ones_like(alpha_out)) * delta_theta_out)).clamp(-_THETA_MAX, _THETA_MAX)
    theta_out_new = torch.where(zero_k_out, torch.full_like(theta_out_new, _THETA_MAX), theta_out_new)
    if hub_out_mask is not None:
        # Hub eta_out is owned by the hub bisection, not by this Newton
        # step (the caller discards the raw value). Pass 2 must therefore
        # see the eta_out that will actually be applied -- the input
        # (previously bisected) value -- not the raw Newton proposal,
        # otherwise every theta_in is corrected against a phantom
        # environment (dominant for near-boundary in-hubs, z=eta_out+eta_in -> 0).
        eta_out_new = torch.where(hub_out_mask, eta_out, eta_out_new)

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

    # ------- Newton deltas of θ_in (k-equation) and η_in (s-equation) -------
    delta_theta_in = ((k_in_hat2 - k_in) / H_k_in2).clamp(-max_step, max_step)
    delta_eta_in = ((s_in_hat2 - s_in) / H_s_in2).clamp(-max_step, max_step)
    gated_in = None
    if block_newton_gate > 0.0:
        g2_ = G2 - 1.0
        mm2_ = mult[:, None]
        Bp_in = ((pq2 * g2_) * mm2_).sum(0) - (pq2 * g2_).diagonal()
        Cp_in = ((pq2 * g2_ * g2_) * mm2_).sum(0) - (pq2 * g2_ * g2_).diagonal()
        E_in = ((P2 * G2 * g2_) * mm2_).sum(0) - (P2 * G2 * g2_).diagonal()
        delta_theta_in, delta_eta_in, gated_in = _block_newton_deltas(
            k_in_hat2 - k_in, s_in_hat2 - s_in, H_k_in2, Bp_in, Cp_in, E_in, H_s_in2,
            block_newton_gate, max_step, delta_theta_in, delta_eta_in,
            exclude=_block_exclude(hub_in_mask, k_in_hat2 - k_in, s_in_hat2 - s_in, k_in, s_in, block_newton_min_rel),
        )

    # ------- Update η_in (with z-floor) -------

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
        torch.ones(M, dtype=delta_eta_in.dtype, device=delta_eta_in.device),
    )
    eta_in_new = (eta_in + alpha_in * delta_eta_in).clamp(_ETA_MIN, _ETA_MAX)
    eta_in_new = torch.where(zero_s_in, torch.full_like(eta_in_new, _ETA_MAX), eta_in_new)
    # ------- Update θ_in -------
    theta_in_new = (theta_in + (delta_theta_in if gated_in is None else torch.where(gated_in, alpha_in, torch.ones_like(alpha_in)) * delta_theta_in)).clamp(-_THETA_MAX, _THETA_MAX)
    theta_in_new = torch.where(zero_k_in, torch.full_like(theta_in_new, _THETA_MAX), theta_in_new)

    theta_new = torch.cat([theta_out_new, theta_in_new, eta_out_new, eta_in_new])
    return theta_new, F_current


# -------------------------------------------------------------------------
# Chunked DECM step (for N > _LARGE_N_THRESHOLD)
# -------------------------------------------------------------------------

def _decm_step_chunked_weighted(
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
    mult: torch.Tensor,
    hub_out_mask: "torch.Tensor | None" = None,
    hub_in_mask: "torch.Tensor | None" = None,
    block_newton_gate: float = 0.0,
    block_newton_min_rel: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunked alternating GS-Newton step for the degeneracy-reduced DECM.

    Mirrors :func:`_decm_step_dense_weighted` but processes the M×M group
    matrices in row chunks to keep memory usage at O(chunk_size × M), the
    same trade :func:`_decm_step_chunked` makes for the per-node case.

    :func:`_decm_step_dense_weighted` has no size guard: it always builds
    the full M×M matrices regardless of M, which is fine when the
    degeneracy reduction is substantial (M << N, the common case for real
    integer degree/strength data with a heavy repeated low-degree bulk)
    but becomes impractically slow/memory-heavy when it isn't (e.g.
    continuous-valued or otherwise low-collision degree/strength
    sequences, where M stays close to N even for N in the tens of
    thousands). :func:`solve_fixed_point_decm` selects this function
    instead of the dense one once M exceeds ``_LARGE_N_THRESHOLD``,
    mirroring exactly the existing dense-vs-chunked selection already
    used for the non-reduced (``mult is None``) path.

    Produces results equal to :func:`_decm_step_dense_weighted` (up to
    floating-point summation order) for the same inputs, for any
    ``chunk_size`` -- unlike :func:`_decm_step_chunked`, which restricts
    its z-floor line-search to "significant" pairs as a performance/
    robustness heuristic, this function tracks the exact same z_min as
    the dense weighted version (including its singleton-diagonal
    handling), since exact equivalence with the already-verified dense
    weighted path was the primary correctness target here.

    Args: same as :func:`_decm_step_dense_weighted` (including ``block_newton_gate``), plus:
        chunk_size: Rows (groups) per processing chunk.

    Returns:
        ``(theta_new, F_current)``, both shape (4M,)/(4M,).
    """
    M = k_out.shape[0]
    theta_out = theta[:M]
    theta_in = theta[M : 2 * M]
    eta_out = theta[2 * M : 3 * M]
    eta_in = theta[3 * M :]

    singleton = mult <= 1.0

    # ------------------------------------------------------------------
    # Pass 1: accumulate row sums (out-direction) + preliminary col sums
    # (in-direction, at the *input* theta -- used for F_current only)
    # ------------------------------------------------------------------
    k_out_hat = torch.zeros(M, dtype=theta.dtype, device=theta.device)
    k_in_hat = torch.zeros(M, dtype=theta.dtype, device=theta.device)
    s_out_hat = torch.zeros(M, dtype=theta.dtype, device=theta.device)
    s_in_hat = torch.zeros(M, dtype=theta.dtype, device=theta.device)
    H_k_out = torch.zeros(M, dtype=theta.dtype, device=theta.device)
    H_s_out = torch.zeros(M, dtype=theta.dtype, device=theta.device)
    Bp_out = torch.zeros(M, dtype=theta.dtype, device=theta.device)
    Cp_out = torch.zeros(M, dtype=theta.dtype, device=theta.device)
    E_out = torch.zeros(M, dtype=theta.dtype, device=theta.device)
    z_min_out = torch.full((M,), float("inf"), dtype=theta.dtype, device=theta.device)

    for i_start in range(0, M, chunk_size):
        i_end = min(i_start + chunk_size, M)
        local_i = torch.arange(i_end - i_start, dtype=torch.long, device=theta.device)
        global_j = torch.arange(i_start, i_end, dtype=torch.long, device=theta.device)

        eta_chunk = eta_out[i_start:i_end, None] + eta_in[None, :]  # (chunk, M)
        eta_safe = eta_chunk.clamp(min=_Z_G_CLAMP)
        G_chunk = -1.0 / torch.expm1(-eta_safe)
        log_q_chunk = -torch.log(torch.expm1(eta_safe))
        logit_p_chunk = (
            -theta_out[i_start:i_end, None] - theta_in[None, :] + log_q_chunk
        )
        p_chunk = torch.sigmoid(logit_p_chunk)
        w_chunk = p_chunk * G_chunk
        pq_chunk = p_chunk * (1.0 - p_chunk)
        H_s_chunk = p_chunk * G_chunk * (G_chunk - 1.0) + pq_chunk * G_chunk.pow(2)

        # Diagonal entries P[g,g] etc. for g in this chunk -- subtracted
        # once below rather than zeroed, the same "weighted sum minus one
        # self-interaction term" generalisation _decm_step_dense_weighted
        # uses for the degeneracy-reduced self-loop exclusion.
        diag_p = p_chunk[local_i, global_j]
        diag_w = w_chunk[local_i, global_j]
        diag_pq = pq_chunk[local_i, global_j]
        diag_H_s = H_s_chunk[local_i, global_j]

        k_out_hat[i_start:i_end] = (p_chunk * mult[None, :]).sum(1) - diag_p
        s_out_hat[i_start:i_end] = (w_chunk * mult[None, :]).sum(1) - diag_w
        H_k_out[i_start:i_end] = (pq_chunk * mult[None, :]).sum(1) - diag_pq
        H_s_out[i_start:i_end] = (H_s_chunk * mult[None, :]).sum(1) - diag_H_s
        if block_newton_gate > 0.0:
            g_chunk = G_chunk - 1.0
            pqg = pq_chunk * g_chunk
            Bp_out[i_start:i_end] = (pqg * mult[None, :]).sum(1) - pqg[local_i, global_j]
            pqgg = pqg * g_chunk
            Cp_out[i_start:i_end] = (pqgg * mult[None, :]).sum(1) - pqgg[local_i, global_j]
            pgg = p_chunk * G_chunk * g_chunk
            E_out[i_start:i_end] = (pgg * mult[None, :]).sum(1) - pgg[local_i, global_j]

        # Column accumulation (in-direction, at input theta), weighted by
        # this chunk's own group multiplicities (the row/g side). The
        # diagonal term for column h=g only ever appears in this chunk
        # (at its own row g), so it is subtracted exactly once, here.
        k_in_hat += (p_chunk * mult[i_start:i_end, None]).sum(0)
        s_in_hat += (w_chunk * mult[i_start:i_end, None]).sum(0)
        k_in_hat[i_start:i_end] -= diag_p
        s_in_hat[i_start:i_end] -= diag_w

        # z_min_out[g] = min over h of eta_gh, excluding the diagonal only
        # for singleton groups (mult==1: no real edge behind that entry;
        # see _decm_step_dense_weighted).
        eta_for_min = eta_safe.clone()
        chunk_singleton = singleton[i_start:i_end]
        if chunk_singleton.any():
            rows = local_i[chunk_singleton]
            eta_for_min[rows, global_j[chunk_singleton]] = float("inf")
        z_min_out[i_start:i_end] = eta_for_min.min(1).values.clamp(min=_Z_G_CLAMP)

    nz_in = s_in > 0
    if nz_in.any():
        z_min_out = torch.minimum(z_min_out, eta_out + eta_in[nz_in].min())

    F_current = torch.cat(
        [k_out_hat - k_out, k_in_hat - k_in,
         s_out_hat - s_out, s_in_hat - s_in]
    )

    H_k_out = H_k_out.clamp(min=1e-15)
    H_s_out = H_s_out.clamp(min=1e-15)

    # ------- Newton deltas of θ_out (k-equation) and η_out (s-equation) -------
    delta_theta_out = ((k_out_hat - k_out) / H_k_out).clamp(-max_step, max_step)
    delta_eta_out = ((s_out_hat - s_out) / H_s_out).clamp(-max_step, max_step)
    gated_out = None
    if block_newton_gate > 0.0:
        delta_theta_out, delta_eta_out, gated_out = _block_newton_deltas(
            k_out_hat - k_out, s_out_hat - s_out, H_k_out, Bp_out, Cp_out, E_out, H_s_out,
            block_newton_gate, max_step, delta_theta_out, delta_eta_out,
            exclude=_block_exclude(hub_out_mask, k_out_hat - k_out, s_out_hat - s_out, k_out, s_out, block_newton_min_rel),
        )

    # ------- Update η_out (with z-floor line-search) -------
    z_floor_out = (z_min_out * _Z_NEWTON_FRAC).clamp(min=_Z_NEWTON_FLOOR)
    available_out = (z_min_out - z_floor_out).clamp(min=0.0)
    alpha_out = torch.where(
        delta_eta_out < 0,
        (available_out / delta_eta_out.abs().clamp(min=1e-30)).clamp(max=1.0),
        torch.ones(M, dtype=delta_eta_out.dtype, device=delta_eta_out.device),
    )
    eta_out_new = (eta_out + alpha_out * delta_eta_out).clamp(_ETA_MIN, _ETA_MAX)
    eta_out_new = torch.where(
        zero_s_out, torch.full_like(eta_out_new, _ETA_MAX), eta_out_new
    )
    # ------- Update θ_out (a jointly-stepped row shares the z-floor scaling of its η step) -------
    theta_out_new = (theta_out + (delta_theta_out if gated_out is None else torch.where(gated_out, alpha_out, torch.ones_like(alpha_out)) * delta_theta_out)).clamp(-_THETA_MAX, _THETA_MAX)
    theta_out_new = torch.where(
        zero_k_out, torch.full_like(theta_out_new, _THETA_MAX), theta_out_new
    )
    if hub_out_mask is not None:
        # Hub eta_out is owned by the hub bisection, not by this Newton
        # step (the caller discards the raw value). Pass 2 must therefore
        # see the eta_out that will actually be applied -- the input
        # (previously bisected) value -- not the raw Newton proposal,
        # otherwise every theta_in is corrected against a phantom
        # environment (dominant for near-boundary in-hubs, z=eta_out+eta_in -> 0).
        eta_out_new = torch.where(hub_out_mask, eta_out, eta_out_new)

    # ------------------------------------------------------------------
    # Pass 2: accumulate col sums using updated (theta_out_new, eta_out_new)
    # ------------------------------------------------------------------
    k_in_hat2 = torch.zeros(M, dtype=theta.dtype, device=theta.device)
    s_in_hat2 = torch.zeros(M, dtype=theta.dtype, device=theta.device)
    H_k_in2 = torch.zeros(M, dtype=theta.dtype, device=theta.device)
    H_s_in2 = torch.zeros(M, dtype=theta.dtype, device=theta.device)
    Bp_in = torch.zeros(M, dtype=theta.dtype, device=theta.device)
    Cp_in = torch.zeros(M, dtype=theta.dtype, device=theta.device)
    E_in = torch.zeros(M, dtype=theta.dtype, device=theta.device)
    z_min_in = torch.full((M,), float("inf"), dtype=theta.dtype, device=theta.device)

    for i_start in range(0, M, chunk_size):
        i_end = min(i_start + chunk_size, M)
        local_i = torch.arange(i_end - i_start, dtype=torch.long, device=theta.device)
        global_j = torch.arange(i_start, i_end, dtype=torch.long, device=theta.device)

        eta2_chunk = eta_out_new[i_start:i_end, None] + eta_in[None, :]
        eta2_safe = eta2_chunk.clamp(min=_Z_G_CLAMP)
        G2_chunk = -1.0 / torch.expm1(-eta2_safe)
        log_q2_chunk = -torch.log(torch.expm1(eta2_safe))
        logit_p2_chunk = (
            -theta_out_new[i_start:i_end, None] - theta_in[None, :] + log_q2_chunk
        )
        p2_chunk = torch.sigmoid(logit_p2_chunk)
        w2_chunk = p2_chunk * G2_chunk
        pq2_chunk = p2_chunk * (1.0 - p2_chunk)
        H_s2_chunk = p2_chunk * G2_chunk * (G2_chunk - 1.0) + pq2_chunk * G2_chunk.pow(2)

        diag_p2 = p2_chunk[local_i, global_j]
        diag_w2 = w2_chunk[local_i, global_j]
        diag_pq2 = pq2_chunk[local_i, global_j]
        diag_H_s2 = H_s2_chunk[local_i, global_j]

        k_in_hat2 += (p2_chunk * mult[i_start:i_end, None]).sum(0)
        s_in_hat2 += (w2_chunk * mult[i_start:i_end, None]).sum(0)
        H_k_in2 += (pq2_chunk * mult[i_start:i_end, None]).sum(0)
        H_s_in2 += (H_s2_chunk * mult[i_start:i_end, None]).sum(0)
        if block_newton_gate > 0.0:
            g2_chunk = G2_chunk - 1.0
            pqg2 = pq2_chunk * g2_chunk
            Bp_in += (pqg2 * mult[i_start:i_end, None]).sum(0)
            pqgg2 = pqg2 * g2_chunk
            Cp_in += (pqgg2 * mult[i_start:i_end, None]).sum(0)
            pgg2 = p2_chunk * G2_chunk * g2_chunk
            E_in += (pgg2 * mult[i_start:i_end, None]).sum(0)
            Bp_in[i_start:i_end] -= pqg2[local_i, global_j]
            Cp_in[i_start:i_end] -= pqgg2[local_i, global_j]
            E_in[i_start:i_end] -= pgg2[local_i, global_j]

        k_in_hat2[i_start:i_end] -= diag_p2
        s_in_hat2[i_start:i_end] -= diag_w2
        H_k_in2[i_start:i_end] -= diag_pq2
        H_s_in2[i_start:i_end] -= diag_H_s2

        eta2_for_min = eta2_safe.clone()
        chunk_singleton = singleton[i_start:i_end]
        if chunk_singleton.any():
            rows = local_i[chunk_singleton]
            eta2_for_min[rows, global_j[chunk_singleton]] = float("inf")
        z_min_in = torch.minimum(z_min_in, eta2_for_min.min(0).values)

    z_min_in = z_min_in.clamp(min=_Z_G_CLAMP)
    nz_out = s_out > 0
    if nz_out.any():
        z_min_in = torch.minimum(z_min_in, eta_out_new[nz_out].min() + eta_in)

    H_k_in2 = H_k_in2.clamp(min=1e-15)
    H_s_in2 = H_s_in2.clamp(min=1e-15)

    # ------- Newton deltas of θ_in (k-equation) and η_in (s-equation) -------
    delta_theta_in = ((k_in_hat2 - k_in) / H_k_in2).clamp(-max_step, max_step)
    delta_eta_in = ((s_in_hat2 - s_in) / H_s_in2).clamp(-max_step, max_step)
    gated_in = None
    if block_newton_gate > 0.0:
        delta_theta_in, delta_eta_in, gated_in = _block_newton_deltas(
            k_in_hat2 - k_in, s_in_hat2 - s_in, H_k_in2, Bp_in, Cp_in, E_in, H_s_in2,
            block_newton_gate, max_step, delta_theta_in, delta_eta_in,
            exclude=_block_exclude(hub_in_mask, k_in_hat2 - k_in, s_in_hat2 - s_in, k_in, s_in, block_newton_min_rel),
        )

    # ------- Update η_in (with z-floor line-search) -------
    z_floor_in = (z_min_in * _Z_NEWTON_FRAC).clamp(min=_Z_NEWTON_FLOOR)
    available_in = (z_min_in - z_floor_in).clamp(min=0.0)
    alpha_in = torch.where(
        delta_eta_in < 0,
        (available_in / delta_eta_in.abs().clamp(min=1e-30)).clamp(max=1.0),
        torch.ones(M, dtype=delta_eta_in.dtype, device=delta_eta_in.device),
    )
    eta_in_new = (eta_in + alpha_in * delta_eta_in).clamp(_ETA_MIN, _ETA_MAX)
    eta_in_new = torch.where(
        zero_s_in, torch.full_like(eta_in_new, _ETA_MAX), eta_in_new
    )
    # ------- Update θ_in -------
    theta_in_new = (theta_in + (delta_theta_in if gated_in is None else torch.where(gated_in, alpha_in, torch.ones_like(alpha_in)) * delta_theta_in)).clamp(-_THETA_MAX, _THETA_MAX)
    theta_in_new = torch.where(
        zero_k_in, torch.full_like(theta_in_new, _THETA_MAX), theta_in_new
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


def _bisect_hub_eta_decm_batch(
    hub_indices: "list[int]",
    theta_self_all: "np.ndarray",
    theta_other: "np.ndarray",
    eta_other: "np.ndarray",
    s_target_all: "np.ndarray",
    n_bisect: int = 60,
    mult: "np.ndarray | None" = None,
    negative_eta_scope: str = "unreachable_only",
    method: str = "bisect",
) -> "np.ndarray":
    """Vectorized batch version of :func:`_bisect_hub_eta_decm`: solves η for
    every node in ``hub_indices`` simultaneously (each bisection step is one
    ``(k, N)`` vectorized array op instead of ``k`` separate per-node Python
    calls), against the *same* snapshot of ``theta_other``/``eta_other``.

    Same equation, same monotonicity as the per-node version; see its
    docstring for the derivation. Only the "unreachable" edge case at the
    low end (``f(eta_lo) <= 0``) is checked, matching the per-node version
    -- ``f(eta_hi) < 0`` always holds for a positive ``s_target``.

    ``eta_lo`` is NOT pinned at ``_ETA_MIN`` (>=0): the model's actual
    requirement is z_ij = eta_val + eta_other_j > 0 for every partner j,
    not eta_val >= 0 in isolation -- eta is a Lagrange multiplier for an
    *equality* constraint, so nothing in the MaxEnt derivation requires its
    sign. eta_val >= 0 was only ever a sufficient (not necessary) way to
    guarantee every pairwise z > 0 without checking per-pair. The true
    per-node floor is the eta_val where z first touches the numerical
    floor for the SINGLE tightest partner (min eta_other): more negative
    than that and MORE than one pairwise z would clamp simultaneously,
    which is a qualitatively different (undesirable) regime. See branch
    decm-negative-eta-gauge. This is a strict relaxation: for any node
    whose old positive-only search already found an interior root, the
    root is unchanged (monotonic f, wider bracket, same crossing) -- only
    genuinely eta>=0-unreachable nodes behave differently.

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
        negative_eta_scope: ``"unreachable_only"`` (default, conservative):
                         only nodes whose OLD eta>=0-only equation is
                         already unreachable get the negative bracket,
                         everything else is byte-identical to before.
                         ``"all"``: every hub node in this batch gets the
                         negative bracket (broader, empirically noisier --
                         see decm-negative-eta-gauge branch notes).
        method:         ``"bisect"`` (default): ``n_bisect`` plain halvings of
                         the full bracket (bit-identical to all earlier
                         versions). ``"illinois"``: Illinois-modified regula
                         falsi on the same bracket, evaluating only the rows
                         not yet converged (|f| <= 1e-14 * max(s, 1) or
                         bracket < 1e-15); same root to ~1e-14 absolute in
                         eta, ~3x fewer row-evaluations on real hub batches
                         (q4, 131/435 hubs). ``n_bisect`` is ignored.

    Returns:
        Array of shape ``(k,)``: η* for each node in ``hub_indices``, in the
        same order.
    """
    import numpy as _np

    if method not in ("bisect", "illinois"):
        raise ValueError(f"unknown method: {method!r}")

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

    # Per-hub-node min(eta_other) EXCLUDING the node's own position -- a hub
    # node's own eta_other[idx] (its "self" entry on the opposing side) is
    # itself near the floor for exactly these pathological nodes, so a
    # naive global min would wrongly use the node's own value and barely
    # move the bracket. Use the smallest-excluding-self via 2nd-smallest
    # fallback (O(N log N) once, not O(k*N)).
    _order = _np.argsort(eta_other)
    _min1_pos, _min1_val = _order[0], eta_other[_order[0]]
    _min2_val = eta_other[_order[1]] if eta_other.shape[0] > 1 else _min1_val
    _min_other_excl_self = _np.where(idx == _min1_pos, _min2_val, _min1_val)
    eta_lo_negative = _np.maximum(_Z_G_CLAMP - _min_other_excl_self, -_ETA_MAX)

    if negative_eta_scope == "all":
        eta_lo_init = eta_lo_negative
    elif negative_eta_scope == "unreachable_only":
        # Auto-scoped (the conservative default): only nodes that the OLD
        # eta>=0-only bracket would already flag unreachable get the
        # negative floor; every other hub node's bracket -- and therefore
        # its bisection result -- is byte-identical to the pre-relaxation
        # behaviour. On q4/e1 this selects exactly the single pathological
        # node out of 21-582 hub-flagged nodes per side (verified
        # empirically 2026-09-01: every other hub node has >1800x slack,
        # there is no natural intermediate tier to size a threshold on).
        eta_lo_old = _np.full(k, _ETA_MIN)
        unreachable_old = _f(eta_lo_old) <= 0.0
        eta_lo_init = _np.where(unreachable_old, eta_lo_negative, eta_lo_old)
    else:
        raise ValueError(f"unknown negative_eta_scope: {negative_eta_scope!r}")

    eta_lo = eta_lo_init.copy()
    eta_hi = _np.full(k, _ETA_MAX)
    f_lo = _f(eta_lo)
    unreachable = f_lo <= 0.0

    if method == "illinois":
        mult_row = None if mult is None else mult[None, :]

        def _f_rows(eta_val: "_np.ndarray", sel: "_np.ndarray") -> "_np.ndarray":
            z = _np.maximum(eta_val[:, None] + eta_other[None, :], _Z_G_CLAMP)
            G = -1.0 / _np.expm1(-z)
            log_q = -_np.log(_np.expm1(z))
            p = 1.0 / (1.0 + _np.exp(-(-theta_self[sel][:, None] - theta_other_row + log_q)))
            W = p * G
            r = _np.arange(sel.shape[0])
            if mult_row is not None:
                total = (W * mult_row).sum(axis=1) - W[r, idx[sel]]
            else:
                W = W.copy()
                W[r, idx[sel]] = 0.0
                total = W.sum(axis=1)
            return total - s_target[sel]

        f_hi = _f(eta_hi)
        result = eta_hi.copy()                       # f(hi) > 0: bisection also ends at ~hi
        active = _np.where(~unreachable & (f_hi <= 0.0))[0]
        side = _np.zeros(k, dtype=_np.int64)
        for _ in range(80):
            if active.size == 0:
                break
            lo_a, hi_a = eta_lo[active], eta_hi[active]
            fl_a, fh_a = f_lo[active], f_hi[active]
            x = (lo_a * fh_a - hi_a * fl_a) / (fh_a - fl_a)
            x = _np.where(_np.isfinite(x) & (x > lo_a) & (x < hi_a), x, 0.5 * (lo_a + hi_a))
            fx = _f_rows(x, active)
            right = fx > 0.0                         # f decreasing: root to the right of x
            sd = side[active]
            eta_lo[active] = _np.where(right, x, lo_a)
            eta_hi[active] = _np.where(right, hi_a, x)
            f_lo[active] = _np.where(right, fx, _np.where(sd == -1, 0.5 * fl_a, fl_a))
            f_hi[active] = _np.where(right, _np.where(sd == 1, 0.5 * fh_a, fh_a), fx)
            side[active] = _np.where(right, 1, -1)
            done = (_np.abs(fx) <= 1e-14 * _np.maximum(s_target[active], 1.0)) | (
                (eta_hi[active] - eta_lo[active]) <= 1e-15 * _np.maximum(1.0, _np.abs(x))
            )
            result[active[done]] = x[done]
            active = active[~done]
        if active.size:
            result[active] = 0.5 * (eta_lo[active] + eta_hi[active])
        return _np.where(unreachable, eta_lo_init, result)

    for _ in range(n_bisect):
        eta_mid = 0.5 * (eta_lo + eta_hi)
        go_lo = _f(eta_mid) > 0.0
        eta_lo = _np.where(go_lo, eta_mid, eta_lo)
        eta_hi = _np.where(go_lo, eta_hi, eta_mid)

    result = 0.5 * (eta_lo + eta_hi)
    return _np.where(unreachable, eta_lo_init, result)


# -------------------------------------------------------------------------
# Main solver
# -------------------------------------------------------------------------

def _solve_decm_reduced(
    theta0: torch.Tensor,
    k_out: torch.Tensor,
    k_in: torch.Tensor,
    s_out: torch.Tensor,
    s_in: torch.Tensor,
    mult: torch.Tensor,
    tol: float = 1e-8,
    max_iter: int = 5000,
    chunk_size: int = 0,
    anderson_depth: int = 10,
    max_step: float = 0.5,
    max_time: float = 0.0,
    verbose: bool = False,
    monitor: bool = False,
    topo_weig: bool = False,
    hub_sk_threshold: float = 0.0,
    hub_bisect_max_sweeps: int = 3,
    hub_root_method: str = "bisect",
    patience_rel_tol: float = 0.0,
    block_newton_gate: float = 0.0,
    block_newton_min_rel: float | None = None,
    stall_block_newton_gate: float = 0.0,
    stall_block_newton_off_mre: float = 2e-4,
    stall_block_newton_revert: bool = True,
    z_clamp: float = 1e-6,
    weight_anderson: bool = True,
    init_best_theta: torch.Tensor | None = None,
    init_best_res: float = float("inf"),
    blowup_factor: float | None = None,
    patience: int = 750,
    noise_base: float = 1e-4,
    noise_cap_mult: float = 16.0,
    noise_growth: float = 2.0,
    max_stalls: int = 5,
    seed: int | None = None,
    diag_callback: Callable[[int, int, float], None] | None = None,
    diag_every: int = 1,
    diag_state_callback: Callable[[int, torch.Tensor, torch.Tensor], None] | None = None,
    streak_fix_threshold: int = 0,
    streak_fix_n_bisect: int = 60,
    streak_fix_n_sweeps: int = 5,
    device: str = "cpu",
    checkpoint_fn: Callable[[int, str, torch.Tensor, float], None] | None = None,
) -> SolverResult:
    """Alternating GS-Newton fixed-point solver for the DECM.

    Each iteration applies a 2-pass GS-Newton step: pass 1 updates
    (θ_out, η_out) from row sums; pass 2 updates (θ_in, η_in) from col
    sums using the freshly updated values.

    Anderson acceleration (depth ``anderson_depth``) is applied to the full
    4N vector with a feasibility guard that rejects infeasible mixes.

    Args:
        theta0:         Initial guess [θ_out|θ_in|η_out|η_in], shape (4N,).
        k_out:          Observed out-degree sequence, shape (N,).
        k_in:           Observed in-degree sequence, shape (N,).
        s_out:          Observed out-strength sequence, shape (N,).
        s_in:           Observed in-strength sequence, shape (N,).
        tol:            Convergence tolerance on the ℓ∞ relative residual (MRE = max|F_i|/target_i).
        max_iter:       Maximum number of iterations.
        chunk_size:     If > 0, use chunked computation with this row-chunk
                        size.  If 0, auto-select: dense for
                        N ≤ ``_LARGE_N_THRESHOLD``, chunked otherwise.
        anderson_depth: Anderson acceleration depth (0 = plain Newton, no blowup guard; 1 = plain Newton WITH the blowup guard / rollback; >= 2 = Anderson mixing + guard).
        max_step:       Maximum |Δ| per node per Newton step.
        max_time:       Wall-clock time limit in seconds (0 = no limit).
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
        patience_rel_tol: Relative-improvement threshold of the stall criterion (default 0.0 = previous behaviour: ANY strictly smaller MRE
                        counts as progress and resets the ``patience`` counter). With a value > 0 (e.g. 0.01) only a record that beats the
                        reference record by more than that fraction resets it, so a plateau whose record keeps creeping down by ~1e-8 per
                        iteration (q4: 1.90e-4 -> 1.77e-4 over 8000 iterations, zero [patience] events) is recognised as a stall and enters
                        the recovery ladder (soft reset, then perturbed restart). ``best_theta`` is still updated on every improvement.
        block_newton_gate: Degeneracy-reduced path only (``mult`` given). 0.0 (default) = previous behaviour, bit-identical. > 0: a group side
                        whose 2x2 (theta, eta) Hessian block is ill-conditioned, det/(A*C) < block_newton_gate, is updated with the joint 2x2
                        Newton step instead of two decoupled scalar steps (see :func:`_block_newton_deltas`). Targets the nearly saturated
                        rows (s ~ k, G ~ 1) where the scalar iteration contracts by ~1 - 1e-5 per step (q4/e1 plateau). ~0.05 selects only
                        those rows; non-hub rows only on the out side.
        stall_block_newton_gate: Degeneracy-reduced path only. 0.0 (default) = off, ladder unchanged (bit-identical). > 0 adds a stall-ladder
                        tier: after the soft reset fails within another ``patience`` window, the joint 2x2 block-Newton step is switched ON
                        (gate = this value, ``block_newton_min_rel`` as usual) from ``best_theta`` and stays on for the rest of the run;
                        if that also fails to set a record within ``patience``, the noisy restart follows. Needs ``patience_rel_tol`` > 0 on plateaus that creep down (see there).
        stall_block_newton_revert: With the block-Newton stall tier: if the ladder moves past it (no significant record within another
                        ``patience`` window after it was switched on), the joint step is switched OFF again before the next tier, so the
                        noisy restarts run with the scalar step (on q1 the joint step kept on made every restart blow up). Default True.
        stall_block_newton_off_mre: With a block-Newton gate active (static or stall tier): once the record falls below this MRE (provided the joint step was switched on above it: q4 turns it on at 2e-5 and needs it) the joint step is switched
                        OFF again (scalar Newton finishes; the joint step is unstable near ~1e-4 on e1). A static ``block_newton_gate`` (on from the start) counts as switched on at the residual the run starts from: a fresh start (large residual) has it switched off once the record falls below this value, a run resumed from below it keeps it. Default 2e-4 (same as ``solve_tool``); 0.0 = never off.
        block_newton_min_rel: With ``block_newton_gate > 0``: rows whose relative residual is already below this value are not stepped jointly
                        (they keep the scalar step). ``None`` (default) = ``0.1 * tol``; ``0.0`` = no restriction. Prevents the s == k rows,
                        whose solution lies at infinity, from marching along the soft direction (theta-c, eta+c) far beyond what their
                        residual needs (e1: eta ~ 37, theta ~ -31, then MRE > 1).
        hub_bisect_max_sweeps: Maximum Gauss-Seidel sweeps (out-hubs then in-hubs) of the per-iteration hub bisection. The first 3 sweeps
                        always run (default 3 = previous behaviour); further sweeps run only while the largest hub-eta change of the
                        last sweep exceeds 1e-13. Raise (e.g. 50) when a relaxed (negative-eta) hub sits next to a tight partner
                        (z = eta_out + eta_in ~ 1e-4): the alternation then converges very slowly and the hub's equations stay off by
                        ~1e-4 for hundreds of iterations (q4 group 1403).
        hub_root_method: Root finder of the per-iteration hub-eta equation: ``"bisect"`` (default, 60 halvings, bit-identical to earlier
                        versions) or ``"illinois"`` (regula falsi on the same bracket, same root to ~1e-14, ~3x cheaper on hub-heavy networks;
                        see :func:`_bisect_hub_eta_decm_batch`).
        hub_sk_threshold: When > 0, nodes whose observed strength-to-degree
                        ratio ``s_i / k_hat_i`` exceeds this value are treated
                        as *hub nodes* and solved exactly each iteration via 1D
                        bisection over η rather than the global Newton step.
                        Use this for networks with a few nodes that have very
                        high ``s/k`` ratios (e.g. ``s/k > 5``) which cause the
                        global solver to stagnate.  Default=0.0 (disabled).
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
                        1e-6 since 2026-09-25 (the value that resolved a
                        real stagnation on the hub-heavy N=28156 network
                        ita_election_dico3, and the one every production
                        run has used); the model's residual
                        (``max_relative_error``) and the targeted streak fix
                        use the same constant, so
                        the solver is judged by the equations it solves.
                        The previous default was 1e-8; there is no general
                        rule for picking another value -- see the DECM
                        z-clamp investigation notes
                        for the full experimental history.
        mult:           Group multiplicities, shape (M,). ``k_out``/``k_in``/``s_out``/``s_in``/``theta0`` are
                        *group-level* quantities (one entry per unique (k_out,s_out,k_in,s_in) 4-tuple) and the
                        step is :func:`_decm_step_dense_weighted` / :func:`_decm_step_chunked_weighted`.
                        ``hub_sk_threshold`` operates at group granularity (the bisection helper accepts ``mult``).
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
                        Default 1e-4 (lowered from the original 1e-2 --
                        2026-09-04: on crisi_dico1, already sitting at its
                        genuine best_theta, 1e-2 wasted 69-148x-of-best
                        recovery excursions per restart with no restart
                        ever finding anything better; 1e-4 recovers to
                        ~4-10x-of-best in the same 5-iteration window,
                        same outcome otherwise -- see
                        decm_patience_restart_instability memory).
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
                        available perturbation has been tried ``max_stalls`` times
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
        diag_callback:  Read-only diagnostic hook, ``None`` by default (no
                        behaviour change whatsoever when omitted -- not
                        even an extra branch cost worth mentioning). If
                        given, called as ``diag_callback(n_iter,
                        argmax_idx, res_norm)`` every ``diag_every``
                        iterations, where ``argmax_idx`` indexes into the
                        4G-length concatenated ``[k_out|k_in|s_out|s_in]``
                        layout (G = len(k_out), i.e. N or M depending on
                        whether this is called directly or via
                        :func:`solve_fixed_point_decm`) --
                        decode with ``block, node = divmod(argmax_idx, G)``
                        (block 0..3 = k_out/k_in/s_out/s_in). Built to
                        answer the node-population-persistence question in
                        decm_low_degree_precision_floor memory (2026-09-10
                        update) without the earlier session's mistake of
                        reimplementing the Newton step in a bespoke loop
                        (which blew up memory/was inexplicably slow) --
                        this reuses the already-correct, already-computed
                        ``F_current``/``_v_targets`` from the real loop, at
                        the cost of one extra ``argmax`` per sampled
                        iteration.
        diag_every:     Call ``diag_callback`` (and ``diag_state_callback``)
                        every this many iterations (default 1 = every
                        iteration). Irrelevant if both are ``None``.
        checkpoint_fn:  ``None`` by default (no behaviour change). If given, called as ``checkpoint_fn(n_iter, event, best_theta, best_res)`` every time
                        the solver restarts from ``best_theta`` (event = "soft_reset", "block_newton", "noisy_restart" or
                        "blowup_rollback"), but only when the record changed since the last call (no duplicate saves). Lets a long run be resumed
                        from the point the restarts return to (the degenerate wrapper passes the theta expanded to the N nodes).
        diag_state_callback: Read-only diagnostic hook, ``None`` by default
                        (no behaviour change when omitted). If given, called
                        as ``diag_state_callback(n_iter, theta, F_current)``
                        right next to ``diag_callback``, with the iterate
                        being evaluated this iteration and its full residual
                        (the exact state whose max relative residual is the
                        MRE logged for ``n_iter``). Do NOT mutate the
                        tensors; clone them if you keep them.
        streak_fix_threshold: PROTOTYPE, experimental (2026-09-11,
                        corrected 2026-09-14) -- ``0`` (default) disables
                        entirely, zero behaviour change. If positive,
                        tracks how many CONSECUTIVE iterations the SAME
                        single (block, node) has been the argmax of the
                        relative residual -- a categorical signal, found
                        empirically (see decm_low_degree_precision_floor
                        memory) to separate genuinely-frozen stretches
                        (streaks of 150+ iterations, <0.15% residual
                        movement) from healthy multi-node-alternating
                        dynamics (streaks under ~30 iterations, large
                        residual movement) far more cleanly than
                        record-based ``patience`` or the (separately,
                        already found unreliable) amplitude-ratio
                        ``freeze_ratio`` gate. Once the streak reaches
                        this threshold, applies
                        :func:`_targeted_bisection_fix` to JUST that one
                        monopolizing group AND JUST the side (out/in) the
                        argmax actually flagged -- e.g. a `k_out`/`s_out`
                        monopoly only touches that group's theta_out/
                        eta_out, never its theta_in/eta_in, since no
                        equation couples both sides of the same node (see
                        2026-09-14 correction in decm_low_degree_
                        precision_floor memory: fixing both sides
                        unconditionally, the original 2026-09-11 design,
                        triggered a persistent post-fix blowup cascade on
                        c1 -- most likely explained by that overreach, not
                        by anything specific to the node tested). O(M) not
                        O(M^2) -- cheap enough to fire often. Clears
                        Anderson history and resets the streak counter
                        after firing. Does NOT replace ``patience`` -- runs
                        alongside it, since it targets a different,
                        finer-grained trigger. quirinale_dico1 converges
                        with ``streak_fix_threshold=10``.
        streak_fix_n_bisect: Bisection halvings per stage in
                        :func:`_targeted_bisection_fix` (default 60,
                        CPU/float64, cheap since only ``streak_fix_
                        threshold``-triggered single-group subsets are
                        touched).
        streak_fix_n_sweeps: Gauss-Seidel sweeps per fix (default 5).
        device:         ``"cpu"`` (default, float64) or a torch device
                        string like ``"mps"``/``"cuda"`` (float32 -- MPS
                        has no float64 support, and CUDA float64 is slow
                        enough that float32 is the practical choice too). Covers
                        the main Newton/Anderson loop and its dense-
                        weighted step; the (CPU-only by design) targeted
                        streak-fix interoperates with any setting (its
                        inputs/outputs are converted at the boundary). The returned
                        ``SolverResult`` is always CPU/float64 regardless
                        of ``device``.

    Returns:
        :class:`~src.solvers.base.SolverResult` with the best iterate found.
    """
    if z_clamp != 1e-6:
        warnings.warn(
            f"z_clamp={z_clamp:g} != 1e-6: the model's residual (max_relative_error) clamps the pair sums at 1e-6, so for pairs "
            f"between the two values the solver's equations differ from the residual it is judged by. Use the default 1e-6.",
            stacklevel=2,
        )
    global _Z_G_CLAMP, _Z_NEWTON_FLOOR
    _Z_G_CLAMP = z_clamp
    _Z_NEWTON_FLOOR = z_clamp

    _dev = torch.device(device)
    _dtype = torch.float64 if _dev.type == "cpu" else torch.float32

    # Convert inputs. NOTE: single combined .to(dtype=, device=) call --
    # unlike the CPU-return conversions elsewhere in this file (which must
    # be split into two .to() calls to avoid a direct MPS->float64 cast),
    # the combined form works fine for CPU-float64 -> MPS-float32 (verified
    # empirically); it's specifically the *chained* `.to(device=X).to(dtype=Y)`
    # form that fails here, since the intermediate step would try to hold a
    # float64 tensor on MPS before the dtype conversion ever runs.
    def _t(x):
        if isinstance(x, torch.Tensor):
            return x.to(dtype=_dtype, device=_dev)
        return torch.tensor(x, dtype=_dtype, device=_dev)

    k_out = _t(k_out)
    k_in = _t(k_in)
    s_out = _t(s_out)
    s_in = _t(s_in)
    theta = _t(theta0).clone()
    mult = _t(mult)

    N = k_out.shape[0]

    zero_k_out = k_out == 0
    zero_k_in = k_in == 0
    zero_s_out = s_out == 0
    zero_s_in = s_in == 0

    # Restore fixed nodes in the initial theta
    _tmax = torch.full((N,), _THETA_MAX, dtype=_dtype, device=_dev)
    _emax = torch.full((N,), _ETA_MAX, dtype=_dtype, device=_dev)
    theta[:N] = torch.where(zero_k_out, _tmax, theta[:N])
    theta[N:2*N] = torch.where(zero_k_in, _tmax, theta[N:2*N])
    theta[2*N:3*N] = torch.where(zero_s_out, _emax, theta[2*N:3*N])
    theta[3*N:] = torch.where(zero_s_in, _emax, theta[3*N:])

    # Clamp to valid range
    theta[:2*N] = theta[:2*N].clamp(-_THETA_MAX, _THETA_MAX)
    theta[2*N:] = theta[2*N:].clamp(_ETA_MIN, _ETA_MAX)

    # Decide chunked vs dense (PyTorch path only)
    if chunk_size == 0:
        effective_chunk = 0 if N <= _LARGE_N_THRESHOLD else _DEFAULT_CHUNK
    else:
        effective_chunk = chunk_size

    # Joint 2x2 step: rows already below 0.1*tol keep the scalar step (see _block_exclude); explicit 0.0 disables the restriction.
    _bn_min_rel = (0.1 * tol if block_newton_min_rel is None else float(block_newton_min_rel)) if block_newton_gate > 0.0 else 0.0

    # Mutable (gate, min_rel) read by the step closures: a stall-ladder tier can switch the joint 2x2 step on mid-run.
    _bn = [block_newton_gate, _bn_min_rel]
    _bn_on_res = [float('inf')]   # record at which the joint step was switched on (off_mre acts only if that was above it); a static gate counts as switched on at the starting residual
    _bn_stall_used = [False]   # stall tier already fired (so switching the gate off again does not re-arm it)
    _stall_bn_min_rel = 0.1 * tol if block_newton_min_rel is None else float(block_newton_min_rel)

    # Step function with bound arguments
    if effective_chunk > 0:
        def _step(th, hub_out_mask=None, hub_in_mask=None):
            return _decm_step_chunked_weighted(
                th, k_out, k_in, s_out, s_in,
                zero_k_out, zero_k_in, zero_s_out, zero_s_in,
                effective_chunk, max_step, mult,
                hub_out_mask=hub_out_mask, hub_in_mask=hub_in_mask, block_newton_gate=_bn[0], block_newton_min_rel=_bn[1],
            )
    else:
        def _step(th, hub_out_mask=None, hub_in_mask=None):
            return _decm_step_dense_weighted(
                th, k_out, k_in, s_out, s_in,
                zero_k_out, zero_k_in, zero_s_out, zero_s_in,
                max_step, mult,
                hub_out_mask=hub_out_mask, hub_in_mask=hub_in_mask, block_newton_gate=_bn[0], block_newton_min_rel=_bn[1],
            )

    # Anderson mixing needs per-component weights in the degeneracy-reduced
    # path: each of the 4 blocks [theta_out|theta_in|eta_out|eta_in] has one
    # entry per group, and a group of multiplicity mult[g] should count
    # mult[g] times in the mixing least-squares fit -- exactly like it would
    # if those mult[g] identical nodes were each their own row in the
    # unreduced (per-node) system (see _anderson_mixing's docstring).
    _anderson_weights = torch.cat([mult, mult, mult, mult]) if weight_anderson else None

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
    if block_newton_gate > 0.0 and best_theta_res != float("inf"):
        _bn_on_res[0] = best_theta_res   # static gate: "switched on" at the residual the run starts from (off_mre acts only above it)

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
    _stall_ref: float = best_theta_res     # reference record of the stall criterion (see patience_rel_tol)
    # 0/1/2: how far the current stagnation episode has escalated since the
    # last genuine improvement -- 0 = nothing tried yet (do the cheap
    # no-noise soft reset next), 1 = soft reset already tried (switch the
    # block-Newton step on, if `stall_block_newton_gate > 0` and it is not
    # on yet; otherwise go straight to noise), 2 = only noise left.
    _stall_tier = 0

    def _revert_stall_bn() -> None:
        """The block-Newton stall tier had its window and set no significant record: back to the scalar step."""
        if stall_block_newton_revert and _bn_stall_used[0] and _bn[0] > 0.0 and block_newton_gate == 0.0:
            _bn[0] = 0.0
            if verbose:
                print(f"[patience] block-Newton OFF (no significant record within patience) at iter {n_iter}.")

    # Argmax-identity streak tracking for `streak_fix_threshold` (see its
    # docstring) -- independent of `_stall_tier`/`iters_since_improve`,
    # since it tracks a different, finer-grained signal (which single
    # equation has been worst, consecutively) rather than lack of a global
    # record. `None` sentinel so the very first iteration never counts as
    # a streak of length 1 against nothing.
    _streak_id: int | None = None
    _streak_len = 0
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

    _ckpt_last_res = float("inf")

    def _ckpt(event: str) -> None:
        """Report `best_theta` to `checkpoint_fn` when a restart returns to it and the record changed since the last report."""
        nonlocal _ckpt_last_res
        if checkpoint_fn is not None and best_theta_res < _ckpt_last_res:
            _ckpt_last_res = best_theta_res
            checkpoint_fn(n_iter, event, best_theta.clone(), best_theta_res)

    def _perturbed_restart() -> tuple[torch.Tensor, bool]:
        """Build a noisy restart around `best_theta`; escalate + maybe give up."""
        _ckpt("noisy_restart")
        nonlocal restarts, stalls_at_cap
        restarts += 1
        noise_mult = min(noise_growth ** (restarts - 1), noise_cap_mult)
        give_up = False
        if noise_mult >= noise_cap_mult:
            stalls_at_cap += 1
            give_up = stalls_at_cap >= max_stalls
        noise_scale = noise_base * noise_mult
        # torch.from_numpy always gives a CPU tensor -- move it onto
        # best_theta's own device/dtype before combining. Order depends on
        # the target device (float64 can never touch MPS, even transiently).
        noise = torch.from_numpy(
            _restart_rng.normal(scale=noise_scale, size=tuple(best_theta.shape))
        )
        noise = (
            noise.to(device=_dev).to(dtype=_dtype) if _dev.type == "cpu"
            else noise.to(dtype=_dtype).to(device=_dev)
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
        _eta_r = theta_restart[2 * N :]
        theta_restart[2 * N :] = torch.where(
            _hub_eta_mask, _eta_r.clamp(-_ETA_MAX, _ETA_MAX), _eta_r.clamp(_ETA_MIN, _ETA_MAX)
        )
        _ero, _eri = _project_pair_floor(
            theta_restart[2 * N : 3 * N], theta_restart[3 * N :], mult, _Z_G_CLAMP, _hub_eta_mask[:N], _hub_eta_mask[N:]
        )
        theta_restart[2 * N : 3 * N] = _ero
        theta_restart[3 * N :] = _eri
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

    _hub_active = hub_sk_threshold > 0.0
    _hub_out_mask: torch.Tensor | None = None   # shape (N,) bool
    _hub_in_mask: torch.Tensor | None = None
    # Group multiplicities as numpy, for the weighted hub-bisection sum
    # (degeneracy-reduced path only; None reproduces the per-node behaviour).
    # (.cpu() first: these may live on a non-CPU device under `device=...`,
    # and hub bisection itself always runs via numpy regardless.)
    _mult_np = mult.detach().to(device="cpu").to(dtype=torch.float64).numpy()
    # Precomputed once: target arrays as numpy, reused by every hub-bisection
    # call each iteration.
    _s_out_np = s_out.detach().to(device="cpu").to(dtype=torch.float64).numpy()
    _s_in_np = s_in.detach().to(device="cpu").to(dtype=torch.float64).numpy()

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
        _hub_4N_mask = torch.zeros(4 * N, dtype=torch.bool, device=_dev)
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
        _hub_4N_mask = torch.zeros(4 * N, dtype=torch.bool, device=_dev)

    # η-only view of the hub mask (2N-length, matches theta[2*N:]) -- used
    # to exempt hub-relaxed components (negative-eta relaxation, see
    # _bisect_hub_eta_decm_batch) from the post-Anderson-mixing η≥_ETA_MIN
    # floors below. Those floors predate the relaxation and are a safety
    # net for the rest of the network; for hub components they must not
    # run, since _apply_hub_bisection (called right after, unconditionally
    # for every hub node) is what actually establishes the correct η --
    # flooring here first would silently discard whatever negative-eta
    # progress the previous iteration's damped bisection made, forcing it
    # to re-climb from _ETA_MIN every single iteration (see
    # decm_hub_eta_floor_reset_bug memory for the empirical trace that
    # found this).
    _hub_eta_mask = _hub_4N_mask[2 * N :]

    # The start-of-solve clamp above floors EVERY eta to _ETA_MIN, which wipes
    # the negative hub eta (negative-eta relaxation) that a checkpoint
    # legitimately carries: resuming e1 from MRE 6.66e-4 started at 0.78, q4
    # from 2.86e-3 at 0.74, and the solver then had to re-climb them under
    # _damp_hub_step. Hub eta are owned by the hub bisection, so take them
    # from the caller's theta0 unclamped (non-hub eta keep the floor).
    if _hub_active and (_hub_out_indices or _hub_in_indices):
        _eta0 = _t(theta0)[2 * N :].clamp(-_ETA_MAX, _ETA_MAX)
        theta[2 * N :] = torch.where(_hub_eta_mask, _eta0, theta[2 * N :])
        if init_best_theta is None:
            best_theta = theta.clone()

    # The pair-sum condition z_ij = eta_out_i + eta_in_j > 0 must hold for the starting point too (theta0 is user input):
    # raise the offending non-hub eta instead of letting the equations clamp the pair to the floor.
    _eo0, _ei0 = _project_pair_floor(
        theta[2 * N : 3 * N], theta[3 * N :], mult, _Z_G_CLAMP, _hub_eta_mask[:N], _hub_eta_mask[N:]
    )
    _n_raised = int((_eo0 != theta[2 * N : 3 * N]).sum() + (_ei0 != theta[3 * N :]).sum())
    if _n_raised:
        theta[2 * N : 3 * N] = _eo0
        theta[3 * N :] = _ei0
        if init_best_theta is None:
            best_theta = theta.clone()
        if verbose:
            print(f"[theta0] {_n_raised} eta raised so that every pair sum eta_out_i + eta_in_j >= {_Z_G_CLAMP:g}.")

    if _hub_active and (_hub_out_indices or _hub_in_indices):
        # The raw per-node Newton step (whichever _decm_step_* variant is
        # bound above -- dense/chunked/weighted, all share the same
        # `eta_*_new = (...).clamp(_ETA_MIN, _ETA_MAX)` pattern) floors
        # EVERY node's η to _ETA_MIN internally, predating the negative-eta
        # relaxation. For hub-flagged nodes that floor silently overwrites
        # whatever negative η the previous iteration's damped bisection
        # established, forcing _apply_hub_bisection to re-climb from
        # _ETA_MIN every single iteration regardless of the Anderson-mixing
        # exemption above (verified empirically: cur_eta reset to exactly
        # 1e-10 every iteration even with that exemption in place -- see
        # decm_hub_eta_floor_reset_bug memory). Hub-flagged η components are
        # exclusively owned by _apply_hub_bisection; wrap _step so the raw
        # map never touches them -- pass the input's own value straight
        # through instead of whatever the generic Newton/floor logic
        # proposes.
        _step_raw = _step

        def _step(th, _step_raw=_step_raw):
            theta_new, F_current = _step_raw(th, hub_out_mask=_hub_out_mask, hub_in_mask=_hub_in_mask)
            theta_new = torch.cat([
                theta_new[: 2 * N],
                torch.where(_hub_eta_mask, th[2 * N :], theta_new[2 * N :]),
            ])
            return theta_new, F_current

    # Every Newton step output must satisfy the pair-sum condition (non-hub eta are raised where needed; hub eta keep the
    # feasible values given by the hub bisection).
    _step_unprojected = _step

    def _step(th, _inner=_step_unprojected):
        theta_new, F_current = _inner(th)
        _eo, _ei = _project_pair_floor(
            theta_new[2 * N : 3 * N], theta_new[3 * N :], mult, _Z_G_CLAMP, _hub_eta_mask[:N], _hub_eta_mask[N:]
        )
        return torch.cat([theta_new[: 2 * N], _eo, _ei]), F_current

    def _damp_hub_step(
        cur_eta: "np.ndarray",
        target_eta: "np.ndarray",
        eta_other: "np.ndarray",
        idx: "np.ndarray",
    ) -> "np.ndarray":
        """Limit a bisected hub-eta update that dips below the old eta>=0
        floor (i.e. one now using the negative-eta relaxation).

        v1 of this damping (kept in decm-negative-eta-gauge history) capped
        the per-call SHRINKAGE OF z_min at 50% (reusing _Z_NEWTON_FRAC) --
        empirically that had ZERO effect (the real jump only needed ~35% of
        z_min) and the network destabilized identically to no damping at
        all. Root cause: G(z) = 1/(1-e^-z) ~ 1/z near small z, so its
        *relative* sensitivity to a z change is what actually perturbs a
        neighbouring node's ordinary (undamped) Newton step -- a 35% drop
        in z_min already produces a ~54% jump in G, i.e. the neighbour's
        expected weight from this hub jumps by more than half in one call.
        v2 (this one) directly bounds the relative growth of G at the
        single tightest partner to _G_DAMP_FRAC per call, converting that
        back to an eta floor via G's exact inverse -- not a z-space proxy.
        Targets that stay at or above the old floor (the overwhelming
        majority of hub nodes) are completely untouched, so this cannot
        change behaviour on any network that has no unreachable node under
        the old eta>=0-only rule (see decm_project_convergence_bar memory:
        already-converging networks must not regress).
        """
        import numpy as _np

        needs_damp = target_eta < _ETA_MIN
        if not needs_damp.any():
            return target_eta
        _order = _np.argsort(eta_other)
        _min1_pos, _min1_val = _order[0], eta_other[_order[0]]
        _min2_val = eta_other[_order[1]] if eta_other.shape[0] > 1 else _min1_val
        _min_other_excl_self = _np.where(idx == _min1_pos, _min2_val, _min1_val)

        z_min_cur = _np.maximum(cur_eta + _min_other_excl_self, _Z_G_CLAMP)
        G_cur = -1.0 / _np.expm1(-z_min_cur)
        G_max_allowed = G_cur * (1.0 + _G_DAMP_FRAC)
        # Inverse of G(z) = 1/(1-e^-z):  z = -log(1 - 1/G), valid for G>1
        # (always true here since G_cur >= 1/z_min_cur and z_min_cur is
        # small). G_max_allowed > G_cur > 1 always holds by construction.
        z_floor_allowed = -_np.log1p(-1.0 / G_max_allowed)
        eta_floor_allowed = z_floor_allowed - _min_other_excl_self

        return _np.where(needs_damp, _np.maximum(target_eta, eta_floor_allowed), target_eta)

    def _apply_hub_bisection(theta_fp: torch.Tensor) -> torch.Tensor:
        """Exact 1D-bisection correction of hub eta components, vectorized
        across all hub nodes at once (see :func:`_bisect_hub_eta_decm_batch`).
        Applied to the *final* ``theta_next`` for this iteration (after
        Anderson mixing, restarts, or rollback have all already settled on
        it -- see the single call site near the end of the main loop) so
        the bisection is always against the opposing values that actually
        carry forward, not a pre-mixing snapshot that then drifts.

        Targets below the old eta>=0 floor (the negative-eta relaxation,
        see _bisect_hub_eta_decm_batch) are damped via _damp_hub_step
        instead of applied instantly -- see its docstring."""
        if not (_hub_active and (_hub_out_indices or _hub_in_indices)):
            return theta_fp
        # This whole function is numpy-based by design (in-place mutation) --
        # bring theta_fp to CPU/float64 first (safe even if it's already
        # there; 2-step to avoid the direct MPS->float64 cast failure) and
        # convert the result back to the caller's own device/dtype at the end.
        _th_fp_np = theta_fp.detach().to(device="cpu").to(dtype=torch.float64).numpy()
        # Sweeps 1-3 always run over ALL hubs (previous behaviour). Further sweeps (up to hub_bisect_max_sweeps) only re-solve the
        # hubs whose eta was still moving (> _HUB_SWEEP_TOL) in the previous sweep -- typically a relaxed hub and its tight partner
        # (z = eta_out + eta_in ~ 1e-4, G ~ 1e4): that pair is hypersensitive and converges slowly under out/in alternation
        # (q4 group 1403, e1 group 3174), while the ~90 settled hubs need no more work.
        def _partner_min(eta_other: "np.ndarray", idx: "np.ndarray") -> "np.ndarray":
            # smallest partner eta of each hub, the hub's own column excluded
            _o = _np_hub.argsort(eta_other)
            if eta_other.shape[0] < 2:
                return _np_hub.zeros(idx.shape)
            return _np_hub.where(idx == _o[0], eta_other[_o[1]], eta_other[_o[0]])

        def _hub_z(eta_new: "np.ndarray", eta_other: "np.ndarray", idx: "np.ndarray") -> "np.ndarray":
            # tightest partner pair sum of each hub (the hub's own column excluded), floored like the damping does
            _o = _np_hub.argsort(eta_other)
            _mo = _np_hub.where(idx == _o[0], eta_other[_o[1]], eta_other[_o[0]]) if eta_other.shape[0] > 1 else _np_hub.zeros(idx.shape)
            return _np_hub.maximum(eta_new + _mo, _Z_G_CLAMP)

        _hub_eta_pos = _np_hub.concatenate([2 * N + _hub_out_idx_arr, 3 * N + _hub_in_idx_arr])
        _xs: list = []
        for _sweep in range(max(3, hub_bisect_max_sweeps)):
            _x_prev = _th_fp_np[_hub_eta_pos].copy()
            if _hub_out_indices:
                _target_eta_out = _bisect_hub_eta_decm_batch(
                    hub_indices=_hub_out_indices,
                    theta_self_all=_th_fp_np[:N],
                    theta_other=_th_fp_np[N:2 * N],
                    eta_other=_th_fp_np[3 * N:],
                    s_target_all=_s_out_np,
                    mult=_mult_np,
                    method=hub_root_method,
                )
                _cur_eta_out = _th_fp_np[2 * N + _hub_out_idx_arr]
                _new_eta_out = _damp_hub_step(
                    _cur_eta_out, _target_eta_out, _th_fp_np[3 * N:], _hub_out_idx_arr
                )
                _th_fp_np[2 * N + _hub_out_idx_arr] = _new_eta_out
            if _hub_in_indices:
                _target_eta_in = _bisect_hub_eta_decm_batch(
                    hub_indices=_hub_in_indices,
                    theta_self_all=_th_fp_np[N:2 * N],
                    theta_other=_th_fp_np[:N],
                    eta_other=_th_fp_np[2 * N:3 * N],
                    s_target_all=_s_in_np,
                    mult=_mult_np,
                    method=hub_root_method,
                )
                _cur_eta_in = _th_fp_np[3 * N + _hub_in_idx_arr]
                _new_eta_in = _damp_hub_step(
                    _cur_eta_in, _target_eta_in, _th_fp_np[2 * N:3 * N], _hub_in_idx_arr
                )
                _th_fp_np[3 * N + _hub_in_idx_arr] = _new_eta_in
            if _sweep < 2:
                continue
            # Sweeps 1-3 are the previous behaviour. Beyond that (hub_bisect_max_sweeps > 3) the out/in alternation of a
            # relaxed hub with a tight partner (z ~ 1e-4, G ~ 1e4) contracts by only ~0.985 per sweep (q4 group 1403, e1
            # group 3174): plain sweeping would need ~700 sweeps. Stop when every hub moved by < _HUB_SWEEP_TOL * z, and
            # otherwise extrapolate the fixed point of the sweep map (vector Aitken, order 1) every 3rd sweep.
            _x_now = _th_fp_np[_hub_eta_pos].copy()
            _zt = _np_hub.concatenate([
                _hub_z(_x_now[:_hub_out_idx_arr.size], _th_fp_np[3 * N:], _hub_out_idx_arr),
                _hub_z(_x_now[_hub_out_idx_arr.size:], _th_fp_np[2 * N:3 * N], _hub_in_idx_arr),
            ])
            _rel_chg = float((_np_hub.abs(_x_now - _x_prev) / _zt).max())
            if _rel_chg < _HUB_SWEEP_TOL:
                break
            _xs = (_xs + [_x_now])[-3:]
            if len(_xs) == 3:
                _d1, _d2 = _xs[1] - _xs[0], _xs[2] - _xs[1]
                _n1 = float(_d1 @ _d1)
                _rho = float(_d2 @ _d1) / _n1 if _n1 > 0.0 else 0.0
                if _rho > 0.3:
                    _rho = min(_rho, 0.999)
                    _xe = _xs[2] + (_rho / (1.0 - _rho)) * _d2
                    _eo, _ei = _th_fp_np[3 * N:], _th_fp_np[2 * N:3 * N]
                    _floor = _np_hub.concatenate([
                        _Z_G_CLAMP - (_zt[:_hub_out_idx_arr.size] * 0 + _partner_min(_eo, _hub_out_idx_arr)),
                        _Z_G_CLAMP - (_zt[_hub_out_idx_arr.size:] * 0 + _partner_min(_ei, _hub_in_idx_arr)),
                    ]).clip(min=-_ETA_MAX)
                    _th_fp_np[_hub_eta_pos] = _np_hub.clip(_xe, _floor, _ETA_MAX)
                _xs = []
        # The order must depend on the target device (float64 can never touch
        # MPS, even transiently) -- here the source is always CPU/float64 (from the
        # numpy round-trip above), so only the "target is non-cpu" direction
        # (dtype first, then device) can actually be exercised, but both are
        # handled for robustness.
        _th_fp_t = torch.from_numpy(_th_fp_np)
        return (
            _th_fp_t.to(device=_dev).to(dtype=_dtype) if _dev.type == "cpu"
            else _th_fp_t.to(dtype=_dtype).to(device=_dev)
        )

    # Anderson exclusion mask: hub nodes get a component solved exactly by
    # bisection *after* mixing (see the main loop), so their raw (unstable,
    # unbisected) Newton-step contribution must not feed into the
    # least-squares mixing fit used for everyone else -- excluded here,
    # corrected properly by bisection afterward.
    _exclude_4N_mask = _hub_4N_mask

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

            # Hub bisection is applied once, at the very end of the loop
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

            _need_argmax = (
                (diag_callback is not None and n_iter % diag_every == 0)
                or streak_fix_threshold > 0
            )
            _argmax_idx = -1
            if _need_argmax:
                _rel_diag = torch.full_like(_v_targets, -1.0)
                _rel_diag[_v_nonzero] = F_current.abs()[_v_nonzero] / _v_targets[_v_nonzero]
                _argmax_idx = int(_rel_diag.argmax().item())

            if diag_callback is not None and n_iter % diag_every == 0:
                diag_callback(n_iter, _argmax_idx, res_norm)
            if diag_state_callback is not None and n_iter % diag_every == 0:
                diag_state_callback(n_iter, theta, F_current)

            if streak_fix_threshold > 0:
                # Key the streak on (node, side), not on the single equation:
                # a node's k and s equations on one side share the same
                # (theta, eta) pair and typically alternate as the argmax
                # (q4 plateau: k_in/s_in of one group, longest same-equation
                # streak = 1, so the fix never fired).
                _streak_key = (_argmax_idx % N) + N * ((_argmax_idx // N) % 2)
                if _streak_key == _streak_id:
                    _streak_len += 1
                else:
                    _streak_id = _streak_key
                    _streak_len = 1

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
                if block_newton_gate > 0.0 and best_theta_res == float("inf"):
                    _bn_on_res[0] = res_norm   # static gate, fresh start: the residual the run starts from
                best_theta_res = res_norm
                best_theta = theta.clone()
                if stall_block_newton_off_mre > 0.0 and _bn[0] > 0.0 and _bn_on_res[0] > stall_block_newton_off_mre > best_theta_res:
                    _bn[0] = 0.0
                    _bn_stall_used[0] = True
                    if verbose:
                        print(f"[patience] block-Newton OFF (record {best_theta_res:.3e} < {stall_block_newton_off_mre:.1e}) at iter {n_iter}.")
                progressed_since_restart = True
                if res_norm < _stall_ref * (1.0 - patience_rel_tol):
                    # A SIGNIFICANT record (beats the reference by more than patience_rel_tol; any improvement when it is 0).
                    _stall_ref = res_norm
                    restarts = 0
                    stalls_at_cap = 0
                    iters_since_improve = 0
                    _stall_tier = 0
                else:
                    iters_since_improve += 1
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
                if _stall_tier == 0:
                    # Tier 1 (cheapest): no-noise soft reset.
                    _ckpt("soft_reset")
                    if verbose:
                        print(
                            f"[patience] stalled {patience} iters with no "
                            f"{'improvement' if patience_rel_tol <= 0 else f'>{patience_rel_tol:.3g} relative improvement'}"
                            f" -- soft reset (Anderson clear, no "
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
                    _stall_tier = 1
                    iters_since_improve = 0
                    _and_g.clear()
                    _and_r.clear()
                elif _stall_tier == 1 and stall_block_newton_gate > 0.0 and _bn[0] == 0.0 and not _bn_stall_used[0]:
                    # Mode switch (not a kick): from now on the nearly saturated rows get the joint 2x2 (theta, eta) step.
                    _ckpt("block_newton")
                    if verbose:
                        print(
                            f"[patience] soft reset failed -- block-Newton ON "
                            f"(gate={stall_block_newton_gate}) from best={best_theta_res:.3e} at iter {n_iter}."
                        )
                    _bn[0] = stall_block_newton_gate
                    _bn_stall_used[0] = True
                    _bn_on_res[0] = best_theta_res
                    _bn[1] = _stall_bn_min_rel
                    _patience_restart_theta = best_theta.clone()
                    _stall_tier += 1
                    iters_since_improve = 0
                    _and_g.clear()
                    _and_r.clear()
                else:
                    # Last tier: escalating noisy restart.
                    _revert_stall_bn()
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
                    _stall_tier = 0
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
            elif anderson_depth >= 1:
                # depth 1 = no mixing (the history never reaches 2 entries) but the blowup guard below IS active: it used to live only
                # in the depth > 1 branch, so a depth-1 run could diverge (e1: MRE 2.6e-2 -> 0.77) with no [blowup] rollback at all.
                # Blowup guard -- skipped for exactly one iteration right
                # after a perturbed restart (see `_post_restart_reset`
                # above): the restart's own residual is deliberately large
                # by design and must not be judged as a "blowup" against the
                # pre-restart record.
                _blowup_recovered = False
                if (
                    (len(_and_g) >= 2 or anderson_depth == 1)     # depth 1 has no history to wait for
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
                    _ckpt("blowup_rollback")
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
                    # Hub exclusion: zero their components in r_k so
                    # Anderson mixing weights focus on convergence of the
                    # rest of the network.
                    if _hub_active:
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

                        # Enforce η ≥ _ETA_MIN after mixing -- hub-relaxed
                        # components exempted, see _hub_eta_mask above.
                        _eta_slice = theta_next[2 * N :]
                        _eta_floored = torch.where(
                            _hub_eta_mask, _eta_slice, _eta_slice.clamp(min=_ETA_MIN)
                        )
                        theta_next[2 * N :] = _eta_floored
                        theta_next = theta_next.clamp(-_THETA_MAX, _THETA_MAX)
                        # Restore η part clamp (same hub exemption)
                        _eta_slice = theta_next[2 * N :]
                        theta_next[2 * N :] = torch.where(
                            _hub_eta_mask, _eta_slice, _eta_slice.clamp(min=_ETA_MIN, max=_ETA_MAX)
                        )

                        # Geometric floor on η: don't go below fraction of
                        # theta_fp's η (same hub exemption)
                        eta_floor_mix = (theta_fp[2 * N :] * _ANDERSON_THETA_FLOOR).clamp(min=_ETA_MIN)
                        _eta_slice = theta_next[2 * N :]
                        theta_next_eta = torch.where(
                            _hub_eta_mask, _eta_slice, torch.maximum(_eta_slice, eta_floor_mix)
                        )
                        theta_next = torch.cat([theta_next[:2 * N], theta_next_eta])

                        # Feasibility guard: if min(η_out) + min(η_in) < floor → reject mix
                        eta_out_mix = theta_next[2 * N : 3 * N]
                        eta_in_mix = theta_next[3 * N :]
                        z_min_and = _min_pair_z(eta_out_mix, eta_in_mix, mult)
                        if z_min_and < _Z_NEWTON_FLOOR:
                            theta_next = theta_fp
                            _and_g.clear()
                            _and_r.clear()
                    else:
                        theta_next = theta_fp
            else:
                theta_next = theta_fp

            # Hub bisection: applied once here, to the *final*
            # theta_next for this iteration -- whichever branch produced it
            # (normal Anderson mix, no-history-yet, isolated-blowup
            # rollback, or a patience/blowup-escalation perturbed restart).
            # Solving against theta_next's own opposing values (not a
            # pre-mixing snapshot) is what makes this self-consistent: the
            # exact fit is against the values that actually carry forward
            # into the next iteration, not ones that then drift during
            # mixing. See _apply_hub_bisection.
            if _hub_active:
                theta_next = _apply_hub_bisection(theta_next)

            if streak_fix_threshold > 0 and _streak_len >= streak_fix_threshold:
                _fix_group = torch.tensor([_streak_id % N], dtype=torch.long)
                _fix_side_idx = _streak_id // N          # 0 = out side, 1 = in side
                _fix_block = "out-side" if _fix_side_idx == 0 else "in-side"
                # A node's own theta_out/eta_out appear ONLY in its own
                # k_out/s_out equations, and theta_in/eta_in ONLY in
                # k_in/s_in -- no equation couples both sides of the same
                # node, so only fix the side the argmax actually flagged
                # (2026-09-14 correction: fixing both sides unconditionally
                # was the likely cause of the post-fix blowup cascade
                # documented in decm_low_degree_precision_floor memory).
                _fix_side = "out" if _fix_side_idx == 0 else "in"
                if verbose:
                    print(
                        f"[streak-fix] group {_fix_group.item()} ({_fix_block}) monopolized "
                        f"the argmax for {_streak_len} consecutive iters -- targeted bisection "
                        f"fix (side={_fix_side!r}) at iter {n_iter}.",
                        flush=True,
                    )
                # _targeted_bisection_fix is CPU/float64-only internally and
                # always returns CPU/float64 -- convert back to this loop's
                # own device/dtype (a no-op when device="cpu", the default).
                # Order depends on the target device (float64 can never touch
                # MPS, even transiently, so "target is cpu" and "target is
                # non-cpu" need opposite .to() call orders).
                _fix_result = _targeted_bisection_fix(
                    theta_next, k_out, k_in, s_out, s_in, mult, _fix_group, _fix_side,
                    n_sweeps=streak_fix_n_sweeps, n_bisect=streak_fix_n_bisect,
                )
                theta_next = (
                    _fix_result.to(device=_dev).to(dtype=_dtype) if _dev.type == "cpu"
                    else _fix_result.to(dtype=_dtype).to(device=_dev)
                )
                _streak_len = 0
                _streak_id = None
                _and_g.clear()
                _and_r.clear()

            theta = theta_next

    finally:
        elapsed = time.perf_counter() - t0
        _peak_ram_monitor.__exit__(None, None, None)
        peak_ram = _peak_ram_monitor.peak_bytes

    return SolverResult(
        # NOTE: must split device+dtype into two .to() calls -- combining them
        # in one call fails when converting FROM MPS to float64 directly.
        theta=theta.detach().to(device="cpu").to(dtype=torch.float64).numpy(),
        best_theta=best_theta.detach().to(device="cpu").to(dtype=torch.float64).numpy(),
        converged=converged,
        iterations=n_iter,
        residuals=residuals,
        elapsed_time=elapsed,
        peak_ram_bytes=peak_ram,
        message=message,
        best_mre=best_theta_res,
    )


def solve_fixed_point_decm(
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
    chunk_size: int = 0,
    verbose: bool = False,
    monitor: bool = False,
    topo_weig: bool = False,
    weight_anderson: bool = True,
    hub_sk_threshold: float = 0.0,
    hub_bisect_max_sweeps: int = 3,
    hub_root_method: str = "bisect",
    patience_rel_tol: float = 0.0,
    block_newton_gate: float = 0.0,
    block_newton_min_rel: float | None = None,
    stall_block_newton_gate: float = 0.0,
    stall_block_newton_off_mre: float = 2e-4,
    stall_block_newton_revert: bool = True,
    z_clamp: float = 1e-6,
    init_best_theta: torch.Tensor | None = None,
    init_best_res: float = float("inf"),
    blowup_factor: float | None = None,
    patience: int = 750,
    noise_base: float = 1e-4,
    noise_cap_mult: float = 16.0,
    noise_growth: float = 2.0,
    max_stalls: int = 5,
    seed: int | None = None,
    diag_callback: Callable[[int, int, float], None] | None = None,
    diag_every: int = 1,
    diag_state_callback: Callable[[int, torch.Tensor, torch.Tensor], None] | None = None,
    streak_fix_threshold: int = 0,
    streak_fix_n_bisect: int = 60,
    streak_fix_n_sweeps: int = 5,
    device: str = "cpu",
    checkpoint_fn: Callable[[int, str, torch.Tensor, float], None] | None = None,
) -> SolverResult:
    """Degeneracy-reduced alternating GS-Newton solver for the DECM.

    Groups nodes that share the exact same (k_out, s_out, k_in, s_in)
    4-tuple -- which are proven to share identical (theta_out, eta_out,
    theta_in, eta_in) at the true DECM solution, see the module-level note
    above :func:`compute_degeneracy_groups` -- and solves the resulting
    M-unknown reduced system (M = number of unique 4-tuples <= N) instead
    of the full 4N-unknown system. The returned ``SolverResult`` is
    expanded back to the original per-node shape (4N,).

    ``hub_sk_threshold`` IS supported: hub identification and its exact 1D
    bisection operate at group granularity, with the bisection sum weighted
    by group multiplicity (see :func:`_bisect_hub_eta_decm`'s ``mult``
    parameter). A group's degree/strength is shared by every member node
    by construction, so hub membership at group granularity is exactly
    equivalent to per-node membership.

    ``device`` ("cpu", "mps", "cuda") selects where the main Newton/Anderson loop runs (float32 off the CPU).

    Args:
        theta0: Initial guess [theta_out|theta_in|eta_out|eta_in], shape
            (4N,). Nodes sharing a 4-tuple should already carry identical
            values here (true for the standard "degrees"-based initial
            guess); if not, one representative member's value is used for
            the whole group.
        k_out, k_in, s_out, s_in: Observed sequences, each shape (N,).
        z_clamp: Forwarded to :func:`_solve_decm_reduced` unchanged
            (operates at group/M granularity here, since the reduced eta
            values are already one entry per degeneracy group). See its
            docstring for the full explanation.
        (all other args): see :func:`_solve_decm_reduced`.
        init_best_theta: Optional externally-supplied record theta, same
            expanded per-node shape (4N,) as ``theta0``/``best_theta``
            (i.e. *not* pre-reduced) -- reduced internally the same way as
            ``theta0``. Since a genuine cross-chunk record from this same
            reduced solver is always exactly uniform within each degeneracy
            group already, this reduction is exact (not an approximation,
            unlike the analogous step for an arbitrary ``theta0``). See
            :func:`_solve_decm_reduced`'s ``init_best_theta`` docs for why
            this matters.
        init_best_res: The residual (MRE) achieved by ``init_best_theta``.
        blowup_factor, patience, noise_base, noise_cap_mult, max_stalls, seed:
            Blowup-guard threshold and perturbed restart on stagnation/
            repeated-blowup, operating at group (M-length) granularity --
            see :func:`_solve_decm_reduced`'s docstring for the full
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

    def _checkpoint_fn_g(n_iter_: int, event: str, theta_m: torch.Tensor, res_: float) -> None:
        """Expand the group-level best_theta to the N nodes before handing it to the user's ``checkpoint_fn``."""
        if checkpoint_fn is not None:
            checkpoint_fn(n_iter_, event, theta_m.reshape(4, M)[:, group_of].reshape(-1).clone(), res_)

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

    result_g = _solve_decm_reduced(
        theta0_g,
        k_out_g, k_in_g, s_out_g, s_in_g, mult,
        tol=tol,
        max_iter=max_iter,
        anderson_depth=anderson_depth,
        max_step=max_step,
        max_time=max_time,
        chunk_size=chunk_size,
        verbose=verbose,
        monitor=monitor,
        topo_weig=topo_weig,
        weight_anderson=weight_anderson,
        hub_sk_threshold=hub_sk_threshold,
        hub_bisect_max_sweeps=hub_bisect_max_sweeps,
        hub_root_method=hub_root_method,
        patience_rel_tol=patience_rel_tol,
        block_newton_gate=block_newton_gate,
        block_newton_min_rel=block_newton_min_rel,
        stall_block_newton_gate=stall_block_newton_gate,
        stall_block_newton_off_mre=stall_block_newton_off_mre,
        stall_block_newton_revert=stall_block_newton_revert,
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
        diag_callback=diag_callback,
        diag_every=diag_every,
        diag_state_callback=diag_state_callback,
        streak_fix_threshold=streak_fix_threshold,
        streak_fix_n_bisect=streak_fix_n_bisect,
        streak_fix_n_sweeps=streak_fix_n_sweeps,
        device=device,
        checkpoint_fn=_checkpoint_fn_g,
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

def _targeted_bisection_fix(
    theta: torch.Tensor,
    k_out: torch.Tensor,
    k_in: torch.Tensor,
    s_out: torch.Tensor,
    s_in: torch.Tensor,
    mult: torch.Tensor,
    target_idx: torch.Tensor,
    side: str,
    n_sweeps: int = 5,
    n_bisect: int = 60,
) -> torch.Tensor:
    """Exact coordinate bisection restricted to ``target_idx`` groups AND
    to only the ``side`` ("out" or "in") that the triggering argmax
    actually flagged -- surgical, cheap (O(K*M) not O(M^2), K =
    len(target_idx)) alternative to a whole-network sweep, for the prototype case where one (or a few)
    node(s) have been identified (via the argmax-streak diagnostic, see
    decm_low_degree_precision_floor memory) as monopolizing the
    worst-violated equation for many consecutive iterations while the
    rest of the network is free to keep moving on its own. CPU/float64
    only -- K is expected to be tiny (single digits), so there is no
    GPU/chunking benefit.

    ``side="out"`` solves ONLY ``target_idx``'s (theta_out, eta_out) pair
    (the k_out/s_out equations) against the CURRENT (frozen) theta_in/
    eta_in of every group, including the target's own; ``side="in"``
    solves ONLY (theta_in, eta_in) (the k_in/s_in equations) against the
    current theta_out/eta_out. The two sides are never touched together:
    a node's own theta_out/eta_out appear ONLY in its own k_out/s_out
    equations and a node's own theta_in/eta_in ONLY in its own k_in/s_in
    equations -- there is no equation coupling both sides of the SAME
    node, so fixing the side the argmax did NOT flag is unmotivated
    (2026-09-14 correction, see decm_low_degree_precision_floor memory:
    the first version of this function fixed both sides unconditionally,
    which triggered a persistent post-fix blowup cascade on c1 -- most
    likely explained by this exact overreach, not by anything specific to
    the particular node tested).

    Does NOT claim to fix the rest of the network (it doesn't touch
    it, nor the untriggered side of the target itself), only that
    ``target_idx``'s flagged-side equations become locally consistent
    with the network's current state, which a plain Newton step has
    apparently failed to do for many consecutive iterations.
    """
    if side not in ("out", "in"):
        raise ValueError(f"side must be 'out' or 'in', got {side!r}")
    # This function is CPU/float64-only by design (see docstring) -- force
    # everything onto CPU/float64 up front so the caller's main loop is free
    # to run on any device (e.g. MPS) without this function needing its own
    # device parameter. Split into two .to() calls: combining device+dtype
    # in one call fails when the source is already on MPS (no direct
    # MPS->float64 cast).
    theta = theta.to(device="cpu").to(dtype=torch.float64)
    k_out = k_out.to(device="cpu").to(dtype=torch.float64)
    k_in = k_in.to(device="cpu").to(dtype=torch.float64)
    s_out = s_out.to(device="cpu").to(dtype=torch.float64)
    s_in = s_in.to(device="cpu").to(dtype=torch.float64)
    mult = mult.to(device="cpu").to(dtype=torch.float64)
    target_idx = target_idx.to(device="cpu")

    M = k_out.shape[0]
    K = target_idx.shape[0]
    rows = torch.arange(K)

    theta_out = theta[:M].clone()
    theta_in = theta[M : 2 * M].clone()
    eta_out = theta[2 * M : 3 * M].clone()
    eta_in = theta[3 * M :].clone()

    z_clamp = _Z_G_CLAMP     # same clamp as the rest of the solver (was a hard-coded 1e-8)

    def _bisect(cand_lo: "float | torch.Tensor", cand_hi: float, f: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        lo = cand_lo.clone() if torch.is_tensor(cand_lo) else torch.full((K,), cand_lo, dtype=torch.float64)
        hi = torch.full((K,), cand_hi, dtype=torch.float64)
        for _ in range(n_bisect):
            mid = 0.5 * (lo + hi)
            go_lo = f(mid) > 0.0
            lo = torch.where(go_lo, mid, lo)
            hi = torch.where(go_lo, hi, mid)
        return 0.5 * (lo + hi)

    def _partner_min(vec: torch.Tensor) -> torch.Tensor:
        """Smallest partner eta of each target: the other side's eta over all partners, the target's own column only
        counted when its group has more than one node (the diagonal pair (i, i) does not exist for mult <= 1)."""
        if M < 2:
            return torch.full((K,), float("inf"), dtype=torch.float64)
        v, ix = torch.topk(vec, 2, largest=False)
        om = torch.where(target_idx == ix[0], v[1].expand(K), v[0].expand(K))
        return torch.where(mult[target_idx] > 1, torch.minimum(om, vec[target_idx]), om)

    def _eta_lo(f: Callable[[torch.Tensor], torch.Tensor], other_min: torch.Tensor) -> torch.Tensor:
        """Lower bracket for eta. Only the pair sum z = eta + eta_partner must stay > 0, NOT eta >= 0: a node that is
        'unreachable' with eta >= 0 (s_model(_ETA_MIN) <= s_target) may go negative, down to the tightest partner's floor
        (same rule as the per-iteration hub bisection). Clamping such a node at _ETA_MIN instead left its s equation
        ~14% off after every streak-fix (q4 group 1403, 2026-09-25). Reachable nodes keep the old bracket."""
        base = torch.full((K,), _ETA_MIN, dtype=torch.float64)
        neg = (z_clamp - other_min).clamp(min=-_ETA_MAX)
        return torch.where((f(base) <= 0.0) & (neg < _ETA_MIN), neg, base)

    for _ in range(n_sweeps):
        if side == "out":
            # theta_out[target]: target rows vs. ALL columns (current, untouched theta_in/eta_in).
            eta_row = eta_out[target_idx][:, None] + eta_in[None, :]  # (K, M)
            log_q = -torch.log(torch.expm1(eta_row.clamp(min=z_clamp)))

            def f_theta_out(cand: torch.Tensor) -> torch.Tensor:
                logit_p = -cand[:, None] - theta_in[None, :] + log_q
                P = torch.sigmoid(logit_p)
                P_self = P[rows, target_idx]
                return (P * mult[None, :]).sum(1) - P_self - k_out[target_idx]

            theta_out[target_idx] = torch.where(
                k_out[target_idx] == 0,
                torch.full((K,), _THETA_MAX, dtype=torch.float64),
                _bisect(-_THETA_MAX, _THETA_MAX, f_theta_out),
            )

            # eta_out[target]: target rows vs. ALL columns (updated theta_out, current eta_in).
            # eta is solved at FIXED phi = theta + eta (theta = phi - eta), not at fixed theta: for large eta p_ij depends on phi
            # only, so moving eta at fixed theta moves phi and breaks the k equation just solved -- the alternation then never
            # converges (q5 group 832: k off by 3.7e-4 after 60 sweeps; with phi fixed it converges by ~0.36 per sweep).
            phi_out = theta_out[target_idx] + eta_out[target_idx]

            def f_eta_out(cand: torch.Tensor) -> torch.Tensor:
                z = cand[:, None] + eta_in[None, :]
                z_safe = z.clamp(min=z_clamp)
                G = -1.0 / torch.expm1(-z_safe)
                log_q2 = -torch.log(torch.expm1(z_safe))
                logit_p = -(phi_out - cand)[:, None] - theta_in[None, :] + log_q2
                P = torch.sigmoid(logit_p)
                W = P * G
                W_self = W[rows, target_idx]
                return (W * mult[None, :]).sum(1) - W_self - s_out[target_idx]

            _eta_new = torch.where(
                s_out[target_idx] == 0,
                torch.full((K,), _ETA_MAX, dtype=torch.float64),
                _bisect(_eta_lo(f_eta_out, _partner_min(eta_in)), _ETA_MAX, f_eta_out),
            )
            theta_out[target_idx] = torch.where(s_out[target_idx] == 0, theta_out[target_idx], phi_out - _eta_new)
            eta_out[target_idx] = _eta_new

        else:  # side == "in"
            # theta_in[target]: ALL rows (current, untouched theta_out/eta_out) vs. target columns.
            eta_col = eta_out[:, None] + eta_in[target_idx][None, :]  # (M, K)
            log_q3 = -torch.log(torch.expm1(eta_col.clamp(min=z_clamp)))

            def f_theta_in(cand: torch.Tensor) -> torch.Tensor:
                logit_p = -theta_out[:, None] - cand[None, :] + log_q3
                P = torch.sigmoid(logit_p)
                P_self = P[target_idx, rows]
                return (P * mult[:, None]).sum(0) - P_self - k_in[target_idx]

            theta_in[target_idx] = torch.where(
                k_in[target_idx] == 0,
                torch.full((K,), _THETA_MAX, dtype=torch.float64),
                _bisect(-_THETA_MAX, _THETA_MAX, f_theta_in),
            )

            # eta_in[target]: ALL rows (current theta_out/eta_out, updated theta_in) vs. target columns (at fixed phi, see the out side).
            phi_in = theta_in[target_idx] + eta_in[target_idx]

            def f_eta_in(cand: torch.Tensor) -> torch.Tensor:
                z = eta_out[:, None] + cand[None, :]
                z_safe = z.clamp(min=z_clamp)
                G = -1.0 / torch.expm1(-z_safe)
                log_q4 = -torch.log(torch.expm1(z_safe))
                logit_p = -theta_out[:, None] - (phi_in - cand)[None, :] + log_q4
                P = torch.sigmoid(logit_p)
                W = P * G
                W_self = W[target_idx, rows]
                return (W * mult[:, None]).sum(0) - W_self - s_in[target_idx]

            _eta_new = torch.where(
                s_in[target_idx] == 0,
                torch.full((K,), _ETA_MAX, dtype=torch.float64),
                _bisect(_eta_lo(f_eta_in, _partner_min(eta_out)), _ETA_MAX, f_eta_in),
            )
            theta_in[target_idx] = torch.where(s_in[target_idx] == 0, theta_in[target_idx], phi_in - _eta_new)
            eta_in[target_idx] = _eta_new

    return torch.cat([theta_out, theta_in, eta_out, eta_in])
