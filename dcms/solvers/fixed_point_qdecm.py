"""Fixed-point iteration solver for the qDECM weight step.

The qDECM weight equations (conditioned on a fixed DCM topology ``p_ij``) are:

    s_out_i = Σ_{j≠i} p_ij / (1 − β_out_i · β_in_j)
    s_in_i  = Σ_{j≠i} p_ji / (1 − β_out_j · β_in_i)

where z_ij = β_out_i · β_in_j and G_ij = 1/(1-z_ij) is the geometric-distribution
correction factor.  Note: the numerator is p_ij (not p_ij · z_ij), because
E[w_ij] = p_ij · E[w_ij | a_ij=1] = p_ij · G_ij.

Nodes sharing the same (k_out, k_in, s_out, s_in) have identical parameters at the solution, so the
weight step works on the M-unknown reduced system and expands the solution back to the N nodes
(the topology step is a plain DCM, itself reduced on the (k_out, k_in) pairs).

The iteration is a θ-space Gauss-Seidel Newton step: out-multipliers are updated first (pass 1), then
in-multipliers are updated using the fresh θ_β_out values (pass 2). For each group g in pass 1:

      Δθ_β_out_g = −F_out_g / F′_out_g
  where:
      F_out_g  = Σ_h m_h p_gh · G_gh − p_gg G_gg − s_out_g
      F′_out_g = −Σ_h m_h p_gh · G_gh · (G_gh − 1) + (diagonal term)   [= ∂F_out/∂θ_β_out_g]
      G_gh     = −1 / expm1(−(θ_β_out_g + θ_β_in_h))  [= 1/(1−exp(−z))]

G > 1 for all z > 0, so F′ < 0 (the system is well-conditioned for Newton in θ-space). The per-group
step is clipped to ``[−max_step, +max_step]`` and accelerated by Anderson mixing.
"""
from __future__ import annotations

import datetime
import math
import sys
import time
import warnings

import torch

from dcms.models.parameters import _ETA_MAX
from dcms.solvers._common import anderson_mixing as _anderson_mixing, expand_result, first_member_index, reduce_blocks
from dcms.solvers.base import SolverResult
from dcms.utils.profiling import _PeakRAMMonitor

_ANDERSON_MAX_NORM: float = 1e6

# Allow the residual to grow transiently large while the θ-Newton solver
# navigates the near-singular region (z → 0⁺, G → ∞).  The per-node alpha
# line-search in _theta_newton_step_chunked keeps z ≥ max(z_prev * _Z_NEWTON_FRAC,
# _Z_NEWTON_FLOOR) after each GS step; the Anderson feasibility guard projects any
# infeasible mix back to z ≥ floor while keeping history intact for acceleration.
_ANDERSON_BLOWUP_FACTOR: float = 5000.0  # clear Anderson history if F > 5000 × best
_TN_RESET_WINDOW: int = 30  # clear Anderson + revert to best_theta if no TN improvement
_Q_MAX: float = 0.9999  # maximum allowed product β_out * β_in

# Minimum z_ij = θ_β_out_i + θ_β_in_j used in G_ij = 1/expm1(z_safe).
# Must be small enough that G(z_clamp) >> s/k for hub nodes (which need
# large G ≈ s/k ≈ 30–100).  With z_clamp=1e-8, G_max ≈ 10^8, and for
# a hub-hub pair with p ≈ 1e-5: p * G_max ≈ 1000 >> s_out ≈ 213.
# This ensures F > 0 at z_clamp, giving delta > 0 so hub nodes always
# self-escape the z = z_clamp boundary state via the doubling mechanism.
_Z_G_CLAMP: float = 1e-8
# Per-node line-search: two constraints keep the Newton step well-behaved.
#
# 1) Hard floor (_Z_NEWTON_FLOOR = _Z_G_CLAMP = 1e-8): z never drops below the
#    clamped region.  At z = z_clamp the doubling mechanism applies (delta ≈ z)
#    so z doubles each GS step, recovering to z* in ~22 iterations.
#
# 2) Relative step limit (_Z_NEWTON_FRAC = 0.5): z may decrease by at most
#    50 % per step.  Without this limit, a negative Newton step can jump z from
#    2·z* all the way to z_clamp in a single step, triggering a new blowup.
#    With the limit: z halves each step from above z*, converging to z* in
#    log₂(z_initial / z*) steps without ever touching z_clamp.
_Z_NEWTON_FLOOR: float = _Z_G_CLAMP  # = 1e-8; hard floor (enables doubling)
_Z_NEWTON_FRAC: float = 0.5          # relative floor; z_new >= z_old * 0.5
_ANDERSON_THETA_FLOOR: float = 0.1



def _bisect_hub_beta(
    p_row: "np.ndarray",
    beta_other: "np.ndarray",
    s_target: float,
    n_bisect: int = 60,
    self_idx: int | None = None,
    mult: "np.ndarray | None" = None,
) -> float:
    """Find β such that Σ_j p_j / (1 − β·b_j) = s_target via bisection.

    Used for hub nodes where s/k >> 1, requiring β close to 1.  The function

        f(β) = Σ_j p_j / (1 − β·b_j)

    is strictly increasing in β ∈ [0, β_max) where β_max = 1/max(b_j).
    For a hub node f(0) = Σ p_j ≈ k_hat << s_target, so the root is
    well-bracketed.  60 bisection steps give relative precision ≈ 2⁻⁶⁰.

    Args:
        p_row:      Connection probabilities to neighbours (length N, diagonal
                    entry excluded and assumed zero -- or, in the
                    degeneracy-reduced path, length M with a real value at
                    ``self_idx`` that must be excluded via ``mult``).
        beta_other: Current β values of the other side (β_in for an out-hub,
                    β_out for an in-hub), length N (or M, reduced path).
        s_target:   Observed strength to match (must be positive).
        n_bisect:   Number of bisection iterations (default 60).
        self_idx:   Internal use by the degeneracy-reduced path: index of
                    this hub's own group within ``p_row``/``beta_other``,
                    whose term must be excluded from the sum (see ``mult``).
                    ``None`` (default) reproduces the old per-node behaviour.
        mult:       Internal use by the degeneracy-reduced path: group
                    multiplicities, length M. When given (together with
                    ``self_idx``), the sum becomes the "weighted sum minus
                    one diagonal term" generalisation used throughout the
                    reduction (see
                    ``dcms.solvers.fixed_point_decm._bisect_hub_eta_decm``
                    for the combinatorial justification). ``None`` (default)
                    reproduces the old unweighted per-node sum exactly.

    Returns:
        Scalar β* such that f(β*) ≈ s_target.
    """
    import numpy as _np

    pos = p_row > 0.0
    if mult is not None and self_idx is not None and mult[self_idx] <= 1:
        pos = pos.copy()
        pos[self_idx] = False       # a single-node group has no pair with itself: it must not bound the bracket
    if not pos.any():
        return 0.0

    max_b = beta_other[pos].max()
    if max_b <= 0.0:
        return 0.0

    beta_lo = 0.0
    beta_hi = (1.0 / max_b) * (1.0 - 1e-9)

    for _ in range(n_bisect):
        beta_mid = 0.5 * (beta_lo + beta_hi)
        z = beta_mid * beta_other
        z = _np.minimum(z, 1.0 - 1e-12)
        terms = p_row / (1.0 - z)
        if mult is not None:
            w = mult.copy()
            w[self_idx] -= 1.0      # (mult_g - 1) partners inside the hub's own group: 0 for a single node (no 1/(1-z) there)
            f = float((terms * w).sum()) - s_target
        else:
            f = terms.sum() - s_target
        if f < 0.0:
            beta_lo = beta_mid
        else:
            beta_hi = beta_mid

    return 0.5 * (beta_lo + beta_hi)


# -------------------------------------------------------------------------
# Degeneracy reduction for the weight step (Vallarano et al., Sci Rep
# 11:15227, 2021)
# -------------------------------------------------------------------------
#
# The qDECM topology step is a plain DCM -- already reduced by
# solve_fixed_point_dcm_degenerate (2-tuple (k_out,k_in) groups). The weight
# step's own unknowns (theta_beta_out, theta_beta_in) additionally depend on
# (s_out, s_in) through the strength targets, so its degeneracy condition is
# the FULL 4-tuple (k_out,k_in,s_out,s_in) -- exactly DECM's condition (see
# compute_degeneracy_groups in fixed_point_decm.py), reused as-is rather
# than re-derived: a node's weight-step equations here are structurally
# identical to DECM's eta-only sub-problem, just with the geometric weight
# multiplied by a *fixed* p_ij (from the already-solved topology) instead of
# p_ij being coupled to the same theta being solved for. Since every node in
# a weight-degeneracy group already shares the same (k_out,k_in) 2-tuple (a
# coarser condition implied by sharing the full 4-tuple), it also shares the
# same theta_topo value -- so reducing theta_topo down to weight-group
# granularity is always exact, never an approximation.

def _qdecm_step_dense_weighted(
    theta_weight: torch.Tensor,
    P: torch.Tensor,
    s_out: torch.Tensor,
    s_in: torch.Tensor,
    max_step: float,
    mult: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One theta-Newton (GS) step for the degeneracy-reduced qDECM weight step.

    Identical algebra to :func:`_theta_newton_step_dense`, generalised so
    each group ``g`` stands in for ``mult[g]`` original nodes -- same
    "weighted sum minus one diagonal term" pattern used throughout the
    reduction (see DECM's ``_decm_step_dense_weighted`` for the
    combinatorial justification). ``P`` here is the group-level topology
    probability matrix (M, M), already fixed (not re-derived from
    ``theta_weight``). With ``mult`` all-ones (M == N) this reduces exactly
    to the per-node formula.

    Args:
        theta_weight: Current 2M parameter vector [theta_b_out|theta_b_in].
        P: Group-level DCM probability matrix, shape (M, M).
        s_out, s_in: Group-level target sequences, shape (M,).
        max_step: Maximum |Delta theta| per group per step.
        mult: Group multiplicities, shape (M,).

    Returns:
        ``(theta_new, F_current)``, both shape (2M,).
    """
    M = s_out.shape[0]
    theta_b_out = theta_weight[:M]
    theta_b_in = theta_weight[M:]

    z = theta_b_out[:, None] + theta_b_in[None, :]
    z_safe = z.clamp(min=_Z_G_CLAMP)
    G = -1.0 / torch.expm1(-z_safe)

    PG = P * G
    PG_diag = PG.diagonal()
    PGG1 = P * G * (G - 1.0)
    PGG1_diag = PGG1.diagonal()

    F_out = (PG * mult[None, :]).sum(1) - PG_diag - s_out
    # F_in_current: the *returned* residual's in-part, evaluated at the
    # INPUT state (PG, pre-update) -- distinct from F_in below, which is
    # evaluated at the GS-updated state and used only for the delta_in
    # Newton step. Mirrors the unreduced dense function's F_in_current vs.
    # F_in split exactly (see _theta_newton_step_dense).
    F_in_current = (PG * mult[:, None]).sum(0) - PG_diag - s_in
    # neg_H_out is the *positive* Hessian magnitude (mirrors the unreduced
    # chunked path's `neg_H_out = sum_PGG1_out.clamp(min=1e-15)` -- PGG1 is
    # positive, so no extra negation is needed here, unlike H_out in the
    # dense per-node function which tracks the raw (<=0) Hessian value).
    neg_H_out = ((PGG1 * mult[None, :]).sum(1) - PGG1_diag).clamp(min=1e-15)
    delta_out = (F_out / neg_H_out).clamp(-max_step, max_step)

    nz_in_mask = s_in > 0
    _z_ls_out = z.clone()
    singleton = mult <= 1.0
    if singleton.any():
        idx = singleton.nonzero(as_tuple=True)[0]
        _z_ls_out[idx, idx] = float("inf")
    if not nz_in_mask.all():
        _z_ls_out[:, ~nz_in_mask] = float("inf")
    z_min_out = _z_ls_out.min(dim=1).values
    z_floor_out = torch.clamp(z_min_out * _Z_NEWTON_FRAC, min=_Z_NEWTON_FLOOR)
    available_out = (z_min_out - z_floor_out).clamp(min=0.0)
    needed_out = delta_out.abs().clamp(min=1e-30)
    alpha_out = torch.where(
        delta_out < 0,
        (available_out / needed_out).clamp(max=1.0),
        torch.ones(M, dtype=theta_b_out.dtype, device=theta_b_out.device),
    )
    theta_b_out_new = (theta_b_out + alpha_out * delta_out).clamp(-_ETA_MAX, _ETA_MAX)
    theta_b_out_new = torch.where(
        s_out == 0, torch.full_like(theta_b_out_new, _ETA_MAX), theta_b_out_new
    )

    z2 = theta_b_out_new[:, None] + theta_b_in[None, :]
    z2_safe = z2.clamp(min=_Z_G_CLAMP)
    G2 = -1.0 / torch.expm1(-z2_safe)

    PG2 = P * G2
    PG2_diag = PG2.diagonal()
    PGG12 = P * G2 * (G2 - 1.0)
    PGG12_diag = PGG12.diagonal()

    F_in = (PG2 * mult[:, None]).sum(0) - PG2_diag - s_in
    neg_H_in = ((PGG12 * mult[:, None]).sum(0) - PGG12_diag).clamp(min=1e-15)
    delta_in = (F_in / neg_H_in).clamp(-max_step, max_step)

    nz_out_mask = s_out > 0
    _z_ls_in = z2.clone()
    if singleton.any():
        idx = singleton.nonzero(as_tuple=True)[0]
        _z_ls_in[idx, idx] = float("inf")
    if not nz_out_mask.all():
        _z_ls_in[~nz_out_mask, :] = float("inf")
    z_min_in = _z_ls_in.min(dim=0).values
    z_floor_in = torch.clamp(z_min_in * _Z_NEWTON_FRAC, min=_Z_NEWTON_FLOOR)
    available_in = (z_min_in - z_floor_in).clamp(min=0.0)
    needed_in = delta_in.abs().clamp(min=1e-30)
    alpha_in = torch.where(
        delta_in < 0,
        (available_in / needed_in).clamp(max=1.0),
        torch.ones(M, dtype=theta_b_in.dtype, device=theta_b_in.device),
    )
    theta_b_in_new = (theta_b_in + alpha_in * delta_in).clamp(-_ETA_MAX, _ETA_MAX)
    theta_b_in_new = torch.where(
        s_in == 0, torch.full_like(theta_b_in_new, _ETA_MAX), theta_b_in_new
    )

    F_current = torch.cat([F_out, F_in_current])
    return torch.cat([theta_b_out_new, theta_b_in_new]), F_current


def _solve_qdecm_weights_reduced(
    theta0: torch.Tensor,
    s_out: torch.Tensor,
    s_in: torch.Tensor,
    theta_topo: torch.Tensor,
    P: torch.Tensor,
    mult: torch.Tensor,
    tol: float,
    max_iter: int,
    anderson_depth: int,
    max_step: float,
    max_time: float,
    verbose: bool,
    monitor: bool,
    hub_sk_threshold: float,
    weight_anderson: bool,
    z_clamp: float,
    device: str,
) -> SolverResult:
    """Newton/Anderson loop of the qDECM weight step on the reduced system (M groups, multiplicities ``mult``,
    group-level topology probabilities ``P`` of shape (M, M)); see :func:`solve_fixed_point_qdecm_weight`."""
    if z_clamp != 1e-8:
        warnings.warn(
            f"z_clamp={z_clamp:g} != 1e-8: qDECMModel's own residual/weight computations "
            f"(dcms/models/qdecm.py) still hard-code 1e-8, so for pairs between the two "
            f"values the solver's equations would differ from the residual it is judged "
            f"by. Use the default 1e-8.",
            stacklevel=3,
        )
    global _Z_G_CLAMP, _Z_NEWTON_FLOOR
    _Z_G_CLAMP = z_clamp
    _Z_NEWTON_FLOOR = z_clamp

    _dev = torch.device(device)
    _dtype = torch.float64 if _dev.type == "cpu" else torch.float32

    # Convert inputs to tensors
    s_out = s_out.to(dtype=_dtype, device=_dev)
    s_in = s_in.to(dtype=_dtype, device=_dev)
    theta = theta0.clone().to(dtype=_dtype, device=_dev)
    mult = mult.to(dtype=_dtype, device=_dev)
    P_mat = P.to(dtype=_dtype, device=_dev)

    N = s_out.shape[0]
    theta = theta.clamp(-_ETA_MAX, _ETA_MAX)  # allow β>1 (negative θ)

    # --- Hub bisection pre-computation ---
    # Identify hub nodes (s/k_hat > hub_sk_threshold) and precompute their
    # full p-rows/columns (one row per out-hub, one col per in-hub).
    # Hub β values are solved exactly via 1D bisection each iteration instead
    # of the global Newton/FP step, which stagnates when z_ij ≈ 0.
    _hub_active = hub_sk_threshold > 0.0
    _hub_out_idx: list[int] = []
    _hub_in_idx: list[int] = []
    _hub_P_out_rows: list[torch.Tensor] = []  # p_ij row for each out-hub
    _hub_P_in_cols: list[torch.Tensor] = []   # p_ij col for each in-hub
    if _hub_active:
        # Expected degree k_hat = sum_j p_ij for each group
        k_hat_out = P_mat.sum(1)   # (N,)
        k_hat_in  = P_mat.sum(0)   # (N,)

        hub_out_mask = (
            (s_out > 0)
            & (k_hat_out > 0)
            & (s_out / k_hat_out.clamp(min=1e-10) > hub_sk_threshold)
        )
        hub_in_mask = (
            (s_in > 0)
            & (k_hat_in > 0)
            & (s_in / k_hat_in.clamp(min=1e-10) > hub_sk_threshold)
        )
        _hub_out_idx = hub_out_mask.nonzero(as_tuple=True)[0].tolist()
        _hub_in_idx  = hub_in_mask.nonzero(as_tuple=True)[0].tolist()

        # Precompute the full p-row / p-col for each hub (small number of rows)
        for _i in _hub_out_idx:
            _hub_P_out_rows.append(P_mat[_i].clone())
        for _j in _hub_in_idx:
            _hub_P_in_cols.append(P_mat[:, _j].clone())

        if verbose and (_hub_out_idx or _hub_in_idx):
            print(
                f"[hub_bisection] out-hubs={_hub_out_idx}, in-hubs={_hub_in_idx} "
                f"(threshold s/k > {hub_sk_threshold})"
            )

    _peak_ram_monitor = _PeakRAMMonitor()
    _peak_ram_monitor.__enter__()
    t0 = time.perf_counter()

    n_iter = 0
    residuals: list[float] = []
    converged = False
    message = "Maximum iterations reached without convergence."

    # Stagnation detection
    _STAGNATION_WINDOW: int = 200
    _STAGNATION_RTOL: float = 0.01
    best_res_recent: float = float("inf")
    best_res_old: float = float("inf")

    # Track the best iterate seen during the run
    best_theta: torch.Tensor = theta.clone()
    best_theta_res: float = float("inf")

    # Anderson acceleration history (in θ-space)
    _and_g: list[torch.Tensor] = []
    _and_r: list[torch.Tensor] = []

    # Running minimum of the equation-residual for blowup detection
    _best_res_for_anderson: float = float("inf")

    # θ-Newton Anderson reset counter (revert to best_theta when stagnant)
    _tn_best_local: float = float("inf")
    _tn_stagnation_count: int = 0

    # Precompute verbose targets once (MRE = max |F_i| / s_i)
    _v_targets = torch.cat([s_out, s_in])
    _v_nonzero = _v_targets > 0

    # Anderson mixing needs per-component weights in the degeneracy-reduced
    # path -- see _anderson_mixing's docstring and the analogous DECM/DCM/DWCM fix.
    _anderson_weights = torch.cat([mult, mult]) if weight_anderson else None
    # Group multiplicities as numpy, for the weighted hub-bisection sum.
    # (.cpu() first: mult may live on a non-CPU device, e.g. under `device="mps"`,
    # and this small, occasional-use array is only ever consumed via numpy.)
    _mult_np = mult.detach().to(device="cpu").to(dtype=torch.float64).numpy()

    try:
        for _ in range(max_iter):
            # Wall-clock time limit
            if max_time > 0 and (time.perf_counter() - t0) > max_time:
                message = (
                    f"Time limit ({max_time:.0f}s) reached at iteration {n_iter}."
                )
                break

            theta_fp, F_current = _qdecm_step_dense_weighted(
                theta, P_mat, s_out, s_in, max_step, mult
            )

            # Universal single-step floor: prevent any θ from dropping below
            # _ANDERSON_THETA_FLOOR × its current value in a single iteration.
            # Only apply when theta_fp > 0 — when the FP output is negative
            # (β > 1 needed), allow it freely; don't push it back above zero.
            _step_floor = (theta * _ANDERSON_THETA_FLOOR).clamp(min=-_ETA_MAX)
            theta_fp = torch.where(
                theta_fp > 0, torch.maximum(theta_fp, _step_floor), theta_fp
            )

            # --- Hub bisection (1D coordinate descent for high s/k nodes) ---
            # For hub nodes, the global step cannot resolve β_i ≈ 1 accurately.
            # Multiple Gauss-Seidel sweeps (out-hubs then in-hubs) until the hub
            # sub-system is self-consistent.  A single sweep is inconsistent when
            # a node appears in both out-hub and in-hub lists: the β_in update in
            # the in-hub pass invalidates the β_out computed in the out-hub pass.
            # 3 sweeps are enough for typical cases (convergence is geometric).
            if _hub_active and (_hub_out_idx or _hub_in_idx):
                import numpy as _hnp
                _beta_out_fp = torch.exp(-theta_fp[:N]).cpu().numpy()
                _beta_in_fp  = torch.exp(-theta_fp[N:]).cpu().numpy()

                _HUB_SWEEPS = 3
                for _sweep in range(_HUB_SWEEPS):
                    for _hk, _hi in enumerate(_hub_out_idx):
                        _p_row = _hub_P_out_rows[_hk].cpu().numpy()
                        _new_beta = _bisect_hub_beta(
                            _p_row, _beta_in_fp, s_out[_hi].item(),
                            self_idx=_hi,
                            mult=_mult_np,
                        )
                        if _new_beta > 0.0:
                            _new_theta = float(-_hnp.log(max(_new_beta, 1e-300)))
                            theta_fp[_hi] = max(_new_theta, -_ETA_MAX)
                            _beta_out_fp[_hi] = _new_beta

                    for _hk, _hj in enumerate(_hub_in_idx):
                        _p_col = _hub_P_in_cols[_hk].cpu().numpy()
                        _new_beta = _bisect_hub_beta(
                            _p_col, _beta_out_fp, s_in[_hj].item(),
                            self_idx=_hj,
                            mult=_mult_np,
                        )
                        if _new_beta > 0.0:
                            _new_theta = float(-_hnp.log(max(_new_beta, 1e-300)))
                            theta_fp[N + _hj] = max(_new_theta, -_ETA_MAX)
                            _beta_in_fp[_hj] = _new_beta

                # All hub constraints satisfied → zero their F_current entries
                for _hi in _hub_out_idx:
                    F_current[_hi] = 0.0
                for _hj in _hub_in_idx:
                    F_current[N + _hj] = 0.0

            # --- Convergence check using the step-computed residual ---
            # Relative: MRE = max|F_i|/target_i
            res_norm = (
                (F_current.abs()[_v_nonzero] / _v_targets[_v_nonzero]).max().item()
                if _v_nonzero.any() else 0.0
            )

            if not math.isfinite(res_norm):
                message = f"NaN/Inf detected at iteration {n_iter}."
                break

            n_iter += 1
            residuals.append(res_norm)

            if verbose:
                _elapsed = time.perf_counter() - t0
                _mre = (
                    (F_current.abs()[_v_nonzero] / _v_targets[_v_nonzero]).max().item()
                    if _v_nonzero.any() else float("nan")
                )
                print(
                    f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] "
                    f"iteration={n_iter:5d}, "
                    f"elapsed time={int(_elapsed // 3600):4d}:{int((_elapsed % 3600) // 60):02d}:{int(_elapsed % 60):02d}, "
                    f"MRE_weights={_mre:.5e}",
                    end="\r" if monitor else "\n",
                )
                sys.stdout.flush()

            # Keep a reference to the iterate with the minimum equation-residual.
            # res_norm = MRE(F(theta)) is measured at the *current* theta (before
            # the step), so we save theta (not theta_fp) to keep best_theta and
            # best_theta_res consistent.
            if res_norm < best_theta_res:
                best_theta_res = res_norm
                best_theta = theta.clone()

            if res_norm < tol:
                converged = True
                message = f"Converged in {n_iter} iteration(s)."
                break

            # --- Stagnation detection ---
            if res_norm < best_res_recent:
                best_res_recent = res_norm
            if n_iter % _STAGNATION_WINDOW == 0:
                if n_iter > _STAGNATION_WINDOW:
                    improvement = (best_res_old - best_res_recent) / max(best_res_old, 1e-30)
                    if 0.0 <= improvement < _STAGNATION_RTOL:
                        message = (
                            f"Stagnation: residual improved by only {improvement:.2%} "
                            f"over last {_STAGNATION_WINDOW} iterations "
                            f"(best={best_res_recent:.3e}). Stopping at iter {n_iter}."
                        )
                        break
                best_res_old = best_res_recent
                best_res_recent = float("inf")

            # θ-Newton Anderson reset: when Anderson has been stagnant for
            # _TN_RESET_WINDOW iterations, clear its history and revert to
            # best_theta so the next Newton step starts from the best-seen
            # iterate rather than a contaminated Anderson mix.
            _tn_revert_triggered = False
            if anderson_depth > 1:
                if res_norm < _tn_best_local * 0.99:
                    _tn_best_local = res_norm
                    _tn_stagnation_count = 0
                else:
                    _tn_stagnation_count += 1
                    if _tn_stagnation_count >= _TN_RESET_WINDOW and res_norm > tol:
                        _tn_stagnation_count = 0
                        _tn_best_local = float("inf")
                        _and_g.clear()
                        _and_r.clear()
                        _tn_revert_triggered = True

            # --- Apply Anderson acceleration or plain update ---
            if _tn_revert_triggered:
                # Anderson was stagnant: restart next iteration from best_theta.
                theta_next = best_theta.clone()
            elif anderson_depth > 1:
                # Blowup guard: when res_norm jumps > 100 × best_seen, Anderson
                # history contains "blowup-phase" iterates that poison future
                # mixes.  Clear the history but DO NOT revert to best_theta —
                # instead continue from theta_fp (the current GS output) so the
                # doubling recovery keeps advancing.  The next iteration starts
                # with an empty history and pure-GS behaviour.
                _blowup_recovered = False
                if (
                    len(_and_g) >= 2
                    and math.isfinite(res_norm)
                    and res_norm > _ANDERSON_BLOWUP_FACTOR * _best_res_for_anderson
                ):
                    _and_g.clear()
                    _and_r.clear()
                    _blowup_recovered = True

                # Update the running best AFTER the blowup check
                _best_res_for_anderson = min(_best_res_for_anderson, res_norm)

                if _blowup_recovered:
                    theta_next = theta_fp  # continue from current GS output
                else:
                    r_k = theta_fp - theta
                    # Exclude hub components from Anderson residuals: hub θ
                    # values are always overridden by bisection, so their
                    # differences would confuse the mixing weights and slow
                    # convergence for the (many more) non-hub components.
                    if _hub_active and (_hub_out_idx or _hub_in_idx):
                        r_k = r_k.clone()
                        for _hi in _hub_out_idx:
                            r_k[_hi] = 0.0
                        for _hj in _hub_in_idx:
                            r_k[N + _hj] = 0.0
                    r_k_norm = r_k.abs().max().item()
                    if math.isfinite(r_k_norm) and r_k_norm < _ANDERSON_MAX_NORM:
                        _and_g.append(theta_fp.clone())
                        _and_r.append(r_k.clone())

                    if len(_and_g) > anderson_depth:
                        _and_g.pop(0)
                        _and_r.pop(0)

                    if len(_and_g) >= 2:
                        theta_next = _anderson_mixing(_and_g, _and_r, weights=_anderson_weights)
                        # Geometric floor: prevent Anderson from extrapolating
                        # any θ_i to near 0 (β_i ≈ 1), which would cause blowups.
                        theta_floor = (theta * _ANDERSON_THETA_FLOOR).clamp(
                            min=-_ETA_MAX
                        )
                        effective_floor = torch.minimum(theta_floor, theta_fp)
                        theta_next = torch.maximum(theta_next, effective_floor)
                        theta_next = theta_next.clamp(-_ETA_MAX, _ETA_MAX)

                        # Feasibility guard: if Anderson extrapolated z_min =
                        # min(θ_out) + min(θ_in) below 0, the mixed iterate is
                        # infeasible (some z_ij < 0 → G → ∞ → immediate blow-up).
                        # Reject the mix and fall back to the Newton proposal
                        # theta_fp, which was computed under the z-floor guarantee.
                        # This is safer than shifting (which distorts the mix) and
                        # faster than the doubling recovery from z = 1e-8.
                        _N2 = theta_next.shape[0] // 2
                        _z_min_and = (
                            theta_next[:_N2].min().item()
                            + theta_next[_N2:].min().item()
                        )
                        if _z_min_and < _Z_NEWTON_FLOOR:
                            # Reject infeasible mix; clear history to avoid
                            # re-mixing infeasible iterates in future steps.
                            theta_next = theta_fp
                            _and_g.clear()
                            _and_r.clear()
                    else:
                        theta_next = theta_fp
            else:
                theta_next = theta_fp

            # --- Freeze hub components after Anderson mixing ---
            # Anderson mixes ALL components, including hubs, which were just
            # solved exactly by bisection.  Override the hub θ values in
            # theta_next with those from theta_fp (post-bisection) so that
            # the next iteration starts with the exact bisection solution,
            # not a contaminated linear combination.
            if _hub_active and theta_next is not theta_fp and (
                _hub_out_idx or _hub_in_idx
            ):
                for _hi in _hub_out_idx:
                    theta_next[_hi] = theta_fp[_hi]
                for _hj in _hub_in_idx:
                    theta_next[N + _hj] = theta_fp[N + _hj]

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
    )


def solve_fixed_point_qdecm(
    theta_topo0: "ArrayLike",  # type: ignore[name-defined]
    theta_weight0: "ArrayLike",  # type: ignore[name-defined]
    k_out: "ArrayLike",  # type: ignore[name-defined]
    k_in: "ArrayLike",  # type: ignore[name-defined]
    s_out: "ArrayLike",  # type: ignore[name-defined]
    s_in: "ArrayLike",  # type: ignore[name-defined]
    tol: float = 1e-8,
    max_iter: int = 10_000,
    topo_max_iter: int = 10_000,
    anderson_depth: int = 10,
    max_step: float = 1.0,
    max_time: float = 0.0,
    verbose: bool = False,
    monitor: bool = False,
    hub_sk_threshold: float = 0.0,
    weight_anderson: bool = True,
    z_clamp: float = 1e-8,
    device: str = "cpu",
) -> SolverResult:
    """Degeneracy-reduced qDECM solver (topology + conditioned-weight step).

    Per the user's guidance: the topology step is a plain DCM, already
    reduced by :func:`~dcms.solvers.fixed_point_dcm.solve_fixed_point_dcm_degenerate`
    (2-tuple (k_out,k_in) groups) -- reused here as-is. Only the
    *conditioned-weight* step needed new reduction work: it depends on the
    full 4-tuple (k_out,k_in,s_out,s_in), exactly DECM's degeneracy
    condition, reused via
    :func:`~dcms.solvers.fixed_point_decm.compute_degeneracy_groups`
    rather than re-derived (see the module-level note above
    :func:`_qdecm_step_dense_weighted`).

    Args:
        theta_topo0: Initial topology guess [theta_out|theta_in], shape (2N,).
        theta_weight0: Initial weight guess [theta_b_out|theta_b_in], shape (2N,).
        k_out, k_in, s_out, s_in: Observed sequences, each shape (N,).
        tol: Convergence tolerance for the weight step (and, since the
            topology step feeds it, indirectly for topology too via
            solve_fixed_point_dcm's own tol, which is set equal).
        max_iter: Max iterations for the weight step.
        topo_max_iter: Max iterations for the topology (DCM) step.
        anderson_depth, max_step, max_time, verbose, monitor, device: as in
            :func:`~dcms.solvers.fixed_point_dcm.solve_fixed_point_dcm`; hub_sk_threshold: see the module docstring of
            the hub bisection (0 = off); weight_anderson: weight the Anderson rows by the group multiplicities;
            z_clamp: minimum z in G = 1/expm1(z) (module constant for the duration of the call).

    Returns:
        :class:`~dcms.solvers.base.SolverResult` for the *weight* step, with
        ``theta``/``best_theta`` expanded to shape (2N,). The topology
        solver's convergence status is folded into ``message``.
    """
    from dcms.solvers.fixed_point_dcm import solve_fixed_point_dcm

    k_out = k_out if isinstance(k_out, torch.Tensor) else torch.tensor(k_out, dtype=torch.float64)
    k_in = k_in if isinstance(k_in, torch.Tensor) else torch.tensor(k_in, dtype=torch.float64)
    k_out, k_in = k_out.to(dtype=torch.float64), k_in.to(dtype=torch.float64)
    theta_topo0 = (
        theta_topo0 if isinstance(theta_topo0, torch.Tensor)
        else torch.tensor(theta_topo0, dtype=torch.float64)
    ).to(dtype=torch.float64)

    # --- Topology step: plain DCM, already reduced (2-tuple groups) ---
    # Solved to the same tolerance as the weight step (2026-10-27: the old tol*1e-3 was checked to be unnecessary: the
    # E[w_ij] matrices agree to the solver tolerance either way).
    topo_res = solve_fixed_point_dcm(
        theta_topo0, k_out, k_in,
        tol=tol, max_iter=topo_max_iter, anderson_depth=anderson_depth,
        max_step=max_step, max_time=max_time, verbose=verbose, monitor=monitor,
        device=device,
    )
    theta_topo = torch.as_tensor(topo_res.best_theta, dtype=torch.float64)

    weight_res = solve_fixed_point_qdecm_weight(
        theta_topo, theta_weight0, k_out, k_in, s_out, s_in,
        tol=tol, max_iter=max_iter, anderson_depth=anderson_depth,
        max_step=max_step, max_time=max_time, verbose=verbose, monitor=monitor,
        hub_sk_threshold=hub_sk_threshold, weight_anderson=weight_anderson,
        z_clamp=z_clamp, device=device,
    )

    return SolverResult(
        theta=weight_res.theta,
        best_theta=weight_res.best_theta,
        converged=weight_res.converged,
        iterations=weight_res.iterations,
        residuals=weight_res.residuals,
        elapsed_time=weight_res.elapsed_time + topo_res.elapsed_time,
        peak_ram_bytes=max(weight_res.peak_ram_bytes, topo_res.peak_ram_bytes),
        message=f"[topo: {topo_res.message}] [weight: {weight_res.message}]",
    )


def solve_fixed_point_qdecm_weight(
    theta_topo: "ArrayLike",  # type: ignore[name-defined]
    theta_weight0: "ArrayLike",  # type: ignore[name-defined]
    k_out: "ArrayLike",  # type: ignore[name-defined]
    k_in: "ArrayLike",  # type: ignore[name-defined]
    s_out: "ArrayLike",  # type: ignore[name-defined]
    s_in: "ArrayLike",  # type: ignore[name-defined]
    tol: float = 1e-8,
    max_iter: int = 10_000,
    anderson_depth: int = 10,
    max_step: float = 1.0,
    max_time: float = 0.0,
    verbose: bool = False,
    monitor: bool = False,
    hub_sk_threshold: float = 0.0,
    weight_anderson: bool = True,
    z_clamp: float = 1e-8,
    device: str = "cpu",
) -> SolverResult:
    """Degeneracy-reduced qDECM **weight step only**, given an already-solved
    (or otherwise fixed) topology.

    Factored out of :func:`solve_fixed_point_qdecm_degenerate` so callers
    that already have `theta_topo` from a separate topology solve (e.g.
    :meth:`~dcms.models.qdecm.qDECMModel.solve_tool`, which tracks topology
    and weight `SolverResult`s separately for its `residuals_topo`/
    `residuals_weights` fields) don't need to re-solve or duplicate the
    reduction logic. `solve_fixed_point_qdecm_degenerate` is the
    convenience entry point when you don't already have a topology
    solution; this is the lower-level piece it's built from.

    Args:
        theta_topo: Topology parameters [theta_out|theta_in], shape (2N,)
            (e.g. from :func:`~dcms.solvers.fixed_point_dcm.solve_fixed_point_dcm_degenerate`).
        theta_weight0: Initial weight guess [theta_b_out|theta_b_in], shape (2N,).
        k_out, k_in, s_out, s_in: Observed sequences, each shape (N,).
        anderson_depth, max_step, max_time, verbose, monitor, device: as in
            :func:`~dcms.solvers.fixed_point_dcm.solve_fixed_point_dcm`; hub_sk_threshold: see the module docstring of
            the hub bisection (0 = off); weight_anderson: weight the Anderson rows by the group multiplicities;
            z_clamp: minimum z in G = 1/expm1(z) (module constant for the duration of the call).

    Returns:
        :class:`~dcms.solvers.base.SolverResult` for the weight step, with
        ``theta``/``best_theta`` expanded to shape (2N,).
    """
    from dcms.solvers.fixed_point_decm import compute_degeneracy_groups

    k_out = k_out if isinstance(k_out, torch.Tensor) else torch.tensor(k_out, dtype=torch.float64)
    k_in = k_in if isinstance(k_in, torch.Tensor) else torch.tensor(k_in, dtype=torch.float64)
    s_out = s_out if isinstance(s_out, torch.Tensor) else torch.tensor(s_out, dtype=torch.float64)
    s_in = s_in if isinstance(s_in, torch.Tensor) else torch.tensor(s_in, dtype=torch.float64)
    k_out, k_in = k_out.to(dtype=torch.float64), k_in.to(dtype=torch.float64)
    s_out, s_in = s_out.to(dtype=torch.float64), s_in.to(dtype=torch.float64)
    theta_topo = (
        theta_topo if isinstance(theta_topo, torch.Tensor)
        else torch.tensor(theta_topo, dtype=torch.float64)
    ).to(dtype=torch.float64)
    theta_weight0 = (
        theta_weight0 if isinstance(theta_weight0, torch.Tensor)
        else torch.tensor(theta_weight0, dtype=torch.float64)
    ).to(dtype=torch.float64)

    N = k_out.shape[0]

    # --- Weight step: full 4-tuple degeneracy (DECM's grouping, reused) ---
    group_of, mult, k_out_g, k_in_g, s_out_g, s_in_g = compute_degeneracy_groups(
        k_out, k_in, s_out, s_in
    )
    M = k_out_g.shape[0]

    _first_idx = first_member_index(group_of, M)

    # Every node in a weight-degeneracy group shares the same (k_out,k_in)
    # 2-tuple too (a coarser condition implied by the shared 4-tuple), so it
    # already shares the same theta_topo value -- this reduction is exact,
    # not an approximation (see module-level note above
    # _qdecm_step_dense_weighted).
    theta_topo_out_g = theta_topo[:N][_first_idx]
    theta_topo_in_g = theta_topo[N:][_first_idx]
    log_xy_g = -theta_topo_out_g[:, None] - theta_topo_in_g[None, :]
    # The diagonal is the REAL probability between two distinct members of the same group: the weighted step counts it
    # (mult_g - 1) times ("weighted sum minus one diagonal term"); zeroing it dropped all within-group pairs.
    P_g = torch.sigmoid(log_xy_g)

    theta_weight0_g = reduce_blocks(theta_weight0, _first_idx, N, 2)

    global _Z_G_CLAMP, _Z_NEWTON_FLOOR
    saved_clamp = (_Z_G_CLAMP, _Z_NEWTON_FLOOR)       # the core sets the module-level z clamp for the duration of the call
    try:
        result_g = _solve_qdecm_weights_reduced(
            theta_weight0_g, s_out_g, s_in_g, torch.cat([theta_topo_out_g, theta_topo_in_g]), P_g, mult,
            tol=tol, max_iter=max_iter, anderson_depth=anderson_depth, max_step=max_step, max_time=max_time,
            verbose=verbose, monitor=monitor, hub_sk_threshold=hub_sk_threshold, weight_anderson=weight_anderson,
            z_clamp=z_clamp, device=device,
        )
    finally:
        _Z_G_CLAMP, _Z_NEWTON_FLOOR = saved_clamp

    return expand_result(result_g, group_of, N, M, 2)
