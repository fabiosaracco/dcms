"""Fixed-point iteration solver for the aDECM weight step.

The aDECM weight equations (conditioned on a fixed DCM topology ``p_ij``) are:

    s_out_i = Σ_{j≠i} p_ij · β_out_i · β_in_j / (1 − β_out_i · β_in_j)
    s_in_i  = Σ_{j≠i} p_ji · β_out_j · β_in_i / (1 − β_out_j · β_in_i)

Two variants are implemented:

* **Gauss-Seidel** — β-space fixed-point; out-multipliers are updated first
  using the update:

      β_out_i^{new} = s_out_i / D_out_i

  where D_out_i = Σ_{j≠i} p_ij · β_in_j / (1 − β_out_i · β_in_j).

  Fresh β_out values are immediately used when computing the in-multiplier
  update:

      β_in_i^{new} = s_in_i / D_in_i

  where D_in_i = Σ_{j≠i} p_ji · β_out_j / (1 − β_out_j · β_in_i).

* **θ-Newton** — θ-space Gauss-Seidel Newton steps (analogous to the DWCM
  θ-Newton variant).  Out-multipliers are updated first (pass 1), then
  in-multipliers are updated using the fresh θ_β_out values (pass 2).

  For each node i in pass 1:

      Δθ_β_out_i = −F_out_i / F′_out_i

  where:
      F_out_i  = Σ_{j≠i} p_ij · G_ij − s_out_i
      F′_out_i = −Σ_{j≠i} p_ij · G_ij · (1 + G_ij)
      G_ij     = 1 / expm1(θ_β_out_i + θ_β_in_j)

  Pass 2 mirrors this for the in-multipliers using the updated θ_β_out.
  The per-node step is clipped to ``[−max_step, +max_step]``.

**Anderson acceleration** is available for all variants (depth 0 = plain FP).

For N > ``_LARGE_N_THRESHOLD`` the N×N matrices are never materialised;
instead row chunks of size ``_DEFAULT_CHUNK`` are used.
"""
from __future__ import annotations

import datetime
import math
import sys
import time
from typing import Callable

import torch

from dcms.models.parameters import aDECM_LARGE_N_THRESHOLD as _LARGE_N_THRESHOLD
from dcms.models.parameters import _DEFAULT_CHUNK, _ETA_MIN, _ETA_MAX
from dcms.solvers.base import SolverResult
from dcms.utils.profiling import _PeakRAMMonitor

_ANDERSON_MAX_NORM: float = 1e6

# Allow the residual to grow transiently large while the θ-Newton solver
# navigates the near-singular region (z → 0⁺, G → ∞).  The per-node alpha
# line-search in _theta_newton_step_chunked keeps z ≥ max(z_prev * _Z_NEWTON_FRAC,
# _Z_NEWTON_FLOOR) after each GS step; the Anderson feasibility guard projects any
# infeasible mix back to z ≥ floor while keeping history intact for acceleration.
_ANDERSON_BLOWUP_FACTOR: float = 5000.0  # clear Anderson history if F > 5000 × best
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
_FP_NEWTON_FALLBACK_DELTA: float = 0.1
_FPGS_NEWTON_RESET_WINDOW: int = 30
_FPGS_NEWTON_STEPS: int = 30
_FPGS_NEWTON_AND_DEPTH: int = 5


def _anderson_mixing(
    fp_outputs: list[torch.Tensor],
    residuals_hist: list[torch.Tensor],
) -> torch.Tensor:
    """Anderson mixing: compute the next iterate from history.

    Solves the *per-component-weighted* constrained least-squares problem:

        min  ‖W⁻¹ Σ_i c_i r_i‖²    s.t.  Σ_i c_i = 1

    where r_i = g(θ_i) − θ_i are the FP residuals and W = diag(w) is a
    per-component scale with w_j = max_i |r_i[j]| (the running max magnitude
    of each coordinate over the history).

    This *weighted* formulation prevents high-strength hub nodes whose
    θ-space FP residuals are much larger than typical nodes from dominating
    the mixing — the unweighted R^T R is ill-conditioned by a factor equal to
    the hub/typical residual ratio squared (easily 10⁶ for N=5000 power-law
    networks), leading to catastrophic extrapolation.  Scaling each row by its
    max magnitude makes the effective R^T R close to the identity (condition ≈ 1),
    giving stable weights while preserving Anderson's acceleration property.

    Args:
        fp_outputs:     List of g(θ_k) values (FP outputs), each shape (2N,).
        residuals_hist: List of r_k = g(θ_k) − θ_k values, each shape (2N,).

    Returns:
        Anderson-mixed next iterate θ_{k+1}, shape (2N,).
    """
    m = len(fp_outputs)
    if m == 1:
        return fp_outputs[0]

    R = torch.stack(residuals_hist, dim=1)  # (2N, m)
    G = torch.stack(fp_outputs, dim=1)      # (2N, m)

    # Per-component weighting: scale each row by its max |r_k| over the history.
    w = R.abs().max(dim=1, keepdim=True).values.clamp(min=1e-15)  # (2N, 1)
    R_w = R / w  # (2N, m): each row scaled to max-absolute-value 1

    # Normal equations on the weighted system: (R_w^T R_w) c ∝ ones
    RtR = R_w.T @ R_w  # (m, m)
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


def _fp_step_dense(
    beta_out: torch.Tensor,
    beta_in: torch.Tensor,
    P: torch.Tensor,
    s_out: torch.Tensor,
    s_in: torch.Tensor,
    variant: str,
    theta: torch.Tensor | None = None,
    max_step: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One dense fixed-point update step for the aDECM weight equations.

    Args:
        beta_out: Current β_out values, shape (N,).
        beta_in:  Current β_in values, shape (N,).
        P:        DCM probability matrix, shape (N, N), diagonal zero.
        s_out:    Observed out-strength sequence, shape (N,).
        s_in:     Observed in-strength sequence, shape (N,).
        variant:  ``"gauss-seidel"`` or ``"jacobi"``.
        theta:    Full parameter vector [θ_out | θ_in], shape (2N,), used
                  by the Newton fallback.  Pass ``None`` to disable.
        max_step: Maximum |Δθ| per node for the Newton fallback step.

    Returns:
        ``(beta_out_new, beta_in_new, F_current)`` where F_current is the
        residual at the pre-update state, shape (2N,).
    """
    N = beta_out.shape[0]

    xy = beta_out[:, None] * beta_in[None, :]
    denom = (1.0 - xy.clamp(max=_Q_MAX)).clamp(min=1e-8)
    # G_new_ij = 1/(1 - β_out_i β_in_j) = 1/denom
    PG_new = P / denom         # p_ij · G_new_ij (diagonal masked by P[i,i]=0)

    # s_out_hat[i] = Σ_{j≠i} p_ij G_new_ij
    s_out_hat = PG_new.sum(dim=1)
    # s_in_hat_orig[j] = Σ_{i≠j} p_ij G_new_ij  (col sums, at current β_out)
    s_in_hat_orig = PG_new.sum(dim=0)

    F_current = torch.cat([s_out_hat - s_out, s_in_hat_orig - s_in])

    beta_out_new = torch.where(s_out_hat > 0, beta_out * s_out / s_out_hat, beta_out)

    if theta is not None:
        theta_out = theta[:N]
        _theta_out_fp = (-torch.log(beta_out_new.clamp(min=1e-300))).clamp(
            -_ETA_MAX, _ETA_MAX
        )
        _use_newton_out = (_theta_out_fp - theta_out).abs() > _FP_NEWTON_FALLBACK_DELTA
        if _use_newton_out.any():
            _delta_out = (s_out_hat - s_out) / s_out_hat.clamp(min=1e-30)
            _delta_out = _delta_out.clamp(-max_step, max_step)
            _theta_out_nt = (theta_out + _delta_out).clamp(-_ETA_MAX, _ETA_MAX)
            beta_out_new = torch.where(
                _use_newton_out, torch.exp(-_theta_out_nt), beta_out_new
            )

    beta_out_upd = beta_out_new if variant == "gauss-seidel" else beta_out

    xy2 = beta_out_upd[:, None] * beta_in[None, :]
    denom2 = (1.0 - xy2.clamp(max=_Q_MAX)).clamp(min=1e-8)
    PG_new2 = P / denom2
    s_in_hat = PG_new2.sum(dim=0)

    beta_in_new = torch.where(s_in_hat > 0, beta_in * s_in / s_in_hat, beta_in)

    if theta is not None:
        theta_in = theta[N:]
        _theta_in_fp = (-torch.log(beta_in_new.clamp(min=1e-300))).clamp(
            -_ETA_MAX, _ETA_MAX
        )
        _use_newton_in = (_theta_in_fp - theta_in).abs() > _FP_NEWTON_FALLBACK_DELTA
        if _use_newton_in.any():
            _delta_in = (s_in_hat - s_in) / s_in_hat.clamp(min=1e-30)
            _delta_in = _delta_in.clamp(-max_step, max_step)
            _theta_in_nt = (theta_in + _delta_in).clamp(-_ETA_MAX, _ETA_MAX)
            beta_in_new = torch.where(
                _use_newton_in, torch.exp(-_theta_in_nt), beta_in_new
            )

    return beta_out_new, beta_in_new, F_current


def _fp_step_chunked(
    beta_out: torch.Tensor,
    beta_in: torch.Tensor,
    theta_topo_out: torch.Tensor,
    theta_topo_in: torch.Tensor,
    s_out: torch.Tensor,
    s_in: torch.Tensor,
    chunk_size: int,
    variant: str,
    theta: torch.Tensor | None = None,
    max_step: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Chunked fixed-point update for aDECM weight equations.

    Avoids materialising the full N×N P and β matrices.

    Args:
        beta_out:       Current β_out values, shape (N,).
        beta_in:        Current β_in values, shape (N,).
        theta_topo_out: Out-topology parameters, shape (N,).
        theta_topo_in:  In-topology parameters, shape (N,).
        s_out:          Observed out-strength sequence, shape (N,).
        s_in:           Observed in-strength sequence, shape (N,).
        chunk_size:     Rows per processing chunk.
        variant:        ``"gauss-seidel"`` or ``"jacobi"``.
        theta:          Full parameter vector [θ_out | θ_in], shape (2N,), used
                        by the Newton fallback.  Pass ``None`` to disable.
        max_step:       Maximum |Δθ| per node for the Newton fallback step.

    Returns:
        ``(beta_out_new, beta_in_new, F_current)`` where F_current is the
        residual at the pre-update state, shape (2N,).
    """
    N = beta_out.shape[0]
    D_out = torch.zeros(N, dtype=torch.float64)    # will hold s_out_hat = Σ p G_new
    s_in_hat_orig = torch.zeros(N, dtype=torch.float64)

    for i_start in range(0, N, chunk_size):
        i_end = min(i_start + chunk_size, N)
        chunk_len = i_end - i_start

        # p_chunk[local_i, j] = p_{i,j} = sigmoid(-θ_out_i - θ_in_j)
        log_xy = (
            -theta_topo_out[i_start:i_end, None]
            - theta_topo_in[None, :]
        )  # (chunk, N)
        p_chunk = torch.sigmoid(log_xy)

        # β_out[i] · β_in[j] products for this chunk
        xy_chunk = beta_out[i_start:i_end, None] * beta_in[None, :]
        denom_chunk = (1.0 - xy_chunk.clamp(max=_Q_MAX)).clamp(min=1e-8)

        # G_new_ij = 1/denom_ij; d_chunk = p_ij G_new_ij
        d_chunk = p_chunk / denom_chunk   # (chunk, N)
        local_i = torch.arange(chunk_len, dtype=torch.long)
        global_j = torch.arange(i_start, i_end, dtype=torch.long)
        d_chunk[local_i, global_j] = 0.0    # zero diagonal

        # D_out[i] = Σ_{j≠i} p_ij G_new_ij  (= s_out_hat[i])
        D_out[i_start:i_end] = d_chunk.sum(dim=1)

        # s_in_hat_orig[j] = Σ_{i≠j} p_ij G_new_ij  (col sums at current β_out)
        s_in_hat_orig += d_chunk.sum(dim=0)

    # Residual at current β (D_out = s_out_hat)
    s_out_hat = D_out
    F_current = torch.cat([s_out_hat - s_out, s_in_hat_orig - s_in])

    beta_out_new = torch.where(D_out > 0, beta_out * s_out / D_out, beta_out)

    if theta is not None:
        _theta_out_fp = (
            -torch.log(beta_out_new.clamp(min=1e-300))
        ).clamp(-_ETA_MAX, _ETA_MAX)
        _use_newton_out = (
            (_theta_out_fp - theta[:N]).abs() > _FP_NEWTON_FALLBACK_DELTA
        )
        if _use_newton_out.any():
            _delta_out = (
                (s_out_hat - s_out) / s_out_hat.clamp(min=1e-30)
            ).clamp(-max_step, max_step)
            _theta_out_nt = (theta[:N] + _delta_out).clamp(-_ETA_MAX, _ETA_MAX)
            beta_out_new = torch.where(
                _use_newton_out, torch.exp(-_theta_out_nt), beta_out_new
            )

    beta_out_upd = beta_out_new if variant == "gauss-seidel" else beta_out

    # D_in[j] = Σ_{i≠j} p[i,j] · G_new_ij  (= s_in_hat[j])
    D_in = torch.zeros(N, dtype=torch.float64)
    for j_start in range(0, N, chunk_size):
        j_end = min(j_start + chunk_size, N)
        chunk_len = j_end - j_start

        log_xy = (
            -theta_topo_out[j_start:j_end, None]
            - theta_topo_in[None, :]
        )  # (chunk, N): rows are j, cols are i
        p_chunk = torch.sigmoid(log_xy)

        xy_chunk = beta_out_upd[j_start:j_end, None] * beta_in[None, :]  # (chunk, N)
        denom_chunk = (1.0 - xy_chunk.clamp(max=_Q_MAX)).clamp(min=1e-8)

        # d_chunk[j,i] = p_ji G_new_ji = p_ji / (1-β_out_j β_in_i)
        d_chunk = p_chunk / denom_chunk   # (chunk, N)
        local_j = torch.arange(chunk_len, dtype=torch.long)
        global_i = torch.arange(j_start, j_end, dtype=torch.long)
        d_chunk[local_j, global_i] = 0.0    # zero diagonal (j==i)
        D_in += d_chunk.sum(dim=0)

    s_in_hat = D_in
    beta_in_new = torch.where(D_in > 0, beta_in * s_in / D_in, beta_in)

    if theta is not None:
        _theta_in_fp = (
            -torch.log(beta_in_new.clamp(min=1e-300))
        ).clamp(-_ETA_MAX, _ETA_MAX)
        _use_newton_in = (
            (_theta_in_fp - theta[N:]).abs() > _FP_NEWTON_FALLBACK_DELTA
        )
        if _use_newton_in.any():
            _delta_in = (
                (s_in_hat - s_in) / s_in_hat.clamp(min=1e-30)
            ).clamp(-max_step, max_step)
            _theta_in_nt = (theta[N:] + _delta_in).clamp(-_ETA_MAX, _ETA_MAX)
            beta_in_new = torch.where(
                _use_newton_in, torch.exp(-_theta_in_nt), beta_in_new
            )

    return beta_out_new, beta_in_new, F_current


def _theta_newton_step_dense(
    theta_weight: torch.Tensor,
    P: torch.Tensor,
    s_out: torch.Tensor,
    s_in: torch.Tensor,
    max_step: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One θ-space coordinate Newton step for the aDECM weight equations (dense).

    For each node i:
        Δθ_β_out_i = (Σ_{j≠i} p_ij G_ij − s_out_i) / Σ_{j≠i} p_ij G_ij(1+G_ij)

    where G_ij = 1/expm1(θ_β_out_i + θ_β_in_j).

    Args:
        theta_weight: Current weight parameters [θ_β_out | θ_β_in], shape (2N,).
        P:            DCM probability matrix, shape (N, N), diagonal zero.
        s_out:        Observed out-strength sequence, shape (N,).
        s_in:         Observed in-strength sequence, shape (N,).
        max_step:     Maximum |Δθ| per node per step.

    Returns:
        Tuple of (updated weight parameters, residual F(θ_current)).
        The residual is evaluated at the *input* theta_weight (before update).
        Shape: ((2N,), (2N,)).
    """
    N = s_out.shape[0]
    theta_b_out = theta_weight[:N]
    theta_b_in = theta_weight[N:]

    z = theta_b_out[:, None] + theta_b_in[None, :]  # (N, N)
    z_safe = z.clamp(min=_Z_G_CLAMP)
    G = -1.0 / torch.expm1(-z_safe)   # G_new = 1/(1-exp(-z))
    G.fill_diagonal_(0.0)

    PG = P * G               # (N, N)  = p_ij G_new_ij
    PGG1 = P * G * (G - 1.0)  # (N, N) = p_ij G_new_ij(G_new_ij-1) = p_ij G_new G_old

    # Residual at current theta (before update)
    F_out = PG.sum(dim=1) - s_out    # shape (N,)
    F_in_current = PG.sum(dim=0) - s_in  # col sums at ORIGINAL theta
    F_current = torch.cat([F_out, F_in_current])

    # Out-multiplier Newton steps
    H_out = -PGG1.sum(dim=1)                          # diagonal Hessian ≤ 0
    neg_H_out = (-H_out).clamp(min=1e-15)
    delta_out = F_out / neg_H_out                     # = -ΔF/|H|
    delta_out = delta_out.clamp(-max_step, max_step)
    theta_b_out_new = (theta_b_out + delta_out).clamp(-_ETA_MAX, _ETA_MAX)
    # Zero-strength nodes: β = 0 exactly → θ = _ETA_MAX
    theta_b_out_new = torch.where(
        s_out == 0, torch.full_like(theta_b_out_new, _ETA_MAX), theta_b_out_new
    )

    # Recompute G with updated θ_β_out (Gauss-Seidel: use fresh θ_β_out)
    z2 = theta_b_out_new[:, None] + theta_b_in[None, :]
    z2_safe = z2.clamp(min=_Z_G_CLAMP)
    G2 = -1.0 / torch.expm1(-z2_safe)
    G2.fill_diagonal_(0.0)

    PG2 = P * G2
    PGG12 = P * G2 * (G2 - 1.0)

    # In-multiplier Newton steps
    F_in = PG2.sum(dim=0) - s_in                      # col sums of PG2
    H_in = -PGG12.sum(dim=0)
    neg_H_in = (-H_in).clamp(min=1e-15)
    delta_in = F_in / neg_H_in
    delta_in = delta_in.clamp(-max_step, max_step)
    theta_b_in_new = (theta_b_in + delta_in).clamp(-_ETA_MAX, _ETA_MAX)
    theta_b_in_new = torch.where(
        s_in == 0, torch.full_like(theta_b_in_new, _ETA_MAX), theta_b_in_new
    )

    return torch.cat([theta_b_out_new, theta_b_in_new]), F_current


def _theta_newton_step_chunked(
    theta_weight: torch.Tensor,
    theta_topo_out: torch.Tensor,
    theta_topo_in: torch.Tensor,
    s_out: torch.Tensor,
    s_in: torch.Tensor,
    chunk_size: int,
    max_step: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunked θ-space Gauss-Seidel Newton step for aDECM weight equations.

    Two passes over the (N, N) product matrix, mirroring the DWCM chunked
    Newton step:

    * **Pass 1** — row sums (F_out, neg_H_out) and column sums for the
      current-state residual (F_in_current).  Accumulates z_min_out[i] =
      min over significant j of z_ij = θ_β_out_i + θ_β_in_j.
      Applies **out-direction** Newton step with a per-pass z-floor
      line-search (z_LS = 0) that prevents any significant pair from
      having z_ij go negative.
    * **Pass 2** — column sums using the updated θ_β_out_new (Gauss-Seidel:
      in-direction sees the fresh out-multipliers).  Accumulates z_min_in[j]
      and applies the **in-direction** step with the same z-floor line-search.

    **Why two separate line-searches (not one combined)**:
    In GS mode the two directions are applied sequentially.  A combined
    (Jacobi) line-search would be unnecessarily conservative for the first
    direction (would account for δ_in that hasn't been applied yet).  Using
    per-pass line-searches gives the largest safe step for each direction
    independently.

    **Why z_LS = 0 (not _Z_G_CLAMP)**:
    Using z_LS = _Z_G_CLAMP caused stagnation: once z hit exactly the clamp
    value, available = z − _Z_G_CLAMP = 0 → alpha_cand = 0 → zero step.
    With z_LS = 0 the available space is z itself, which is always positive
    until the physical boundary z = 0 is hit.  The G-computation clamp
    (_Z_G_CLAMP) remains separate and only caps G numerically.

    **z_min is computed over significant pairs only** (p_ij > 0.5/N), so
    disconnected-pair columns/rows with extreme θ_β values do not force
    alpha to near-zero unnecessarily.

    Args:
        theta_weight:   Current weight parameters [θ_β_out | θ_β_in], shape (2N,).
        theta_topo_out: Out-topology parameters, shape (N,).
        theta_topo_in:  In-topology parameters, shape (N,).
        s_out:          Observed out-strength sequence, shape (N,).
        s_in:           Observed in-strength sequence, shape (N,).
        chunk_size:     Rows per processing chunk.
        max_step:       Maximum |Δθ| per node per step.

    Returns:
        Tuple of (updated weight parameters, residual F(θ_current)).
        F is evaluated at the *input* theta_weight (before update).
        Shape: ((2N,), (2N,)).
    """
    N = s_out.shape[0]
    theta_b_out = theta_weight[:N]
    theta_b_in = theta_weight[N:]

    sig_log_thresh: float = math.log(0.5 / N) - math.log(1.0 - 0.5 / N)

    # ------------------------------------------------------------------
    # Pass 1: out-direction Newton step
    # Accumulate row sums (F_out, neg_H_out), col sums for F_current,
    # and z_min_out[i] = min over significant j of z_ij (for line-search).
    # ------------------------------------------------------------------
    sum_PG_out = torch.zeros(N, dtype=torch.float64)
    sum_PGG1_out = torch.zeros(N, dtype=torch.float64)
    sum_PG_in_current = torch.zeros(N, dtype=torch.float64)
    z_min_out = torch.full((N,), float("inf"), dtype=torch.float64)

    for i_start in range(0, N, chunk_size):
        i_end = min(i_start + chunk_size, N)
        chunk_len = i_end - i_start

        log_xy = (
            -theta_topo_out[i_start:i_end, None]
            - theta_topo_in[None, :]
        )
        p_chunk = torch.sigmoid(log_xy)  # (chunk, N)
        sig_mask = log_xy > sig_log_thresh  # p > 0.5/N

        z_chunk = theta_b_out[i_start:i_end, None] + theta_b_in[None, :]
        z_safe = z_chunk.clamp(min=_Z_G_CLAMP)
        G_chunk = -1.0 / torch.expm1(-z_safe)  # G_new = 1/(1-exp(-z))

        local_i = torch.arange(chunk_len, dtype=torch.long)
        global_j = torch.arange(i_start, i_end, dtype=torch.long)
        G_chunk[local_i, global_j] = 0.0  # zero diagonal

        PG_chunk = p_chunk * G_chunk
        PGG1_chunk = PG_chunk * (G_chunk - 1.0)  # p G_new (G_new-1) = p G_new G_old

        sum_PG_out[i_start:i_end] = PG_chunk.sum(dim=1)
        sum_PGG1_out[i_start:i_end] = PGG1_chunk.sum(dim=1)
        sum_PG_in_current += PG_chunk.sum(dim=0)

        # z_min_out[i]: min over significant j of (theta_b_out_i + theta_b_in_j)
        z_sig = torch.where(sig_mask, z_chunk, torch.full_like(z_chunk, float("inf")))
        z_min_out[i_start:i_end] = torch.minimum(
            z_min_out[i_start:i_end], z_sig.min(dim=1).values
        )

    # Global z-floor guard: significant-pair tracking misses insignificant pairs
    # (p < 0.5/N) that can still blow up G when z < 0.  Use the global minimum
    # of theta_b_in over non-zero-strength nodes to tighten z_min_out[i].
    # z_min_out[i] = theta_b_out_i + min_j(theta_b_in_j) (all j with s_in_j > 0).
    nz_in_mask = s_in > 0
    if nz_in_mask.any():
        min_theta_b_in_nz = theta_b_in[nz_in_mask].min()
        z_min_out = torch.minimum(z_min_out, theta_b_out + min_theta_b_in_nz)

    F_out = sum_PG_out - s_out
    F_in_current = sum_PG_in_current - s_in
    F_current = torch.cat([F_out, F_in_current])

    neg_H_out = sum_PGG1_out.clamp(min=1e-15)
    delta_out = (F_out / neg_H_out).clamp(-max_step, max_step)

    # Per-node z-floor line-search for out-direction.
    # Effective floor = max(z_min * _Z_NEWTON_FRAC, _Z_NEWTON_FLOOR).
    # The relative term prevents z from jumping more than 50 % downward in one
    # step (stops oscillation between z* and z_clamp); the hard floor enables
    # the doubling self-escape at z = z_clamp.
    z_floor_out = torch.clamp(z_min_out * _Z_NEWTON_FRAC, min=_Z_NEWTON_FLOOR)
    available_out = (z_min_out - z_floor_out).clamp(min=0.0)
    needed_out = delta_out.abs().clamp(min=1e-30)
    alpha_out = torch.where(
        delta_out < 0,
        (available_out / needed_out).clamp(max=1.0),
        torch.ones(N, dtype=torch.float64),
    )

    theta_b_out_new = (theta_b_out + alpha_out * delta_out).clamp(-_ETA_MAX, _ETA_MAX)
    theta_b_out_new = torch.where(
        s_out == 0, torch.full_like(theta_b_out_new, _ETA_MAX), theta_b_out_new
    )

    # ------------------------------------------------------------------
    # Pass 2: in-direction Newton step (GS: uses theta_b_out_new).
    # Accumulate col sums (F_in, neg_H_in) and z_min_in[j] for line-search.
    # ------------------------------------------------------------------
    sum_PG_in = torch.zeros(N, dtype=torch.float64)
    sum_PGG1_in = torch.zeros(N, dtype=torch.float64)
    z_min_in = torch.full((N,), float("inf"), dtype=torch.float64)

    for i_start in range(0, N, chunk_size):
        i_end = min(i_start + chunk_size, N)
        chunk_len = i_end - i_start

        log_xy = (
            -theta_topo_out[i_start:i_end, None]
            - theta_topo_in[None, :]
        )
        p_chunk = torch.sigmoid(log_xy)  # (chunk, N)
        sig_mask = log_xy > sig_log_thresh

        z2_chunk = theta_b_out_new[i_start:i_end, None] + theta_b_in[None, :]
        z2_safe = z2_chunk.clamp(min=_Z_G_CLAMP)
        G2_chunk = -1.0 / torch.expm1(-z2_safe)  # G_new

        local_i = torch.arange(chunk_len, dtype=torch.long)
        global_j = torch.arange(i_start, i_end, dtype=torch.long)
        G2_chunk[local_i, global_j] = 0.0  # zero diagonal

        PG2_chunk = p_chunk * G2_chunk
        PGG12_chunk = PG2_chunk * (G2_chunk - 1.0)  # p G_new (G_new-1)

        sum_PG_in += PG2_chunk.sum(dim=0)    # col sums
        sum_PGG1_in += PGG12_chunk.sum(dim=0)

        # z_min_in[j]: min over significant i of (theta_b_out_new_i + theta_b_in_j)
        z2_sig = torch.where(sig_mask, z2_chunk, torch.full_like(z2_chunk, float("inf")))
        z_min_in = torch.minimum(z_min_in, z2_sig.min(dim=0).values)

    # Global z-floor guard for in-direction (mirrors out-direction guard above).
    nz_out_mask = s_out > 0
    if nz_out_mask.any():
        min_theta_b_out_new_nz = theta_b_out_new[nz_out_mask].min()
        z_min_in = torch.minimum(z_min_in, min_theta_b_out_new_nz + theta_b_in)

    F_in = sum_PG_in - s_in
    neg_H_in = sum_PGG1_in.clamp(min=1e-15)
    delta_in = (F_in / neg_H_in).clamp(-max_step, max_step)

    # Per-node z-floor line-search for in-direction (same two-floor logic).
    z_floor_in = torch.clamp(z_min_in * _Z_NEWTON_FRAC, min=_Z_NEWTON_FLOOR)
    available_in = (z_min_in - z_floor_in).clamp(min=0.0)
    needed_in = delta_in.abs().clamp(min=1e-30)
    alpha_in = torch.where(
        delta_in < 0,
        (available_in / needed_in).clamp(max=1.0),
        torch.ones(N, dtype=torch.float64),
    )

    theta_b_in_new = (theta_b_in + alpha_in * delta_in).clamp(-_ETA_MAX, _ETA_MAX)
    theta_b_in_new = torch.where(
        s_in == 0, torch.full_like(theta_b_in_new, _ETA_MAX), theta_b_in_new
    )

    return torch.cat([theta_b_out_new, theta_b_in_new]), F_current


def solve_fixed_point_adecm(
    residual_fn: Callable[[torch.Tensor], torch.Tensor],
    theta0: "ArrayLike",  # type: ignore[name-defined]
    s_out: "ArrayLike",  # type: ignore[name-defined]
    s_in: "ArrayLike",  # type: ignore[name-defined]
    theta_topo: "ArrayLike",  # type: ignore[name-defined]
    P: "ArrayLike | None" = None,
    tol: float = 1e-8,
    max_iter: int = 10_000,
    damping: float = 1.0,
    variant: str = "gauss-seidel",
    chunk_size: int = 0,
    anderson_depth: int = 0,
    max_step: float = 1.0,
    max_time: float = 0.0,
    backend: str = "auto",
    num_threads: int = 0,
    verbose: bool = False,
) -> SolverResult:
    """Fixed-point iteration for the aDECM weight step.

    Args:
        residual_fn: Function F_w(θ_β) → strength residual tensor (2N,).
                     Used for convergence checks.
        theta0:      Initial weight parameter vector [θ_β_out | θ_β_in], shape (2N,).
                     All entries must be strictly positive.
        s_out:       Observed out-strength sequence, shape (N,).
        s_in:        Observed in-strength sequence, shape (N,).
        theta_topo:  Fixed topology parameters [θ_out | θ_in] from the DCM
                     solution, shape (2N,).  Used to compute p_ij at each step.
        P:           Pre-computed DCM probability matrix, shape (N, N).
                     If ``None``, computed from ``theta_topo`` at the start.
                     Pass a pre-computed matrix to avoid recomputing every call.
        tol:         Convergence tolerance on the ℓ∞ residual norm.
        max_iter:    Maximum number of iterations.
        damping:     Damping factor α ∈ (0, 1] for the ``"gauss-seidel"`` and
                     ``"jacobi"`` variants.  α=1 → no damping.  Not used by
                     ``"theta-newton"``; use ``max_step`` instead.
        variant:     One of ``"gauss-seidel"``, ``"jacobi"``, or
                     ``"theta-newton"``.
        chunk_size:  If > 0, process the N×N products in chunks of this size.
                     If 0, auto-select: dense for N ≤ ``_LARGE_N_THRESHOLD``,
                     chunked (``_DEFAULT_CHUNK``) otherwise.
        anderson_depth: Anderson acceleration depth.  0 = plain FP.
        max_step:    Maximum per-node Newton step in ``"theta-newton"`` variant.
        max_time:    Wall-clock time limit in seconds.  If > 0, the solver
                     stops after this many seconds even if ``max_iter`` has not
                     been reached.  Default 0 (no time limit).
        backend:     Compute backend: ``"auto"`` (default), ``"pytorch"``, or
                     ``"numba"``.  ``"auto"`` uses PyTorch for N ≤ 5 000 and
                     Numba for larger networks.  Falls back automatically with
                     a warning if the requested backend is unavailable.
        num_threads: Number of Numba parallel threads.  0 (default) leaves
                     the global Numba thread count unchanged.  Only takes
                     effect when ``backend="numba"`` (or ``"auto"`` at large N).
        verbose:     If ``True``, print a progress line at every iteration
                     showing timestamp, iteration count, elapsed time, and MRE.

    Returns:
        :class:`~src.solvers.base.SolverResult` instance.
    """
    if variant not in ("jacobi", "gauss-seidel", "theta-newton"):
        raise ValueError(
            f"Unknown variant {variant!r}. "
            "Choose 'jacobi', 'gauss-seidel', or 'theta-newton'."
        )
    if not (0.0 < damping <= 1.0):
        raise ValueError(f"damping must be in (0, 1], got {damping}")
    if chunk_size < 0:
        raise ValueError(f"chunk_size must be ≥ 0 (0 = auto), got {chunk_size}")

    # Convert inputs to tensors
    if not isinstance(s_out, torch.Tensor):
        s_out = torch.tensor(s_out, dtype=torch.float64)
    else:
        s_out = s_out.to(dtype=torch.float64)
    if not isinstance(s_in, torch.Tensor):
        s_in = torch.tensor(s_in, dtype=torch.float64)
    else:
        s_in = s_in.to(dtype=torch.float64)
    if not isinstance(theta_topo, torch.Tensor):
        theta_topo = torch.tensor(theta_topo, dtype=torch.float64)
    else:
        theta_topo = theta_topo.to(dtype=torch.float64)
    if not isinstance(theta0, torch.Tensor):
        theta = torch.tensor(theta0, dtype=torch.float64)
    else:
        theta = theta0.clone().to(dtype=torch.float64)

    N = s_out.shape[0]
    theta = theta.clamp(-_ETA_MAX, _ETA_MAX)  # allow β>1 (negative θ)

    # Resolve compute backend
    from dcms.utils.backend import resolve_backend
    _backend = resolve_backend(backend, N)
    _use_numba = (_backend == "numba")
    _prev_numba_threads: int | None = None
    if _use_numba:
        import numpy as np
        from dcms.solvers._numba_kernels import (
            _adecm_theta_newton_numba,
            _adecm_fp_gs_numba,
        )
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

    # Pre-compute DCM probability matrix (or use caller-supplied one)
    if _use_numba:
        theta_topo_out_chunked = theta_topo[:N]
        theta_topo_in_chunked = theta_topo[N:]
    elif effective_chunk == 0:  # dense path: pre-compute P once
        if P is None:
            theta_topo_out = theta_topo[:N]
            theta_topo_in = theta_topo[N:]
            log_xy = -theta_topo_out[:, None] - theta_topo_in[None, :]
            P_mat = torch.sigmoid(log_xy)  # (N, N)
            P_mat.fill_diagonal_(0.0)
        elif not isinstance(P, torch.Tensor):
            P_mat = torch.tensor(P, dtype=torch.float64)
        else:
            P_mat: torch.Tensor = P.to(dtype=torch.float64)
    else:
        theta_topo_out_chunked = theta_topo[:N]
        theta_topo_in_chunked = theta_topo[N:]

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

    # FP-GS periodic Newton correction stagnation counter
    _fpgs_best_local: float = float("inf")
    _fpgs_stagnation_count: int = 0

    # Precompute verbose targets once (MRE = max |F_i| / s_i)
    _v_targets = torch.cat([s_out, s_in])
    _v_nonzero = _v_targets > 0

    try:
        for _ in range(max_iter):
            # Wall-clock time limit
            if max_time > 0 and (time.perf_counter() - t0) > max_time:
                message = (
                    f"Time limit ({max_time:.0f}s) reached at iteration {n_iter}."
                )
                break

            if variant == "theta-newton":
                # θ-space coordinate Newton step.
                if _use_numba:
                    tbo = theta[:N].numpy()
                    tbi = theta[N:].numpy()
                    tbo_new, tbi_new, fo, fi = _adecm_theta_newton_numba(
                        tbo, tbi,
                        theta_topo_out_chunked.numpy(),
                        theta_topo_in_chunked.numpy(),
                        s_out.numpy(), s_in.numpy(),
                        max_step, _ETA_MIN, _ETA_MAX,
                        _Z_G_CLAMP, _Z_NEWTON_FLOOR, _Z_NEWTON_FRAC,
                    )
                    theta_fp = torch.from_numpy(np.concatenate([tbo_new, tbi_new]))
                    F_current = torch.from_numpy(np.concatenate([fo, fi]))
                elif effective_chunk > 0:
                    theta_fp, F_current = _theta_newton_step_chunked(
                        theta,
                        theta_topo_out_chunked,
                        theta_topo_in_chunked,
                        s_out,
                        s_in,
                        effective_chunk,
                        max_step,
                    )
                else:
                    theta_fp, F_current = _theta_newton_step_dense(
                        theta, P_mat, s_out, s_in, max_step
                    )
            else:
                # β-space fixed-point iteration (Gauss-Seidel or Jacobi)
                if _use_numba:
                    bo = torch.exp(-theta[:N]).numpy()
                    bi = torch.exp(-theta[N:]).numpy()
                    bon, bin_, fo, fi = _adecm_fp_gs_numba(
                        bo, bi,
                        theta_topo_out_chunked.numpy(),
                        theta_topo_in_chunked.numpy(),
                        s_out.numpy(), s_in.numpy(),
                        theta[:N].numpy(), theta[N:].numpy(),
                        max_step, _ETA_MIN, _ETA_MAX,
                        _Q_MAX, _FP_NEWTON_FALLBACK_DELTA, True,
                    )
                    beta_out_new = torch.from_numpy(bon)
                    beta_in_new = torch.from_numpy(bin_)
                    F_current = torch.from_numpy(np.concatenate([fo, fi]))
                elif effective_chunk > 0:
                    beta_out = torch.exp(-theta[:N])
                    beta_in = torch.exp(-theta[N:])
                    beta_out_new, beta_in_new, F_current = _fp_step_chunked(
                        beta_out, beta_in,
                        theta_topo_out_chunked, theta_topo_in_chunked,
                        s_out, s_in, effective_chunk, variant,
                        theta=theta, max_step=max_step,
                    )
                else:
                    beta_out = torch.exp(-theta[:N])
                    beta_in = torch.exp(-theta[N:])
                    beta_out_new, beta_in_new, F_current = _fp_step_dense(
                        beta_out, beta_in, P_mat, s_out, s_in, variant,
                        theta=theta, max_step=max_step,
                    )

                # Clamp β to (0, ∞) to maintain non-negativity; product cap is
                # enforced via _Q_MAX in the step functions.
                beta_out_new = beta_out_new.clamp(min=1e-300)
                beta_in_new = beta_in_new.clamp(min=1e-300)

                # Convert to θ-space and apply damping
                theta_out_new = (-torch.log(beta_out_new)).clamp(-_ETA_MAX, _ETA_MAX)
                theta_in_new = (-torch.log(beta_in_new)).clamp(-_ETA_MAX, _ETA_MAX)
                fp_raw = torch.cat([theta_out_new, theta_in_new])

                # Damped FP output: g(θ) = α * FP_raw(θ) + (1−α) * θ
                theta_fp = (damping * fp_raw + (1.0 - damping) * theta).clamp(
                    -_ETA_MAX, _ETA_MAX
                )

            # Universal single-step floor: prevent any θ from dropping below
            # _ANDERSON_THETA_FLOOR × its current value in a single iteration.
            # Only apply when theta_fp > 0 — when the FP output is negative
            # (β > 1 needed), allow it freely; don't push it back above zero.
            _step_floor = (theta * _ANDERSON_THETA_FLOOR).clamp(min=-_ETA_MAX)
            theta_fp = torch.where(
                theta_fp > 0, torch.maximum(theta_fp, _step_floor), theta_fp
            )

            # --- Convergence check using the step-computed residual ---
            res_norm = F_current.abs().max().item()

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
                    f"[{datetime.datetime.now():%H:%M:%S}] "
                    f"iteration={n_iter}, "
                    f"elapsed time={int(_elapsed // 3600):d}:{int((_elapsed % 3600) // 60):d}:{_elapsed % 60:.0f}, "
                    f"MRE={_mre:.3e}"
                )
                sys.stdout.flush()

            # Keep a reference to the iterate with the minimum equation-residual.
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

            _fpgs_newton_fired = False
            if variant != "theta-newton" and anderson_depth > 1:
                if res_norm < _fpgs_best_local * 0.99:
                    _fpgs_best_local = res_norm
                    _fpgs_stagnation_count = 0
                else:
                    _fpgs_stagnation_count += 1
                    if (
                        _fpgs_stagnation_count >= _FPGS_NEWTON_RESET_WINDOW
                        and res_norm > tol
                    ):
                        _fpgs_newton_fired = True
                        _fpgs_stagnation_count = 0
                        _fpgs_best_local = float("inf")
                        _and_g.clear()
                        _and_r.clear()

            # --- Apply Anderson acceleration or plain update ---
            if anderson_depth > 1:
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
                    r_k_norm = r_k.abs().max().item()
                    if math.isfinite(r_k_norm) and (
                        variant != "theta-newton" or r_k_norm < _ANDERSON_MAX_NORM
                    ):
                        _and_g.append(theta_fp.clone())
                        _and_r.append(r_k.clone())

                    if len(_and_g) > anderson_depth:
                        _and_g.pop(0)
                        _and_r.pop(0)

                    if len(_and_g) >= 2:
                        theta_next = _anderson_mixing(_and_g, _and_r)
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

            # --- FP-GS post-Anderson Newton-Anderson mini-loop ---
            if _fpgs_newton_fired:
                theta_nt = best_theta.clone()
                _nt_and_g: list[torch.Tensor] = []
                _nt_and_r: list[torch.Tensor] = []
                for _ in range(_FPGS_NEWTON_STEPS):
                    if _use_numba:
                        tbo = theta_nt[:N].numpy()
                        tbi = theta_nt[N:].numpy()
                        tbo_new, tbi_new, fo, fi = _adecm_theta_newton_numba(
                            tbo, tbi,
                            theta_topo_out_chunked.numpy(),
                            theta_topo_in_chunked.numpy(),
                            s_out.numpy(), s_in.numpy(),
                            max_step, _ETA_MIN, _ETA_MAX,
                            _Z_G_CLAMP, _Z_NEWTON_FLOOR, _Z_NEWTON_FRAC,
                        )
                        theta_nt_fp = torch.from_numpy(np.concatenate([tbo_new, tbi_new]))
                        F_nt = torch.from_numpy(np.concatenate([fo, fi]))
                    elif effective_chunk > 0:
                        theta_nt_fp, F_nt = _theta_newton_step_chunked(
                            theta_nt,
                            theta_topo_out_chunked,
                            theta_topo_in_chunked,
                            s_out,
                            s_in,
                            effective_chunk,
                            max_step,
                        )
                    else:
                        theta_nt_fp, F_nt = _theta_newton_step_dense(
                            theta_nt, P_mat, s_out, s_in, max_step
                        )
                    # Per-step floor (same as main loop): only when positive
                    _nt_floor = (theta_nt * _ANDERSON_THETA_FLOOR).clamp(min=-_ETA_MAX)
                    theta_nt_fp = torch.where(
                        theta_nt_fp > 0,
                        torch.maximum(theta_nt_fp, _nt_floor),
                        theta_nt_fp,
                    ).clamp(-_ETA_MAX, _ETA_MAX)
                    nt_res = F_nt.abs().max().item()
                    if nt_res < tol:
                        theta_nt = theta_nt_fp
                        break
                    # Anderson mixing within the mini-loop
                    r_nt = theta_nt_fp - theta_nt
                    r_nt_norm = r_nt.abs().max().item()
                    if math.isfinite(r_nt_norm) and r_nt_norm < _ANDERSON_MAX_NORM:
                        _nt_and_g.append(theta_nt_fp.clone())
                        _nt_and_r.append(r_nt.clone())
                    if len(_nt_and_g) > _FPGS_NEWTON_AND_DEPTH:
                        _nt_and_g.pop(0)
                        _nt_and_r.pop(0)
                    if len(_nt_and_g) >= 2:
                        theta_nt_next = _anderson_mixing(_nt_and_g, _nt_and_r)
                        eff_floor = torch.minimum(
                            _nt_floor, theta_nt_fp
                        )
                        theta_nt_next = torch.maximum(theta_nt_next, eff_floor).clamp(
                            -_ETA_MAX, _ETA_MAX
                        )
                    else:
                        theta_nt_next = theta_nt_fp
                    theta_nt = theta_nt_next
                theta_next = theta_nt
                # Reset blowup threshold so post-Newton residuals don't trigger
                # a false blowup on the very next main-loop iteration.
                _best_res_for_anderson = float("inf")

            theta = theta_next
    finally:
        elapsed = time.perf_counter() - t0
        _peak_ram_monitor.__exit__(None, None, None)
        peak_ram = _peak_ram_monitor.peak_bytes
        if _prev_numba_threads is not None:
            import numba as _numba_mod
            _numba_mod.set_num_threads(_prev_numba_threads)

    return SolverResult(
        theta=best_theta.detach().numpy(),
        converged=converged,
        iterations=n_iter,
        residuals=residuals,
        elapsed_time=elapsed,
        peak_ram_bytes=peak_ram,
        message=message,
    )
