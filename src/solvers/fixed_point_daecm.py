"""Fixed-point iteration solver for the DaECM weight step.

The DaECM weight equations (conditioned on a fixed DCM topology ``p_ij``) are:

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

* **θ-Newton** — θ-space coordinate Newton steps (analogous to the DWCM
  θ-Newton variant).  For each node i:

      Δθ_β_out_i = −F_out_i / F′_out_i

  where:
      F_out_i  = Σ_{j≠i} p_ij · G_ij − s_out_i
      F′_out_i = −Σ_{j≠i} p_ij · G_ij · (1 + G_ij)
      G_ij     = 1 / expm1(θ_β_out_i + θ_β_in_j)

  The per-node step is clipped to ``[−max_step, +max_step]``.

**Anderson acceleration** is available for all variants (depth 0 = plain FP).

For N > ``_LARGE_N_THRESHOLD`` the N×N matrices are never materialised;
instead row chunks of size ``_DEFAULT_CHUNK`` are used.
"""
from __future__ import annotations

import math
import time
import tracemalloc
from typing import Callable

import torch

from src.models.daecm import _LARGE_N_THRESHOLD, _DEFAULT_CHUNK, _ETA_MIN, _ETA_MAX
from src.solvers.base import SolverResult

_ANDERSON_MAX_NORM: float = 1e6

_ANDERSON_BLOWUP_FACTOR: float = 5.0
_Q_MAX: float = 0.9999  # maximum allowed product β_out * β_in
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
    """One dense fixed-point update step for the DaECM weight equations.

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

    # D_out[i] = Σ_{j≠i} P_ij · β_in_j / (1 - β_out_i · β_in_j)
    D_out = (P * beta_in[None, :] / denom).sum(dim=1)
    s_out_hat = beta_out * D_out

    # D_in_orig uses ORIGINAL beta_out (for residual at current state).
    # denom.T[m,j] = denom[j,m] = 1 - β_out_j * β_in_m ← correct denominator for D_in
    D_in_orig = (P.T * beta_out[None, :] / denom.T).sum(dim=1)
    s_in_hat_orig = beta_in * D_in_orig

    F_current = torch.cat([s_out_hat - s_out, s_in_hat_orig - s_in])

    beta_out_new = torch.where(D_out > 0, s_out / D_out, beta_out)

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
    D_in = (P.T * beta_out_upd[None, :] / denom2.T).sum(dim=1)
    s_in_hat = beta_in * D_in

    beta_in_new = torch.where(D_in > 0, s_in / D_in, beta_in)

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
    """Chunked fixed-point update for DaECM weight equations.

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
    D_out = torch.zeros(N, dtype=torch.float64)
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

        # D_out[i] = Σ_{j≠i} p[i,j] · β_in[j] / (1 - β_out[i] β_in[j])
        d_chunk = p_chunk * beta_in[None, :] / denom_chunk  # (chunk, N)
        local_i = torch.arange(chunk_len, dtype=torch.long)
        global_j = torch.arange(i_start, i_end, dtype=torch.long)
        d_chunk[local_i, global_j] = 0.0    # zero diagonal
        D_out[i_start:i_end] = d_chunk.sum(dim=1)

        # Accumulate s_in_hat_orig[j] = β_in[j] * Σ_i p[i,j] * β_out[i] / (1-β_out[i]*β_in[j])
        # = β_in[j] * Σ_i p[i,j] * β_out[i] / denom[i,j]
        # d_chunk[i,j] = p[i,j] * β_in[j] / denom[i,j]
        # so d_chunk.T[j,i] * β_out[i_chunk] sums to Σ_i p[i,j] * β_in[j] * β_out[i] / denom[i,j]
        b_out_chunk = beta_out[i_start:i_end]
        s_in_hat_orig += d_chunk.T @ b_out_chunk  # (N,)

    # Residual at current β
    s_out_hat = beta_out * D_out
    F_current = torch.cat([s_out_hat - s_out, s_in_hat_orig - s_in])

    beta_out_new = torch.where(D_out > 0, s_out / D_out, beta_out)

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

    # D_in[i] = Σ_{j≠i} p[j,i] · β_out[j] / (1 - β_out[j] β_in[i])
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

        d_chunk = p_chunk * beta_out_upd[j_start:j_end, None] / denom_chunk  # (chunk, N)
        local_j = torch.arange(chunk_len, dtype=torch.long)
        global_i = torch.arange(j_start, j_end, dtype=torch.long)
        d_chunk[local_j, global_i] = 0.0    # zero diagonal (j==i)
        D_in += d_chunk.sum(dim=0)

    s_in_hat = beta_in * D_in
    beta_in_new = torch.where(D_in > 0, s_in / D_in, beta_in)

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
    """One θ-space coordinate Newton step for the DaECM weight equations (dense).

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
    z_safe = z.clamp(min=1e-8)
    G = 1.0 / torch.expm1(z_safe)   # G_ij = β_β / (1-β_β), diagonal irrelevant
    G.fill_diagonal_(0.0)

    PG = P * G               # (N, N)  = p_ij G_ij
    PGG1 = P * G * (1.0 + G)  # (N, N) = p_ij G_ij(1+G_ij)

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
    z2_safe = z2.clamp(min=1e-8)
    G2 = 1.0 / torch.expm1(z2_safe)
    G2.fill_diagonal_(0.0)

    PG2 = P * G2
    PGG12 = P * G2 * (1.0 + G2)

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
    """Chunked θ-space coordinate Newton step for the DaECM weight equations.

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
        The residual is evaluated at the *input* theta_weight (before update).
        Shape: ((2N,), (2N,)).
    """
    N = s_out.shape[0]
    theta_b_out = theta_weight[:N]
    theta_b_in = theta_weight[N:]

    # First pass: compute out-Newton steps and accumulate F_in at original theta
    sum_PG_out = torch.zeros(N, dtype=torch.float64)
    sum_PGG1_out = torch.zeros(N, dtype=torch.float64)
    sum_PG_in_orig = torch.zeros(N, dtype=torch.float64)

    for i_start in range(0, N, chunk_size):
        i_end = min(i_start + chunk_size, N)
        chunk_len = i_end - i_start

        log_xy = (
            -theta_topo_out[i_start:i_end, None]
            - theta_topo_in[None, :]
        )
        p_chunk = torch.sigmoid(log_xy)  # (chunk, N)

        z_chunk = theta_b_out[i_start:i_end, None] + theta_b_in[None, :]
        z_safe = z_chunk.clamp(min=1e-8)
        G_chunk = 1.0 / torch.expm1(z_safe)  # (chunk, N)

        local_i = torch.arange(chunk_len, dtype=torch.long)
        global_j = torch.arange(i_start, i_end, dtype=torch.long)
        G_chunk[local_i, global_j] = 0.0

        PG_chunk = p_chunk * G_chunk
        PGG1_chunk = p_chunk * G_chunk * (1.0 + G_chunk)

        sum_PG_out[i_start:i_end] = PG_chunk.sum(dim=1)
        sum_PGG1_out[i_start:i_end] = PGG1_chunk.sum(dim=1)
        # Column sums give F_in at original theta (free from same PG_chunk)
        sum_PG_in_orig += PG_chunk.sum(dim=0)

    F_out = sum_PG_out - s_out
    F_in_current = sum_PG_in_orig - s_in
    F_current = torch.cat([F_out, F_in_current])

    neg_H_out = sum_PGG1_out.clamp(min=1e-15)
    delta_out = (F_out / neg_H_out).clamp(-max_step, max_step)
    theta_b_out_new = (theta_b_out + delta_out).clamp(-_ETA_MAX, _ETA_MAX)
    theta_b_out_new = torch.where(
        s_out == 0, torch.full_like(theta_b_out_new, _ETA_MAX), theta_b_out_new
    )

    # Second pass: compute in-Newton steps (with updated θ_β_out)
    sum_PG_in = torch.zeros(N, dtype=torch.float64)
    sum_PGG1_in = torch.zeros(N, dtype=torch.float64)

    for j_start in range(0, N, chunk_size):
        j_end = min(j_start + chunk_size, N)
        chunk_len = j_end - j_start

        # p[j, i] for j in chunk, all i
        log_xy = (
            -theta_topo_out[j_start:j_end, None]
            - theta_topo_in[None, :]
        )
        p_chunk = torch.sigmoid(log_xy)  # (chunk, N): p[j, i]

        z_chunk = theta_b_out_new[j_start:j_end, None] + theta_b_in[None, :]
        z_safe = z_chunk.clamp(min=1e-8)
        G_chunk = 1.0 / torch.expm1(z_safe)  # G[j, i]

        local_j = torch.arange(chunk_len, dtype=torch.long)
        global_i = torch.arange(j_start, j_end, dtype=torch.long)
        G_chunk[local_j, global_i] = 0.0

        PG_chunk = p_chunk * G_chunk        # p[j,i] G[j,i]
        PGG1_chunk = p_chunk * G_chunk * (1.0 + G_chunk)

        # s_in[i] = Σ_j p[j,i] G[j,i]; column sums
        sum_PG_in += PG_chunk.sum(dim=0)
        sum_PGG1_in += PGG1_chunk.sum(dim=0)

    F_in = sum_PG_in - s_in
    neg_H_in = sum_PGG1_in.clamp(min=1e-15)
    delta_in = (F_in / neg_H_in).clamp(-max_step, max_step)
    theta_b_in_new = (theta_b_in + delta_in).clamp(-_ETA_MAX, _ETA_MAX)
    theta_b_in_new = torch.where(
        s_in == 0, torch.full_like(theta_b_in_new, _ETA_MAX), theta_b_in_new
    )

    return torch.cat([theta_b_out_new, theta_b_in_new]), F_current


def solve_fixed_point_daecm(
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
) -> SolverResult:
    """Fixed-point iteration for the DaECM weight step.

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
    theta = theta.clamp(_ETA_MIN, _ETA_MAX)

    # Decide chunked vs dense
    if chunk_size == 0:
        effective_chunk = 0 if N <= _LARGE_N_THRESHOLD else _DEFAULT_CHUNK
    else:
        effective_chunk = chunk_size

    # Pre-compute DCM probability matrix (or use caller-supplied one)
    if effective_chunk == 0:  # dense path: pre-compute P once
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

    tracemalloc.start()
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
                if effective_chunk > 0:
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
                beta_out = torch.exp(-theta[:N])
                beta_in = torch.exp(-theta[N:])

                if effective_chunk > 0:
                    beta_out_new, beta_in_new, F_current = _fp_step_chunked(
                        beta_out, beta_in,
                        theta_topo_out_chunked, theta_topo_in_chunked,
                        s_out, s_in, effective_chunk, variant,
                        theta=theta, max_step=max_step,
                    )
                else:
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
                    if improvement < _STAGNATION_RTOL:
                        message = (
                            f"Stagnation: residual improved by only {improvement:.2%} "
                            f"over last {_STAGNATION_WINDOW} iterations "
                            f"(best={best_res_recent:.3e}). Stopping at iter {n_iter}."
                        )
                        break
                best_res_old = best_res_recent
                best_res_recent = float("inf")

            # --- FP-GS periodic Newton correction ---
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
                # Blowup guard: clear history and reset to best_theta when
                # the residual jumps catastrophically above the best seen.
                _blowup_recovered = False
                if (
                    len(_and_g) >= 2
                    and math.isfinite(res_norm)
                    and res_norm > _ANDERSON_BLOWUP_FACTOR * _best_res_for_anderson
                ):
                    _and_g.clear()
                    _and_r.clear()
                    theta = best_theta.clone()
                    _blowup_recovered = True

                # Update the running best AFTER the blowup check
                _best_res_for_anderson = min(_best_res_for_anderson, res_norm)

                if _blowup_recovered:
                    theta_next = theta  # = best_theta
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
                    if effective_chunk > 0:
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
        _, peak_ram = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    return SolverResult(
        theta=best_theta.detach().numpy(),
        converged=converged,
        iterations=n_iter,
        residuals=residuals,
        elapsed_time=elapsed,
        peak_ram_bytes=peak_ram,
        message=message,
    )
