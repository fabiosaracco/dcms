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

_STAGNATION_WINDOW: int = 200
_STAGNATION_RTOL: float = 0.01


# -------------------------------------------------------------------------
# Anderson mixing (identical to the version in fixed_point_qdecm.py)
# -------------------------------------------------------------------------

def _anderson_mixing(
    fp_outputs: list[torch.Tensor],
    residuals_hist: list[torch.Tensor],
) -> torch.Tensor:
    """Anderson mixing: compute the next iterate from FP history.

    Uses per-component-weighted least-squares to compute mixing coefficients,
    preventing high-magnitude components from dominating the mix.

    Args:
        fp_outputs:     List of g(θ_k) values, each shape (M,).
        residuals_hist: List of r_k = g(θ_k) − θ_k values, each shape (M,).

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
        W[self_idx] = 0.0                           # exclude diagonal
        return float(W.sum()) - s_target

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
    hub_sk_threshold: float = 0.0,
    backtracking_gamma: float = 0.0,
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
        hub_sk_threshold: When > 0, nodes whose observed strength-to-degree
                        ratio ``s_i / k_hat_i`` exceeds this value are treated
                        as *hub nodes* and solved exactly each iteration via 1D
                        bisection over η rather than the global Newton step.
                        Use this for networks with a few nodes that have very
                        high ``s/k`` ratios (e.g. ``s/k > 5``) which cause the
                        global solver to stagnate.  Default=0.0 (disabled).
        backtracking_gamma (float): When > 0, after each Newton step a
                        backtracking line search is applied: the residual at
                        the proposed ``theta_fp`` is evaluated, and if it
                        exceeds ``backtracking_gamma × res_norm`` (residual at
                        the current iterate), the step is halved up to 5 times
                        until the condition is met.  Typical value: ``2.0``.
                        Default 0 (disabled).  Anderson history is cleared
                        whenever a dampened step is accepted.

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
    if _use_numba and variant == "theta-newton":
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
    if effective_chunk > 0:
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

    _peak_ram_monitor = _PeakRAMMonitor()
    _peak_ram_monitor.__enter__()
    t0 = time.perf_counter()

    # Scale-adaptive Anderson blowup: generous for tiny networks (where wild
    # Anderson excursions can find good iterates), strict for large ones
    # (where residual cascades corrupt history for hundreds of iterations).
    eff_blowup = max(50.0, min(5000.0, 200_000.0 / N))

    n_iter = 0
    residuals: list[float] = []
    converged = False
    message = "Maximum iterations reached without convergence."

    best_theta: torch.Tensor = theta.clone()
    best_theta_res: float = float("inf")
    best_res_recent: float = float("inf")
    best_res_old: float = float("inf")

    _and_g: list[torch.Tensor] = []
    _and_r: list[torch.Tensor] = []
    _best_res_for_anderson: float = float("inf")

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

            # ------------------------------------------------------------------
            # Hub bisection: for nodes with s/k > threshold, find η exactly
            # via 1D bisection (3 GS sweeps for consistency between out/in).
            # ------------------------------------------------------------------
            if _hub_active and (_hub_out_indices or _hub_in_indices):
                _th_fp_np = theta_fp.numpy()
                for _sweep in range(3):
                    # --- Out-hub pass ---
                    for _i in _hub_out_indices:
                        _eta_in_cur = _th_fp_np[3 * N:]       # current η_in
                        _tho_cur = _th_fp_np[N:2 * N]         # current θ_in
                        _eta_out_new = _bisect_hub_eta_decm(
                            theta_self=float(_th_fp_np[_i]),   # θ_out_i
                            theta_other=_tho_cur,
                            eta_other=_eta_in_cur,
                            s_target=float(s_out[_i].item()),
                            self_idx=_i,
                        )
                        _th_fp_np[2 * N + _i] = _eta_out_new
                    # --- In-hub pass ---
                    for _j in _hub_in_indices:
                        _eta_out_cur = _th_fp_np[2 * N:3 * N] # current η_out
                        _tho_cur = _th_fp_np[:N]               # current θ_out
                        _eta_in_new = _bisect_hub_eta_decm(
                            theta_self=float(_th_fp_np[N + _j]),   # θ_in_j
                            theta_other=_tho_cur,
                            eta_other=_eta_out_cur,
                            s_target=float(s_in[_j].item()),
                            self_idx=_j,
                        )
                        _th_fp_np[3 * N + _j] = _eta_in_new
                theta_fp = torch.from_numpy(_th_fp_np)

                # Zero out F_current for hub nodes (their strength equation is
                # exactly satisfied after bisection; keep residual for bookkeeping)
                for _i in _hub_out_indices:
                    F_current[2 * N + _i] = 0.0
                for _j in _hub_in_indices:
                    F_current[3 * N + _j] = 0.0


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
                    f"[{datetime.datetime.now():%H:%M:%S}] "
                    f"iteration={n_iter:5d}, "
                    f"elapsed time={int(_elapsed // 3600):4d}:{int((_elapsed % 3600) // 60):02d}:{int(_elapsed % 60):02d}, "
                    f"MRE_topo={_mre_topo:.2e}, MRE_weights={_mre_weights:.2e}",
                    end="\r" if monitor else "\n",
                )
                sys.stdout.flush()

            if res_norm < best_theta_res:
                best_theta_res = res_norm
                best_theta = theta.clone()

            if res_norm < tol:
                converged = True
                message = f"Converged in {n_iter} iteration(s)."
                break

            # Stagnation detection
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

            # Anderson acceleration
            if anderson_depth > 1:
                # Blowup guard
                _blowup_recovered = False
                if (
                    len(_and_g) >= 2
                    and math.isfinite(res_norm)
                    and res_norm > eff_blowup * _best_res_for_anderson
                ):
                    _and_g.clear()
                    _and_r.clear()
                    _blowup_recovered = True

                _best_res_for_anderson = min(_best_res_for_anderson, res_norm)

                if _blowup_recovered:
                    theta_next = theta_fp
                else:
                    r_k = theta_fp - theta
                    # Hub exclusion: zero hub components in r_k so Anderson
                    # mixing weights focus on non-hub convergence
                    if _hub_active:
                        r_k[_hub_4N_mask] = 0.0
                    r_k_norm = r_k.abs().max().item()
                    if math.isfinite(r_k_norm) and r_k_norm < _ANDERSON_MAX_NORM:
                        _and_g.append(theta_fp.clone())
                        _and_r.append(r_k.clone())

                    if len(_and_g) > anderson_depth:
                        _and_g.pop(0)
                        _and_r.pop(0)

                    if len(_and_g) >= 2:
                        theta_next = _anderson_mixing(_and_g, _and_r)

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

                        # Anderson freeze: restore hub η to bisection values
                        if _hub_active:
                            theta_next[_hub_4N_mask] = theta_fp[_hub_4N_mask]
                    else:
                        theta_next = theta_fp
            else:
                theta_next = theta_fp

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
    )
