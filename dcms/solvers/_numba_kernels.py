"""Numba JIT-compiled parallel kernels for the fixed-point solvers.

These kernels mirror the PyTorch dense/chunked implementations but use
Numba ``@njit(parallel=True)`` loops to avoid materialising N×N matrices.
Memory usage is O(N) and the loops are compiled to native code and run in
parallel across all available CPU threads.

Parallelism strategy
--------------------
Every double sum over (i, j) pairs is split into two independent passes:

* **Row-pass** (parallel over *i*): accumulates per-row quantities
  (out-direction sums). Each thread owns a private accumulator for row ``i``
  and reads shared but *read-only* column vectors.
* **Col-pass** (parallel over *j*): accumulates per-column quantities
  (in-direction sums). Each thread owns a private accumulator for column ``j``
  and reads shared but *read-only* row vectors.

This avoids the race condition that would arise from parallelising the outer
loop of the original fused ``for i / for j`` nest (where `out[j] += f(i,j)`
would be written by multiple threads).

Z-floor optimisation
--------------------
The aDECM and DECM kernels need ``min_{j≠i}(θ_in[j])`` for each node ``i``.
The naïve O(N) inner search per node makes the update step O(N²).  We instead
precompute the global minimum *and* second minimum in O(N), so each node can
look up the answer in O(1).

Thread count
------------
Set the number of Numba threads with ``numba.set_num_threads(n)`` before
calling a kernel.  The default is ``numba.config.NUMBA_NUM_THREADS``
(equal to ``os.cpu_count()``).  Passing ``num_threads=0`` leaves the global
setting unchanged.

All functions accept and return plain NumPy arrays (float64).  The calling
solver is responsible for converting between ``torch.Tensor`` and NumPy.

If Numba is not installed this module raises ``ImportError`` at import time;
the backend dispatcher in :mod:`dcms.utils.backend` ensures it is never
imported when Numba is unavailable.
"""
from __future__ import annotations

import math

import numpy as np
import numba


# ── helpers ────────────────────────────────────────────────────────────────
@numba.njit(cache=True, fastmath=False)
def _expm1(x: float) -> float:  # pragma: no cover – JIT-compiled
    """``math.expm1`` equivalent inside Numba."""
    return math.expm1(x)


@numba.njit(cache=True, fastmath=False)
def _clamp(x: float, lo: float, hi: float) -> float:  # pragma: no cover
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


@numba.njit(cache=True, fastmath=False)
def _min2(arr: np.ndarray) -> tuple:  # pragma: no cover
    """Return ``(idx_min, val_min, val_second_min)`` over *arr* in one pass."""
    N = arr.shape[0]
    idx_min = 0
    val_min = arr[0]
    val_2nd = math.inf
    for k in range(1, N):
        v = arr[k]
        if v < val_min:
            val_2nd = val_min
            val_min = v
            idx_min = k
        elif v < val_2nd:
            val_2nd = v
    return idx_min, val_min, val_2nd


# ═══════════════════════════════════════════════════════════════════════════
# DCM kernels
# ═══════════════════════════════════════════════════════════════════════════

@numba.njit(cache=True, fastmath=False, parallel=True)
def _dcm_theta_newton_numba(
    theta_out: np.ndarray,
    theta_in: np.ndarray,
    k_out: np.ndarray,
    k_in: np.ndarray,
    max_step: float,
    ETA_MAX: float,
) -> tuple:  # pragma: no cover
    """Jacobi θ-Newton step for the DCM (parallel scalar loops).

    Returns ``(theta_out_new, theta_in_new, F_out, F_in)``.
    """
    N = theta_out.shape[0]
    x = np.empty(N)
    y = np.empty(N)
    for i in numba.prange(N):
        x[i] = math.exp(-theta_out[i])
        y[i] = math.exp(-theta_in[i])

    # ── Row-pass (parallel over i): out-direction sums ──
    k_out_hat = np.empty(N)
    sum_p1p_out = np.empty(N)
    for i in numba.prange(N):
        acc_k = 0.0
        acc_h = 0.0
        xi = x[i]
        for j in range(N):
            if i == j:
                continue
            xy = xi * y[j]
            p = xy / (1.0 + xy)
            acc_k += p
            acc_h += p * (1.0 - p)
        k_out_hat[i] = acc_k
        sum_p1p_out[i] = acc_h

    # ── Col-pass (parallel over j): in-direction sums ──
    k_in_hat = np.empty(N)
    sum_p1p_in = np.empty(N)
    for j in numba.prange(N):
        acc_k = 0.0
        acc_h = 0.0
        yj = y[j]
        for i in range(N):
            if i == j:
                continue
            xy = x[i] * yj
            p = xy / (1.0 + xy)
            acc_k += p
            acc_h += p * (1.0 - p)
        k_in_hat[j] = acc_k
        sum_p1p_in[j] = acc_h

    # ── Newton update ──
    F_out = np.empty(N)
    F_in = np.empty(N)
    theta_out_new = np.empty(N)
    theta_in_new = np.empty(N)
    for i in numba.prange(N):
        F_out[i] = k_out_hat[i] - k_out[i]
        H_out = max(sum_p1p_out[i], 1e-15)
        delta_out = _clamp(F_out[i] / H_out, -max_step, max_step)
        theta_out_new[i] = ETA_MAX if k_out[i] == 0.0 else _clamp(
            theta_out[i] + delta_out, -ETA_MAX, ETA_MAX
        )

        F_in[i] = k_in_hat[i] - k_in[i]
        H_in = max(sum_p1p_in[i], 1e-15)
        delta_in = _clamp(F_in[i] / H_in, -max_step, max_step)
        theta_in_new[i] = ETA_MAX if k_in[i] == 0.0 else _clamp(
            theta_in[i] + delta_in, -ETA_MAX, ETA_MAX
        )

    return theta_out_new, theta_in_new, F_out, F_in


@numba.njit(cache=True, fastmath=False, parallel=True)
def _dcm_fp_gs_numba(
    x: np.ndarray,
    y: np.ndarray,
    k_out: np.ndarray,
    k_in: np.ndarray,
    max_step: float,
    ETA_MAX: float,
    FP_NEWTON_FALLBACK_DELTA: float,
    use_newton_fallback: bool,
) -> tuple:  # pragma: no cover
    """Gauss-Seidel FP step for the DCM (parallel scalar loops).

    Returns ``(x_new, y_new, F_out, F_in)``.
    """
    N = x.shape[0]

    # ── Pass 1, row-sum (parallel over i): D_out ──
    D_out = np.empty(N)
    for i in numba.prange(N):
        acc = 0.0
        xi = x[i]
        for j in range(N):
            if i == j:
                continue
            acc += y[j] / (1.0 + xi * y[j])
        D_out[i] = acc

    # ── Pass 1, col-sum (parallel over j): k_in_hat ──
    k_in_hat = np.empty(N)
    for j in numba.prange(N):
        acc = 0.0
        yj = y[j]
        for i in range(N):
            if i == j:
                continue
            xy = x[i] * yj
            acc += xy / (1.0 + xy)
        k_in_hat[j] = acc

    # ── Compute x_new and F_out ──
    k_out_hat = np.empty(N)
    x_new = np.empty(N)
    F_out = np.empty(N)
    for i in numba.prange(N):
        k_out_hat[i] = x[i] * D_out[i]
        x_new[i] = k_out[i] / D_out[i] if D_out[i] > 0 else x[i]
        F_out[i] = k_out_hat[i] - k_out[i]

    # Newton fallback for out-direction (uses x_new[i], private per i)
    if use_newton_fallback:
        for i in numba.prange(N):
            theta_out_i = -math.log(max(x[i], 1e-300))
            theta_fp = _clamp(-math.log(max(x_new[i], 1e-300)), -ETA_MAX, ETA_MAX)
            if abs(theta_fp - theta_out_i) > FP_NEWTON_FALLBACK_DELTA:
                s_hat = max(k_out_hat[i], 1e-30)
                delta = _clamp((k_out_hat[i] - k_out[i]) / s_hat, -max_step, max_step)
                x_new[i] = math.exp(-_clamp(theta_out_i + delta, -ETA_MAX, ETA_MAX))

    # ── Pass 2 (parallel over i): D_in using updated x_new ──
    D_in = np.empty(N)
    for i in numba.prange(N):
        acc = 0.0
        yi = y[i]
        for j in range(N):
            if j == i:
                continue
            acc += x_new[j] / (1.0 + x_new[j] * yi)
        D_in[i] = acc

    # ── Compute y_new and F_in ──
    y_new = np.empty(N)
    k_in_hat_upd = np.empty(N)
    F_in = np.empty(N)
    for i in numba.prange(N):
        k_in_hat_upd[i] = y[i] * D_in[i]
        y_new[i] = k_in[i] / D_in[i] if D_in[i] > 0 else y[i]
        F_in[i] = k_in_hat[i] - k_in[i]

    # Newton fallback for in-direction
    if use_newton_fallback:
        for i in numba.prange(N):
            theta_in_i = -math.log(max(y[i], 1e-300))
            theta_fp = _clamp(-math.log(max(y_new[i], 1e-300)), -ETA_MAX, ETA_MAX)
            if abs(theta_fp - theta_in_i) > FP_NEWTON_FALLBACK_DELTA:
                s_hat = max(k_in_hat_upd[i], 1e-30)
                delta = _clamp((k_in_hat_upd[i] - k_in[i]) / s_hat, -max_step, max_step)
                y_new[i] = math.exp(-_clamp(theta_in_i + delta, -ETA_MAX, ETA_MAX))

    return x_new, y_new, F_out, F_in


# ═══════════════════════════════════════════════════════════════════════════
# DWCM kernels
# ═══════════════════════════════════════════════════════════════════════════

@numba.njit(cache=True, fastmath=False, parallel=True)
def _dwcm_theta_newton_numba(
    theta_out: np.ndarray,
    theta_in: np.ndarray,
    s_out: np.ndarray,
    s_in: np.ndarray,
    max_step: float,
    ETA_MIN: float,
    ETA_MAX: float,
) -> tuple:  # pragma: no cover
    """θ-Newton GS step for the DWCM (parallel scalar loops).

    Returns ``(theta_out_new, theta_in_new, F_out, F_in)``.
    """
    N = theta_out.shape[0]

    # ── Pass 1, row-sum (parallel over i): F_out, h_out ──
    F_out = np.empty(N)
    h_out = np.empty(N)
    for i in numba.prange(N):
        acc_f = 0.0
        acc_h = 0.0
        to_i = theta_out[i]
        for j in range(N):
            if i == j:
                continue
            z = to_i + theta_in[j]
            z_safe = max(z, 1e-15)
            W = 1.0 / _expm1(z_safe)
            acc_f += W
            acc_h -= W * (1.0 + W)
        F_out[i] = acc_f - s_out[i]
        h_out[i] = acc_h

    # ── Pass 1, col-sum (parallel over j): F_in at old θ ──
    F_in_current = np.empty(N)
    for j in numba.prange(N):
        acc = 0.0
        ti_j = theta_in[j]
        for i in range(N):
            if i == j:
                continue
            z = theta_out[i] + ti_j
            z_safe = max(z, 1e-15)
            acc += 1.0 / _expm1(z_safe)
        F_in_current[j] = acc - s_in[j]

    # ── Update θ_out ──
    theta_out_new = np.empty(N)
    for i in numba.prange(N):
        if s_out[i] == 0.0:
            theta_out_new[i] = ETA_MAX
        else:
            denom = h_out[i] - 1e-30
            delta = _clamp(-F_out[i] / denom, -max_step, max_step)
            theta_out_new[i] = _clamp(theta_out[i] + delta, ETA_MIN, ETA_MAX)

    # ── Pass 2, col-sum (parallel over j): F_in, h_in with new θ_out ──
    F_in2 = np.empty(N)
    h_in = np.empty(N)
    for j in numba.prange(N):
        acc_f = 0.0
        acc_h = 0.0
        ti_j = theta_in[j]
        for i in range(N):
            if i == j:
                continue
            z = theta_out_new[i] + ti_j
            z_safe = max(z, 1e-15)
            W = 1.0 / _expm1(z_safe)
            acc_f += W
            acc_h -= W * (1.0 + W)
        F_in2[j] = acc_f - s_in[j]
        h_in[j] = acc_h

    # ── Update θ_in ──
    theta_in_new = np.empty(N)
    for j in numba.prange(N):
        if s_in[j] == 0.0:
            theta_in_new[j] = ETA_MAX
        else:
            denom = h_in[j] - 1e-30
            delta = _clamp(-F_in2[j] / denom, -max_step, max_step)
            theta_in_new[j] = _clamp(theta_in[j] + delta, ETA_MIN, ETA_MAX)

    return theta_out_new, theta_in_new, F_out, F_in_current


@numba.njit(cache=True, fastmath=False, parallel=True)
def _dwcm_fp_gs_numba(
    beta_out: np.ndarray,
    beta_in: np.ndarray,
    s_out: np.ndarray,
    s_in: np.ndarray,
    theta_out: np.ndarray,
    theta_in: np.ndarray,
    max_step: float,
    ETA_MIN: float,
    ETA_MAX: float,
    FP_NEWTON_FALLBACK_DELTA: float,
    use_newton_fallback: bool,
) -> tuple:  # pragma: no cover
    """Gauss-Seidel FP step for the DWCM (parallel scalar loops).

    Returns ``(beta_out_new, beta_in_new, F_out, F_in)``.
    """
    N = beta_out.shape[0]

    # ── Pass 1, row-sum (parallel over i): D_out ──
    D_out = np.empty(N)
    for i in numba.prange(N):
        acc = 0.0
        bo_i = beta_out[i]
        for j in range(N):
            if i == j:
                continue
            xy = bo_i * beta_in[j]
            denom = max(1.0 - xy, 1e-15)
            acc += beta_in[j] / denom
        D_out[i] = acc

    # ── Pass 1, col-sum (parallel over j): s_in_hat at old β ──
    s_in_hat_orig = np.empty(N)
    for j in numba.prange(N):
        acc = 0.0
        bi_j = beta_in[j]
        for i in range(N):
            if i == j:
                continue
            xy = beta_out[i] * bi_j
            denom = max(1.0 - xy, 1e-15)
            acc += beta_out[i] * bi_j / denom
        s_in_hat_orig[j] = acc

    # ── Compute beta_out_new, F_out, F_in ──
    s_out_hat = np.empty(N)
    beta_out_new = np.empty(N)
    F_out = np.empty(N)
    F_in = np.empty(N)
    for i in numba.prange(N):
        s_out_hat[i] = beta_out[i] * D_out[i]
        beta_out_new[i] = s_out[i] / D_out[i] if D_out[i] > 0 else beta_out[i]
        F_out[i] = s_out_hat[i] - s_out[i]
        F_in[i] = s_in_hat_orig[i] - s_in[i]

    # Newton fallback for out-direction
    if use_newton_fallback:
        for i in numba.prange(N):
            b_clamped = max(min(beta_out_new[i], 1.0 - 1e-15), 1e-300)
            theta_fp = _clamp(-math.log(b_clamped), ETA_MIN, ETA_MAX)
            if abs(theta_fp - theta_out[i]) > FP_NEWTON_FALLBACK_DELTA:
                s_hat = max(s_out_hat[i], 1e-30)
                delta = _clamp((s_out_hat[i] - s_out[i]) / s_hat, -max_step, max_step)
                theta_nt = _clamp(theta_out[i] + delta, ETA_MIN, ETA_MAX)
                beta_out_new[i] = math.exp(-theta_nt)

    # Clamp beta_out_new
    for i in numba.prange(N):
        beta_out_new[i] = max(min(beta_out_new[i], 1.0 - 1e-15), 1e-300)

    # ── Pass 2, col-sum (parallel over j): D_in with updated β_out ──
    D_in = np.empty(N)
    for j in numba.prange(N):
        acc = 0.0
        bi_j = beta_in[j]
        for i in range(N):
            if i == j:
                continue
            xy = beta_out_new[i] * bi_j
            denom = max(1.0 - xy, 1e-15)
            acc += beta_out_new[i] / denom
        D_in[j] = acc

    # ── Compute beta_in_new ──
    s_in_hat_upd = np.empty(N)
    beta_in_new = np.empty(N)
    for i in numba.prange(N):
        s_in_hat_upd[i] = beta_in[i] * D_in[i]
        beta_in_new[i] = s_in[i] / D_in[i] if D_in[i] > 0 else beta_in[i]

    # Newton fallback for in-direction
    if use_newton_fallback:
        for i in numba.prange(N):
            b_clamped = max(min(beta_in_new[i], 1.0 - 1e-15), 1e-300)
            theta_fp = _clamp(-math.log(b_clamped), ETA_MIN, ETA_MAX)
            if abs(theta_fp - theta_in[i]) > FP_NEWTON_FALLBACK_DELTA:
                s_hat = max(s_in_hat_upd[i], 1e-30)
                delta = _clamp((s_in_hat_upd[i] - s_in[i]) / s_hat, -max_step, max_step)
                theta_nt = _clamp(theta_in[i] + delta, ETA_MIN, ETA_MAX)
                beta_in_new[i] = math.exp(-theta_nt)

    for i in numba.prange(N):
        beta_in_new[i] = max(min(beta_in_new[i], 1.0 - 1e-15), 1e-300)

    return beta_out_new, beta_in_new, F_out, F_in


# ═══════════════════════════════════════════════════════════════════════════
# aDECM kernels
# ═══════════════════════════════════════════════════════════════════════════

@numba.njit(cache=True, fastmath=False, parallel=True)
def _adecm_theta_newton_numba(
    theta_beta_out: np.ndarray,
    theta_beta_in: np.ndarray,
    theta_topo_out: np.ndarray,
    theta_topo_in: np.ndarray,
    s_out: np.ndarray,
    s_in: np.ndarray,
    max_step: float,
    ETA_MIN: float,
    ETA_MAX: float,
    Z_G_CLAMP: float,
    Z_NEWTON_FLOOR: float,
    Z_NEWTON_FRAC: float,
) -> tuple:  # pragma: no cover
    """θ-Newton GS step for the aDECM weight equations (parallel scalar loops).

    Returns ``(theta_beta_out_new, theta_beta_in_new, F_out, F_in)``.
    """
    N = theta_beta_out.shape[0]

    # Precompute global min/second-min of theta_beta_in for z-floor (O(N) total)
    idx_min_in, val_min_in, val_2nd_in = _min2(theta_beta_in)

    # ── Pass 1, row-sum (parallel over i): F_out, h_out ──
    F_out = np.empty(N)
    h_out = np.empty(N)
    for i in numba.prange(N):
        acc_f = 0.0
        acc_h = 0.0
        tbo_i = theta_beta_out[i]
        tto_i = theta_topo_out[i]
        for j in range(N):
            if i == j:
                continue
            logit = -tto_i - theta_topo_in[j]
            p = 1.0 / (1.0 + math.exp(-logit))
            z = tbo_i + theta_beta_in[j]
            z_safe = max(z, Z_G_CLAMP)
            em1_neg = _expm1(-z_safe)
            G = -1.0 / em1_neg if em1_neg < -1e-300 else 1e15
            pG = p * G
            acc_f += pG
            acc_h -= p * G * (G - 1.0)
        F_out[i] = acc_f - s_out[i]
        h_out[i] = acc_h

    # ── Pass 1, col-sum (parallel over j): F_in at old θ ──
    F_in_current = np.empty(N)
    for j in numba.prange(N):
        acc = 0.0
        tbi_j = theta_beta_in[j]
        tti_j = theta_topo_in[j]
        for i in range(N):
            if i == j:
                continue
            logit = -theta_topo_out[i] - tti_j
            p = 1.0 / (1.0 + math.exp(-logit))
            z = theta_beta_out[i] + tbi_j
            z_safe = max(z, Z_G_CLAMP)
            em1_neg = _expm1(-z_safe)
            G = -1.0 / em1_neg if em1_neg < -1e-300 else 1e15
            acc += p * G
        F_in_current[j] = acc - s_in[j]

    # ── Update θ_β_out (z-floor via precomputed min) ──
    theta_beta_out_new = np.empty(N)
    for i in numba.prange(N):
        if s_out[i] == 0.0:
            theta_beta_out_new[i] = ETA_MAX
        else:
            denom = h_out[i] - 1e-30
            delta = _clamp(-F_out[i] / denom, -max_step, max_step)
            new_val = theta_beta_out[i] + delta
            # z-floor: use precomputed min (O(1) per node)
            min_in = val_2nd_in if i == idx_min_in else val_min_in
            z_floor = max(Z_NEWTON_FLOOR, theta_beta_out[i] * Z_NEWTON_FRAC)
            new_val = max(new_val, z_floor - min_in)
            theta_beta_out_new[i] = _clamp(new_val, -ETA_MAX, ETA_MAX)

    # Precompute global min/second-min of theta_beta_out_new for in-direction z-floor
    idx_min_out_new, val_min_out_new, val_2nd_out_new = _min2(theta_beta_out_new)

    # ── Pass 2, col-sum (parallel over j): F_in, h_in with new θ_β_out ──
    F_in2 = np.empty(N)
    h_in = np.empty(N)
    for j in numba.prange(N):
        acc_f = 0.0
        acc_h = 0.0
        tbi_j = theta_beta_in[j]
        tti_j = theta_topo_in[j]
        for i in range(N):
            if i == j:
                continue
            logit = -theta_topo_out[i] - tti_j
            p = 1.0 / (1.0 + math.exp(-logit))
            z = theta_beta_out_new[i] + tbi_j
            z_safe = max(z, Z_G_CLAMP)
            em1_neg = _expm1(-z_safe)
            G = -1.0 / em1_neg if em1_neg < -1e-300 else 1e15
            pG = p * G
            acc_f += pG
            acc_h -= p * G * (G - 1.0)
        F_in2[j] = acc_f - s_in[j]
        h_in[j] = acc_h

    # ── Update θ_β_in ──
    theta_beta_in_new = np.empty(N)
    for j in numba.prange(N):
        if s_in[j] == 0.0:
            theta_beta_in_new[j] = ETA_MAX
        else:
            denom = h_in[j] - 1e-30
            delta = _clamp(-F_in2[j] / denom, -max_step, max_step)
            new_val = theta_beta_in[j] + delta
            # z-floor using precomputed min of θ_β_out_new
            min_out = val_2nd_out_new if j == idx_min_out_new else val_min_out_new
            z_floor = max(Z_NEWTON_FLOOR, theta_beta_in[j] * Z_NEWTON_FRAC)
            new_val = max(new_val, z_floor - min_out)
            theta_beta_in_new[j] = _clamp(new_val, -ETA_MAX, ETA_MAX)

    return theta_beta_out_new, theta_beta_in_new, F_out, F_in_current


@numba.njit(cache=True, fastmath=False, parallel=True)
def _adecm_fp_gs_numba(
    beta_out: np.ndarray,
    beta_in: np.ndarray,
    theta_topo_out: np.ndarray,
    theta_topo_in: np.ndarray,
    s_out: np.ndarray,
    s_in: np.ndarray,
    theta_beta_out: np.ndarray,
    theta_beta_in: np.ndarray,
    max_step: float,
    ETA_MIN: float,
    ETA_MAX: float,
    Q_MAX: float,
    FP_NEWTON_FALLBACK_DELTA: float,
    use_newton_fallback: bool,
) -> tuple:  # pragma: no cover
    """Gauss-Seidel FP step for the aDECM weight equations (parallel scalar loops).

    Returns ``(beta_out_new, beta_in_new, F_out, F_in)``.
    """
    N = beta_out.shape[0]

    # ── Pass 1, row-sum (parallel over i): s_out_hat ──
    s_out_hat = np.empty(N)
    for i in numba.prange(N):
        acc = 0.0
        bo_i = beta_out[i]
        tto_i = theta_topo_out[i]
        for j in range(N):
            if i == j:
                continue
            logit = -tto_i - theta_topo_in[j]
            p = 1.0 / (1.0 + math.exp(-logit))
            xy = bo_i * beta_in[j]
            denom = max(1.0 - min(xy, Q_MAX), 1e-8)
            acc += p / denom
        s_out_hat[i] = acc

    # ── Pass 1, col-sum (parallel over j): s_in_hat_orig ──
    s_in_hat_orig = np.empty(N)
    for j in numba.prange(N):
        acc = 0.0
        bi_j = beta_in[j]
        tti_j = theta_topo_in[j]
        for i in range(N):
            if i == j:
                continue
            logit = -theta_topo_out[i] - tti_j
            p = 1.0 / (1.0 + math.exp(-logit))
            xy = beta_out[i] * bi_j
            denom = max(1.0 - min(xy, Q_MAX), 1e-8)
            acc += p / denom
        s_in_hat_orig[j] = acc

    # ── Compute beta_out_new, F_out, F_in ──
    beta_out_new = np.empty(N)
    F_out = np.empty(N)
    F_in = np.empty(N)
    for i in numba.prange(N):
        F_out[i] = s_out_hat[i] - s_out[i]
        F_in[i] = s_in_hat_orig[i] - s_in[i]
        beta_out_new[i] = beta_out[i] * s_out[i] / s_out_hat[i] if s_out_hat[i] > 0 else beta_out[i]

    # Newton fallback for out-direction
    if use_newton_fallback:
        for i in numba.prange(N):
            b_clamped = max(min(beta_out_new[i], 1.0 - 1e-15), 1e-300)
            theta_fp = _clamp(-math.log(b_clamped), -ETA_MAX, ETA_MAX)
            if abs(theta_fp - theta_beta_out[i]) > FP_NEWTON_FALLBACK_DELTA:
                s_hat = max(s_out_hat[i], 1e-30)
                delta = _clamp((s_out_hat[i] - s_out[i]) / s_hat, -max_step, max_step)
                theta_nt = _clamp(theta_beta_out[i] + delta, -ETA_MAX, ETA_MAX)
                beta_out_new[i] = math.exp(-theta_nt)

    for i in numba.prange(N):
        beta_out_new[i] = max(min(beta_out_new[i], 1.0 - 1e-15), 1e-300)

    # ── Pass 2, col-sum (parallel over j): s_in_hat with updated β_out ──
    s_in_hat = np.empty(N)
    for j in numba.prange(N):
        acc = 0.0
        bi_j = beta_in[j]
        tti_j = theta_topo_in[j]
        for i in range(N):
            if i == j:
                continue
            logit = -theta_topo_out[i] - tti_j
            p = 1.0 / (1.0 + math.exp(-logit))
            xy = beta_out_new[i] * bi_j
            denom = max(1.0 - min(xy, Q_MAX), 1e-8)
            acc += p / denom
        s_in_hat[j] = acc

    # ── Compute beta_in_new ──
    beta_in_new = np.empty(N)
    for i in numba.prange(N):
        beta_in_new[i] = beta_in[i] * s_in[i] / s_in_hat[i] if s_in_hat[i] > 0 else beta_in[i]

    # Newton fallback for in-direction
    if use_newton_fallback:
        for i in numba.prange(N):
            b_clamped = max(min(beta_in_new[i], 1.0 - 1e-15), 1e-300)
            theta_fp = _clamp(-math.log(b_clamped), -ETA_MAX, ETA_MAX)
            if abs(theta_fp - theta_beta_in[i]) > FP_NEWTON_FALLBACK_DELTA:
                s_hat = max(s_in_hat[i], 1e-30)
                delta = _clamp((s_in_hat[i] - s_in[i]) / s_hat, -max_step, max_step)
                theta_nt = _clamp(theta_beta_in[i] + delta, -ETA_MAX, ETA_MAX)
                beta_in_new[i] = math.exp(-theta_nt)

    for i in numba.prange(N):
        beta_in_new[i] = max(min(beta_in_new[i], 1.0 - 1e-15), 1e-300)

    return beta_out_new, beta_in_new, F_out, F_in


# ═══════════════════════════════════════════════════════════════════════════
# DECM kernels
# ═══════════════════════════════════════════════════════════════════════════

@numba.njit(cache=True, fastmath=False, parallel=True)
def _decm_step_numba(
    theta_out: np.ndarray,
    theta_in: np.ndarray,
    eta_out: np.ndarray,
    eta_in: np.ndarray,
    k_out: np.ndarray,
    k_in: np.ndarray,
    s_out: np.ndarray,
    s_in: np.ndarray,
    zero_k_out: np.ndarray,
    zero_k_in: np.ndarray,
    zero_s_out: np.ndarray,
    zero_s_in: np.ndarray,
    max_step: float,
    THETA_MAX: float,
    ETA_MAX: float,
    Z_G_CLAMP: float,
    Z_NEWTON_FLOOR: float,
    Z_NEWTON_FRAC: float,
) -> tuple:  # pragma: no cover
    """Alternating GS-Newton step for the DECM (parallel scalar loops).

    Returns ``(theta_out_new, theta_in_new, eta_out_new, eta_in_new,
               F_k_out, F_k_in, F_s_out, F_s_in)``.
    """
    N = theta_out.shape[0]

    # Precompute global min/second-min of eta_in for z-floor (O(N))
    idx_min_eta_in, val_min_eta_in, val_2nd_eta_in = _min2(eta_in)

    # ── Pass 1, row-sum (parallel over i): out-direction quantities ──
    k_out_hat = np.empty(N)
    s_out_hat = np.empty(N)
    H_k_out = np.empty(N)
    H_s_out = np.empty(N)
    for i in numba.prange(N):
        acc_k = 0.0
        acc_s = 0.0
        acc_hk = 0.0
        acc_hs = 0.0
        to_i = theta_out[i]
        eo_i = eta_out[i]
        for j in range(N):
            if i == j:
                continue
            eta = eo_i + eta_in[j]
            eta_safe = max(eta, Z_G_CLAMP)
            em1 = _expm1(eta_safe)
            G = 1.0 / em1 if em1 > 0 else 1e15
            log_q = -math.log(em1) if em1 > 1e-300 else 300.0
            logit_p = -to_i - theta_in[j] + log_q
            if logit_p >= 0:
                p = 1.0 / (1.0 + math.exp(-logit_p))
            else:
                ep = math.exp(logit_p)
                p = ep / (1.0 + ep)
            acc_k += p
            pG = p * G
            acc_s += pG
            pq = p * (1.0 - p)
            acc_hk += pq
            acc_hs += p * G * (G - 1.0) + pq * G * G
        k_out_hat[i] = acc_k
        s_out_hat[i] = acc_s
        H_k_out[i] = acc_hk
        H_s_out[i] = acc_hs

    # ── Update θ_out and η_out ──
    theta_out_new = np.empty(N)
    eta_out_new = np.empty(N)
    F_k_out = np.empty(N)
    F_s_out = np.empty(N)
    for i in numba.prange(N):
        F_k_out[i] = k_out_hat[i] - k_out[i]
        F_s_out[i] = s_out_hat[i] - s_out[i]

        hk = max(H_k_out[i], 1e-15)
        delta_theta = _clamp(F_k_out[i] / hk, -max_step, max_step)
        theta_out_new[i] = THETA_MAX if zero_k_out[i] else _clamp(
            theta_out[i] + delta_theta, -THETA_MAX, THETA_MAX
        )

        hs = max(H_s_out[i], 1e-15)
        delta_eta = _clamp(F_s_out[i] / hs, -max_step, max_step)
        if zero_s_out[i]:
            eta_out_new[i] = ETA_MAX
        else:
            new_eta = eta_out[i] + delta_eta
            # z-floor: O(1) per node via precomputed min
            min_in = val_2nd_eta_in if i == idx_min_eta_in else val_min_eta_in
            z_floor = max(Z_NEWTON_FLOOR, eta_out[i] * Z_NEWTON_FRAC)
            new_eta = max(new_eta, z_floor - min_in)
            eta_out_new[i] = _clamp(new_eta, Z_G_CLAMP, ETA_MAX)

    # Precompute min of eta_out_new for in-direction z-floor
    idx_min_eta_out_new, val_min_eta_out_new, val_2nd_eta_out_new = _min2(eta_out_new)

    # ── Pass 2, col-sum (parallel over j): in-direction quantities ──
    k_in_hat2 = np.empty(N)
    s_in_hat2 = np.empty(N)
    H_k_in = np.empty(N)
    H_s_in = np.empty(N)
    for j in numba.prange(N):
        acc_k = 0.0
        acc_s = 0.0
        acc_hk = 0.0
        acc_hs = 0.0
        ti_j = theta_in[j]
        ei_j = eta_in[j]
        for i in range(N):
            if i == j:
                continue
            eta = eta_out_new[i] + ei_j
            eta_safe = max(eta, Z_G_CLAMP)
            em1 = _expm1(eta_safe)
            G = 1.0 / em1 if em1 > 0 else 1e15
            log_q = -math.log(em1) if em1 > 1e-300 else 300.0
            logit_p = -theta_out_new[i] - ti_j + log_q
            if logit_p >= 0:
                p = 1.0 / (1.0 + math.exp(-logit_p))
            else:
                ep = math.exp(logit_p)
                p = ep / (1.0 + ep)
            acc_k += p
            pG = p * G
            acc_s += pG
            pq = p * (1.0 - p)
            acc_hk += pq
            acc_hs += p * G * (G - 1.0) + pq * G * G
        k_in_hat2[j] = acc_k
        s_in_hat2[j] = acc_s
        H_k_in[j] = acc_hk
        H_s_in[j] = acc_hs

    # ── Update θ_in and η_in ──
    theta_in_new = np.empty(N)
    eta_in_new = np.empty(N)
    F_k_in = np.empty(N)
    F_s_in = np.empty(N)
    for j in numba.prange(N):
        F_k_in[j] = k_in_hat2[j] - k_in[j]
        F_s_in[j] = s_in_hat2[j] - s_in[j]

        hk = max(H_k_in[j], 1e-15)
        delta_theta = _clamp(F_k_in[j] / hk, -max_step, max_step)
        theta_in_new[j] = THETA_MAX if zero_k_in[j] else _clamp(
            theta_in[j] + delta_theta, -THETA_MAX, THETA_MAX
        )

        hs = max(H_s_in[j], 1e-15)
        delta_eta = _clamp(F_s_in[j] / hs, -max_step, max_step)
        if zero_s_in[j]:
            eta_in_new[j] = ETA_MAX
        else:
            new_eta = eta_in[j] + delta_eta
            # z-floor: O(1) per node via precomputed min of eta_out_new
            min_out = val_2nd_eta_out_new if j == idx_min_eta_out_new else val_min_eta_out_new
            z_floor = max(Z_NEWTON_FLOOR, eta_in[j] * Z_NEWTON_FRAC)
            new_eta = max(new_eta, z_floor - min_out)
            eta_in_new[j] = _clamp(new_eta, Z_G_CLAMP, ETA_MAX)

    return (theta_out_new, theta_in_new, eta_out_new, eta_in_new,
            F_k_out, F_k_in, F_s_out, F_s_in)

