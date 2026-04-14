"""Numba JIT-compiled scalar kernels for the fixed-point solvers.

These kernels mirror the PyTorch dense/chunked implementations but use
Numba ``@njit`` loops to avoid materialising N×N matrices entirely.
Memory usage is O(N) and the loops are compiled to native code, making
them competitive with (and often faster than) chunked PyTorch for large N.

All functions accept and return plain NumPy arrays (float64).  The calling
solver is responsible for converting between ``torch.Tensor`` and NumPy.

If Numba is not installed this module raises ``ImportError`` at import time;
the backend dispatcher in :mod:`dcms.utils.backend` ensures it is never
imported when Numba is unavailable.
"""
from __future__ import annotations

import math

import numpy as np
import numba  # noqa: F401 — guaranteed available by the backend dispatcher


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


# ═══════════════════════════════════════════════════════════════════════════
# DCM kernels
# ═══════════════════════════════════════════════════════════════════════════

@numba.njit(cache=True, fastmath=False)
def _dcm_theta_newton_numba(
    theta_out: np.ndarray,
    theta_in: np.ndarray,
    k_out: np.ndarray,
    k_in: np.ndarray,
    max_step: float,
    ETA_MAX: float,
) -> tuple:  # pragma: no cover
    """Jacobi θ-Newton step for the DCM (scalar loops).

    Returns ``(theta_out_new, theta_in_new, F_out, F_in)``.
    """
    N = theta_out.shape[0]
    x = np.empty(N)
    y = np.empty(N)
    for i in range(N):
        x[i] = math.exp(-theta_out[i])
        y[i] = math.exp(-theta_in[i])

    k_out_hat = np.zeros(N)
    sum_p1p_out = np.zeros(N)
    k_in_hat = np.zeros(N)
    sum_p1p_in = np.zeros(N)

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            xy = x[i] * y[j]
            p = xy / (1.0 + xy)
            p1p = p * (1.0 - p)
            k_out_hat[i] += p
            sum_p1p_out[i] += p1p
            k_in_hat[j] += p
            sum_p1p_in[j] += p1p

    F_out = np.empty(N)
    F_in = np.empty(N)
    theta_out_new = np.empty(N)
    theta_in_new = np.empty(N)

    for i in range(N):
        F_out[i] = k_out_hat[i] - k_out[i]
        F_in[i] = k_in_hat[i] - k_in[i]

        H_out = max(sum_p1p_out[i], 1e-15)
        delta_out = _clamp(F_out[i] / H_out, -max_step, max_step)
        if k_out[i] == 0.0:
            theta_out_new[i] = ETA_MAX
        else:
            theta_out_new[i] = _clamp(theta_out[i] + delta_out, -ETA_MAX, ETA_MAX)

        H_in = max(sum_p1p_in[i], 1e-15)
        delta_in = _clamp(F_in[i] / H_in, -max_step, max_step)
        if k_in[i] == 0.0:
            theta_in_new[i] = ETA_MAX
        else:
            theta_in_new[i] = _clamp(theta_in[i] + delta_in, -ETA_MAX, ETA_MAX)

    return theta_out_new, theta_in_new, F_out, F_in


@numba.njit(cache=True, fastmath=False)
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
    """Gauss-Seidel FP step for the DCM (scalar loops).

    Returns ``(x_new, y_new, F_out, F_in)``.
    """
    N = x.shape[0]
    D_out = np.zeros(N)
    k_in_hat = np.zeros(N)

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            xy = x[i] * y[j]
            denom = 1.0 + xy
            D_out[i] += y[j] / denom
            p = xy / denom
            k_in_hat[j] += p

    k_out_hat = np.empty(N)
    x_new = np.empty(N)
    for i in range(N):
        k_out_hat[i] = x[i] * D_out[i]

    F_out = np.empty(N)
    for i in range(N):
        F_out[i] = k_out_hat[i] - k_out[i]

    for i in range(N):
        if D_out[i] > 0:
            x_new[i] = k_out[i] / D_out[i]
        else:
            x_new[i] = x[i]

    # Newton fallback for out-direction
    if use_newton_fallback:
        theta_out = np.empty(N)
        for i in range(N):
            theta_out[i] = -math.log(max(x[i], 1e-300))

        for i in range(N):
            theta_out_fp = _clamp(-math.log(max(x_new[i], 1e-300)), -ETA_MAX, ETA_MAX)
            if abs(theta_out_fp - theta_out[i]) > FP_NEWTON_FALLBACK_DELTA:
                s_hat = max(k_out_hat[i], 1e-30)
                delta = _clamp((k_out_hat[i] - k_out[i]) / s_hat, -max_step, max_step)
                theta_nt = _clamp(theta_out[i] + delta, -ETA_MAX, ETA_MAX)
                x_new[i] = math.exp(-theta_nt)

    # Gauss-Seidel: use updated x immediately for D_in
    D_in = np.zeros(N)
    for j in range(N):
        for i in range(N):
            if j == i:
                continue
            D_in[i] += x_new[j] / (1.0 + x_new[j] * y[i])

    y_new = np.empty(N)
    k_in_hat_upd = np.empty(N)
    for i in range(N):
        k_in_hat_upd[i] = y[i] * D_in[i]
        if D_in[i] > 0:
            y_new[i] = k_in[i] / D_in[i]
        else:
            y_new[i] = y[i]

    # Newton fallback for in-direction
    if use_newton_fallback:
        theta_in = np.empty(N)
        for i in range(N):
            theta_in[i] = -math.log(max(y[i], 1e-300))

        for i in range(N):
            theta_in_fp = _clamp(-math.log(max(y_new[i], 1e-300)), -ETA_MAX, ETA_MAX)
            if abs(theta_in_fp - theta_in[i]) > FP_NEWTON_FALLBACK_DELTA:
                s_hat = max(k_in_hat_upd[i], 1e-30)
                delta = _clamp((k_in_hat_upd[i] - k_in[i]) / s_hat, -max_step, max_step)
                theta_nt = _clamp(theta_in[i] + delta, -ETA_MAX, ETA_MAX)
                y_new[i] = math.exp(-theta_nt)

    F_in = np.empty(N)
    for i in range(N):
        F_in[i] = k_in_hat[i] - k_in[i]

    return x_new, y_new, F_out, F_in


# ═══════════════════════════════════════════════════════════════════════════
# DWCM kernels
# ═══════════════════════════════════════════════════════════════════════════

@numba.njit(cache=True, fastmath=False)
def _dwcm_theta_newton_numba(
    theta_out: np.ndarray,
    theta_in: np.ndarray,
    s_out: np.ndarray,
    s_in: np.ndarray,
    max_step: float,
    ETA_MIN: float,
    ETA_MAX: float,
) -> tuple:  # pragma: no cover
    """θ-Newton GS step for the DWCM (scalar loops).

    Returns ``(theta_out_new, theta_in_new, F_out, F_in)``.
    """
    N = theta_out.shape[0]

    # ── Pass 1: out-direction at current θ ──
    F_out = np.zeros(N)
    h_out = np.zeros(N)
    F_in_current = np.zeros(N)

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            z = theta_out[i] + theta_in[j]
            z_safe = max(z, 1e-15)
            em1 = _expm1(z_safe)
            W = 1.0 / em1
            F_out[i] += W
            h_out[i] -= W * (1.0 + W)
            F_in_current[j] += W

    theta_out_new = np.empty(N)
    for i in range(N):
        F_out[i] -= s_out[i]
        if s_out[i] == 0.0:
            theta_out_new[i] = ETA_MAX
        else:
            denom = h_out[i] - 1e-30
            delta = _clamp(-F_out[i] / denom, -max_step, max_step)
            theta_out_new[i] = _clamp(theta_out[i] + delta, ETA_MIN, ETA_MAX)

    # ── Pass 2: in-direction with updated θ_out ──
    F_in2 = np.zeros(N)
    h_in = np.zeros(N)

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            z = theta_out_new[i] + theta_in[j]
            z_safe = max(z, 1e-15)
            em1 = _expm1(z_safe)
            W = 1.0 / em1
            F_in2[j] += W
            h_in[j] -= W * (1.0 + W)

    theta_in_new = np.empty(N)
    for j in range(N):
        F_in2[j] -= s_in[j]
        F_in_current[j] -= s_in[j]
        if s_in[j] == 0.0:
            theta_in_new[j] = ETA_MAX
        else:
            denom = h_in[j] - 1e-30
            delta = _clamp(-F_in2[j] / denom, -max_step, max_step)
            theta_in_new[j] = _clamp(theta_in[j] + delta, ETA_MIN, ETA_MAX)

    return theta_out_new, theta_in_new, F_out, F_in_current


@numba.njit(cache=True, fastmath=False)
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
    """Gauss-Seidel FP step for the DWCM (scalar loops).

    Returns ``(beta_out_new, beta_in_new, F_out, F_in)``.
    """
    N = beta_out.shape[0]
    D_out = np.zeros(N)
    s_in_hat = np.zeros(N)

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            xy = beta_out[i] * beta_in[j]
            denom = max(1.0 - xy, 1e-15)
            d = beta_in[j] / denom
            D_out[i] += d
            s_in_hat[j] += beta_out[i] * d

    s_out_hat = np.empty(N)
    beta_out_new = np.empty(N)
    for i in range(N):
        s_out_hat[i] = beta_out[i] * D_out[i]
        if D_out[i] > 0:
            beta_out_new[i] = s_out[i] / D_out[i]
        else:
            beta_out_new[i] = beta_out[i]

    F_out = np.empty(N)
    F_in = np.empty(N)
    for i in range(N):
        F_out[i] = s_out_hat[i] - s_out[i]
        F_in[i] = s_in_hat[i] - s_in[i]

    # Newton fallback for out-direction
    if use_newton_fallback:
        for i in range(N):
            b_clamped = max(min(beta_out_new[i], 1.0 - 1e-15), 1e-300)
            theta_fp = _clamp(-math.log(b_clamped), ETA_MIN, ETA_MAX)
            if abs(theta_fp - theta_out[i]) > FP_NEWTON_FALLBACK_DELTA:
                s_hat = max(s_out_hat[i], 1e-30)
                delta = _clamp((s_out_hat[i] - s_out[i]) / s_hat, -max_step, max_step)
                theta_nt = _clamp(theta_out[i] + delta, ETA_MIN, ETA_MAX)
                beta_out_new[i] = math.exp(-theta_nt)

    # Clamp beta_out_new
    for i in range(N):
        beta_out_new[i] = max(min(beta_out_new[i], 1.0 - 1e-15), 1e-300)

    # Gauss-Seidel: use updated β_out for D_in
    D_in = np.zeros(N)
    for j in range(N):
        for i in range(N):
            if j == i:
                continue
            xy = beta_out_new[j] * beta_in[i]
            denom = max(1.0 - xy, 1e-15)
            D_in[i] += beta_out_new[j] / denom

    beta_in_new = np.empty(N)
    for i in range(N):
        if D_in[i] > 0:
            beta_in_new[i] = s_in[i] / D_in[i]
        else:
            beta_in_new[i] = beta_in[i]

    # Newton fallback for in-direction
    if use_newton_fallback:
        for i in range(N):
            s_in_hat_cur = beta_in[i] * D_in[i]
            b_clamped = max(min(beta_in_new[i], 1.0 - 1e-15), 1e-300)
            theta_fp = _clamp(-math.log(b_clamped), ETA_MIN, ETA_MAX)
            if abs(theta_fp - theta_in[i]) > FP_NEWTON_FALLBACK_DELTA:
                s_hat = max(s_in_hat_cur, 1e-30)
                delta = _clamp((s_in_hat_cur - s_in[i]) / s_hat, -max_step, max_step)
                theta_nt = _clamp(theta_in[i] + delta, ETA_MIN, ETA_MAX)
                beta_in_new[i] = math.exp(-theta_nt)

    for i in range(N):
        beta_in_new[i] = max(min(beta_in_new[i], 1.0 - 1e-15), 1e-300)

    return beta_out_new, beta_in_new, F_out, F_in


# ═══════════════════════════════════════════════════════════════════════════
# aDECM kernels
# ═══════════════════════════════════════════════════════════════════════════

@numba.njit(cache=True, fastmath=False)
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
    """θ-Newton GS step for the aDECM weight equations (scalar loops).

    Returns ``(theta_beta_out_new, theta_beta_in_new, F_out, F_in)``.
    """
    N = theta_beta_out.shape[0]

    # ── Pass 1: out-direction ──
    F_out = np.zeros(N)
    h_out = np.zeros(N)
    F_in_current = np.zeros(N)

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            # p_ij = sigmoid(-θ_topo_out_i - θ_topo_in_j)
            logit = -theta_topo_out[i] - theta_topo_in[j]
            p = 1.0 / (1.0 + math.exp(-logit))

            z = theta_beta_out[i] + theta_beta_in[j]
            z_safe = max(z, Z_G_CLAMP)
            em1 = _expm1(z_safe)
            G = 1.0 / em1

            pG = p * G
            F_out[i] += pG
            h_out[i] -= p * G * (1.0 + G)
            F_in_current[j] += pG

    theta_beta_out_new = np.empty(N)
    for i in range(N):
        F_out[i] -= s_out[i]
        if s_out[i] == 0.0:
            theta_beta_out_new[i] = ETA_MAX
        else:
            denom = h_out[i] - 1e-30
            delta = _clamp(-F_out[i] / denom, -max_step, max_step)
            new_val = theta_beta_out[i] + delta
            # z-floor line search
            min_in = ETA_MAX
            for j in range(N):
                if j != i and theta_beta_in[j] < min_in:
                    min_in = theta_beta_in[j]
            z_floor = max(Z_NEWTON_FLOOR, theta_beta_out[i] * Z_NEWTON_FRAC)
            min_allowed = z_floor - min_in
            new_val = max(new_val, min_allowed)
            theta_beta_out_new[i] = _clamp(new_val, ETA_MIN, ETA_MAX)

    # ── Pass 2: in-direction with updated θ_β_out ──
    F_in2 = np.zeros(N)
    h_in = np.zeros(N)

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            logit = -theta_topo_out[i] - theta_topo_in[j]
            p = 1.0 / (1.0 + math.exp(-logit))

            z = theta_beta_out_new[i] + theta_beta_in[j]
            z_safe = max(z, Z_G_CLAMP)
            em1 = _expm1(z_safe)
            G = 1.0 / em1

            pG = p * G
            F_in2[j] += pG
            h_in[j] -= p * G * (1.0 + G)

    theta_beta_in_new = np.empty(N)
    for j in range(N):
        F_in2[j] -= s_in[j]
        F_in_current[j] -= s_in[j]
        if s_in[j] == 0.0:
            theta_beta_in_new[j] = ETA_MAX
        else:
            denom = h_in[j] - 1e-30
            delta = _clamp(-F_in2[j] / denom, -max_step, max_step)
            new_val = theta_beta_in[j] + delta
            min_out = ETA_MAX
            for i in range(N):
                if i != j and theta_beta_out_new[i] < min_out:
                    min_out = theta_beta_out_new[i]
            z_floor = max(Z_NEWTON_FLOOR, theta_beta_in[j] * Z_NEWTON_FRAC)
            min_allowed = z_floor - min_out
            new_val = max(new_val, min_allowed)
            theta_beta_in_new[j] = _clamp(new_val, ETA_MIN, ETA_MAX)

    return theta_beta_out_new, theta_beta_in_new, F_out, F_in_current


@numba.njit(cache=True, fastmath=False)
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
    """Gauss-Seidel FP step for the aDECM weight equations (scalar loops).

    Returns ``(beta_out_new, beta_in_new, F_out, F_in)``.
    """
    N = beta_out.shape[0]
    s_out_hat = np.zeros(N)
    s_in_hat_orig = np.zeros(N)

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            logit = -theta_topo_out[i] - theta_topo_in[j]
            p = 1.0 / (1.0 + math.exp(-logit))
            xy = beta_out[i] * beta_in[j]
            denom = max(1.0 - min(xy, Q_MAX), 1e-8)
            val = p / denom
            s_out_hat[i] += val
            s_in_hat_orig[j] += val

    F_out = np.empty(N)
    F_in = np.empty(N)
    beta_out_new = np.empty(N)
    for i in range(N):
        F_out[i] = s_out_hat[i] - s_out[i]
        F_in[i] = s_in_hat_orig[i] - s_in[i]
        if s_out_hat[i] > 0:
            beta_out_new[i] = beta_out[i] * s_out[i] / s_out_hat[i]
        else:
            beta_out_new[i] = beta_out[i]

    # Newton fallback for out-direction
    if use_newton_fallback:
        for i in range(N):
            b_clamped = max(min(beta_out_new[i], 1.0 - 1e-15), 1e-300)
            theta_fp = _clamp(-math.log(b_clamped), -ETA_MAX, ETA_MAX)
            if abs(theta_fp - theta_beta_out[i]) > FP_NEWTON_FALLBACK_DELTA:
                s_hat = max(s_out_hat[i], 1e-30)
                delta = _clamp((s_out_hat[i] - s_out[i]) / s_hat, -max_step, max_step)
                theta_nt = _clamp(theta_beta_out[i] + delta, -ETA_MAX, ETA_MAX)
                beta_out_new[i] = math.exp(-theta_nt)

    for i in range(N):
        beta_out_new[i] = max(min(beta_out_new[i], 1.0 - 1e-15), 1e-300)

    # GS: use updated β_out for D_in
    s_in_hat = np.zeros(N)
    for j in range(N):
        for i in range(N):
            if j == i:
                continue
            logit = -theta_topo_out[j] - theta_topo_in[i]
            p = 1.0 / (1.0 + math.exp(-logit))
            xy = beta_out_new[j] * beta_in[i]
            denom = max(1.0 - min(xy, Q_MAX), 1e-8)
            s_in_hat[i] += p / denom

    beta_in_new = np.empty(N)
    for i in range(N):
        if s_in_hat[i] > 0:
            beta_in_new[i] = beta_in[i] * s_in[i] / s_in_hat[i]
        else:
            beta_in_new[i] = beta_in[i]

    if use_newton_fallback:
        for i in range(N):
            b_clamped = max(min(beta_in_new[i], 1.0 - 1e-15), 1e-300)
            theta_fp = _clamp(-math.log(b_clamped), -ETA_MAX, ETA_MAX)
            if abs(theta_fp - theta_beta_in[i]) > FP_NEWTON_FALLBACK_DELTA:
                s_hat = max(s_in_hat[i], 1e-30)
                delta = _clamp((s_in_hat[i] - s_in[i]) / s_hat, -max_step, max_step)
                theta_nt = _clamp(theta_beta_in[i] + delta, -ETA_MAX, ETA_MAX)
                beta_in_new[i] = math.exp(-theta_nt)

    for i in range(N):
        beta_in_new[i] = max(min(beta_in_new[i], 1.0 - 1e-15), 1e-300)

    return beta_out_new, beta_in_new, F_out, F_in


# ═══════════════════════════════════════════════════════════════════════════
# DECM kernels
# ═══════════════════════════════════════════════════════════════════════════

@numba.njit(cache=True, fastmath=False)
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
    """Alternating GS-Newton step for the DECM (scalar loops).

    Returns ``(theta_out_new, theta_in_new, eta_out_new, eta_in_new,
               F_k_out, F_k_in, F_s_out, F_s_in)``.
    """
    N = theta_out.shape[0]

    # ── Pass 1: compute sums at current state ──
    k_out_hat = np.zeros(N)
    k_in_hat = np.zeros(N)
    s_out_hat = np.zeros(N)
    s_in_hat = np.zeros(N)

    H_k_out = np.zeros(N)
    H_s_out = np.zeros(N)

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            eta = eta_out[i] + eta_in[j]
            eta_safe = max(eta, Z_G_CLAMP)
            em1 = _expm1(eta_safe)
            G = 1.0 / em1 if em1 > 0 else 1e15
            log_q = -math.log(em1) if em1 > 1e-300 else 300.0
            logit_p = -theta_out[i] - theta_in[j] + log_q
            # sigmoid
            if logit_p >= 0:
                p = 1.0 / (1.0 + math.exp(-logit_p))
            else:
                ep = math.exp(logit_p)
                p = ep / (1.0 + ep)

            k_out_hat[i] += p
            k_in_hat[j] += p
            pG = p * G
            s_out_hat[i] += pG
            s_in_hat[j] += pG

            pq = p * (1.0 - p)
            H_k_out[i] += pq
            pGG1 = p * G * (G - 1.0)
            corr = pq * G * G
            H_s_out[i] += pGG1 + corr

    # ── Update θ_out and η_out ──
    theta_out_new = np.empty(N)
    eta_out_new = np.empty(N)
    F_k_out = np.empty(N)
    F_s_out = np.empty(N)

    for i in range(N):
        F_k_out[i] = k_out_hat[i] - k_out[i]
        F_s_out[i] = s_out_hat[i] - s_out[i]

        hk = max(H_k_out[i], 1e-15)
        delta_theta = _clamp(F_k_out[i] / hk, -max_step, max_step)
        if zero_k_out[i]:
            theta_out_new[i] = THETA_MAX
        else:
            theta_out_new[i] = _clamp(theta_out[i] + delta_theta, -THETA_MAX, THETA_MAX)

        hs = max(H_s_out[i], 1e-15)
        delta_eta = _clamp(F_s_out[i] / hs, -max_step, max_step)
        if zero_s_out[i]:
            eta_out_new[i] = ETA_MAX
        else:
            new_eta = eta_out[i] + delta_eta
            # z-floor
            min_in = ETA_MAX
            for j in range(N):
                if j != i and eta_in[j] < min_in:
                    min_in = eta_in[j]
            z_floor = max(Z_NEWTON_FLOOR, eta_out[i] * Z_NEWTON_FRAC)
            min_allowed = z_floor - min_in
            new_eta = max(new_eta, min_allowed)
            eta_out_new[i] = _clamp(new_eta, Z_G_CLAMP, ETA_MAX)

    # ── Pass 2: recompute col sums with updated θ_out, η_out ──
    k_in_hat2 = np.zeros(N)
    s_in_hat2 = np.zeros(N)
    H_k_in = np.zeros(N)
    H_s_in = np.zeros(N)

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            eta = eta_out_new[i] + eta_in[j]
            eta_safe = max(eta, Z_G_CLAMP)
            em1 = _expm1(eta_safe)
            G = 1.0 / em1 if em1 > 0 else 1e15
            log_q = -math.log(em1) if em1 > 1e-300 else 300.0
            logit_p = -theta_out_new[i] - theta_in[j] + log_q
            if logit_p >= 0:
                p = 1.0 / (1.0 + math.exp(-logit_p))
            else:
                ep = math.exp(logit_p)
                p = ep / (1.0 + ep)

            k_in_hat2[j] += p
            s_in_hat2[j] += p * G

            pq = p * (1.0 - p)
            H_k_in[j] += pq
            H_s_in[j] += p * G * (G - 1.0) + pq * G * G

    # ── Update θ_in and η_in ──
    theta_in_new = np.empty(N)
    eta_in_new = np.empty(N)
    F_k_in = np.empty(N)
    F_s_in = np.empty(N)

    for i in range(N):
        F_k_in[i] = k_in_hat[i] - k_in[i]
        F_s_in[i] = s_in_hat[i] - s_in[i]

    for j in range(N):
        hk = max(H_k_in[j], 1e-15)
        delta_theta = _clamp((k_in_hat2[j] - k_in[j]) / hk, -max_step, max_step)
        if zero_k_in[j]:
            theta_in_new[j] = THETA_MAX
        else:
            theta_in_new[j] = _clamp(theta_in[j] + delta_theta, -THETA_MAX, THETA_MAX)

        hs = max(H_s_in[j], 1e-15)
        delta_eta = _clamp((s_in_hat2[j] - s_in[j]) / hs, -max_step, max_step)
        if zero_s_in[j]:
            eta_in_new[j] = ETA_MAX
        else:
            new_eta = eta_in[j] + delta_eta
            min_out = ETA_MAX
            for i in range(N):
                if i != j and eta_out_new[i] < min_out:
                    min_out = eta_out_new[i]
            z_floor = max(Z_NEWTON_FLOOR, eta_in[j] * Z_NEWTON_FRAC)
            min_allowed = z_floor - min_out
            new_eta = max(new_eta, min_allowed)
            eta_in_new[j] = _clamp(new_eta, Z_G_CLAMP, ETA_MAX)

    return (theta_out_new, theta_in_new, eta_out_new, eta_in_new,
            F_k_out, F_k_in, F_s_out, F_s_in)
