"""Fixed-point iteration solver for the DCM binary model.

The DCM connection-probability equations are:

    p_ij = x_i · y_j / (1 + x_i · y_j)   where x_i = exp(−θ_out_i),
                                                 y_j = exp(−θ_in_j)

leading to the degree constraints:

    k_out_i = Σ_{j≠i} p_ij
    k_in_i  = Σ_{j≠i} p_ji

Two variants are implemented:

* **Gauss-Seidel** — x-space fixed-point; out-multipliers are updated first
  using x_i^{new} = k_out_i / D_out_i where
  D_out_i = Σ_{j≠i} y_j / (1 + x_i y_j).  Fresh x values are immediately
  used when computing the in-multiplier update.

* **theta-newton** — θ-space coordinate Newton step.  For each node i:

      Δθ_out_i = (k_out_hat_i − k_out_i) / Σ_{j≠i} p_ij(1−p_ij)

  The per-node step is clipped to ``[−max_step, +max_step]``.

**Anderson acceleration** is available for all variants (depth 0 = plain FP).

Includes all PR#12 robustness fixes:
- Weighted Anderson mixing (prevents hub-node domination)
- θ-floor after Anderson steps (prevents β → 1 blowups)
- Blowup reset (clears Anderson history when residual spikes)
- Newton fallback in FP-GS (spectral radius > 1 near hub nodes)
- FP-GS Newton-Anderson mini-loop (escape stagnation)
- best_theta tracking (returns best θ seen, not blown-up final θ)

Chunked implementation only — never materialises N×N matrix.
"""
from __future__ import annotations

import datetime
import math
import sys
import time
from typing import Callable

import torch

from dcms.models.parameters import DCM_LARGE_N_THRESHOLD as _LARGE_N_THRESHOLD
from dcms.models.parameters import _DEFAULT_CHUNK
from dcms.solvers.base import SolverResult
from dcms.utils.profiling import _PeakRAMMonitor

_ETA_MIN: float = 1e-10
_ETA_MAX: float = 50.0

_ANDERSON_MAX_NORM: float = 1e6
_ANDERSON_BLOWUP_FACTOR: float = 100.0
_ANDERSON_THETA_FLOOR: float = 0.1
_FP_NEWTON_FALLBACK_DELTA: float = 0.1
_FPGS_NEWTON_RESET_WINDOW: int = 30
_FPGS_NEWTON_STEPS: int = 30
_FPGS_NEWTON_AND_DEPTH: int = 5


def _anderson_mixing(
    fp_outputs: list[torch.Tensor],
    residuals_hist: list[torch.Tensor],
) -> torch.Tensor:
    """Anderson mixing with per-component row-weighting (PR#12).

    Solves the *per-component-weighted* constrained least-squares problem:

        min  ‖W⁻¹ Σ_i c_i r_i‖²    s.t.  Σ_i c_i = 1

    where w_j = max_i |r_i[j]| prevents hub nodes from dominating the mixing.

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


def _fp_step_chunked_dcm(
    x: torch.Tensor,
    y: torch.Tensor,
    k_out: torch.Tensor,
    k_in: torch.Tensor,
    chunk_size: int,
    theta: torch.Tensor | None = None,
    max_step: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One chunked Gauss-Seidel fixed-point update step for the DCM degree equations.

    Computes D_out and D_in without materialising the full N×N p matrix.
    Also computes the residual F(θ_current) at the pre-update state.

    When *theta* is provided, a Newton fallback is applied for hub nodes where
    the x-FP step would produce a large θ-step (|Δθ| > threshold).

    Args:
        x:          Current x_i = exp(−θ_out_i), shape (N,).
        y:          Current y_i = exp(−θ_in_i), shape (N,).
        k_out:      Observed out-degree sequence, shape (N,).
        k_in:       Observed in-degree sequence, shape (N,).
        chunk_size: Number of rows (or columns) per chunk.
        theta:      Full parameter vector [θ_out | θ_in], shape (2N,), used
                    by the Newton fallback.  Pass ``None`` to disable.
        max_step:   Maximum |Δθ| per node for the Newton fallback step.

    Returns:
        ``(x_new, y_new, F_current)`` where F_current is the residual at the
        pre-update state, shape (2N,).
    """
    N = x.shape[0]
    D_out = torch.zeros(N, dtype=torch.float64)
    k_in_hat = torch.zeros(N, dtype=torch.float64)  # col sums of p = k_in_hat

    # --- D_out pass: also accumulates k_in_hat (col sums at current x, y) ---
    for i_start in range(0, N, chunk_size):
        i_end = min(i_start + chunk_size, N)
        chunk_len = i_end - i_start

        xy_chunk = x[i_start:i_end, None] * y[None, :]  # (chunk, N)
        p_chunk = xy_chunk / (1.0 + xy_chunk)            # p[i,j] = x_i y_j / (1+x_i y_j)

        # D_out[i] contribution: Σ_{j≠i} y_j / (1 + x_i y_j)
        dy = y[None, :] / (1.0 + xy_chunk)               # (chunk, N)

        # Zero out diagonal (j == i in global coords)
        local_i = torch.arange(chunk_len, dtype=torch.long)
        global_j = torch.arange(i_start, i_end, dtype=torch.long)
        dy[local_i, global_j] = 0.0
        p_chunk[local_i, global_j] = 0.0

        D_out[i_start:i_end] = dy.sum(dim=1)
        k_in_hat += p_chunk.sum(dim=0)  # col sums: k_in_hat[j] += Σ_{i in chunk} p[i,j]

    k_out_hat = x * D_out  # k_out_hat[i] = x_i * D_out_i = Σ_{j≠i} p_ij
    F_current = torch.cat([k_out_hat - k_out, k_in_hat - k_in])

    x_new = torch.where(D_out > 0, k_out / D_out, x)

    # Newton fallback for out-direction
    if theta is not None:
        _theta_out_fp = (
            -torch.log(x_new.clamp(1e-300, 1e300))
        ).clamp(-_ETA_MAX, _ETA_MAX)
        _use_newton_out = (
            (_theta_out_fp - theta[:N]).abs() > _FP_NEWTON_FALLBACK_DELTA
        )
        if _use_newton_out.any():
            _delta_out = (
                (k_out_hat - k_out) / k_out_hat.clamp(min=1e-30)
            ).clamp(-max_step, max_step)
            _theta_out_nt = (theta[:N] + _delta_out).clamp(-_ETA_MAX, _ETA_MAX)
            x_new = torch.where(_use_newton_out, torch.exp(-_theta_out_nt), x_new)

    x_upd = x_new  # Gauss-Seidel: use updated x immediately

    # --- D_in pass: Σ_{j≠i} x_j / (1 + x_j y_i), iterate over chunks of j ---
    D_in = torch.zeros(N, dtype=torch.float64)

    for j_start in range(0, N, chunk_size):
        j_end = min(j_start + chunk_size, N)
        chunk_len = j_end - j_start

        # xy_chunk[j_local, i] = x_upd[j] * y[i]
        xy_chunk = x_upd[j_start:j_end, None] * y[None, :]  # (chunk_len, N)

        # D_in[i] contribution: Σ_{j≠i} x_upd[j] / (1 + x_upd[j] y[i])
        dx = x_upd[j_start:j_end, None] / (1.0 + xy_chunk)  # (chunk_len, N)

        # Zero out diagonal
        local_j = torch.arange(chunk_len, dtype=torch.long)
        global_i = torch.arange(j_start, j_end, dtype=torch.long)
        dx[local_j, global_i] = 0.0

        # Accumulate D_in column-wise: D_in[i] += Σ_{j in chunk, j≠i} x_upd[j]/(1+x_upd[j]y_i)
        D_in += dx.sum(dim=0)

    k_in_hat_upd = y * D_in  # k_in_hat[i] = y_i * D_in[i]
    y_new = torch.where(D_in > 0, k_in / D_in, y)

    # Newton fallback for in-direction
    if theta is not None:
        _theta_in_fp = (
            -torch.log(y_new.clamp(1e-300, 1e300))
        ).clamp(-_ETA_MAX, _ETA_MAX)
        _use_newton_in = (
            (_theta_in_fp - theta[N:]).abs() > _FP_NEWTON_FALLBACK_DELTA
        )
        if _use_newton_in.any():
            _delta_in = (
                (k_in_hat_upd - k_in) / k_in_hat_upd.clamp(min=1e-30)
            ).clamp(-max_step, max_step)
            _theta_in_nt = (theta[N:] + _delta_in).clamp(-_ETA_MAX, _ETA_MAX)
            y_new = torch.where(_use_newton_in, torch.exp(-_theta_in_nt), y_new)

    return x_new, y_new, F_current


def _theta_newton_step_chunked_dcm(
    theta: torch.Tensor,
    k_out: torch.Tensor,
    k_in: torch.Tensor,
    chunk_size: int,
    max_step: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunked θ-space coordinate Newton step for the DCM (Jacobi ordering).

    For each node i:
        Δθ_out_i = (k_out_hat_i − k_out_i) / Σ_{j≠i} p_ij(1−p_ij)
        Δθ_in_i  = (k_in_hat_i  − k_in_i)  / Σ_{j≠i} p_ji(1−p_ji)

    Uses **Jacobi ordering**: all Hessians and residuals are evaluated at the
    *same* θ, and the θ_out and θ_in updates are applied simultaneously.  This
    avoids the Gauss-Seidel feedback loop where the θ_in second pass exactly
    compensates the θ_out update for high-degree hub nodes, preventing
    convergence.  A single pass through the N×N chunk grid accumulates both
    row sums (for θ_out) and column sums (for θ_in), halving the work vs. the
    old two-pass GS formulation.

    Args:
        theta:      Current parameter vector [θ_out | θ_in], shape (2N,).
        k_out:      Observed out-degree sequence, shape (N,).
        k_in:       Observed in-degree sequence, shape (N,).
        chunk_size: Rows per processing chunk.
        max_step:   Maximum allowed |Δθ| per node per step.

    Returns:
        Tuple of (updated parameter vector, residual F(θ_current)).
        The residual is evaluated at the *input* theta (before the update).
        Shape: ((2N,), (2N,)).
    """
    N = k_out.shape[0]
    theta_out = theta[:N]
    theta_in = theta[N:]
    x = torch.exp(-theta_out)
    y = torch.exp(-theta_in)

    # Single pass: accumulate row sums (k_out_hat, H_out) AND col sums (k_in_hat, H_in)
    # using the same x, y — true Jacobi ordering with no GS feedback.
    k_out_hat = torch.zeros(N, dtype=torch.float64)   # row sums of p
    sum_p1p_out = torch.zeros(N, dtype=torch.float64) # row sums of p(1-p)
    k_in_hat = torch.zeros(N, dtype=torch.float64)    # col sums of p
    sum_p1p_in = torch.zeros(N, dtype=torch.float64)  # col sums of p(1-p)

    for i_start in range(0, N, chunk_size):
        i_end = min(i_start + chunk_size, N)
        chunk_len = i_end - i_start

        xy_chunk = x[i_start:i_end, None] * y[None, :]  # (chunk, N)
        p_chunk = xy_chunk / (1.0 + xy_chunk)

        # Zero diagonal
        local_i = torch.arange(chunk_len, dtype=torch.long)
        global_j = torch.arange(i_start, i_end, dtype=torch.long)
        p_chunk[local_i, global_j] = 0.0

        p1p_chunk = p_chunk * (1.0 - p_chunk)

        k_out_hat[i_start:i_end] = p_chunk.sum(dim=1)       # row sums
        sum_p1p_out[i_start:i_end] = p1p_chunk.sum(dim=1)   # row sums of p(1-p)
        k_in_hat += p_chunk.sum(dim=0)                        # col sums
        sum_p1p_in += p1p_chunk.sum(dim=0)                   # col sums of p(1-p)

    F_out = k_out_hat - k_out
    F_in = k_in_hat - k_in
    F_current = torch.cat([F_out, F_in])

    # Simultaneous Jacobi updates
    neg_H_out = sum_p1p_out.clamp(min=1e-15)
    delta_out = (F_out / neg_H_out).clamp(-max_step, max_step)
    theta_out_new = (theta_out + delta_out).clamp(-_ETA_MAX, _ETA_MAX)
    theta_out_new = torch.where(
        k_out == 0, torch.full_like(theta_out_new, _ETA_MAX), theta_out_new
    )

    neg_H_in = sum_p1p_in.clamp(min=1e-15)
    delta_in = (F_in / neg_H_in).clamp(-max_step, max_step)
    theta_in_new = (theta_in + delta_in).clamp(-_ETA_MAX, _ETA_MAX)
    theta_in_new = torch.where(
        k_in == 0, torch.full_like(theta_in_new, _ETA_MAX), theta_in_new
    )

    return torch.cat([theta_out_new, theta_in_new]), F_current


def solve_fixed_point_dcm(
    residual_fn: Callable[[torch.Tensor], torch.Tensor],
    theta0: "ArrayLike",  # type: ignore[name-defined]
    k_out: "ArrayLike",  # type: ignore[name-defined]
    k_in: "ArrayLike",  # type: ignore[name-defined]
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
    monitor: bool = False,
) -> SolverResult:
    """Fixed-point iteration for the DCM binary model.

    Args:
        residual_fn: Function F(θ) → residual tensor, used for convergence check.
        theta0:     Initial parameter vector [θ_out | θ_in], shape (2N,).
                    All entries should be strictly positive.
        k_out:      Observed out-degree sequence, shape (N,).
        k_in:       Observed in-degree sequence, shape (N,).
        tol:        Convergence tolerance on the ℓ∞ residual norm.
        max_iter:   Maximum number of iterations.
        damping:    Damping factor α ∈ (0, 1] for the ``"gauss-seidel"``
                    variant.  α=1 → no damping.
        variant:    One of ``"gauss-seidel"`` or ``"theta-newton"``.
        chunk_size: If > 0, process the N×N products in chunks of this size.
                    If 0, auto-select based on ``_LARGE_N_THRESHOLD``.
        anderson_depth: Anderson acceleration depth.  0 = plain FP.
        max_step:   Maximum per-node Newton step in ``"theta-newton"`` variant.
        max_time:   Wall-clock time limit in seconds.  0 = no limit.
        backend:    Compute backend: ``"auto"`` (default), ``"pytorch"``, or
                    ``"numba"``.  ``"auto"`` uses PyTorch for N ≤ 5 000 and
                    Numba for larger networks.  If the requested backend is
                    unavailable the solver falls back automatically with a
                    warning.
        num_threads: Number of Numba parallel threads.  0 (default) leaves
                    the global Numba thread count unchanged.  Only takes
                    effect when ``backend="numba"`` (or ``"auto"`` at large N).
        verbose:    If ``True``, print a progress line at every iteration
                    showing timestamp, iteration count, elapsed time, and MRE.
        monitor:    If ``True`` (and ``verbose=True``), overwrite the same
                    terminal line at each iteration (``end='\\r'``) so only
                    the latest status is visible.  Useful for long runs where
                    per-iteration scrolling would be noisy.  Default=False.

    Returns:
        :class:`~src.solvers.base.SolverResult` instance.
    """
    if variant not in ("gauss-seidel", "theta-newton"):
        raise ValueError(
            f"Unknown variant {variant!r}. "
            "Choose 'gauss-seidel' or 'theta-newton'."
        )
    if not (0.0 < damping <= 1.0):
        raise ValueError(f"damping must be in (0, 1], got {damping}")
    if chunk_size < 0:
        raise ValueError(f"chunk_size must be ≥ 0 (0 = auto), got {chunk_size}")

    if not isinstance(k_out, torch.Tensor):
        k_out = torch.tensor(k_out, dtype=torch.float64)
    else:
        k_out = k_out.to(dtype=torch.float64)
    if not isinstance(k_in, torch.Tensor):
        k_in = torch.tensor(k_in, dtype=torch.float64)
    else:
        k_in = k_in.to(dtype=torch.float64)
    if not isinstance(theta0, torch.Tensor):
        theta = torch.tensor(theta0, dtype=torch.float64)
    else:
        theta = theta0.clone().to(dtype=torch.float64)

    N = k_out.shape[0]
    theta = theta.clamp(-_ETA_MAX, _ETA_MAX)

    # Resolve compute backend
    from dcms.utils.backend import resolve_backend
    _backend = resolve_backend(backend, N)
    _use_numba = (_backend == "numba")
    _prev_numba_threads: int | None = None
    if _use_numba:
        import numpy as np
        from dcms.solvers._numba_kernels import (
            _dcm_theta_newton_numba,
            _dcm_fp_gs_numba,
        )
        from dcms.utils.backend import resolve_num_threads as _rnt
        _safe_threads = _rnt(num_threads)
        import numba as _numba_mod
        _prev_numba_threads = _numba_mod.get_num_threads()
        _numba_mod.set_num_threads(_safe_threads)

    # Decide whether to use chunked computation (PyTorch path only)
    if chunk_size == 0:
        effective_chunk = _DEFAULT_CHUNK if N > _LARGE_N_THRESHOLD else max(N, 1)
    else:
        effective_chunk = chunk_size

    _peak_ram_monitor = _PeakRAMMonitor()
    _peak_ram_monitor.__enter__()
    t0 = time.perf_counter()

    n_iter = 0
    residuals: list[float] = []
    converged = False
    message = "Maximum iterations reached without convergence."

    _STAGNATION_WINDOW: int = 200
    _STAGNATION_RTOL: float = 0.01
    best_res_recent: float = float("inf")
    best_res_old: float = float("inf")

    best_theta: torch.Tensor = theta.clone()
    best_theta_res: float = float("inf")

    _and_g: list[torch.Tensor] = []
    _and_r: list[torch.Tensor] = []

    _best_res_for_anderson: float = float("inf")

    _fpgs_best_local: float = float("inf")
    _fpgs_stagnation_count: int = 0

    # Precompute verbose targets once (MRE = max |F_i| / k_i)
    _v_targets = torch.cat([k_out, k_in])
    _v_nonzero = _v_targets > 0

    try:
        for _ in range(max_iter):
            if max_time > 0 and (time.perf_counter() - t0) > max_time:
                message = (
                    f"Time limit ({max_time:.0f}s) reached at iteration {n_iter}."
                )
                break

            if variant == "theta-newton":
                if _use_numba:
                    to = theta[:N].numpy()
                    ti = theta[N:].numpy()
                    to_new, ti_new, fo, fi = _dcm_theta_newton_numba(
                        to, ti, k_out.numpy(), k_in.numpy(), max_step, _ETA_MAX,
                    )
                    theta_fp = torch.from_numpy(np.concatenate([to_new, ti_new]))
                    F_current = torch.from_numpy(np.concatenate([fo, fi]))
                else:
                    theta_fp, F_current = _theta_newton_step_chunked_dcm(
                        theta, k_out, k_in, effective_chunk, max_step
                    )
            else:
                if _use_numba:
                    x_np = np.exp(-theta[:N].numpy())
                    y_np = np.exp(-theta[N:].numpy())
                    xn, yn, fo, fi = _dcm_fp_gs_numba(
                        x_np, y_np, k_out.numpy(), k_in.numpy(),
                        max_step, _ETA_MAX, _FP_NEWTON_FALLBACK_DELTA,
                        True,
                    )
                    x_new = torch.from_numpy(xn).clamp(1e-300, 1e300)
                    y_new = torch.from_numpy(yn).clamp(1e-300, 1e300)
                    F_current = torch.from_numpy(np.concatenate([fo, fi]))
                else:
                    x = torch.exp(-theta[:N])
                    y = torch.exp(-theta[N:])
                    x_new, y_new, F_current = _fp_step_chunked_dcm(
                        x, y, k_out, k_in, effective_chunk,
                        theta=None, max_step=max_step,
                    )

                    # Clamp to valid range
                    x_new = x_new.clamp(1e-300, 1e300)
                    y_new = y_new.clamp(1e-300, 1e300)

                # Convert to θ-space and apply damping
                theta_out_new = (-torch.log(x_new)).clamp(-_ETA_MAX, _ETA_MAX)
                theta_in_new = (-torch.log(y_new)).clamp(-_ETA_MAX, _ETA_MAX)
                fp_raw = torch.cat([theta_out_new, theta_in_new])

                theta_fp = (damping * fp_raw + (1.0 - damping) * theta).clamp(
                    -_ETA_MAX, _ETA_MAX
                )

            # Convergence check
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
                    f"iteration={n_iter:5d}, "
                    f"elapsed time={int(_elapsed // 3600):4d}:{int((_elapsed % 3600) // 60):02d}:{int(_elapsed % 60):02d}, "
                    f"MRE_topo={_mre:.2e}",
                    end="\r" if monitor else "\n",
                )
                sys.stdout.flush()

            if res_norm < best_theta_res:
                best_theta_res = res_norm
                best_theta = theta.clone()

            if res_norm < tol:
                converged = True
                message = f"Converged in {n_iter} iteration(s)."+" "*50
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

            # Apply Anderson acceleration or plain update
            if anderson_depth > 1:
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

                _best_res_for_anderson = min(_best_res_for_anderson, res_norm)

                if _blowup_recovered:
                    theta_next = theta
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
                        theta_next = theta_next.clamp(-_ETA_MAX, _ETA_MAX)
                    else:
                        theta_next = theta_fp
            else:
                theta_next = theta_fp

            # FP-GS post-Anderson Newton-Anderson mini-loop
            if _fpgs_newton_fired:
                theta_nt = best_theta.clone()
                _nt_and_g: list[torch.Tensor] = []
                _nt_and_r: list[torch.Tensor] = []
                for _ in range(_FPGS_NEWTON_STEPS):
                    if _use_numba:
                        to = theta_nt[:N].numpy()
                        ti = theta_nt[N:].numpy()
                        to_new, ti_new, fo, fi = _dcm_theta_newton_numba(
                            to, ti, k_out.numpy(), k_in.numpy(), max_step, _ETA_MAX,
                        )
                        theta_nt_fp = torch.from_numpy(np.concatenate([to_new, ti_new]))
                        F_nt = torch.from_numpy(np.concatenate([fo, fi]))
                    else:
                        theta_nt_fp, F_nt = _theta_newton_step_chunked_dcm(
                            theta_nt, k_out, k_in, effective_chunk, max_step
                        )
                    _nt_floor = torch.full_like(theta_nt, -_ETA_MAX)
                    theta_nt_fp = theta_nt_fp.clamp(-_ETA_MAX, _ETA_MAX)
                    nt_res = F_nt.abs().max().item()
                    if nt_res < tol:
                        theta_nt = theta_nt_fp
                        break
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
                        theta_nt_next = theta_nt_next.clamp(-_ETA_MAX, _ETA_MAX)
                    else:
                        theta_nt_next = theta_nt_fp
                    theta_nt = theta_nt_next
                theta_next = theta_nt
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
