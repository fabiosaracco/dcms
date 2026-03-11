"""Fixed-point iteration solver for the DWCM (Directed Weighted Configuration Model).

The DWCM fixed-point update isolates each multiplier from its own equation.

For out-multipliers (β_out_i = exp(−θ_out_i)):

    D_out_i = Σ_{j≠i} β_in_j / (1 − β_out_i · β_in_j)
    β_out_i^new = s_out_i / D_out_i

For in-multipliers (β_in_i = exp(−θ_in_i)):

    D_in_i = Σ_{j≠i} β_out_j / (1 − β_out_j · β_in_i)
    β_in_i^new = s_in_i / D_in_i

Note: D_out_i depends on β_out_i itself (appearing in each denominator), so
this is not a pure decoupled fixed point — it is an iterative approximation
where β_out_i in the denominator is held at the *previous* value.  This is
the standard approach used in the literature (e.g., NEMtropy).

Two variants:

* **Gauss-Seidel** — β_out is updated first; the updated values are
  immediately used when computing D_in (and thus β_in^new).
* **Jacobi** — β_out and β_in are both computed from old values, then
  updated simultaneously.

Both variants support a damping factor α ∈ (0, 1]:

    θ^(t+1) = α · θ^new + (1−α) · θ^(t)

For N > ``_LARGE_N_THRESHOLD`` the update denominators are computed in row
chunks to avoid materialising the full N×N matrix.
"""
from __future__ import annotations

import math
import time
import tracemalloc
from typing import Callable

import torch

from .base import SolverResult
from src.models.dwcm import _LARGE_N_THRESHOLD, _DEFAULT_CHUNK

# Clamp θ to avoid exp overflow/underflow
_THETA_CLAMP = 50.0
# Minimum beta value to prevent log(0) in the theta update
_BETA_MIN: float = 1e-300


def solve_fixed_point_dwcm(
    residual_fn: Callable[[torch.Tensor], torch.Tensor],
    theta0: "ArrayLike",  # type: ignore[name-defined]
    s_out: "ArrayLike",  # type: ignore[name-defined]
    s_in: "ArrayLike",  # type: ignore[name-defined]
    tol: float = 1e-8,
    max_iter: int = 10_000,
    damping: float = 1.0,
    variant: str = "gauss-seidel",
    chunk_size: int = 0,
) -> SolverResult:
    """Fixed-point iteration for the DWCM.

    Args:
        residual_fn: Function F(θ) → residual tensor (used for convergence check).
        theta0:  Initial parameter vector [θ_out | θ_in], shape (2N,).
        s_out:   Observed out-strength sequence, shape (N,).
        s_in:    Observed in-strength sequence, shape (N,).
        tol:     Convergence tolerance on the ℓ∞ residual norm.
        max_iter: Maximum number of iterations.
        damping: Damping factor α ∈ (0, 1].  α=1 → no damping.
        variant: ``"jacobi"`` or ``"gauss-seidel"``.
        chunk_size: If > 0, process N×N products in chunks of this size to
            avoid materialising the full matrix.  If 0, auto-select: dense
            for N ≤ ``_LARGE_N_THRESHOLD``, chunked otherwise.

    Returns:
        :class:`~src.solvers.base.SolverResult` instance.
    """
    if variant not in ("jacobi", "gauss-seidel"):
        raise ValueError(
            f"Unknown variant {variant!r}. Choose 'jacobi' or 'gauss-seidel'."
        )
    if not (0.0 < damping <= 1.0):
        raise ValueError(f"damping must be in (0, 1], got {damping}")
    if chunk_size < 0:
        raise ValueError(f"chunk_size must be ≥ 0 (0 = auto), got {chunk_size}")

    if not isinstance(s_out, torch.Tensor):
        s_out = torch.tensor(s_out, dtype=torch.float64)
    else:
        s_out = s_out.to(dtype=torch.float64)
    if not isinstance(s_in, torch.Tensor):
        s_in = torch.tensor(s_in, dtype=torch.float64)
    else:
        s_in = s_in.to(dtype=torch.float64)

    N = s_out.shape[0]

    if not isinstance(theta0, torch.Tensor):
        theta = torch.tensor(theta0, dtype=torch.float64)
    else:
        theta = theta0.clone().to(dtype=torch.float64)

    # Decide whether to use chunked computation
    if chunk_size == 0:
        effective_chunk = 0 if N <= _LARGE_N_THRESHOLD else _DEFAULT_CHUNK
    else:
        effective_chunk = chunk_size

    tracemalloc.start()
    t0 = time.perf_counter()

    n_iter = 0
    residuals: list[float] = []
    converged = False
    message = "Maximum iterations reached without convergence."

    try:
        for _ in range(max_iter):
            beta_out = torch.exp(-theta[:N])
            beta_in = torch.exp(-theta[N:])

            if effective_chunk > 0:
                beta_out_new, beta_in_new = _fp_step_chunked_dwcm(
                    beta_out, beta_in, s_out, s_in, effective_chunk, variant
                )
            else:
                beta_out_new, beta_in_new = _fp_step_dense_dwcm(
                    beta_out, beta_in, s_out, s_in, variant
                )

            # Damped update in θ-space
            theta_out_new = -torch.log(beta_out_new.clamp(min=_BETA_MIN))
            theta_in_new = -torch.log(beta_in_new.clamp(min=_BETA_MIN))
            theta_out_new = theta_out_new.clamp(-_THETA_CLAMP, _THETA_CLAMP)
            theta_in_new = theta_in_new.clamp(-_THETA_CLAMP, _THETA_CLAMP)

            theta_new = torch.cat([theta_out_new, theta_in_new])
            theta = damping * theta_new + (1.0 - damping) * theta

            # Convergence check (ℓ∞ norm of residual)
            res = residual_fn(theta)
            res_norm = res.abs().max().item()

            if not math.isfinite(res_norm):
                message = (
                    f"NaN/Inf detected in residual at iteration {n_iter} "
                    f"(max |β_out|={beta_out_new.abs().max().item():.2e}, "
                    f"max |β_in|={beta_in_new.abs().max().item():.2e})."
                )
                break

            n_iter += 1
            residuals.append(res_norm)

            if res_norm < tol:
                converged = True
                message = f"Converged in {n_iter} iteration(s)."
                break
    finally:
        elapsed = time.perf_counter() - t0
        _, peak_ram = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    return SolverResult(
        theta=theta.detach().numpy(),
        converged=converged,
        iterations=n_iter,
        residuals=residuals,
        elapsed_time=elapsed,
        peak_ram_bytes=peak_ram,
        message=message,
    )


def _fp_step_dense_dwcm(
    beta_out: torch.Tensor,
    beta_in: torch.Tensor,
    s_out: torch.Tensor,
    s_in: torch.Tensor,
    variant: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One DWCM fixed-point update step using dense matrix products.

    Args:
        beta_out: Current out-multipliers β_out_i = exp(−θ_out_i), shape (N,).
        beta_in:  Current in-multipliers β_in_i = exp(−θ_in_i), shape (N,).
        s_out:    Observed out-strength sequence, shape (N,).
        s_in:     Observed in-strength sequence, shape (N,).
        variant:  ``"jacobi"`` or ``"gauss-seidel"``.

    Returns:
        ``(beta_out_new, beta_in_new)`` updated multipliers.
    """
    N = beta_out.shape[0]
    # z[i, j] = beta_out_i * beta_in_j
    z = beta_out[:, None] * beta_in[None, :]  # (N, N)
    # denominator clamp to avoid z >= 1
    denom = (1.0 - z).clamp(min=1e-15)

    # D_out[i] = Σ_{j≠i} beta_in_j / (1 - beta_out_i * beta_in_j)
    contrib = beta_in[None, :] / denom  # (N, N)
    # Zero out diagonal (no self-loops)
    diag_idx = torch.arange(N)
    contrib[diag_idx, diag_idx] = 0.0
    D_out = contrib.sum(dim=1)

    beta_out_new = torch.where(D_out > 0, s_out / D_out, beta_out)

    # Gauss-Seidel: use updated beta_out_new immediately for D_in
    beta_out_use = beta_out_new if variant == "gauss-seidel" else beta_out

    z2 = beta_out_use[:, None] * beta_in[None, :]
    denom2 = (1.0 - z2).clamp(min=1e-15)
    # D_in[i] = Σ_{j≠i} beta_out_j / (1 - beta_out_j * beta_in_i)
    # = col sums of (beta_out_j / (1 - beta_out_j * beta_in_i)) excluding diagonal
    contrib2 = beta_out_use[:, None] / denom2  # (N, N)
    contrib2[diag_idx, diag_idx] = 0.0
    D_in = contrib2.sum(dim=0)  # col sums

    beta_in_new = torch.where(D_in > 0, s_in / D_in, beta_in)
    return beta_out_new, beta_in_new


def _fp_step_chunked_dwcm(
    beta_out: torch.Tensor,
    beta_in: torch.Tensor,
    s_out: torch.Tensor,
    s_in: torch.Tensor,
    chunk_size: int,
    variant: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One DWCM fixed-point update step using chunked matrix products.

    Avoids materialising the full N×N matrix; uses O(chunk_size × N) RAM.

    Args:
        beta_out:   Current out-multipliers, shape (N,).
        beta_in:    Current in-multipliers, shape (N,).
        s_out:      Observed out-strength sequence, shape (N,).
        s_in:       Observed in-strength sequence, shape (N,).
        chunk_size: Number of rows per chunk.
        variant:    ``"jacobi"`` or ``"gauss-seidel"``.

    Returns:
        ``(beta_out_new, beta_in_new)`` updated multipliers.
    """
    N = beta_out.shape[0]
    D_out = torch.zeros(N, dtype=torch.float64)

    for i_start in range(0, N, chunk_size):
        i_end = min(i_start + chunk_size, N)
        chunk_len = i_end - i_start
        bo_chunk = beta_out[i_start:i_end]           # (chunk,)
        z_chunk = bo_chunk[:, None] * beta_in[None, :]  # (chunk, N)
        denom = (1.0 - z_chunk).clamp(min=1e-15)
        contrib = beta_in[None, :] / denom           # (chunk, N)
        # Zero diagonal
        local_i = torch.arange(chunk_len, dtype=torch.long)
        global_j = torch.arange(i_start, i_end, dtype=torch.long)
        contrib[local_i, global_j] = 0.0
        D_out[i_start:i_end] = contrib.sum(dim=1)

    beta_out_new = torch.where(D_out > 0, s_out / D_out, beta_out)

    # Gauss-Seidel: use updated beta_out_new immediately
    beta_out_use = beta_out_new if variant == "gauss-seidel" else beta_out

    D_in = torch.zeros(N, dtype=torch.float64)
    for i_start in range(0, N, chunk_size):
        i_end = min(i_start + chunk_size, N)
        chunk_len = i_end - i_start
        bo_chunk = beta_out_use[i_start:i_end]        # (chunk,)
        z_chunk = bo_chunk[:, None] * beta_in[None, :]  # (chunk, N)
        denom = (1.0 - z_chunk).clamp(min=1e-15)
        contrib = bo_chunk[:, None] / denom           # (chunk, N)
        # Zero diagonal
        local_i = torch.arange(chunk_len, dtype=torch.long)
        global_j = torch.arange(i_start, i_end, dtype=torch.long)
        contrib[local_i, global_j] = 0.0
        D_in += contrib.sum(dim=0)                    # accumulate col sums

    beta_in_new = torch.where(D_in > 0, s_in / D_in, beta_in)
    return beta_out_new, beta_in_new
