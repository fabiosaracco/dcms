"""Fixed-point iteration solver for the DWCM model.

The DWCM strength equations are

    s_out_i = Σ_{j≠i} β_out_i · β_in_j / (1 − β_out_i · β_in_j)
    s_in_i  = Σ_{j≠i} β_out_j · β_in_i / (1 − β_out_j · β_in_i)

with β = exp(−θ).  Isolating β_out_i gives the fixed-point update

    β_out_i^{new} = s_out_i / D_out_i
    where  D_out_i = Σ_{j≠i} β_in_j / (1 − β_out_i · β_in_j)

and analogously for β_in_i.

Two variants are implemented:

* **Gauss-Seidel** — out-multipliers are updated first; the updated values
  are immediately used when computing new in-multipliers.
* **Jacobi**       — all multipliers are updated simultaneously using values
  from the *previous* iteration.

Both variants support an optional damping factor α ∈ (0, 1].  The update
is carried out in θ-space:

    θ_i^{t+1} = α · (−log β_i^{new}) + (1−α) · θ_i^{t}

θ is clamped to [_ETA_MIN, _ETA_MAX] after each step to maintain the
feasibility constraint β_out_i · β_in_j < 1 for all i, j.

For N > ``_LARGE_N_THRESHOLD`` the N×N intermediate matrix is never
materialised; instead the update denominators are computed in row/column
chunks of size ``_DEFAULT_CHUNK``, using O(chunk × N) RAM per chunk.
"""
from __future__ import annotations

import math
import time
import tracemalloc
from typing import Callable

import torch

from .base import SolverResult
from src.models.dwcm import _LARGE_N_THRESHOLD, _DEFAULT_CHUNK, _ETA_MIN, _ETA_MAX


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
        theta0: Initial parameter vector [θ_out | θ_in], shape (2N,).
                All entries should be strictly positive.
        s_out:  Observed out-strength sequence, shape (N,).
        s_in:   Observed in-strength sequence, shape (N,).
        tol:    Convergence tolerance on the ℓ∞ residual norm.
        max_iter: Maximum number of iterations.
        damping: Damping factor α ∈ (0, 1].  α=1 → no damping.
        variant: ``"jacobi"`` or ``"gauss-seidel"``.
        chunk_size: If > 0, process the N×N products in chunks of this size
            to avoid materialising the full matrix (useful for large N).
            If 0, auto-select: dense for N ≤ ``_LARGE_N_THRESHOLD``, chunked
            (with ``_DEFAULT_CHUNK``) otherwise.

    Returns:
        :class:`~src.solvers.base.SolverResult` instance.
    """
    if variant not in ("jacobi", "gauss-seidel"):
        raise ValueError(f"Unknown variant {variant!r}. Choose 'jacobi' or 'gauss-seidel'.")
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

    # Clamp initial theta to feasible range
    theta = theta.clamp(_ETA_MIN, _ETA_MAX)

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
            # Physical multipliers: β = exp(-θ), β ∈ (0, 1)
            beta_out = torch.exp(-theta[:N])
            beta_in = torch.exp(-theta[N:])

            if effective_chunk > 0:
                beta_out_new, beta_in_new = _fp_step_chunked_dwcm(
                    beta_out, beta_in, s_out, s_in, effective_chunk, variant
                )
            else:
                # Dense path (materialises full N×N matrix)
                # xy[i, j] = β_out_i · β_in_j, shape (N, N)
                xy = beta_out[:, None] * beta_in[None, :]

                # D_out[i] = Σ_{j≠i} β_in_j / (1 - β_out_i * β_in_j)
                denom = (1.0 - xy).clamp(min=1e-15)
                D_out_mat = beta_in[None, :] / denom   # (N, N)
                D_out_mat.fill_diagonal_(0.0)
                D_out = D_out_mat.sum(dim=1)

                beta_out_new = torch.where(D_out > 0, s_out / D_out, beta_out)

                if variant == "gauss-seidel":
                    beta_out_upd = beta_out_new
                else:
                    beta_out_upd = beta_out  # Jacobi: keep old values

                # D_in[i] = Σ_{j≠i} β_out_j / (1 - β_out_j * β_in_i)
                xy2 = beta_out_upd[:, None] * beta_in[None, :]
                denom2 = (1.0 - xy2).clamp(min=1e-15)
                D_in_mat = beta_out_upd[:, None] / denom2  # (N, N)
                D_in_mat.fill_diagonal_(0.0)
                D_in = D_in_mat.sum(dim=0)

                beta_in_new = torch.where(D_in > 0, s_in / D_in, beta_in)

            # Clamp β to (0, 1) strictly to maintain feasibility
            beta_out_new = beta_out_new.clamp(1e-300, 1.0 - 1e-15)
            beta_in_new = beta_in_new.clamp(1e-300, 1.0 - 1e-15)

            # Convert to θ-space and apply damping
            theta_out_new = (-torch.log(beta_out_new)).clamp(_ETA_MIN, _ETA_MAX)
            theta_in_new = (-torch.log(beta_in_new)).clamp(_ETA_MIN, _ETA_MAX)

            theta_new = torch.cat([theta_out_new, theta_in_new])
            theta = damping * theta_new + (1.0 - damping) * theta
            # Re-clamp after damped blend to maintain feasibility
            theta = theta.clamp(_ETA_MIN, _ETA_MAX)

            # Convergence check (ℓ∞ norm of residual)
            res = residual_fn(theta)
            res_norm = res.abs().max().item()

            if not math.isfinite(res_norm):
                message = f"NaN/Inf detected at iteration {n_iter}."
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


def _fp_step_chunked_dwcm(
    beta_out: torch.Tensor,
    beta_in: torch.Tensor,
    s_out: torch.Tensor,
    s_in: torch.Tensor,
    chunk_size: int,
    variant: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One fixed-point update step for DWCM using chunked matrix products.

    Computes the update denominators D_out and D_in without ever building
    the full N×N product matrix, using at most O(chunk_size × N) RAM.

    Args:
        beta_out:   Current out-strength multipliers β_out_i = exp(−θ_out_i),
                    shape (N,), values in (0, 1).
        beta_in:    Current in-strength multipliers β_in_i = exp(−θ_in_i),
                    shape (N,), values in (0, 1).
        s_out:      Observed out-strength sequence, shape (N,).
        s_in:       Observed in-strength sequence, shape (N,).
        chunk_size: Number of rows (or columns) per chunk.
        variant:    ``"jacobi"`` or ``"gauss-seidel"``.

    Returns:
        ``(beta_out_new, beta_in_new)`` updated multipliers.
    """
    N = beta_out.shape[0]

    # -------------------------------------------------------------------
    # D_out[i] = Σ_{j≠i} β_in_j / (1 - β_out_i * β_in_j)
    # Process rows i in chunks.
    # -------------------------------------------------------------------
    D_out = torch.zeros(N, dtype=torch.float64)
    for i_start in range(0, N, chunk_size):
        i_end = min(i_start + chunk_size, N)
        chunk_len = i_end - i_start
        b_out_chunk = beta_out[i_start:i_end]           # (chunk,)
        xy_chunk = b_out_chunk[:, None] * beta_in[None, :]  # (chunk, N)
        denom = (1.0 - xy_chunk).clamp(min=1e-15)
        d_chunk = beta_in[None, :] / denom              # (chunk, N)
        # Zero out diagonal entries (j == i)
        local_i = torch.arange(chunk_len, dtype=torch.long)
        global_j = torch.arange(i_start, i_end, dtype=torch.long)
        d_chunk[local_i, global_j] = 0.0
        D_out[i_start:i_end] = d_chunk.sum(dim=1)

    beta_out_new = torch.where(D_out > 0, s_out / D_out, beta_out)

    # Gauss-Seidel: use updated β_out immediately; Jacobi: keep old values
    beta_out_upd = beta_out_new if variant == "gauss-seidel" else beta_out

    # -------------------------------------------------------------------
    # D_in[i] = Σ_{j≠i} β_out_j / (1 - β_out_j * β_in_i)
    # Process chunks of j (sources) and accumulate column sums into D_in.
    # -------------------------------------------------------------------
    D_in = torch.zeros(N, dtype=torch.float64)
    for j_start in range(0, N, chunk_size):
        j_end = min(j_start + chunk_size, N)
        chunk_len = j_end - j_start
        b_out_chunk = beta_out_upd[j_start:j_end]       # (chunk,)
        # xy_chunk[j_local, i] = β_out_j * β_in_i
        xy_chunk = b_out_chunk[:, None] * beta_in[None, :]  # (chunk, N)
        denom = (1.0 - xy_chunk).clamp(min=1e-15)
        d_chunk = b_out_chunk[:, None] / denom          # (chunk, N)
        # Zero out diagonal entries (j == i, i.e., global j == i)
        local_j = torch.arange(chunk_len, dtype=torch.long)
        global_i = torch.arange(j_start, j_end, dtype=torch.long)
        d_chunk[local_j, global_i] = 0.0
        # Accumulate column sums into D_in[i]
        D_in += d_chunk.sum(dim=0)

    beta_in_new = torch.where(D_in > 0, s_in / D_in, beta_in)
    return beta_out_new, beta_in_new
