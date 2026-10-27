"""Fixed-point iteration solver for the DCM binary model.

The DCM connection-probability equations are:

    p_ij = x_i · y_j / (1 + x_i · y_j)   where x_i = exp(−θ_out_i),
                                                 y_j = exp(−θ_in_j)

leading to the degree constraints:

    k_out_i = Σ_{j≠i} p_ij
    k_in_i  = Σ_{j≠i} p_ji

Nodes sharing the same (k_out, k_in) pair have identical parameters at the solution (see
:func:`compute_degeneracy_groups_dcm`), so the solver works on the M-unknown reduced system
(M = number of distinct pairs <= N) and expands the solution back to the N nodes.

The iteration is a θ-space Newton (Jacobi) step: for each group g

      Δθ_out_g = (k_out_hat_g − k_out_g) / (Σ_h m_h p_gh(1−p_gh) − p_gg(1−p_gg))

clipped to ``[−max_step, +max_step]``, accelerated by Anderson mixing, with:
- weighted Anderson mixing (prevents hub-node domination),
- blowup reset (clears Anderson history when the residual spikes),
- best_theta tracking (returns the best θ seen, not a blown-up final θ),
- a stagnation stop.
"""
from __future__ import annotations

import datetime
import math
import sys
import time

import torch

from dcms.solvers.base import SolverResult
from dcms.utils.profiling import _PeakRAMMonitor

_ETA_MAX: float = 50.0

_ANDERSON_MAX_NORM: float = 1e6
_ANDERSON_BLOWUP_FACTOR: float = 100.0


def _anderson_mixing(
    fp_outputs: list[torch.Tensor],
    residuals_hist: list[torch.Tensor],
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Anderson mixing with per-component row-weighting (PR#12).

    Solves the *per-component-weighted* constrained least-squares problem:

        min  ‖W⁻¹ Σ_i c_i r_i‖²    s.t.  Σ_i c_i = 1

    where w_j = max_i |r_i[j]| prevents hub nodes from dominating the mixing.

    Args:
        fp_outputs:     List of g(θ_k) values (FP outputs), each shape (2N,).
        residuals_hist: List of r_k = g(θ_k) − θ_k values, each shape (2N,).
        weights:        Optional per-row (per-component) multiplicity
                        weights, shape (2N,). Used by the degeneracy-reduced
                        (``mult``) path: in the unreduced system, a
                        degeneracy class of ``mult[g]`` identical nodes
                        contributes ``mult[g]`` identical rows to the
                        least-squares fit below; in the reduced system that
                        class is a single row, so without this weight the
                        fit implicitly under-counts high-multiplicity
                        classes relative to the unreduced system. ``None``
                        (default) reproduces the old unweighted behaviour
                        exactly, and passing all-ones is mathematically
                        identical to ``None``.

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



def compute_degeneracy_groups_dcm(
    k_out: torch.Tensor,
    k_in: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group nodes by identical (k_out, k_in).

    Args:
        k_out, k_in: Observed sequences, each shape (N,).

    Returns:
        group_of: (N,) long tensor mapping each original node to its group
            index in [0, M).
        mult:     (M,) float tensor, multiplicity (node count) of each group.
        k_out_g, k_in_g: (M,) float tensors, the shared target values.
    """
    N = k_out.shape[0]
    key = torch.stack([k_out, k_in], dim=1)
    uniq, group_of = torch.unique(key, dim=0, return_inverse=True)
    M = uniq.shape[0]
    mult = torch.zeros(M, dtype=torch.float64)
    mult.scatter_add_(0, group_of, torch.ones(N, dtype=torch.float64))
    k_out_g, k_in_g = uniq[:, 0], uniq[:, 1]
    return group_of, mult, k_out_g, k_in_g



def _dcm_step_dense_weighted(
    theta: torch.Tensor,
    k_out: torch.Tensor,
    k_in: torch.Tensor,
    max_step: float,
    mult: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One theta-Newton (Jacobi) step for the degeneracy-reduced DCM.

    Per-node theta-Newton algebra, generalised
    so each group ``g`` stands in for ``mult[g]`` original nodes -- same
    "weighted sum minus one diagonal term" pattern used in DECM's
    ``_decm_step_dense_weighted`` (see that function's docstring for the
    combinatorial justification). With ``mult`` all-ones (M == N) this
    reduces exactly to the per-node formula.

    Args:
        theta:  Current 2M parameter vector [theta_out|theta_in], one entry
            per group.
        k_out, k_in: Group-level target sequences, shape (M,).
        max_step: Maximum |Delta theta| per group per step.
        mult: Group multiplicities (node count per group), shape (M,).

    Returns:
        ``(theta_new, F_current)``, both shape (2M,); ``F_current`` is the
        exact per-member residual, evaluated at the input theta.
    """
    M = k_out.shape[0]
    theta_out = theta[:M]
    theta_in = theta[M:]

    logit_p = -theta_out[:, None] - theta_in[None, :]
    P = torch.sigmoid(logit_p)
    pq = P * (1.0 - P)

    P_diag = P.diagonal()
    pq_diag = pq.diagonal()

    k_out_hat = (P * mult[None, :]).sum(1) - P_diag
    k_in_hat = (P * mult[:, None]).sum(0) - P_diag

    H_out = ((pq * mult[None, :]).sum(1) - pq_diag).clamp(min=1e-15)
    H_in = ((pq * mult[:, None]).sum(0) - pq_diag).clamp(min=1e-15)

    F_out = k_out_hat - k_out
    F_in = k_in_hat - k_in
    F_current = torch.cat([F_out, F_in])

    delta_out = (F_out / H_out).clamp(-max_step, max_step)
    theta_out_new = (theta_out + delta_out).clamp(-_ETA_MAX, _ETA_MAX)
    theta_out_new = torch.where(k_out == 0, torch.full_like(theta_out_new, _ETA_MAX), theta_out_new)

    delta_in = (F_in / H_in).clamp(-max_step, max_step)
    theta_in_new = (theta_in + delta_in).clamp(-_ETA_MAX, _ETA_MAX)
    theta_in_new = torch.where(k_in == 0, torch.full_like(theta_in_new, _ETA_MAX), theta_in_new)

    return torch.cat([theta_out_new, theta_in_new]), F_current



def _solve_dcm_reduced(
    theta0: torch.Tensor,
    k_out: torch.Tensor,
    k_in: torch.Tensor,
    mult: torch.Tensor,
    tol: float,
    max_iter: int,
    anderson_depth: int,
    max_step: float,
    max_time: float,
    verbose: bool,
    monitor: bool,
    weight_anderson: bool,
    device: str,
) -> SolverResult:
    """Newton/Anderson loop on the reduced system (M groups, multiplicities ``mult``); see :func:`solve_fixed_point_dcm`."""
    _dev = torch.device(device)
    _dtype = torch.float64 if _dev.type == "cpu" else torch.float32

    k_out = k_out.to(dtype=_dtype, device=_dev)
    k_in = k_in.to(dtype=_dtype, device=_dev)
    theta = theta0.clone().to(dtype=_dtype, device=_dev)
    mult = mult.to(dtype=_dtype, device=_dev)

    theta = theta.clamp(-_ETA_MAX, _ETA_MAX)

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

    # Precompute verbose targets once (MRE = max |F_i| / k_i)
    _v_targets = torch.cat([k_out, k_in])
    _v_nonzero = _v_targets > 0

    # Anderson mixing needs per-component weights in the reduced system.
    _anderson_weights = torch.cat([mult, mult]) if weight_anderson else None

    try:
        for _ in range(max_iter):
            if max_time > 0 and (time.perf_counter() - t0) > max_time:
                message = (
                    f"Time limit ({max_time:.0f}s) reached at iteration {n_iter}."
                )
                break

            theta_fp, F_current = _dcm_step_dense_weighted(theta, k_out, k_in, max_step, mult)

            # Convergence check (relative: MRE = max|F_i|/target_i)
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
                print(
                    f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] "
                    f"iteration={n_iter:5d}, "
                    f"elapsed time={int(_elapsed // 3600):4d}:{int((_elapsed % 3600) // 60):02d}:{int(_elapsed % 60):02d}, "
                    f"MRE_topo={res_norm:.5e}",
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
                    if math.isfinite(r_k_norm) and r_k_norm < _ANDERSON_MAX_NORM:
                        _and_g.append(theta_fp.clone())
                        _and_r.append(r_k.clone())

                    if len(_and_g) > anderson_depth:
                        _and_g.pop(0)
                        _and_r.pop(0)

                    if len(_and_g) >= 2:
                        theta_next = _anderson_mixing(_and_g, _and_r, weights=_anderson_weights)
                        theta_next = theta_next.clamp(-_ETA_MAX, _ETA_MAX)
                    else:
                        theta_next = theta_fp
            else:
                theta_next = theta_fp

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


def solve_fixed_point_dcm(
    theta0: "ArrayLike",  # type: ignore[name-defined]
    k_out: "ArrayLike",  # type: ignore[name-defined]
    k_in: "ArrayLike",  # type: ignore[name-defined]
    tol: float = 1e-8,
    max_iter: int = 10_000,
    anderson_depth: int = 10,
    max_step: float = 1.0,
    max_time: float = 0.0,
    verbose: bool = False,
    monitor: bool = False,
    weight_anderson: bool = True,
    device: str = "cpu",
) -> SolverResult:
    """Degeneracy-reduced theta-Newton solver for the DCM.

    Groups nodes that share the exact same (k_out, k_in) pair -- which are
    proven to share identical (theta_out, theta_in) at the true DCM
    solution, see :func:`compute_degeneracy_groups_dcm` -- and solves the
    resulting M-unknown reduced system (M = number of unique pairs <= N).
    The returned ``SolverResult`` is expanded back to the original per-node
    shape (2N,).

    Args:
        theta0: Initial guess [theta_out|theta_in], shape (2N,), strictly positive entries. Nodes
            sharing a (k_out,k_in) pair should already carry identical
            values here (true for the standard "degrees"-based initial
            guess); if not, one representative member's value is used for
            the whole group.
        k_out, k_in: Observed sequences, each shape (N,).
        tol: Convergence tolerance on the relative residual (MRE = max|F_i|/target_i).
        max_iter: Maximum number of iterations.
        anderson_depth: Anderson mixing depth (<= 1 = plain Newton iteration).
        max_step: Maximum |Delta theta| per group per step.
        max_time: Wall-clock limit in seconds (0 = none).
        verbose, monitor: Print one line per iteration (``monitor``: overwrite the same line).
        weight_anderson: Weight the Anderson least-squares rows by the group multiplicities.
        device: ``"cpu"`` (default, float64) or a torch device string like
            ``"mps"``/``"cuda"`` (float32 -- MPS has no float64 support at
            all, and CUDA float64 is slow enough that float32 is the
            practical choice too). The returned ``SolverResult`` is always
            CPU/float64 regardless of ``device``.

    Returns:
        :class:`~dcms.solvers.base.SolverResult` with ``theta``/``best_theta``
        of shape (2N,).
    """
    k_out = k_out if isinstance(k_out, torch.Tensor) else torch.tensor(k_out, dtype=torch.float64)
    k_in = k_in if isinstance(k_in, torch.Tensor) else torch.tensor(k_in, dtype=torch.float64)
    k_out = k_out.to(dtype=torch.float64)
    k_in = k_in.to(dtype=torch.float64)
    theta0 = (
        theta0 if isinstance(theta0, torch.Tensor) else torch.tensor(theta0, dtype=torch.float64)
    ).to(dtype=torch.float64)

    N = k_out.shape[0]
    group_of, mult, k_out_g, k_in_g = compute_degeneracy_groups_dcm(k_out, k_in)
    M = k_out_g.shape[0]

    _first_idx = torch.zeros(M, dtype=torch.long)
    _seen = torch.zeros(M, dtype=torch.bool)
    _group_of_list = group_of.tolist()
    for _i, _g in enumerate(_group_of_list):
        if not _seen[_g]:
            _seen[_g] = True
            _first_idx[_g] = _i
    theta0_g = torch.cat([theta0[:N][_first_idx], theta0[N:][_first_idx]])

    result_g = _solve_dcm_reduced(
        theta0_g, k_out_g, k_in_g, mult,
        tol=tol, max_iter=max_iter, anderson_depth=anderson_depth, max_step=max_step, max_time=max_time,
        verbose=verbose, monitor=monitor, weight_anderson=weight_anderson, device=device,
    )

    def _expand(theta_m):
        import numpy as np
        g = group_of.numpy()
        to_ = theta_m[:M][g]
        ti_ = theta_m[M:][g]
        return np.concatenate([to_, ti_])

    return SolverResult(
        theta=_expand(result_g.theta),
        best_theta=_expand(result_g.best_theta),
        converged=result_g.converged,
        iterations=result_g.iterations,
        residuals=result_g.residuals,
        elapsed_time=result_g.elapsed_time,
        peak_ram_bytes=result_g.peak_ram_bytes,
        message=result_g.message + f" [degeneracy-reduced: N={N} -> M={M}]",
    )
