"""Fixed-point iteration solver for the DWCM model.

The DWCM strength equations are

    s_out_i = Σ_{j≠i} β_out_i · β_in_j / (1 − β_out_i · β_in_j)
    s_in_i  = Σ_{j≠i} β_out_j · β_in_i / (1 − β_out_j · β_in_i)

with β = exp(−θ).  Two variants are implemented:

* **Gauss-Seidel** — β-space fixed-point; out-multipliers are updated
  first; the updated values are immediately used when computing new
  in-multipliers.  Update: β_out_i^{new} = s_out_i / D_out_i.
* **theta-newton** — θ-space coordinate Newton step.  For each node i
  the exact 1-D Newton step is computed directly in θ-space:

      Δθ_out_i = −F_i / F′_i
      where F_i  = Σ_{j≠i} W_ij − s_out_i   (residual)
            F′_i = −Σ_{j≠i} W_ij(1+W_ij)    (diagonal Hessian, ≤ 0)
            W_ij = 1/expm1(θ_out_i + θ_in_j)

  The per-node step is clipped to ``[−max_step, +max_step]``.  Because
  the update is additive in θ, β > 1 is never produced — this variant
  is robust to high-strength hub nodes that cause the β-space variants
  to oscillate.

The Gauss-Seidel variant supports an optional damping factor α ∈ (0, 1]:

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
from src.models.parameters import DWCM_LARGE_N_THRESHOLD as _LARGE_N_THRESHOLD
from src.models.parameters import _DEFAULT_CHUNK, _ETA_MIN, _ETA_MAX
# Maximum ℓ∞ norm of the Anderson FP-residual r_k = g(θ_k) − θ_k accepted
# into the mixing history.  Iterates beyond this threshold are skipped to
# prevent Anderson mixing from being contaminated by extreme values.  The
# threshold is most relevant for the theta-newton variant, which can produce
# large transient steps during the initial phase when θ is far from the
# solution, but the guard applies uniformly to all variants.
_ANDERSON_MAX_NORM: float = 1e6

# When the ℓ∞ equation-residual jumps by more than this factor above the
# best residual seen so far, the Anderson mixing on the previous step
# produced a bad iterate (β ≈ 1 nodes, w_ij → ∞).  The history is cleared
# immediately to prevent contamination of future mixing steps.  A factor of
# 100 is conservative enough not to fire during normal oscillations while
# catching all catastrophic blowups observed in practice.
_ANDERSON_BLOWUP_FACTOR: float = 100.0

# Geometric lower-bound on any Anderson step in θ-space.  For each component
# i the Anderson-mixed iterate must satisfy θ_next[i] ≥ floor × θ_current[i].
# This prevents hub nodes (small θ, β near 1 at convergence) from being
# extrapolated past θ = 0 into the infeasible region: if Anderson computes
# θ_hub = -4 from θ_hub = 0.5, the result is clamped to 0.5 × 0.1 = 0.05
# rather than hitting _ETA_MIN = 1e-10 (β ≈ 1, w_ij → ∞).
# A value of 0.1 allows up to 10× acceleration per step while keeping θ > 0.
_ANDERSON_THETA_FLOOR: float = 0.1

# In the β-space FP variants (gauss-seidel, jacobi), the FP map β_new = s/D(β)
# can have spectral radius > 1 for high-strength hub nodes (β* ≈ 1), causing
# slowly-growing oscillations that Anderson cannot accelerate away.  When the
# β-FP step would move θ by more than this amount (|Δθ| > threshold), we fall
# back to a bounded θ-Newton step for that node instead.
#
# The threshold is chosen to catch:
#   • Infeasible overshoots  (β_new ≥ 1) → |Δθ| >> 0.1
#   • Non-contractive oscillations for hub nodes → |Δθ| ≈ 0.12–0.15 per step
# while leaving normal nodes (|Δθ| ≪ 0.1) untouched after warm-up.
_FP_NEWTON_FALLBACK_DELTA: float = 0.1

# When FP-GS Anderson stagnates (residual not improved by ≥ 1 % over this many
# consecutive iterations), run a Newton-Anderson mini-loop from best_theta to
# escape the non-contractive oscillation region.
#
# Plain Newton (diagonal approximation) oscillates on extreme hub networks
# without Anderson mixing — the off-diagonal Jacobian terms are too large to
# ignore.  Combining diagonal Newton with Anderson(m=_FPGS_NEWTON_AND_DEPTH)
# mirrors exactly how the standalone θ-Newton Anderson(10) method converges:
# Anderson corrects the direction bias introduced by the diagonal approximation.
# _FPGS_NEWTON_STEPS = 30 gives a comfortable margin over the 17–30 iterations
# that θ-Newton Anderson(10) needs from a cold start; from best_theta (closer
# to the solution) far fewer steps are typically required.
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

    The solution is obtained via the normal equations of the m×m system
    (R_w^T R_w) c ∝ 1 where R_w = R / w[:,None], then normalised so Σ c_i = 1.

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
    # This prevents hub nodes (large |r_k|) from dominating R^T R and avoids
    # the ill-conditioning that causes Anderson to produce β ≈ 1 iterates.
    w = R.abs().max(dim=1, keepdim=True).values.clamp(min=1e-15)  # (2N, 1)
    R_w = R / w  # (2N, m): each row scaled to max-absolute-value 1

    # Normal equations on the weighted system: (R_w^T R_w) c ∝ ones
    RtR = R_w.T @ R_w  # (m, m)
    # Regularise to avoid singularity when recent steps are nearly collinear.
    RtR = RtR + 1e-10 * torch.eye(m, dtype=RtR.dtype)
    ones = torch.ones(m, dtype=RtR.dtype)
    try:
        c = torch.linalg.solve(RtR, ones)  # (m,): c ∝ (R_w^T R_w)^{-1} 1
        c_sum = c.sum().item()
        if abs(c_sum) < 1e-14 or not math.isfinite(c_sum):
            raise RuntimeError("Degenerate Anderson weights.")
        c = c / c_sum  # normalise: Σ c_i = 1
        # Clamp to prevent wild extrapolation, then re-normalise safely
        c = c.clamp(-10.0, 10.0)
        c_sum_clamped = c.sum().item()
        if abs(c_sum_clamped) < 1e-14 or not math.isfinite(c_sum_clamped):
            # Post-clamp weights are degenerate; fall back to uniform mixing.
            c = ones / m
        else:
            c = c / c_sum_clamped
    except RuntimeError:
        c = ones / m  # fallback: simple uniform mixing

    return G @ c


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
    anderson_depth: int = 0,
    max_step: float = 1.0,
    max_time: float = 0.0,
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
        damping: Damping factor α ∈ (0, 1] for the ``"gauss-seidel"``
            variant.  α=1 → no damping (pure FP update).
            Not used by the ``"theta-newton"`` variant; use ``max_step``
            instead to control the Newton step size for that variant.
        variant: One of ``"gauss-seidel"`` or ``"theta-newton"``.
            ``"gauss-seidel"`` solves in β-space; ``"theta-newton"``
            performs coordinate Newton steps directly in θ-space, which avoids
            the β > 1 clamping oscillation that affects high-strength hub nodes.
        chunk_size: If > 0, process the N×N products in chunks of this size
            to avoid materialising the full matrix (useful for large N).
            If 0, auto-select: dense for N ≤ ``_LARGE_N_THRESHOLD``, chunked
            (with ``_DEFAULT_CHUNK``) otherwise.
        anderson_depth: Depth of Anderson acceleration history.  0 disables
            Anderson mixing (plain fixed-point).  Values of 5–10 typically
            give the best convergence acceleration.  Uses the m most recent
            FP outputs to compute the next iterate via constrained
            least-squares mixing (Walker & Ni 2011).
        max_step: Maximum per-node Newton step |Δθ| allowed in the
            ``"theta-newton"`` variant.  Ignored by ``"gauss-seidel"``.  Default 1.0 (one unit in log-space);
            reduce to ~0.5 for very heterogeneous networks to improve
            robustness at the cost of slower convergence per iteration.
        max_time: Wall-clock time limit in seconds.  If > 0, the solver
            stops after this many seconds even if max_iter has not been
            reached.  Default 0 (no time limit).

    Returns:
        :class:`~src.solvers.base.SolverResult` instance.
    """
    if variant not in ("gauss-seidel", "theta-newton"):
        raise ValueError(
            f"Unknown variant {variant!r}. Choose 'gauss-seidel' or 'theta-newton'."
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

    # Stagnation detection: stop if the best residual over the last
    # _STAGNATION_WINDOW iterations has not improved by at least
    # _STAGNATION_RTOL relative to _STAGNATION_WINDOW iterations ago.
    _STAGNATION_WINDOW: int = 200
    _STAGNATION_RTOL: float = 0.01  # 1% improvement threshold
    best_res_recent: float = float("inf")
    best_res_old: float = float("inf")

    # Track the best iterate seen during the run so that SolverResult.theta
    # is always the θ with the minimum residual, not the (possibly blown-up)
    # final θ.  This is critical for multi-start warm restarts: starting from
    # the best intermediate θ rather than the final θ can skip the "recovery
    # from blowup" phase that wastes dozens of iterations.
    best_theta: torch.Tensor = theta.clone()
    best_theta_res: float = float("inf")

    # Anderson acceleration history (in θ-space)
    _and_g: list[torch.Tensor] = []  # g(θ_k) — damped FP outputs
    _and_r: list[torch.Tensor] = []  # r_k = g(θ_k) − θ_k

    # Running minimum of the equation-residual for blowup detection.
    _best_res_for_anderson: float = float("inf")

    # FP-GS periodic Newton correction: stagnation counter that is NEVER reset
    # by blowup recovery (only by Newton firing or genuine ≥ 1 % improvement).
    # This ensures the counter accumulates across 3-period blowup cycles and
    # Newton reliably fires after _FPGS_NEWTON_RESET_WINDOW stagnant iterations.
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
                # Returns (theta_new, F_current) where F_current = F(θ)
                # at the *input* θ (before the update).
                if effective_chunk > 0:
                    theta_fp, F_current = _theta_newton_step_chunked(
                        theta, s_out, s_in, effective_chunk, max_step
                    )
                else:
                    theta_fp, F_current = _theta_newton_step_dense(
                        theta, s_out, s_in, max_step
                    )
            else:
                # β-space fixed-point iteration (Gauss-Seidel or Jacobi)
                beta_out = torch.exp(-theta[:N])
                beta_in = torch.exp(-theta[N:])

                if effective_chunk > 0:
                    beta_out_new, beta_in_new, F_current = _fp_step_chunked_dwcm(
                        beta_out, beta_in, s_out, s_in, effective_chunk,
                        theta=theta, max_step=max_step,
                    )
                else:
                    # Dense path (materialises full N×N matrix)
                    xy = beta_out[:, None] * beta_in[None, :]

                    # D_out[i] = Σ_{j≠i} β_in_j / (1 - β_out_i * β_in_j)
                    denom = (1.0 - xy).clamp(min=1e-15)
                    D_out_mat = beta_in[None, :] / denom   # (N, N)
                    D_out_mat.fill_diagonal_(0.0)
                    D_out = D_out_mat.sum(dim=1)

                    # Residual at current β (free from D_out_mat):
                    # s_out_hat = β_out * D_out,  s_in_hat = Σ_i β_out_i * D_out_mat[i,j]
                    s_out_hat = beta_out * D_out
                    s_in_hat = D_out_mat.T @ beta_out     # (N,) — O(N²) FLOPS, O(N) memory
                    F_current = torch.cat([s_out_hat - s_out, s_in_hat - s_in])

                    beta_out_new = torch.where(D_out > 0, s_out / D_out, beta_out)

                    # Newton fallback: replace the β-FP step with a bounded exact
                    # θ-Newton step for any node where the β-FP map would produce
                    # a large θ-step (|Δθ| > _FP_NEWTON_FALLBACK_DELTA).  This
                    # catches two failure modes:
                    #   1. Infeasible overshoot (β_new ≥ 1): |Δθ| >> threshold.
                    #   2. Non-contractive oscillation (SR_FP > 1): hub nodes
                    #      oscillate with |Δθ| ≈ 0.12–0.15 per step even when
                    #      β_new < 1; their spectral radius just exceeds 1.0 so
                    #      Anderson(10) cannot converge the growing oscillation.
                    # We use the *exact* diagonal Newton step
                    #   Δθ_out_i = (ŝ_i − s_i) / H_out_i
                    # where H_out_i = Σ_j W_ij(1+W_ij) = β_out_i·D_out_i + β_out_i²·‖D_out_mat[i,:]‖².
                    # Using the approximation H ≈ ŝ (first-order) would give a 2×
                    # over-step for extreme hubs (high-W pairs), keeping SR ≈ 1.
                    _s_hat_out = beta_out * D_out  # s_hat_out[i] = β_out_i·D_out_i
                    _theta_out_fp_val = (
                        -torch.log(beta_out_new.clamp(1e-300, 1.0 - 1e-15))
                    ).clamp(_ETA_MIN, _ETA_MAX)
                    _use_newton_out = (
                        (_theta_out_fp_val - theta[:N]).abs() > _FP_NEWTON_FALLBACK_DELTA
                    )
                    if _use_newton_out.any():
                        # H_out[i] = β_out_i·D_out[i] + β_out_i²·Σ_j D_out_mat[i,j]²
                        _H_out = (
                            _s_hat_out
                            + beta_out ** 2 * (D_out_mat ** 2).sum(dim=1)
                        ).clamp(min=1e-30)
                        _delta_nt_out = (
                            (_s_hat_out - s_out) / _H_out
                        ).clamp(-max_step, max_step)
                        _theta_out_nt = (theta[:N] + _delta_nt_out).clamp(
                            _ETA_MIN, _ETA_MAX
                        )
                        beta_out_new = torch.where(
                            _use_newton_out, torch.exp(-_theta_out_nt), beta_out_new
                        )

                    beta_out_upd = beta_out_new  # Gauss-Seidel: use updated values immediately

                    # D_in[i] = Σ_{j≠i} β_out_j / (1 - β_out_j * β_in_i)
                    xy2 = beta_out_upd[:, None] * beta_in[None, :]
                    denom2 = (1.0 - xy2).clamp(min=1e-15)
                    D_in_mat = beta_out_upd[:, None] / denom2  # (N, N)
                    D_in_mat.fill_diagonal_(0.0)
                    D_in = D_in_mat.sum(dim=0)

                    beta_in_new = torch.where(D_in > 0, s_in / D_in, beta_in)

                    # Newton fallback for in-direction (same |Δθ| criterion,
                    # exact H_in[j] = β_in_j·D_in[j] + β_in_j²·Σ_i D_in_mat[i,j]²)
                    _s_hat_in = beta_in * D_in
                    _theta_in_fp_val = (
                        -torch.log(beta_in_new.clamp(1e-300, 1.0 - 1e-15))
                    ).clamp(_ETA_MIN, _ETA_MAX)
                    _use_newton_in = (
                        (_theta_in_fp_val - theta[N:]).abs() > _FP_NEWTON_FALLBACK_DELTA
                    )
                    if _use_newton_in.any():
                        _H_in = (
                            _s_hat_in
                            + beta_in ** 2 * (D_in_mat ** 2).sum(dim=0)
                        ).clamp(min=1e-30)
                        _delta_nt_in = (
                            (_s_hat_in - s_in) / _H_in
                        ).clamp(-max_step, max_step)
                        _theta_in_nt = (theta[N:] + _delta_nt_in).clamp(
                            _ETA_MIN, _ETA_MAX
                        )
                        beta_in_new = torch.where(
                            _use_newton_in, torch.exp(-_theta_in_nt), beta_in_new
                        )

                # Clamp β to (0, 1) strictly to maintain feasibility
                beta_out_new = beta_out_new.clamp(1e-300, 1.0 - 1e-15)
                beta_in_new = beta_in_new.clamp(1e-300, 1.0 - 1e-15)

                # Convert to θ-space and apply damping
                theta_out_new = (-torch.log(beta_out_new)).clamp(_ETA_MIN, _ETA_MAX)
                theta_in_new = (-torch.log(beta_in_new)).clamp(_ETA_MIN, _ETA_MAX)
                fp_raw = torch.cat([theta_out_new, theta_in_new])

                # Damped FP output: g(θ) = α * FP_raw(θ) + (1−α) * θ
                theta_fp = (damping * fp_raw + (1.0 - damping) * theta).clamp(
                    _ETA_MIN, _ETA_MAX
                )

            # Universal single-step floor: prevent any θ from dropping below
            # _ANDERSON_THETA_FLOOR × its current value in a single iteration.
            # This stops θ → ETA_MIN (β → 1, W_ij → ∞) blowups that occur when
            # the Newton fallback produces δθ = −max_step and θ ≤ max_step
            # (e.g. θ=0.5, δ=−0.5 → 0 → ETA_MIN → β≈1 → 5×10⁹ residual).
            # The floor applies whether or not Anderson is active, so it guards
            # both the plain-step path (len < 2) and the Anderson path.
            _step_floor = (theta * _ANDERSON_THETA_FLOOR).clamp(min=_ETA_MIN)
            theta_fp = torch.maximum(theta_fp, _step_floor)

            # --- Convergence check using the step-computed residual ---
            # F_current = F(θ) at the *current* iterate (before update).
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
            # When the β-space FP map is non-contractive (spectral radius > 1
            # for extreme hub nodes), Anderson acceleration stagnates: the
            # oscillation lives in a subspace larger than the Anderson history
            # window.  A single full θ-Newton step jumps the iterate into the
            # FP-Anderson convergence basin; the cleared history then lets
            # Anderson restart cleanly from the better point.
            #
            # Design: the stagnation counter is NEVER reset by blowup recovery
            # (only by Newton firing or genuine improvement).  In a 3-period
            # blowup cycle (blowup every 3 iters) the residual never improves
            # by ≥ 1 %, so the counter increments on every iteration — blowup
            # or not — and Newton fires after exactly _FPGS_NEWTON_RESET_WINDOW
            # iterations.  For a genuinely-converging network the residual
            # decreases by > 1 % per step, the counter resets frequently, and
            # Newton never fires (no regression).
            #
            # The firing flag drives a post-Anderson override (see below) that
            # replaces theta_next unconditionally, so blowup recovery cannot
            # silently discard the Newton correction.
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
                        # Reset for fresh tracking after Newton correction.
                        _fpgs_stagnation_count = 0
                        _fpgs_best_local = float("inf")
                        # Clear Anderson history now so the Anderson section
                        # falls through to the plain-update path and does not
                        # mix Newton-corrected and FP-oscillation iterates.
                        _and_g.clear()
                        _and_r.clear()

            # --- Apply Anderson acceleration or plain update ---
            if anderson_depth > 1:
                # Blowup guard: when the equation-residual jumps catastrophically
                # above the best seen so far, the previous Anderson-mixed iterate
                # pushed some θ_i → 0 (β_i ≈ 1, W_ij → ∞).  Clear the history
                # AND reset θ to best_theta so the next iteration recomputes
                # theta_fp from a safe, low-residual starting point rather than
                # from the blown-up θ.  This breaks the blowup-recovery-blowup
                # cycle that occurs when theta_fp is computed from a corrupted θ.
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

                # Update the running best AFTER the blowup check so that a blowup
                # event does not raise the detection threshold for the next reset.
                _best_res_for_anderson = min(_best_res_for_anderson, res_norm)

                if _blowup_recovered:
                    # Skip the Anderson mixing for this step: stay at best_theta.
                    # theta_fp was computed from the blown-up iterate and is
                    # unsafe; the next iteration will produce a clean theta_fp.
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
                        # Geometric floor for ALL variants: prevent Anderson from
                        # extrapolating any θ_i to near 0 (β_i ≈ 1), which would
                        # make W_ij → ∞ for hub-node pairs and cause blowups.
                        # The floor is capped at theta_fp so Anderson is never
                        # worse than the plain step.
                        # Previously this was theta-newton only because for pure
                        # β-FP variants theta_fp changes by ~1 % per step and the
                        # floor would block Anderson acceleration.  With the Newton
                        # fallback now active for hub nodes, theta_fp for those
                        # nodes can change by up to max_step per iteration, so the
                        # floor (10 % of current θ) is not restrictive and
                        # correctly prevents the β≈1 blowups.
                        theta_floor = (theta * _ANDERSON_THETA_FLOOR).clamp(
                            min=_ETA_MIN
                        )
                        effective_floor = torch.minimum(theta_floor, theta_fp)
                        theta_next = torch.maximum(theta_next, effective_floor)
                        theta_next = theta_next.clamp(_ETA_MIN, _ETA_MAX)
                    else:
                        theta_next = theta_fp
            else:
                theta_next = theta_fp

            # --- FP-GS post-Anderson Newton-Anderson mini-loop ---
            # When FP-GS stagnates, run a θ-Newton Anderson(_FPGS_NEWTON_AND_DEPTH)
            # mini-loop from best_theta.  Plain Newton (diagonal approximation)
            # oscillates on extreme hub networks without Anderson mixing; adding
            # Anderson(m=5) mirrors the standalone θ-Newton Anderson(10) method
            # that reliably converges on all these seeds in 17–30 iterations.
            # Placed AFTER the Anderson/blowup section so blowup recovery cannot
            # silently discard the correction.
            if _fpgs_newton_fired:
                theta_nt = best_theta.clone()
                _nt_and_g: list[torch.Tensor] = []
                _nt_and_r: list[torch.Tensor] = []
                for _ in range(_FPGS_NEWTON_STEPS):
                    if effective_chunk > 0:
                        theta_nt_fp, F_nt = _theta_newton_step_chunked(
                            theta_nt, s_out, s_in, effective_chunk, max_step
                        )
                    else:
                        theta_nt_fp, F_nt = _theta_newton_step_dense(
                            theta_nt, s_out, s_in, max_step
                        )
                    # Per-step floor (same as main loop)
                    _nt_floor = (theta_nt * _ANDERSON_THETA_FLOOR).clamp(min=_ETA_MIN)
                    theta_nt_fp = torch.maximum(theta_nt_fp, _nt_floor).clamp(
                        _ETA_MIN, _ETA_MAX
                    )
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
                            _ETA_MIN, _ETA_MAX
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


def _fp_step_chunked_dwcm(
    beta_out: torch.Tensor,
    beta_in: torch.Tensor,
    s_out: torch.Tensor,
    s_in: torch.Tensor,
    chunk_size: int,
    theta: torch.Tensor | None = None,
    max_step: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One Gauss-Seidel fixed-point update step for DWCM using chunked matrix products.

    Computes the update denominators D_out and D_in without ever building
    the full N×N product matrix, using at most O(chunk_size × N) RAM per chunk.

    Also computes the residual F(θ_current) = [s_out_hat − s_out | s_in_hat − s_in]
    at the *current* (pre-update) β values, at negligible extra cost.

    When *theta* is provided, a Newton fallback is applied for any node where
    the β-FP step would produce β_new ≥ 1 (hub overshoot).  For those nodes a
    bounded θ-space Newton step replaces the infeasible β-FP result.

    Args:
        beta_out:   Current out-strength multipliers β_out_i = exp(−θ_out_i),
                    shape (N,), values in (0, 1).
        beta_in:    Current in-strength multipliers β_in_i = exp(−θ_in_i),
                    shape (N,), values in (0, 1).
        s_out:      Observed out-strength sequence, shape (N,).
        s_in:       Observed in-strength sequence, shape (N,).
        chunk_size: Number of rows (or columns) per chunk.
        theta:      Full parameter vector [θ_out | θ_in], shape (2N,), used
                    by the Newton fallback.  Pass ``None`` to disable.
        max_step:   Maximum |Δθ| per node for the Newton fallback step.

    Returns:
        ``(beta_out_new, beta_in_new, F_current)`` where F_current is the
        residual at the pre-update state, shape (2N,).
    """
    N = beta_out.shape[0]

    # -------------------------------------------------------------------
    # D_out[i] = Σ_{j≠i} β_in_j / (1 - β_out_i * β_in_j)
    # Process rows i in chunks.
    # Also accumulate s_in_hat = Σ_i w_ij for the residual at current β.
    # -------------------------------------------------------------------
    D_out = torch.zeros(N, dtype=torch.float64)
    s_in_hat = torch.zeros(N, dtype=torch.float64)
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
        # w_ij = β_out_i * d_chunk[i,j]; accumulate column sums for s_in_hat
        s_in_hat += d_chunk.T @ b_out_chunk              # (N,)

    # Residual at current β
    s_out_hat = beta_out * D_out
    F_current = torch.cat([s_out_hat - s_out, s_in_hat - s_in])

    beta_out_new = torch.where(D_out > 0, s_out / D_out, beta_out)

    # Newton fallback: for hub nodes where the β-FP step would produce a large
    # θ-step (|Δθ| > _FP_NEWTON_FALLBACK_DELTA), use a bounded θ-Newton step
    # instead.  This catches both infeasible overshoots (β_new ≥ 1) and slowly
    # diverging oscillations (spectral radius of the β-FP map slightly > 1).
    if theta is not None:
        _theta_out_fp_val = (
            -torch.log(beta_out_new.clamp(1e-300, 1.0 - 1e-15))
        ).clamp(_ETA_MIN, _ETA_MAX)
        _use_newton_out = (
            (_theta_out_fp_val - theta[:N]).abs() > _FP_NEWTON_FALLBACK_DELTA
        )
        if _use_newton_out.any():
            _delta_out = (
                (s_out_hat - s_out) / s_out_hat.clamp(min=1e-30)
            ).clamp(-max_step, max_step)
            _theta_out_nt = (theta[:N] + _delta_out).clamp(_ETA_MIN, _ETA_MAX)
            beta_out_new = torch.where(
                _use_newton_out, torch.exp(-_theta_out_nt), beta_out_new
            )

    # Gauss-Seidel: use updated β_out immediately for the D_in computation
    beta_out_upd = beta_out_new

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

    # Newton fallback for in-direction (same |Δθ| criterion)
    if theta is not None:
        _s_in_hat_cur = beta_in * D_in
        _theta_in_fp_val = (
            -torch.log(beta_in_new.clamp(1e-300, 1.0 - 1e-15))
        ).clamp(_ETA_MIN, _ETA_MAX)
        _use_newton_in = (
            (_theta_in_fp_val - theta[N:]).abs() > _FP_NEWTON_FALLBACK_DELTA
        )
        if _use_newton_in.any():
            _delta_in = (
                (_s_in_hat_cur - s_in) / _s_in_hat_cur.clamp(min=1e-30)
            ).clamp(-max_step, max_step)
            _theta_in_nt = (theta[N:] + _delta_in).clamp(_ETA_MIN, _ETA_MAX)
            beta_in_new = torch.where(
                _use_newton_in, torch.exp(-_theta_in_nt), beta_in_new
            )

    return beta_out_new, beta_in_new, F_current


# ---------------------------------------------------------------------------
# θ-space coordinate Newton helpers
# ---------------------------------------------------------------------------

def _theta_newton_step_dense(
    theta: torch.Tensor,
    s_out: torch.Tensor,
    s_in: torch.Tensor,
    max_step: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One θ-space coordinate Newton step (dense N×N path).

    For each node i, computes the exact Newton step on the 1-D equation::

        F_i(θ_out_i) = Σ_{j≠i} W_ij(θ) − s_out_i = 0

    where W_ij = 1 / expm1(θ_out_i + θ_in_j).  The Newton step is::

        Δθ_out_i = −F_i / F′_i = (Σ_j W_ij − s_out_i) / Σ_j W_ij(1+W_ij)

    Because the update is additive in θ (not in β), it naturally stays in the
    feasible domain after clamping to [_ETA_MIN, _ETA_MAX] — unlike the β-space
    fixed-point which can produce β > 1 for high-strength hub nodes.

    Zero-strength nodes (s_i = 0) are kept at _ETA_MAX throughout.

    The per-node step is clipped to ``[-max_step, +max_step]`` to prevent
    overshooting when the initial θ is far from the solution.  The GS ordering
    (update θ_out first, then θ_in using the fresh θ_out) is always used.

    Args:
        theta:    Current parameter vector [θ_out | θ_in], shape (2N,).
        s_out:    Observed out-strength sequence, shape (N,).
        s_in:     Observed in-strength sequence, shape (N,).
        max_step: Maximum allowed |Δθ| per node per step.

    Returns:
        Tuple of (updated parameter vector, residual F(θ_current)).
        The residual is F(θ) evaluated at the *input* θ (before the update),
        shape (2N,), clamped to [_ETA_MIN, _ETA_MAX].
    """
    N = s_out.shape[0]
    theta_out = theta[:N]
    theta_in = theta[N:]

    # W_ij = 1/expm1(θ_out_i + θ_in_j),  W[i,i] = 0 (no self-loops)
    z = theta_out[:, None] + theta_in[None, :]          # (N, N)
    W = 1.0 / torch.expm1(z.clamp(min=1e-15))          # (N, N)
    W.fill_diagonal_(0.0)

    # Residual at current θ (free from the already-computed W)
    F_out = W.sum(dim=1) - s_out                        # (N,)
    F_in = W.sum(dim=0) - s_in                          # (N,)
    F_current = torch.cat([F_out, F_in])

    # Out-direction Newton step (using current θ_in)
    h_out = -(W * (1.0 + W)).sum(dim=1)                 # (N,), ≤ 0
    delta_out = (-F_out / (h_out - 1e-30)).clamp(-max_step, max_step)
    theta_out_new = (theta_out + delta_out).clamp(_ETA_MIN, _ETA_MAX)
    # Zero-strength nodes: β = 0 exactly → θ = _ETA_MAX
    theta_out_new = torch.where(
        s_out == 0, torch.full_like(theta_out_new, _ETA_MAX), theta_out_new
    )

    # In-direction Newton step (GS: use updated θ_out_new)
    z2 = theta_out_new[:, None] + theta_in[None, :]     # (N, N)
    W2 = 1.0 / torch.expm1(z2.clamp(min=1e-15))        # (N, N)
    W2.fill_diagonal_(0.0)

    F_in2 = W2.sum(dim=0) - s_in                        # (N,)
    h_in = -(W2 * (1.0 + W2)).sum(dim=0)                # (N,), ≤ 0
    delta_in = (-F_in2 / (h_in - 1e-30)).clamp(-max_step, max_step)
    theta_in_new = (theta_in + delta_in).clamp(_ETA_MIN, _ETA_MAX)
    theta_in_new = torch.where(
        s_in == 0, torch.full_like(theta_in_new, _ETA_MAX), theta_in_new
    )

    return torch.cat([theta_out_new, theta_in_new]), F_current


def _theta_newton_step_chunked(
    theta: torch.Tensor,
    s_out: torch.Tensor,
    s_in: torch.Tensor,
    chunk_size: int,
    max_step: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunked version of :func:`_theta_newton_step_dense` for large N.

    Computes the θ-space Newton step without materialising the full N×N
    matrix, using O(chunk_size × N) working memory.

    Args:
        theta:      Current parameter vector [θ_out | θ_in], shape (2N,).
        s_out:      Observed out-strength sequence, shape (N,).
        s_in:       Observed in-strength sequence, shape (N,).
        chunk_size: Number of rows processed per chunk (≥ 1).
        max_step:   Maximum allowed |Δθ| per node per step.

    Returns:
        Tuple of (updated parameter vector, residual F(θ_current)).
        The residual is F(θ) evaluated at the *input* θ (before the update),
        shape (2N,), clamped to [_ETA_MIN, _ETA_MAX].
    """
    N = s_out.shape[0]
    theta_out = theta[:N]
    theta_in = theta[N:]

    # -------------------------------------------------------------------
    # Out-direction: accumulate F_out, h_out, and F_in_current row-by-row
    # -------------------------------------------------------------------
    F_out = torch.zeros(N, dtype=torch.float64)
    h_out = torch.zeros(N, dtype=torch.float64)
    F_in_current = torch.zeros(N, dtype=torch.float64)
    for i_start in range(0, N, chunk_size):
        i_end = min(i_start + chunk_size, N)
        chunk_len = i_end - i_start
        z_chunk = theta_out[i_start:i_end, None] + theta_in[None, :]   # (chunk, N)
        W_chunk = 1.0 / torch.expm1(z_chunk.clamp(min=1e-15))
        local_i = torch.arange(chunk_len, dtype=torch.long)
        global_j = torch.arange(i_start, i_end, dtype=torch.long)
        W_chunk[local_i, global_j] = 0.0
        F_out[i_start:i_end] = W_chunk.sum(dim=1) - s_out[i_start:i_end]
        h_out[i_start:i_end] = -(W_chunk * (1.0 + W_chunk)).sum(dim=1)
        # Accumulate column sums for F_in at current θ (free from same W_chunk)
        F_in_current += W_chunk.sum(dim=0)

    F_in_current -= s_in
    F_current = torch.cat([F_out, F_in_current])

    delta_out = (-F_out / (h_out - 1e-30)).clamp(-max_step, max_step)
    theta_out_new = (theta_out + delta_out).clamp(_ETA_MIN, _ETA_MAX)
    theta_out_new = torch.where(
        s_out == 0, torch.full_like(theta_out_new, _ETA_MAX), theta_out_new
    )

    # -------------------------------------------------------------------
    # In-direction: use updated θ_out_new (GS), accumulate column sums
    # -------------------------------------------------------------------
    F_in = torch.zeros(N, dtype=torch.float64)
    h_in = torch.zeros(N, dtype=torch.float64)
    for j_start in range(0, N, chunk_size):
        j_end = min(j_start + chunk_size, N)
        chunk_len = j_end - j_start
        # z2[j_local, i] = θ_out_new[j] + θ_in[i]
        z2_chunk = theta_out_new[j_start:j_end, None] + theta_in[None, :]  # (chunk, N)
        W2_chunk = 1.0 / torch.expm1(z2_chunk.clamp(min=1e-15))
        local_j = torch.arange(chunk_len, dtype=torch.long)
        global_i = torch.arange(j_start, j_end, dtype=torch.long)
        W2_chunk[local_j, global_i] = 0.0
        F_in += W2_chunk.sum(dim=0)
        h_in += -(W2_chunk * (1.0 + W2_chunk)).sum(dim=0)
    F_in -= s_in

    delta_in = (-F_in / (h_in - 1e-30)).clamp(-max_step, max_step)
    theta_in_new = (theta_in + delta_in).clamp(_ETA_MIN, _ETA_MAX)
    theta_in_new = torch.where(
        s_in == 0, torch.full_like(theta_in_new, _ETA_MAX), theta_in_new
    )

    return torch.cat([theta_out_new, theta_in_new]), F_current

