"""Fixed-point iteration solver for the DWCM model.

The DWCM strength equations are

    s_out_i = Σ_{j≠i} β_out_i · β_in_j / (1 − β_out_i · β_in_j)
    s_in_i  = Σ_{j≠i} β_out_j · β_in_i / (1 − β_out_j · β_in_i)

with β = exp(−θ). Nodes sharing the same (s_out, s_in) pair have identical parameters at the
solution (see :func:`compute_degeneracy_groups_dwcm`), so the solver works on the M-unknown
reduced system (M = number of distinct pairs <= N) and expands the solution back to the N nodes.

The iteration is a θ-space coordinate Newton step. For each group g the exact 1-D Newton step is

      Δθ_out_g = −F_g / F′_g
      where F_g  = Σ_h m_h W_gh − W_gg − s_out_g   (residual)
            F′_g = −(Σ_h m_h W_gh(1+W_gh) − W_gg(1+W_gg))   (diagonal Hessian, ≤ 0)
            W_gh = 1/expm1(θ_out_g + θ_in_h)

clipped to ``[−max_step, +max_step]`` and accelerated by Anderson mixing. Because the update is
additive in θ, β > 1 is never produced, which makes the iteration robust to high-strength hub nodes.
θ is clamped to [_ETA_MIN, _ETA_MAX] after each step to maintain the feasibility constraint
β_out_i · β_in_j < 1 for all i, j.
"""
from __future__ import annotations

import datetime
import math
import sys
import time

import torch

from .base import SolverResult
from dcms.utils.profiling import _PeakRAMMonitor
from dcms.models.parameters import _ETA_MIN, _ETA_MAX
# Maximum ℓ∞ norm of the Anderson FP-residual r_k = g(θ_k) − θ_k accepted
# into the mixing history.  Iterates beyond this threshold are skipped to
# prevent Anderson mixing from being contaminated by extreme values.  The
# threshold is most relevant for the theta-newton step, which can produce
# large transient steps during the initial phase when θ is far from the
# solution.
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

# θ-Newton z = theta_out_i + theta_in_j floor against the W = 1/expm1(z)
# singularity as z -> 0 (the same role DECM's `_Z_G_CLAMP` and qDECM's
# `_Z_G_CLAMP` play for their own z = eta_out_i + eta_in_j -- named here for
# the first time (2026-09-29 audit) purely for discoverability/consistency,
# same value as before this rename (no behaviour change). DWCM has never
# needed to relax this the way DECM's hub bisection does, so it has stayed
# a plain floor; the three models' values (DWCM 1e-15, qDECM 1e-8, DECM
# 1e-6) are NOT unified on purpose -- each was tuned independently for its
# own numerics and there is no evidence one value would suit all three.
_Z_G_CLAMP: float = 1e-15

# When the iteration stagnates (residual not improved by ≥ 1 % over this many
# consecutive iterations), run a Newton-Anderson mini-loop from best_theta to
# escape the non-contractive oscillation region (limit cycles on hub networks).
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

# Retry ladder for a diverging mini-loop correction (see the `if _fpgs_newton_fired:` block):
# a single fixed +-1e-6 relative jitter retry was NOT enough on the real failure this targets
# (ita_elections_dico1 at 16/24 CPU threads, 2026-10-07 remote report: the jittered retry's own
# divergence improved ~20x, 2.8e6 -> 1.1e5, but stayed ~5 orders of magnitude above the accept
# threshold) -- escalating RANDOM multiplicative noise, same shape as DECM's own validated
# perturbed-restart ladder (see fixed_point_decm.py's `noise_base`/`noise_growth`/`max_stalls`),
# replaces it. Values match DECM's own batch-recipe defaults for consistency, not because DWCM's
# dynamics are known to need exactly these -- retune if the real failure case still isn't fixed.
_FPGS_NEWTON_RETRY_NOISE_BASE: float = 2e-3
_FPGS_NEWTON_RETRY_NOISE_GROWTH: float = 2.0
_FPGS_NEWTON_RETRY_MAX: int = 5
# After this many consecutive firings in which ALL retries diverged, the mini-loop is disabled for the rest of the run: falling back to
# best_theta just re-enters the same cycle, so further firings only add up to _FPGS_NEWTON_RETRY_MAX * _FPGS_NEWTON_STEPS wasted steps each.
_FPGS_NEWTON_MAX_FAILED_FIRINGS: int = 3



def _anderson_mixing(
    fp_outputs: list[torch.Tensor],
    residuals_hist: list[torch.Tensor],
    weights: torch.Tensor | None = None,
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
        weights:        Optional per-row multiplicity weight, shape (2N,).
                        Used by the degeneracy-reduced (``mult``) path: a
                        group of ``mult[g]`` identical nodes contributes
                        ``mult[g]`` identical rows to this fit in the
                        unreduced system but only one row in the reduced
                        system, so without this weight the fit implicitly
                        under-counts high-multiplicity groups (see the
                        analogous DECM fix,
                        ``dcms.solvers.fixed_point_decm._anderson_mixing``).
                        ``None`` (default) reproduces the old behaviour.

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
    if weights is not None:
        RtR = R_w.T @ (weights[:, None] * R_w)
    else:
        RtR = R_w.T @ R_w  # (m, m)
    # Regularise to avoid singularity when recent steps are nearly collinear.
    RtR = RtR + 1e-10 * torch.eye(m, dtype=RtR.dtype, device=RtR.device)
    ones = torch.ones(m, dtype=RtR.dtype, device=RtR.device)
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


# -------------------------------------------------------------------------
# Degeneracy reduction (Vallarano et al., Sci Rep 11:15227, 2021)
# -------------------------------------------------------------------------
#
# Nodes sharing the exact same (s_out, s_in) pair provably share identical
# (theta_out, theta_in) at the DWCM's unique solution -- same label-swap
# symmetry + strict concavity argument as DCM's degeneracy (see
# compute_degeneracy_groups_dcm in fixed_point_dcm.py), specialised to
# strength constraints on the geometric-weight likelihood instead of degree
# constraints on the Bernoulli likelihood. A 2-tuple (s_out, s_in) is the
# right key here, mirroring DCM's (k_out, k_in) exactly -- DWCM has no
# degree channel to interleave with, unlike DECM's 4-tuple case.

def compute_degeneracy_groups_dwcm(
    s_out: torch.Tensor,
    s_in: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group nodes by identical (s_out, s_in).

    Returns:
        group_of: (N,) long tensor mapping each node to its group in [0, M).
        mult:     (M,) float tensor, multiplicity of each group.
        s_out_g, s_in_g: (M,) float tensors, the shared target values.
    """
    N = s_out.shape[0]
    key = torch.stack([s_out, s_in], dim=1)
    uniq, group_of = torch.unique(key, dim=0, return_inverse=True)
    M = uniq.shape[0]
    mult = torch.zeros(M, dtype=torch.float64)
    mult.scatter_add_(0, group_of, torch.ones(N, dtype=torch.float64))
    s_out_g, s_in_g = uniq[:, 0], uniq[:, 1]
    return group_of, mult, s_out_g, s_in_g


def _dwcm_step_dense_weighted(
    theta: torch.Tensor,
    s_out: torch.Tensor,
    s_in: torch.Tensor,
    max_step: float,
    mult: torch.Tensor,
    eta_lo: float = _ETA_MIN,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One theta-Newton (GS) step for the degeneracy-reduced DWCM.

    Identical algebra to :func:`_theta_newton_step_dense`, generalised so
    each group ``g`` stands in for ``mult[g]`` original nodes -- same
    "weighted sum minus one diagonal term" pattern used in DECM's
    ``_decm_step_dense_weighted`` and DCM's ``_dcm_step_dense_weighted``
    (see either docstring for the combinatorial justification). With
    ``mult`` all-ones (M == N) this reduces exactly to the per-node formula.

    Args:
        theta: Current 2M parameter vector [theta_out|theta_in].
        s_out, s_in: Group-level target sequences, shape (M,).
        max_step: Maximum |Delta theta| per group per step.
        mult: Group multiplicities, shape (M,).
        eta_lo: Lower clamp bound (see _theta_newton_step_dense).

    Returns:
        ``(theta_new, F_current)``, both shape (2M,).
    """
    M = s_out.shape[0]
    theta_out = theta[:M]
    theta_in = theta[M:]

    z = theta_out[:, None] + theta_in[None, :]
    W = 1.0 / torch.expm1(z.clamp(min=_Z_G_CLAMP))
    W_diag = W.diagonal()
    W1W = W * (1.0 + W)
    W1W_diag = W1W.diagonal()

    F_out = (W * mult[None, :]).sum(1) - W_diag - s_out
    # The returned residual must be F(theta) at the INPUT theta for both halves (it drives the convergence test and the best-iterate
    # record); F_in below is evaluated after the out half-step and is used only for the in Newton step.
    F_in_current = (W * mult[:, None]).sum(0) - W_diag - s_in
    h_out = -((W1W * mult[None, :]).sum(1) - W1W_diag)
    delta_out = (-F_out / (h_out - 1e-30)).clamp(-max_step, max_step)
    theta_out_new = (theta_out + delta_out).clamp(eta_lo, _ETA_MAX)
    theta_out_new = torch.where(s_out == 0, torch.full_like(theta_out_new, _ETA_MAX), theta_out_new)

    z2 = theta_out_new[:, None] + theta_in[None, :]
    W2 = 1.0 / torch.expm1(z2.clamp(min=_Z_G_CLAMP))
    W2_diag = W2.diagonal()
    W1W2 = W2 * (1.0 + W2)
    W1W2_diag = W1W2.diagonal()

    F_in = (W2 * mult[:, None]).sum(0) - W2_diag - s_in
    h_in = -((W1W2 * mult[:, None]).sum(0) - W1W2_diag)
    delta_in = (-F_in / (h_in - 1e-30)).clamp(-max_step, max_step)
    theta_in_new = (theta_in + delta_in).clamp(eta_lo, _ETA_MAX)
    theta_in_new = torch.where(s_in == 0, torch.full_like(theta_in_new, _ETA_MAX), theta_in_new)

    F_current = torch.cat([F_out, F_in_current])
    return torch.cat([theta_out_new, theta_in_new]), F_current


def _solve_dwcm_reduced(
    theta0: torch.Tensor,
    s_out: torch.Tensor,
    s_in: torch.Tensor,
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
    seed: int | None,
) -> SolverResult:
    """Newton/Anderson loop on the reduced system (M groups, multiplicities ``mult``); see :func:`solve_fixed_point_dwcm`."""
    _mini_loop_failed = 0          # consecutive firings in which every retry diverged
    _mini_loop_disabled = False
    _mini_loop_gen = torch.Generator()
    if seed is not None:
        _mini_loop_gen.manual_seed(seed)

    _dev = torch.device(device)
    _dtype = torch.float64 if _dev.type == "cpu" else torch.float32

    if not isinstance(s_out, torch.Tensor):
        s_out = torch.tensor(s_out, dtype=_dtype, device=_dev)
    else:
        s_out = s_out.to(dtype=_dtype, device=_dev)
    if not isinstance(s_in, torch.Tensor):
        s_in = torch.tensor(s_in, dtype=_dtype, device=_dev)
    else:
        s_in = s_in.to(dtype=_dtype, device=_dev)
    mult = mult.to(dtype=_dtype, device=_dev)


    if not isinstance(theta0, torch.Tensor):
        theta = torch.tensor(theta0, dtype=_dtype, device=_dev)
    else:
        theta = theta0.clone().to(dtype=_dtype, device=_dev)

    # Clamp initial theta to feasible range
    _eta_lo = _ETA_MIN
    theta = theta.clamp(_eta_lo, _ETA_MAX)

    _peak_ram_monitor = _PeakRAMMonitor()
    _peak_ram_monitor.__enter__()
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

    # Precompute verbose targets once (MRE = max |F_i| / s_i)
    _v_targets = torch.cat([s_out, s_in])
    _v_nonzero = _v_targets > 0

    # Anderson mixing needs per-component weights in the degeneracy-reduced
    # path -- see _anderson_mixing's docstring and the analogous DECM/DCM fix.
    _anderson_weights = torch.cat([mult, mult]) if weight_anderson else None

    try:
        for _ in range(max_iter):
            # Wall-clock time limit
            if max_time > 0 and (time.perf_counter() - t0) > max_time:
                message = (
                    f"Time limit ({max_time:.0f}s) reached at iteration {n_iter}."
                )
                break

            theta_fp, F_current = _dwcm_step_dense_weighted(
                theta, s_out, s_in, max_step, mult, eta_lo=_ETA_MIN
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
            # Relative: MRE = max|F_i|/target_i
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
                _mre = (
                    (F_current.abs()[_v_nonzero] / _v_targets[_v_nonzero]).max().item()
                    if _v_nonzero.any() else float("nan")
                )
                print(
                    f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] "
                    f"iteration={n_iter:5d}, "
                    f"elapsed time={int(_elapsed // 3600):4d}:{int((_elapsed % 3600) // 60):02d}:{int(_elapsed % 60):02d}, "
                    f"MRE_weights={_mre:.5e}",
                    end="\r" if monitor else "\n",
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
            #
            _fpgs_newton_fired = False
            if anderson_depth > 1:
                if res_norm < _fpgs_best_local * 0.99:
                    _fpgs_best_local = res_norm
                    _fpgs_stagnation_count = 0
                else:
                    _fpgs_stagnation_count += 1
                    if (
                        _fpgs_stagnation_count >= _FPGS_NEWTON_RESET_WINDOW
                        and res_norm > tol
                        and not _mini_loop_disabled
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
                    if math.isfinite(r_k_norm) and r_k_norm < _ANDERSON_MAX_NORM:
                        _and_g.append(theta_fp.clone())
                        _and_r.append(r_k.clone())

                    if len(_and_g) > anderson_depth:
                        _and_g.pop(0)
                        _and_r.pop(0)

                    if len(_and_g) >= 2:
                        theta_next = _anderson_mixing(_and_g, _and_r, weights=_anderson_weights)
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
                def _nt_step(th: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                    """One theta-Newton substep from `th`, with the mult-weighted step the main loop uses (the plain per-group
                    formula would silently drop each group's multiplicity and solve the wrong equations)."""
                    return _dwcm_step_dense_weighted(th, s_out, s_in, max_step, mult, eta_lo=_ETA_MIN)

                def _run_mini_loop(start: torch.Tensor) -> tuple[torch.Tensor, float]:
                    """30 steps of theta-Newton + Anderson(_FPGS_NEWTON_AND_DEPTH) from
                    `start`. Returns (theta_nt, nt_res_final) where nt_res_final is the
                    ACTUAL final output's residual -- not the stale nt_res from one substep
                    earlier, which belongs to theta_nt's PREVIOUS value unless the loop broke
                    out on the tol check below -- computed with one extra _nt_step call."""
                    theta_nt = start.clone()
                    _nt_and_g: list[torch.Tensor] = []
                    _nt_and_r: list[torch.Tensor] = []
                    for _ in range(_FPGS_NEWTON_STEPS):
                        theta_nt_fp, F_nt = _nt_step(theta_nt)
                        # Per-step floor (same as main loop)
                        _nt_floor = (theta_nt * _ANDERSON_THETA_FLOOR).clamp(min=_ETA_MIN)
                        theta_nt_fp = torch.maximum(theta_nt_fp, _nt_floor).clamp(
                            _ETA_MIN, _ETA_MAX
                        )
                        nt_res = (F_nt.abs()[_v_nonzero] / _v_targets[_v_nonzero]).max().item() if _v_nonzero.any() else 0.0
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
                            theta_nt_next = _anderson_mixing(
                                _nt_and_g, _nt_and_r, weights=_anderson_weights
                            )
                            eff_floor = torch.minimum(
                                _nt_floor, theta_nt_fp
                            )
                            theta_nt_next = torch.maximum(theta_nt_next, eff_floor).clamp(
                                _ETA_MIN, _ETA_MAX
                            )
                        else:
                            theta_nt_next = theta_nt_fp
                        theta_nt = theta_nt_next
                    _nt_fp_final, _F_nt_final = _nt_step(theta_nt)
                    nt_res_final = (
                        (_F_nt_final.abs()[_v_nonzero] / _v_targets[_v_nonzero]).max().item()
                        if _v_nonzero.any() else 0.0
                    )
                    return theta_nt, nt_res_final

                # Found empirically (2026-09-30, DWCM benchmark on a second machine): on a
                # network stuck in a near-exact floating-point limit cycle, this mini-loop's
                # own internal Anderson mixing can itself diverge -- observed growing
                # 0.98 -> 2.8e4 -> ... -> 2.8e7 over 4 substeps under a thread count (hence
                # float64 summation order) a different thread count did not trigger on the
                # exact same network/recipe. The old code accepted theta_nt unconditionally,
                # so a diverging mini-loop discarded 30 iterations of real progress and handed
                # the main loop a far worse starting point than it had before the "rescue" --
                # and also disarmed the main loop's own blowup guard right after (see below),
                # so nothing caught it.
                #
                # First attempt, from best_theta exactly. Accept unless it's a genuine blowup
                # (same _ANDERSON_BLOWUP_FACTOR tolerance the main loop already judges
                # divergence by, not a strict "must already beat the record" bar -- the
                # mini-loop's whole point is incremental progress a stagnant main loop could
                # not make, and rejecting anything that hasn't yet beaten best_theta_res
                # outright would also discard real, useful partial progress on every normal
                # firing, not just the pathological one this guard exists for).
                # Found 2026-10-07 (remote benchmark, 16/24 CPU threads -- a thread count we
                # cannot reproduce locally): on a network stuck in a near-EXACT limit cycle
                # (not just near-exact), falling straight back to best_theta here just
                # re-enters the identical cycle, burns another _FPGS_NEWTON_RESET_WINDOW
                # iterations counting back up to 30, and repeats indefinitely until the outer
                # give-up check kills the run -- a plain rollback cannot escape a genuine
                # limit cycle any more than it could in the main loop itself. A SINGLE fixed
                # deterministic jitter (+-1e-6 relative, tried first) was not enough: remote's
                # retest showed it only takes the divergence from 2.8e6 down to 1.1e5, still
                # ~5 orders of magnitude above the acceptance bar. Escalating to the same
                # perturbed-restart LADDER already validated in DECM's own solver (random,
                # log-scale multiplicative noise, growing geometrically, several attempts
                # before giving up) gives the symmetry break more room to actually clear the
                # cycle instead of landing on another point of it.
                for _attempt in range(_FPGS_NEWTON_RETRY_MAX):
                    if _attempt == 0:
                        _start = best_theta
                        _noise_scale = 0.0
                    else:
                        _noise_scale = _FPGS_NEWTON_RETRY_NOISE_BASE * (
                            _FPGS_NEWTON_RETRY_NOISE_GROWTH ** (_attempt - 1)
                        )
                        _noise = (
                            torch.randn(best_theta.shape, generator=_mini_loop_gen) * _noise_scale
                        ).to(device=best_theta.device, dtype=best_theta.dtype)
                        _start = (best_theta * torch.exp(_noise)).clamp(_ETA_MIN, _ETA_MAX)
                    theta_nt, nt_res_final = _run_mini_loop(_start)
                    if math.isfinite(nt_res_final) and nt_res_final <= _ANDERSON_BLOWUP_FACTOR * best_theta_res:
                        theta_next = theta_nt
                        _mini_loop_failed = 0
                        if verbose and _attempt > 0:
                            print(
                                f"[fpgs-newton] mini-loop correction diverged on attempt "
                                f"{_attempt}/{_FPGS_NEWTON_RETRY_MAX} -- retry {_attempt + 1} "
                                f"(noise_scale={_noise_scale:.1e}) succeeded "
                                f"(nt_res={nt_res_final:.3e})."
                            )
                        break
                    if verbose:
                        print(
                            f"[fpgs-newton] mini-loop correction attempt {_attempt + 1}/"
                            f"{_FPGS_NEWTON_RETRY_MAX} diverged "
                            f"(nt_res={nt_res_final:.3e} >> best={best_theta_res:.3e}, "
                            f"noise_scale={_noise_scale:.1e})."
                        )
                else:
                    if verbose:
                        print(
                            f"[fpgs-newton] mini-loop correction diverged on all "
                            f"{_FPGS_NEWTON_RETRY_MAX} attempts -- falling back to best_theta."
                        )
                    theta_next = best_theta.clone()
                    _mini_loop_failed += 1
                    if _mini_loop_failed >= _FPGS_NEWTON_MAX_FAILED_FIRINGS:
                        _mini_loop_disabled = True
                        if verbose:
                            print(
                                f"[fpgs-newton] {_mini_loop_failed} consecutive firings with every retry diverged "
                                f"-- mini-loop disabled for the rest of this run."
                            )
                # Reset blowup threshold so post-Newton residuals don't trigger
                # a false blowup on the very next main-loop iteration.
                _best_res_for_anderson = float("inf")

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


def solve_fixed_point_dwcm(
    theta0: "ArrayLike",  # type: ignore[name-defined]
    s_out: "ArrayLike",  # type: ignore[name-defined]
    s_in: "ArrayLike",  # type: ignore[name-defined]
    tol: float = 1e-8,
    max_iter: int = 10_000,
    anderson_depth: int = 10,
    max_step: float = 1.0,
    max_time: float = 0.0,
    verbose: bool = False,
    monitor: bool = False,
    weight_anderson: bool = True,
    device: str = "cpu",
    seed: int | None = None,
) -> SolverResult:
    """Degeneracy-reduced theta-Newton solver for the DWCM.

    Groups nodes that share the exact same (s_out, s_in) pair -- which are
    proven to share identical (theta_out, theta_in) at the true DWCM
    solution, see :func:`compute_degeneracy_groups_dwcm` -- and solves the
    resulting M-unknown reduced system (M = number of unique pairs <= N)
    The returned ``SolverResult`` is expanded back to the original per-node
    shape (2N,).

    Args:
        theta0: Initial guess [theta_out|theta_in], shape (2N,). Nodes
            sharing an (s_out,s_in) pair should already carry identical
            values here (true for the standard "degrees"-based initial
            guess); if not, one representative member's value is used for
            the whole group.
        s_out, s_in: Observed sequences, each shape (N,).
        tol, max_iter, anderson_depth, max_step, max_time, verbose, monitor, weight_anderson, device: as in
            :func:`~dcms.solvers.fixed_point_dcm.solve_fixed_point_dcm` (``device`` != ``"cpu"`` runs in float32).
        seed: Seed for the retry-noise RNG of the stagnation-escape mini-loop (see ``_FPGS_NEWTON_RETRY_NOISE_BASE`` and the
            ``if _fpgs_newton_fired:`` block); ``None`` (default) uses the fixed default state of a fresh
            ``torch.Generator``, so the run is deterministic anyway.

    Returns:
        :class:`~dcms.solvers.base.SolverResult` with ``theta``/``best_theta``
        expanded back to shape (2N,).
    """
    s_out = s_out if isinstance(s_out, torch.Tensor) else torch.tensor(s_out, dtype=torch.float64)
    s_in = s_in if isinstance(s_in, torch.Tensor) else torch.tensor(s_in, dtype=torch.float64)
    s_out = s_out.to(dtype=torch.float64)
    s_in = s_in.to(dtype=torch.float64)
    theta0 = (
        theta0 if isinstance(theta0, torch.Tensor) else torch.tensor(theta0, dtype=torch.float64)
    ).to(dtype=torch.float64)

    N = s_out.shape[0]
    group_of, mult, s_out_g, s_in_g = compute_degeneracy_groups_dwcm(s_out, s_in)
    M = s_out_g.shape[0]

    _first_idx = torch.zeros(M, dtype=torch.long)
    _seen = torch.zeros(M, dtype=torch.bool)
    _group_of_list = group_of.tolist()
    for _i, _g in enumerate(_group_of_list):
        if not _seen[_g]:
            _seen[_g] = True
            _first_idx[_g] = _i
    theta0_g = torch.cat([theta0[:N][_first_idx], theta0[N:][_first_idx]])

    result_g = _solve_dwcm_reduced(
        theta0_g, s_out_g, s_in_g, mult,
        tol=tol, max_iter=max_iter, anderson_depth=anderson_depth, max_step=max_step, max_time=max_time,
        verbose=verbose, monitor=monitor, weight_anderson=weight_anderson, device=device, seed=seed,
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
