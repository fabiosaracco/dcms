"""Approximated Directed Enhanced Configuration Model (aDECM).

The aDECM fixes *four* sequences per node: out-degree, in-degree,
out-strength and in-strength (k_out, k_in, s_out, s_in).  It is solved in
two sequential steps:

1. **Topology step** — solve the DCM to find ``2N`` multipliers ``(x_i, y_i)``
   (equivalently ``θ_out_i, θ_in_i``) reproducing the degree sequences.  The
   resulting link probabilities are:

       p_ij = x_i · y_j / (1 + x_i · y_j)   (i ≠ j)

2. **Weight step** — solve a DWCM conditioned on the DCM topology to find
   ``2N`` additional multipliers ``(β_out_i, β_in_i)`` reproducing the strength
   sequences.  The expected weight of arc i→j is the DCM connection probability
   times the conditional expected weight given the link exists:

       E[w_ij] = p_ij · E[w_ij | a_ij = 1]
                = p_ij / (1 − β_out_i · β_in_j)

   leading to the strength equations:

       s_out_i = Σ_{j≠i} p_ij / (1 − β_out_i · β_in_j)
       s_in_i  = Σ_{j≠i} p_ji / (1 − β_out_j · β_in_i)

This class encapsulates *only* the weight step (step 2).  The topology
parameters ``theta_topo = [θ_out | θ_in]`` (from a DCM solver) are passed
as a fixed argument to every method; the unknowns are the weight parameters
``theta_weight = [θ_β_out | θ_β_in]`` where ``β = exp(−θ_β)``.

**Feasibility constraint:** β_out_i · β_in_j < 1 for all i ≠ j.  Keeping
all ``θ_β > 0`` (equivalently β < 1) is sufficient.

Reference:
    Vallarano, N. et al. (2021). Fast and scalable likelihood maximisation for
    exponential random graph models with local constraints.  *Scientific
    Reports*, 11, 15227.  https://doi.org/10.1038/s41598-021-94118-5
"""
from __future__ import annotations

from typing import Union

import torch
import functools

from dcms.solvers.base import SolverResult

from dcms.models.dcm import DCMModel, _THETA_MAX

# Type alias for inputs: accept both numpy arrays and torch tensors.
_ArrayLike = Union[torch.Tensor, "numpy.ndarray"]  # type: ignore[name-defined]

from dcms.models.parameters import aDECM_LARGE_N_THRESHOLD as _LARGE_N_THRESHOLD
from dcms.models.parameters import _DEFAULT_CHUNK, _ETA_MIN, _ETA_MAX


# Maximum allowed β_out * β_in product; individual β may exceed 1 as long
# as the pairwise product stays below this threshold.
_Q_MAX: float = 0.9999



def _to_tensor(x: _ArrayLike, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Convert *x* to a float64 CPU torch.Tensor (no-copy if already correct)."""
    if isinstance(x, torch.Tensor):
        return x.to(device="cpu", dtype=dtype)
    return torch.tensor(x, dtype=dtype, device="cpu")


class ADECMModel:
    """Encapsulates the aDECM weight-step equations for a network of *N* nodes.

    The topology step (DCM) is handled separately via the ``DCMModel`` class.
    This class provides residuals, Jacobians and initialisations for the
    *weight step only*: given fixed topology parameters ``theta_topo``, find
    ``theta_weight`` such that the strength constraints are satisfied.

    Args:
        k_out: Observed out-degree sequence, shape (N,).
        k_in:  Observed in-degree sequence, shape (N,).
        s_out: Observed out-strength sequence, shape (N,).
        s_in:  Observed in-strength sequence, shape (N,).
    """

    def __init__(
        self,
        k_out: _ArrayLike,
        k_in: _ArrayLike,
        s_out: _ArrayLike,
        s_in: _ArrayLike,
    ) -> None:
        self.k_out: torch.Tensor = _to_tensor(k_out)
        self.k_in: torch.Tensor = _to_tensor(k_in)
        self.s_out: torch.Tensor = _to_tensor(s_out)
        self.s_in: torch.Tensor = _to_tensor(s_in)
        self.N: int = int(self.k_out.shape[0])
        if any(
            t.shape[0] != self.N for t in [self.k_in, self.s_out, self.s_in]
        ):
            raise ValueError(
                "k_out, k_in, s_out and s_in must all have the same length."
            )
        # Internal DCM model for the topology step.
        self._dcm: DCMModel = DCMModel(k_out, k_in)
        # Nodes with zero strength (β = 0 exactly, θ_β → +∞).
        self.zero_s_out: torch.Tensor = (self.s_out == 0)
        self.zero_s_in: torch.Tensor = (self.s_in == 0)
        self.sol_topo: SolverResult | None = None
        self.sol_weights: SolverResult | None = None

    # ------------------------------------------------------------------
    # Core matrices
    # ------------------------------------------------------------------

    def pij_matrix(self, theta_topo: _ArrayLike) -> torch.Tensor:
        """Return the N×N DCM probability matrix p_ij.

        Args:
            theta_topo: Topology parameters [θ_out | θ_in], shape (2N,).

        Returns:
            Probability matrix P, shape (N, N), diagonal zero.
        """
        return self._dcm.pij_matrix(theta_topo)

    def wij_matrix_conditioned(
        self,
        theta_topo: _ArrayLike,
        theta_weight: _ArrayLike,
    ) -> torch.Tensor:
        """Return the N×N conditioned expected-weight matrix.

        The expected weight of arc i→j in the aDECM approximation is:

            W_ij = p_ij · β_out_i · β_in_j / (1 − β_out_i · β_in_j)
                 = p_ij / expm1(θ_β_out_i + θ_β_in_j)

        Diagonal entries are zero (no self-loops).

        Args:
            theta_topo:   Topology parameters [θ_out | θ_in], shape (2N,).
            theta_weight: Weight parameters [θ_β_out | θ_β_in], shape (2N,).
                          All entries must be strictly positive.

        Returns:
            Conditioned weight matrix W, shape (N, N), dtype torch.float64.
        """
        theta_weight = _to_tensor(theta_weight)
        N = self.N
        theta_b_out = theta_weight[:N]
        theta_b_in = theta_weight[N:]

        # G_ij = 1 / (1 - β_out_i β_in_j) = -1/expm1(-z_ij)
        z = theta_b_out[:, None] + theta_b_in[None, :]  # (N, N)
        z_safe = z.clamp(min=1e-8)
        G = -1.0 / torch.expm1(-z_safe)                 # (N, N)
        G.fill_diagonal_(0.0)

        # Apply zero-strength masks
        if self.zero_s_out.any():
            G[self.zero_s_out] = 0.0
        if self.zero_s_in.any():
            G[:, self.zero_s_in] = 0.0

        # Multiply elementwise by DCM probability
        P = self.pij_matrix(theta_topo)  # (N, N), diagonal already 0
        W = P * G
        return W

    # ------------------------------------------------------------------
    # Residual of the strength equations
    # ------------------------------------------------------------------

    def residual_strength(
        self,
        theta_topo: _ArrayLike,
        theta_weight: _ArrayLike,
        P: "Optional[torch.Tensor]" = None,  # type: ignore[name-defined]
    ) -> torch.Tensor:
        """Return the strength-equation residual F_w(θ_β).

        F_w = [s_out_expected − s_out_obs | s_in_expected − s_in_obs]

        For N > ``_LARGE_N_THRESHOLD`` the computation is done in row chunks
        to avoid materialising the full N×N matrix.

        Args:
            theta_topo:   Topology parameters, shape (2N,).
            theta_weight: Weight parameters [θ_β_out | θ_β_in], shape (2N,).
            P:            Optional pre-computed DCM probability matrix (N, N).
                          If provided, avoids recomputing ``p_ij`` each call.

        Returns:
            Residual vector F_w, shape (2N,).
        """
        if self.N > _LARGE_N_THRESHOLD:
            return self._residual_strength_chunked(theta_topo, theta_weight)
        if P is not None:
            # Fast path: use pre-computed P
            P_mat = P if isinstance(P, torch.Tensor) else torch.tensor(P, dtype=torch.float64)
            theta_weight = _to_tensor(theta_weight)
            N = self.N
            tb_out = theta_weight[:N]
            tb_in = theta_weight[N:]
            z = tb_out[:, None] + tb_in[None, :]
            z_safe = z.clamp(min=1e-8)
            G = -1.0 / torch.expm1(-z_safe)
            G.fill_diagonal_(0.0)
            if self.zero_s_out.any():
                G[self.zero_s_out] = 0.0
            if self.zero_s_in.any():
                G[:, self.zero_s_in] = 0.0
            W = P_mat * G
            s_out_hat = W.sum(dim=1)
            s_in_hat = W.sum(dim=0)
            F = torch.empty(2 * N, dtype=torch.float64)
            F[:N] = s_out_hat - self.s_out
            F[N:] = s_in_hat - self.s_in
            return F
        W = self.wij_matrix_conditioned(theta_topo, theta_weight)
        s_out_hat = W.sum(dim=1)   # expected out-strength
        s_in_hat = W.sum(dim=0)    # expected in-strength
        F = torch.empty(2 * self.N, dtype=torch.float64)
        F[: self.N] = s_out_hat - self.s_out
        F[self.N :] = s_in_hat - self.s_in
        return F

    def _residual_strength_chunked(
        self,
        theta_topo: _ArrayLike,
        theta_weight: _ArrayLike,
        chunk_size: int = _DEFAULT_CHUNK,
    ) -> torch.Tensor:
        """Compute strength residual without materialising the full N×N matrix.

        Args:
            theta_topo:   Topology parameters, shape (2N,).
            theta_weight: Weight parameters, shape (2N,).
            chunk_size:   Rows per processing chunk (must be ≥ 1).

        Returns:
            Residual vector F_w, shape (2N,).
        """
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be ≥ 1, got {chunk_size}")
        theta_topo = _to_tensor(theta_topo)
        theta_weight = _to_tensor(theta_weight)
        N = self.N

        theta_topo_out = theta_topo[:N]
        theta_topo_in = theta_topo[N:]
        theta_b_out = theta_weight[:N]
        theta_b_in = theta_weight[N:]

        s_out_hat = torch.zeros(N, dtype=torch.float64)
        s_in_hat = torch.zeros(N, dtype=torch.float64)

        for i_start in range(0, N, chunk_size):
            i_end = min(i_start + chunk_size, N)
            chunk_len = i_end - i_start

            # DCM probabilities for chunk of rows
            log_xy = (
                -theta_topo_out[i_start:i_end, None]
                - theta_topo_in[None, :]
            )  # (chunk, N)
            p_chunk = torch.sigmoid(log_xy)

            # DWCM G factors for chunk of rows
            z_chunk = (
                theta_b_out[i_start:i_end, None] + theta_b_in[None, :]
            )  # (chunk, N)
            z_safe = z_chunk.clamp(min=1e-8)
            g_chunk = -1.0 / torch.expm1(-z_safe)  # (chunk, N): G_new = 1/(1-exp(-z))

            w_chunk = p_chunk * g_chunk           # (chunk, N)

            # Zero out diagonal entries (i == j)
            local_idx = torch.arange(chunk_len, dtype=torch.long)
            global_idx = torch.arange(i_start, i_end, dtype=torch.long)
            w_chunk[local_idx, global_idx] = 0.0

            # Apply zero-degree (topology) and zero-strength masks
            if self._dcm.zero_out[i_start:i_end].any():
                w_chunk[self._dcm.zero_out[i_start:i_end]] = 0.0
            if self._dcm.zero_in.any():
                w_chunk[:, self._dcm.zero_in] = 0.0
            if self.zero_s_out[i_start:i_end].any():
                w_chunk[self.zero_s_out[i_start:i_end]] = 0.0
            if self.zero_s_in.any():
                w_chunk[:, self.zero_s_in] = 0.0

            s_out_hat[i_start:i_end] = w_chunk.sum(dim=1)
            s_in_hat += w_chunk.sum(dim=0)

        F = torch.empty(2 * N, dtype=torch.float64)
        F[:N] = s_out_hat - self.s_out
        F[N:] = s_in_hat - self.s_in
        return F

    # ------------------------------------------------------------------
    # Jacobian of the strength residual
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Negative log-likelihood for the weight step
    # ------------------------------------------------------------------

    def neg_log_likelihood_strength(
        self,
        theta_topo: _ArrayLike,
        theta_weight: _ArrayLike,
    ) -> float:
        """Return −L_w(θ_β), the quantity to be minimised by L-BFGS.

        The weight-step log-likelihood is:

            L_w(θ_β) = −Σ_i θ_β_out_i · s_out_i
                       − Σ_i θ_β_in_i · s_in_i
                       + Σ_{i≠j} p_ij · log(1 − exp(−θ_β_out_i − θ_β_in_j))

        so

            −L_w = Σ_i θ_β_out_i · s_out_i + Σ_i θ_β_in_i · s_in_i
                   − Σ_{i≠j} p_ij · log(1 − exp(−θ_β_out_i − θ_β_in_j))

        For N > ``_LARGE_N_THRESHOLD`` the computation is done in row chunks.

        Args:
            theta_topo:   Topology parameters, shape (2N,).
            theta_weight: Weight parameters, shape (2N,).

        Returns:
            Scalar −L_w(θ_β) (convex, to be minimised).
        """
        if self.N > _LARGE_N_THRESHOLD:
            return self._neg_log_likelihood_strength_chunked(
                theta_topo, theta_weight
            )
        theta_weight = _to_tensor(theta_weight)
        N = self.N
        theta_b_out = theta_weight[:N]
        theta_b_in = theta_weight[N:]

        P = self.pij_matrix(theta_topo)  # (N, N), diagonal 0
        k_out_exp = P.sum(dim=1)   # E[k_out_i] from DCM
        k_in_exp = P.sum(dim=0)    # E[k_in_j]  from DCM

        z = theta_b_out[:, None] + theta_b_in[None, :]  # (N, N)
        z_safe = z.clamp(min=1e-8)
        # log G_new = −log(1 − exp(−z))
        log_G_new = -torch.log1p(-torch.exp(-z_safe))   # (N, N), ≥ 0
        log_G_new.fill_diagonal_(0.0)

        weighted_log = P * log_G_new
        weighted_log.fill_diagonal_(0.0)

        # NLL = θ·(s − k_exp) + Σ p·log G_new  →  d(NLL)/dθ = −F_new  ✓
        dot_term = theta_b_out @ (self.s_out - k_out_exp) + theta_b_in @ (self.s_in - k_in_exp)
        return (dot_term + weighted_log.sum()).item()

    def _neg_log_likelihood_strength_chunked(
        self,
        theta_topo: _ArrayLike,
        theta_weight: _ArrayLike,
        chunk_size: int = _DEFAULT_CHUNK,
    ) -> float:
        """Compute −L_w without materialising the full N×N matrix."""
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be ≥ 1, got {chunk_size}")
        theta_topo = _to_tensor(theta_topo)
        theta_weight = _to_tensor(theta_weight)
        N = self.N

        theta_topo_out = theta_topo[:N]
        theta_topo_in = theta_topo[N:]
        theta_b_out = theta_weight[:N]
        theta_b_in = theta_weight[N:]

        k_out_exp = torch.zeros(N, dtype=torch.float64)
        k_in_exp = torch.zeros(N, dtype=torch.float64)
        log_total = 0.0

        for i_start in range(0, N, chunk_size):
            i_end = min(i_start + chunk_size, N)
            chunk_len = i_end - i_start

            log_xy = (
                -theta_topo_out[i_start:i_end, None]
                - theta_topo_in[None, :]
            )
            p_chunk = torch.sigmoid(log_xy)  # (chunk, N)

            z_chunk = (
                theta_b_out[i_start:i_end, None] + theta_b_in[None, :]
            )  # (chunk, N)
            z_safe = z_chunk.clamp(min=1e-8)
            # log G_new = −log1p(−exp(−z))
            log_G_chunk = -torch.log1p(-torch.exp(-z_safe))  # (chunk, N) ≥ 0

            local_idx = torch.arange(chunk_len, dtype=torch.long)
            global_idx = torch.arange(i_start, i_end, dtype=torch.long)
            log_G_chunk[local_idx, global_idx] = 0.0
            p_chunk_clean = p_chunk.clone()
            p_chunk_clean[local_idx, global_idx] = 0.0

            k_out_exp[i_start:i_end] = p_chunk_clean.sum(dim=1)
            k_in_exp += p_chunk_clean.sum(dim=0)
            log_total += (p_chunk_clean * log_G_chunk).sum().item()

        # NLL = θ·(s − k_exp) + Σ p·log G_new
        dot_term = (
            theta_b_out @ (self.s_out - k_out_exp)
            + theta_b_in @ (self.s_in - k_in_exp)
        ).item()
        return dot_term + log_total

    # ------------------------------------------------------------------
    # Initial-guess utilities
    # ------------------------------------------------------------------

    def initial_theta_topo(self, method: str = "degrees") -> torch.Tensor:
        """Return an initial guess for the topology parameters θ_topo.

        Delegates to :meth:`DCMModel.initial_theta`.

        Args:
            method: ``"degrees"`` (default) or ``"random"``.

        Returns:
            Initial parameter vector θ_topo, shape (2N,).
        """
        return self._dcm.initial_theta(method)

    def initial_theta_weight(
        self,
        theta_topo: _ArrayLike,
        method: str = "topology",
    ) -> torch.Tensor:
        """Return a sensible starting point θ_weight₀ for the weight solvers.

        All strategies are topology-aware: they use the observed degree sequences
        (and optionally the DCM probability matrix p_ij) to estimate β values
        consistent with the aDECM weight equation E[w_ij] = p_ij/(1−β_out_i β_in_j).

        * ``"topology"`` (default): β = sqrt(1 − k/s) per node, derived from the
          mean-field identity s_i/k_i ≈ 1/(1−β²).
        * ``"topology_node"``: per-node Newton solve.  For each node i, solves
          D_i(β_out_i) = s_out_i exactly (given β_in fixed at ``"topology"``
          values) via 5 Newton iterations.  O(N²) total; gives the most accurate
          starting point.

        Zero-strength nodes always have θ_β = +_ETA_MAX (β = 0 exactly).

        Args:
            theta_topo: Current (or initial) topology parameters [θ_out|θ_in],
                        shape (2N,).  Used by ``"topology_scale"`` and
                        ``"topology_node"`` to evaluate p_ij.
            method: Initialisation method name (see above).

        Returns:
            Initial weight parameter vector θ_weight₀, shape (2N,).
        """
        theta_topo = _to_tensor(theta_topo)
        N = self.N
        k_out_safe = self.k_out.clamp(min=1.0)
        k_in_safe = self.k_in.clamp(min=1.0)
        s_out_safe = self.s_out.clamp(min=1e-15)
        s_in_safe = self.s_in.clamp(min=1e-15)
        ratio_out = (k_out_safe / s_out_safe).clamp(max=1.0 - 1e-9)
        ratio_in = (k_in_safe / s_in_safe).clamp(max=1.0 - 1e-9)

        if method == "topology":
            # β = sqrt(1 - k/s): mean-field inversion of s = k/(1-β²)
            beta_out = torch.sqrt(1.0 - ratio_out)
            beta_in = torch.sqrt(1.0 - ratio_in)

        elif method == "topology_node":
            # Per-node Newton solve: for each i, solve Σ_j p_ij/(1-β_out_i·b_j) = s_out_i
            # given b_j = β_in_j^0 from "topology".  5 Newton steps per node.
            b_in = torch.sqrt(1.0 - ratio_in).clamp(min=1e-15, max=1.0 - 1e-9)
            b_out = torch.sqrt(1.0 - ratio_out).clamp(min=1e-15, max=1.0 - 1e-9)
            beta_out = b_out.clone()
            beta_in = b_in.clone()  # β_in solved symmetrically using p_ji and b_out
            theta_out_t = theta_topo[:N]
            theta_in_t = theta_topo[N:]
            for _ in range(5):
                # Solve for β_out given b_in fixed (chunked to avoid N×N alloc at large N)
                F_i = torch.zeros(N, dtype=torch.float64)
                Fp_i = torch.zeros(N, dtype=torch.float64)
                D_in = torch.zeros(N, dtype=torch.float64)
                Dp_in = torch.zeros(N, dtype=torch.float64)
                for i_start in range(0, N, _DEFAULT_CHUNK):
                    i_end = min(i_start + _DEFAULT_CHUNK, N)
                    chunk_len = i_end - i_start
                    local_idx = torch.arange(chunk_len, dtype=torch.long)
                    global_idx = torch.arange(i_start, i_end, dtype=torch.long)
                    p_chunk = torch.sigmoid(-theta_out_t[i_start:i_end, None] - theta_in_t[None, :])
                    p_chunk[local_idx, global_idx] = 0.0
                    z_chunk = (beta_out[i_start:i_end, None] * b_in[None, :]).clamp(max=_Q_MAX)
                    G_chunk = 1.0 / (1.0 - z_chunk)
                    G_chunk[local_idx, global_idx] = 0.0
                    F_i[i_start:i_end] = (p_chunk * G_chunk).sum(1) - s_out_safe[i_start:i_end]
                    dG_chunk = G_chunk ** 2 * b_in[None, :]
                    dG_chunk[local_idx, global_idx] = 0.0
                    Fp_i[i_start:i_end] = (p_chunk * dG_chunk).sum(1)
                    # Accumulate D_in for β_in solve (same pass)
                    z_in_chunk = (b_out[i_start:i_end, None] * beta_in[None, :]).clamp(max=_Q_MAX)
                    Gi_chunk = 1.0 / (1.0 - z_in_chunk)
                    Gi_chunk[local_idx, global_idx] = 0.0
                    D_in += (p_chunk * Gi_chunk).sum(0)
                    Dp_in += (p_chunk * Gi_chunk ** 2 * b_out[i_start:i_end, None]).sum(0)
                beta_out = (beta_out - F_i / Fp_i.clamp(min=1e-15)).clamp(min=1e-15, max=1.0 - 1e-9)
                F_in = D_in - s_in_safe
                beta_in = (beta_in - F_in / Dp_in.clamp(min=1e-15)).clamp(min=1e-15, max=1.0 - 1e-9)

        else:
            raise ValueError(f"Unknown initial-guess method: {method!r}")

        # Convert β → θ_β
        beta_out = beta_out.clamp(min=1e-15)
        beta_in = beta_in.clamp(min=1e-15)
        theta_b_out = (-torch.log(beta_out)).clamp(-_ETA_MAX, _ETA_MAX)
        theta_b_in = (-torch.log(beta_in)).clamp(-_ETA_MAX, _ETA_MAX)
        # Zero-strength nodes: β = 0 exactly ↔ θ_β → +∞
        theta_b_out = torch.where(self.zero_s_out, torch.full_like(theta_b_out, _ETA_MAX), theta_b_out)
        theta_b_in = torch.where(self.zero_s_in, torch.full_like(theta_b_in, _ETA_MAX), theta_b_in)
        return torch.cat([theta_b_out, theta_b_in])

    # ------------------------------------------------------------------
    # Evaluation of constraint satisfaction: topology
    # ------------------------------------------------------------------

    def constraint_error_topology(self, theta: _ArrayLike) -> float:
        """Return the maximum absolute error on all topology constraints.

        Args:
            theta: Parameter vector, shape (2N,).

        Returns:
            Max-abs constraint error (scalar).
        """
        return self._dcm.constraint_error(theta)


    # ------------------------------------------------------------------
    # Constraint evaluation: strength
    # ------------------------------------------------------------------

    def constraint_error_strength(
        self,
        theta_topo: _ArrayLike,
        theta_weight: _ArrayLike,
    ) -> float:
        """Return the max-abs error on the strength constraints.

        Args:
            theta_topo:   Topology parameters, shape (2N,).
            theta_weight: Weight parameters, shape (2N,).

        Returns:
            max|F_w(θ_β)| (scalar).
        """
        return self.residual_strength(theta_topo, theta_weight).abs().max().item()

    def max_relative_error(
        self,
        theta_topo: _ArrayLike,
        theta_weight: _ArrayLike,
    ) -> float:
        """Return the max relative error over all non-zero constraints.

        Evaluates both topology (degree) and strength constraints:

            MRE_topo  = max_{i: k_i>0}  |F_topo_i| / k_i
            MRE_str   = max_{i: s_i>0}  |F_str_i|  / s_i
            MRE       = max(MRE_topo, MRE_str)

        Args:
            theta_topo:   Topology parameters, shape (2N,).
            theta_weight: Weight parameters, shape (2N,).

        Returns:
            Maximum relative constraint error (scalar), or 0.0 if all
            constraints are trivially zero.
        """
        # Topology MRE
        F_topo = self._dcm.residual(theta_topo).abs()
        targets_topo = torch.cat([self.k_out, self.k_in]).to(torch.float64)
        nonzero_topo = targets_topo > 0
        if nonzero_topo.any():
            mre_topo = (F_topo[nonzero_topo] / targets_topo[nonzero_topo]).max().item()
        else:
            mre_topo = 0.0

        # Strength MRE
        F_str = self.residual_strength(theta_topo, theta_weight).abs()
        targets_str = torch.cat([self.s_out, self.s_in]).to(torch.float64)
        nonzero_str = targets_str > 0
        if nonzero_str.any():
            mre_str = (F_str[nonzero_str] / targets_str[nonzero_str]).max().item()
        else:
            mre_str = 0.0

        return max(mre_topo, mre_str)

    # ------------------------------------------------------------------
    # Using the solve function
    # ------------------------------------------------------------------

    def solve_tool(self, ic_topo:str='degrees', ic_weights:str='topology', tol:float=1e-6, max_iter:int=2000, max_time:int=0, variant:str='theta-newton', anderson_depth:int=10, backend:str='auto', num_threads:int=0, verbose:bool=False)-> SolverResult:
        """Select an initial condition on thetas and solve the equation, using the fixed-point solvers.

        Args:
            ic_topo (str): the initial condition on theta for the topology. Default="degrees", another possible choice is "random".
            ic_wei (str): the initial condition on theta for the weights. Default="topology", another possible choice is "random".
            tol (float): the maximum tolerance allowed on the residual. Default=1e-6.
            max_iter (int): the maximum number of iterations. Default=2000.
            variant (str): the numerical method implemented. Default="theta-newton", another possible choice is "gauss-seidel".
            anderson_depth (int): Anderson acceleration depth. Default=10.
            backend (str): Compute backend: ``"auto"`` (default), ``"pytorch"``,
                or ``"numba"``.  ``"auto"`` uses PyTorch for N ≤ 5 000 and
                Numba for larger networks.
            num_threads (int): Number of Numba parallel threads. 0 (default)
                means *auto*: uses all CPUs available to the current process
                (respects ``taskset``/cgroup quotas on Linux). Positive values
                are clamped to the available CPU count to avoid thread-creation
                errors on resource-limited servers. Only has effect when Numba
                is selected as the backend.
            verbose (bool): If ``True``, print a progress line at every
                iteration (timestamp, iteration count, elapsed time, MRE)
                for both the topology and weight steps. Default=False.

        Returns:
            :class:`~src.solvers.base.SolverResult` instance.
        """
        # Step 1: solve the DCM topology
        self.ic_topo=self.initial_theta_topo(ic_topo)
        from dcms.solvers.fixed_point_dcm import solve_fixed_point_dcm  # lazy import to avoid circular dependency
        self.sol_topo = solve_fixed_point_dcm(self._dcm.residual, self.ic_topo, self.k_out, self.k_in, tol=tol, max_iter=max_iter, max_time=max_time, variant=variant, anderson_depth=anderson_depth, backend=backend, num_threads=num_threads, verbose=verbose)
        
        if len(self.sol_topo.message)>0:
            print(f'Topology: {self.sol_topo.message}')

       

        # Step 2: solve the conditioned weight equations
        self.ic_weig = self.initial_theta_weight(theta_topo = self.sol_topo.theta, method=ic_weights)

        # Build the residual function that fixes theta_topo
        res_weight = functools.partial(self.residual_strength, theta_topo=self.sol_topo.theta)

        from dcms.solvers.fixed_point_adecm import solve_fixed_point_adecm  # lazy import to avoid circular dependency
        self.sol_weights = solve_fixed_point_adecm(res_weight, self.ic_weig, self.s_out, self.s_in, theta_topo=self.sol_topo.theta, P=None, tol=tol, max_iter=max_iter, max_time=max_time, variant=variant, anderson_depth=anderson_depth, backend=backend, num_threads=num_threads, verbose=verbose)
        if len(self.sol_weights.message)>0:
            print(f'Weights: {self.sol_weights.message}')

        return self.sol_topo.converged and self.sol_weights.converged

    def sample(self, seed: int | None = None, chunk_size: int = 512) -> list:
        """Sample a weighted directed network from the fitted aDECM.

        Two-step procedure mirroring the aDECM factorisation:

        1. **Topology** — draw ``A_ij ~ Bernoulli(p_ij)`` where
           ``p_ij = x_i y_j / (1 + x_i y_j)`` comes from the DCM solution.
        2. **Weights** — for each present link draw ``w_ij`` from a geometric
           distribution starting at 1 (the conditional distribution given the
           link exists)::

               P(w_ij = k | A_ij = 1) = (1 − β_ij) β_ij^{k−1},   k = 1, 2, …
               β_ij = β_out_i β_in_j = exp(−η_out_i − η_in_j)

        Args:
            seed: Random seed for reproducibility.
            chunk_size: Number of source rows processed at a time.

        Returns:
            Weighted edge list: list of ``[source, target, weight]`` integer triples.

        Raises:
            RuntimeError: if :meth:`solve_tool` has not been called yet.
        """
        if self.sol_topo is None or self.sol_weights is None:
            raise RuntimeError("Call solve_tool() first.")
        import numpy as np
        rng = np.random.default_rng(seed)
        N = self.N
        theta_topo = np.asarray(self.sol_topo.theta, dtype=np.float64)
        theta_out  = theta_topo[:N]
        theta_in   = theta_topo[N:]
        theta_w   = np.asarray(self.sol_weights.theta, dtype=np.float64)
        beta_out  = np.exp(-theta_w[:N])
        beta_in   = np.exp(-theta_w[N:])
        edges: list = []
        for i_start in range(0, N, chunk_size):
            i_end = min(i_start + chunk_size, N)
            # Step 1: topology
            logit = -theta_out[i_start:i_end, None] - theta_in[None, :]
            p = 1.0 / (1.0 + np.exp(-logit))
            for k, i in enumerate(range(i_start, i_end)):
                p[k, i] = 0.0
            A = rng.random(p.shape) < p  # (chunk, N) bool
            # Step 2: weights on present links — Geom(1-β) starting at 1
            b = (beta_out[i_start:i_end, None] * beta_in[None, :]).clip(0.0, 1.0 - 1e-12)
            rows, cols = np.where(A)
            for k, j in zip(rows, cols):
                w = int(rng.geometric(1.0 - b[k, j]))
                edges.append([i_start + int(k), int(j), w])
        return edges

