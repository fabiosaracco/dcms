"""Directed approximated Enhanced Configuration Model (DaECM).

The DaECM fixes *four* sequences per node: out-degree, in-degree,
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

import math
from typing import Union

import torch

from src.models.dcm import DCMModel, _THETA_MAX

# Type alias for inputs: accept both numpy arrays and torch tensors.
_ArrayLike = Union[torch.Tensor, "numpy.ndarray"]  # type: ignore[name-defined]

# θ_β lower bound: ensures β = exp(−θ) < 1 and avoids div-by-zero in w_ij.
_ETA_MIN: float = 1e-10
# θ_β upper bound: exp(−50) ≈ 2e-22, essentially zero weight contribution.
_ETA_MAX: float = 50.0
# Maximum allowed β_out * β_in product; individual β may exceed 1 as long
# as the pairwise product stays below this threshold.
_Q_MAX: float = 0.9999

# For N > this threshold, use chunked computation to avoid N×N materialisation.
_LARGE_N_THRESHOLD: int = 2_000

# Number of rows processed per chunk when using memory-efficient mode.
_DEFAULT_CHUNK: int = 512


def _to_tensor(x: _ArrayLike, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Convert *x* to a float64 CPU torch.Tensor (no-copy if already correct)."""
    if isinstance(x, torch.Tensor):
        return x.to(device="cpu", dtype=dtype)
    return torch.tensor(x, dtype=dtype, device="cpu")


class DaECMModel:
    """Encapsulates the DaECM weight-step equations for a network of *N* nodes.

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

        The expected weight of arc i→j in the DaECM approximation is:

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

    def jacobian_strength(
        self,
        theta_topo: _ArrayLike,
        theta_weight: _ArrayLike,
    ) -> torch.Tensor:
        """Return the Jacobian J_w = ∂F_w/∂θ_β, shape (2N, 2N).

        Denoting H_ij = p_ij · G_ij · (G_ij − 1) (elementwise, diagonal zero),
        where G_ij = 1 / (1 − β_out_i β_in_j):

            J_out,out = −diag(Σ_{j≠i} H_ij)   [diagonal, negative]
            J_out,in  = −H                      [off-diagonal]
            J_in,out  = −Hᵀ                    [off-diagonal]
            J_in,in   = −diag(Σ_{j≠i} H_ji)   [diagonal, negative]

        Args:
            theta_topo:   Topology parameters, shape (2N,).
            theta_weight: Weight parameters, shape (2N,).

        Returns:
            Jacobian matrix, shape (2N, 2N), dtype torch.float64.
        """
        N = self.N
        theta_weight = _to_tensor(theta_weight)
        theta_b_out = theta_weight[:N]
        theta_b_in = theta_weight[N:]

        # G_ij = 1/(1 - β_out_i β_in_j) = -1/expm1(-z_ij)
        z = theta_b_out[:, None] + theta_b_in[None, :]  # (N, N)
        z_safe = z.clamp(min=1e-8)
        G = -1.0 / torch.expm1(-z_safe)
        G.fill_diagonal_(0.0)

        # H_ij = p_ij · G_ij · (G_ij − 1)  (= p_ij · G_new · G_old, diagonal zero)
        P = self.pij_matrix(theta_topo)
        H = P * G * (G - 1.0)   # H[i,i] = 0 since G[i,i]=0

        idx = torch.arange(N)
        J = torch.zeros(2 * N, 2 * N, dtype=torch.float64)
        J[idx, idx] = -H.sum(dim=1)        # top-left diagonal
        J[:N, N:] = -H                     # top-right off-diagonal
        J[N:, :N] = -H.T                  # bottom-left off-diagonal
        J[N + idx, N + idx] = -H.sum(dim=0)  # bottom-right diagonal
        return J

    # ------------------------------------------------------------------
    # Diagonal Hessian of the strength log-likelihood
    # ------------------------------------------------------------------

    def hessian_diag_strength(
        self,
        theta_topo: _ArrayLike,
        theta_weight: _ArrayLike,
    ) -> torch.Tensor:
        """Return the diagonal of Hess_w(L) = ∂²L_w/∂θ_β², shape (2N,).

        The entries are:

            ∂²L_w/∂θ_β_out_i² = −Σ_{j≠i} p_ij · G_ij · (G_ij − 1)
            ∂²L_w/∂θ_β_in_i²  = −Σ_{j≠i} p_ji · G_ji · (G_ji − 1)

        Args:
            theta_topo:   Topology parameters, shape (2N,).
            theta_weight: Weight parameters, shape (2N,).

        Returns:
            Diagonal of Hess_w, shape (2N,), all entries ≤ 0.
        """
        N = self.N
        theta_weight = _to_tensor(theta_weight)
        theta_b_out = theta_weight[:N]
        theta_b_in = theta_weight[N:]

        z = theta_b_out[:, None] + theta_b_in[None, :]
        z_safe = z.clamp(min=1e-8)
        G = -1.0 / torch.expm1(-z_safe)
        G.fill_diagonal_(0.0)

        P = self.pij_matrix(theta_topo)
        H = P * G * (G - 1.0)
        h_out = -H.sum(dim=1)
        h_in = -H.sum(dim=0)
        return torch.cat([h_out, h_in])

    # ------------------------------------------------------------------
    # Negative log-likelihood for the weight step (for L-BFGS)
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
        method: str = "strengths",
    ) -> torch.Tensor:
        """Return a sensible starting point θ_weight₀ for the weight solvers.

        Several initialisation strategies are supported:

        * ``"strengths"`` (default): β_i ≈ sqrt(s_i / (s_i + N − 1))
          (same mean-field as DWCM, ignoring the p_ij factor).
        * ``"topology"``: mean-field init β = sqrt(s/(s+k)) where k is the
          observed degree.  Uses k_out/k_in rather than N-1 so the prior
          is correct for sparse networks where hubs connect to k << N nodes.
        * ``"normalized"``: β_i^{out} = s_i^{out} / Σ_j s_j^{out}.
        * ``"uniform"``: all betas set to the median of the strengths init.
        * ``"random"``: uniform θ_β ∈ [0.1, 2.0].

        Zero-strength nodes always have θ_β = +_ETA_MAX (β = 0 exactly).

        Args:
            theta_topo: Current (or initial) topology parameters — used when
                        method relies on the DCM probability scale.
            method: Initialisation method name.

        Returns:
            Initial weight parameter vector θ_weight₀, shape (2N,).
        """
        N = self.N
        if method == "strengths":
            s_out_safe = self.s_out.clamp(min=1e-15)
            s_in_safe = self.s_in.clamp(min=1e-15)
            beta_out = torch.sqrt(s_out_safe / (s_out_safe + (N - 1)))
            beta_in = torch.sqrt(s_in_safe / (s_in_safe + (N - 1)))
        elif method == "normalized":
            S_out = self.s_out.sum().clamp(min=1e-15)
            S_in = self.s_in.sum().clamp(min=1e-15)
            beta_out = self.s_out.clamp(min=1e-15) / S_out
            beta_in = self.s_in.clamp(min=1e-15) / S_in
        elif method == "uniform":
            s_out_safe = self.s_out.clamp(min=1e-15)
            s_in_safe = self.s_in.clamp(min=1e-15)
            beta_ref_out = torch.sqrt(s_out_safe / (s_out_safe + (N - 1)))
            beta_ref_in = torch.sqrt(s_in_safe / (s_in_safe + (N - 1)))
            pos_out = beta_ref_out[~self.zero_s_out]
            pos_in = beta_ref_in[~self.zero_s_in]
            med_out = pos_out.median().item() if pos_out.numel() > 0 else 0.5
            med_in = pos_in.median().item() if pos_in.numel() > 0 else 0.5
            beta_out = torch.full((N,), med_out, dtype=torch.float64)
            beta_in = torch.full((N,), med_in, dtype=torch.float64)
        elif method == "topology":
            # Mean-field init for the new formula: s/(k) = 1/(1-β²) → β = sqrt(1 - k/s).
            # This is the correct prior for the DaECM weight step where
            # E[w_ij] = p_ij/(1 - β_out_i β_in_j).
            k_out_safe = self.k_out.clamp(min=1.0)
            k_in_safe = self.k_in.clamp(min=1.0)
            s_out_safe = self.s_out.clamp(min=1e-15)
            s_in_safe = self.s_in.clamp(min=1e-15)
            ratio_out = (k_out_safe / s_out_safe).clamp(max=1.0 - 1e-9)
            ratio_in = (k_in_safe / s_in_safe).clamp(max=1.0 - 1e-9)
            beta_out = torch.sqrt(1.0 - ratio_out)
            beta_in = torch.sqrt(1.0 - ratio_in)
        elif method == "balanced":
            # Uniform z0 init: place ALL pairs at z_ij = z0 at start so that
            # no pair is near the z=0 singularity (which freezes Newton steps).
            # For the new formula: G_new(z0) = mean_weight → z0 = -log(1 - 1/mw).
            # theta_b_out_i = theta_b_in_j = 0.5 * z0 for all non-zero nodes.
            k_total = float(self.k_out.double()[~self.zero_s_out].sum().clamp(min=1.0))
            s_total = float(self.s_out.double()[~self.zero_s_out].sum().clamp(min=1e-15))
            mean_weight = max(s_total / k_total, 1.0 + 1e-9)
            z0 = -math.log(1.0 - 1.0 / mean_weight)  # G_new(z0) = mean_weight
            half_z0 = z0 * 0.5
            theta_b_out = torch.full((N,), half_z0, dtype=torch.float64)
            theta_b_in = torch.full((N,), half_z0, dtype=torch.float64)
            theta_b_out = torch.where(
                self.zero_s_out, torch.full_like(theta_b_out, _ETA_MAX), theta_b_out
            )
            theta_b_in = torch.where(
                self.zero_s_in, torch.full_like(theta_b_in, _ETA_MAX), theta_b_in
            )
            return torch.cat([
                theta_b_out.clamp(-_ETA_MAX, _ETA_MAX),
                theta_b_in.clamp(-_ETA_MAX, _ETA_MAX),
            ])
        elif method == "random":
            theta_b_out = torch.empty(N, dtype=torch.float64).uniform_(0.1, 2.0)
            theta_b_in = torch.empty(N, dtype=torch.float64).uniform_(0.1, 2.0)
            theta_b_out = torch.where(
                self.zero_s_out,
                torch.full_like(theta_b_out, _ETA_MAX),
                theta_b_out,
            )
            theta_b_in = torch.where(
                self.zero_s_in,
                torch.full_like(theta_b_in, _ETA_MAX),
                theta_b_in,
            )
            return torch.cat([
                theta_b_out.clamp(_ETA_MIN, _ETA_MAX),
                theta_b_in.clamp(_ETA_MIN, _ETA_MAX),
            ])
        else:
            raise ValueError(f"Unknown initial-guess method: {method!r}")

        # Convert β → θ_β; allow β > 1 (θ_β < 0) for nodes whose solution
        # requires it (e.g. high s/k where the coupled β is small).
        beta_out = beta_out.clamp(min=1e-15)
        beta_in = beta_in.clamp(min=1e-15)
        theta_b_out = (-torch.log(beta_out)).clamp(-_ETA_MAX, _ETA_MAX)
        theta_b_in = (-torch.log(beta_in)).clamp(-_ETA_MAX, _ETA_MAX)

        # Zero-strength nodes: β = 0 exactly ↔ θ_β → +∞
        theta_b_out = torch.where(
            self.zero_s_out, torch.full_like(theta_b_out, _ETA_MAX), theta_b_out
        )
        theta_b_in = torch.where(
            self.zero_s_in, torch.full_like(theta_b_in, _ETA_MAX), theta_b_in
        )
        return torch.cat([theta_b_out, theta_b_in])

    # ------------------------------------------------------------------
    # Joint 4N residual and NLL (for full L-BFGS over all parameters)
    # ------------------------------------------------------------------

    def residual_joint(self, theta_full: _ArrayLike) -> torch.Tensor:
        """Return the full 4N residual [F_topo | F_strength].

        The parameter vector is ``theta_full = [θ_out | θ_in | θ_β_out | θ_β_in]``
        of length 4N.

        Args:
            theta_full: Joint parameter vector, shape (4N,).

        Returns:
            Residual vector, shape (4N,).
        """
        theta_full = _to_tensor(theta_full)
        N = self.N
        theta_topo = theta_full[:2 * N]
        theta_weight = theta_full[2 * N:]
        F_topo = self._dcm.residual(theta_topo)
        F_str = self.residual_strength(theta_topo, theta_weight)
        return torch.cat([F_topo, F_str])

    def neg_log_likelihood_joint(self, theta_full: _ArrayLike) -> float:
        """Return the joint negative log-likelihood −L_topo − L_weight.

        The topology NLL is the standard DCM NLL; the weight NLL is the
        conditioned DWCM NLL.  The parameter vector is
        ``theta_full = [θ_out | θ_in | θ_β_out | θ_β_in]`` of length 4N.

        Args:
            theta_full: Joint parameter vector, shape (4N,).

        Returns:
            Scalar −L(θ) to be minimised.
        """
        theta_full = _to_tensor(theta_full)
        N = self.N
        theta_topo = theta_full[:2 * N]
        theta_weight = theta_full[2 * N:]
        nll_topo = self._dcm.neg_log_likelihood(theta_topo)
        nll_str = self.neg_log_likelihood_strength(theta_topo, theta_weight)
        return nll_topo + nll_str

    def constraint_error_joint(self, theta_full: _ArrayLike) -> float:
        """Return max-abs residual over all 4N constraints.

        Args:
            theta_full: Joint parameter vector, shape (4N,).

        Returns:
            max|F(θ)| (scalar).
        """
        return self.residual_joint(theta_full).abs().max().item()

    # ------------------------------------------------------------------
    # Constraint evaluation
    # ------------------------------------------------------------------

    def constraint_error_topo(self, theta_topo: _ArrayLike) -> float:
        """Return the max-abs error on the topology (degree) constraints.

        Args:
            theta_topo: Topology parameters, shape (2N,).

        Returns:
            max|F_topo(θ)| (scalar).
        """
        return self._dcm.constraint_error(theta_topo)

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
