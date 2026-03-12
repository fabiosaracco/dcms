"""Directed Weighted Configuration Model (DWCM) — weights on directed arcs.

The DWCM fixes the out- and in-strength sequences (s_out, s_in).  The
maximum-entropy expected weight of arc i→j is given by a geometric distribution:

    w_ij = β_out_i * β_in_j / (1 − β_out_i * β_in_j)   (i ≠ j)

where β_out_i = exp(−θ_out_i) and β_in_j = exp(−θ_in_j) are the Lagrange
multipliers in exponential parametrisation.  All β must satisfy 0 < β < 1.

Feasibility constraint: β_out_i * β_in_j < 1 for all i ≠ j.  Since all
θ_i > 0 ensures β_i < 1, keeping θ ≥ _ETA_MIN > 0 is sufficient.

The system of equations to solve is F(θ) = 0, where

    F_i(θ)     = Σ_{j≠i} w_ij  − s_out_i      for i = 0 … N-1
    F_{N+i}(θ) = Σ_{j≠i} w_ji  − s_in_i       for i = 0 … N-1

**Zero-strength nodes**: if s_out_i = 0 then β_out_i = 0 exactly
(θ_out_i → +∞), so w_ij = 0 for all j.  Analogously for s_in_i = 0.

Reference:
    Squartini & Garlaschelli, New J. Phys. 13 (2011) 083001.
"""
from __future__ import annotations

from typing import Union

import torch


# Type alias for inputs: accept both numpy arrays and torch tensors.
_ArrayLike = Union[torch.Tensor, "numpy.ndarray"]  # type: ignore[name-defined]

# θ lower bound: ensures β = exp(−θ) < 1 and avoids div-by-zero in w_ij.
_ETA_MIN: float = 1e-10
# θ upper bound: exp(−50) ≈ 2e-22, essentially zero weight contribution.
_ETA_MAX: float = 50.0

# For N > this threshold, residual() and neg_log_likelihood() use chunked
# computation to avoid materialising the full N×N matrix.
_LARGE_N_THRESHOLD: int = 5_000

# Number of rows processed per chunk when using memory-efficient mode.
_DEFAULT_CHUNK: int = 512


def _to_tensor(x: _ArrayLike, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Convert *x* to a float64 CPU torch.Tensor (no-copy if already correct)."""
    if isinstance(x, torch.Tensor):
        return x.to(device="cpu", dtype=dtype)
    return torch.tensor(x, dtype=dtype, device="cpu")


class DWCMModel:
    """Encapsulates all DWCM equations for a network of *N* nodes.

    Internally all quantities are stored and computed as ``torch.float64``
    tensors.  All parameters θ must be strictly positive (θ > 0) to satisfy
    the feasibility constraint β_out_i * β_in_j < 1.

    Args:
        s_out: Observed out-strength sequence, shape (N,).
        s_in:  Observed in-strength sequence, shape (N,).
    """

    def __init__(self, s_out: _ArrayLike, s_in: _ArrayLike) -> None:
        self.s_out: torch.Tensor = _to_tensor(s_out)
        self.s_in: torch.Tensor = _to_tensor(s_in)
        self.N: int = int(self.s_out.shape[0])
        if self.s_in.shape[0] != self.N:
            raise ValueError("s_out and s_in must have the same length.")
        # Nodes with strength 0: β is exactly 0 (θ → +∞).
        self.zero_out: torch.Tensor = (self.s_out == 0)
        self.zero_in: torch.Tensor = (self.s_in == 0)

    # ------------------------------------------------------------------
    # Core expected-weight matrix
    # ------------------------------------------------------------------

    def wij_matrix(self, theta: _ArrayLike) -> torch.Tensor:
        """Compute the N×N expected weight matrix W_ij = β_ij / (1 − β_ij).

        Uses the numerically stable form W_ij = 1 / expm1(z_ij) where
        z_ij = θ_out_i + θ_in_j > 0.  Diagonal entries are 0 (no self-loops).

        Args:
            theta: Parameter vector [θ_out | θ_in], shape (2N,).
                   All entries should be strictly positive.

        Returns:
            Expected weight matrix W, shape (N, N), dtype torch.float64.
        """
        theta = _to_tensor(theta)
        N = self.N
        theta_out = theta[:N]
        theta_in = theta[N:]
        # z_ij = θ_out_i + θ_in_j; clamp to avoid expm1 = 0
        z = theta_out[:, None] + theta_in[None, :]  # (N, N)
        z_safe = z.clamp(min=1e-15)
        # W_ij = exp(-z) / (1 - exp(-z)) = 1 / (exp(z) - 1) = 1 / expm1(z)
        W = 1.0 / torch.expm1(z_safe)
        W.fill_diagonal_(0.0)
        # Zero-strength nodes contribute exactly zero weight.
        if self.zero_out.any():
            W[self.zero_out] = 0.0
        if self.zero_in.any():
            W[:, self.zero_in] = 0.0
        return W

    # ------------------------------------------------------------------
    # Residual (system of equations)
    # ------------------------------------------------------------------

    def residual(self, theta: _ArrayLike) -> torch.Tensor:
        """Return F(θ) = [s_out_expected − s_out_obs | s_in_expected − s_in_obs].

        For N > ``_LARGE_N_THRESHOLD`` the computation is done in row chunks
        to avoid materialising the full N×N matrix.

        Args:
            theta: Parameter vector [θ_out | θ_in], shape (2N,).

        Returns:
            Residual vector F(θ), shape (2N,).
        """
        if self.N > _LARGE_N_THRESHOLD:
            return self._residual_chunked(theta)
        W = self.wij_matrix(theta)
        s_out_hat = W.sum(dim=1)   # row sums = expected out-strengths
        s_in_hat = W.sum(dim=0)    # col sums = expected in-strengths
        F = torch.empty(2 * self.N, dtype=torch.float64)
        F[: self.N] = s_out_hat - self.s_out
        F[self.N :] = s_in_hat - self.s_in
        return F

    def _residual_chunked(
        self, theta: _ArrayLike, chunk_size: int = _DEFAULT_CHUNK
    ) -> torch.Tensor:
        """Compute F(θ) without materialising the full N×N matrix.

        Uses O(chunk_size × N) RAM instead of O(N²).

        Args:
            theta:      Parameter vector [θ_out | θ_in], shape (2N,).
            chunk_size: Number of rows per processing chunk (must be ≥ 1).

        Returns:
            Residual vector F(θ), shape (2N,).

        Raises:
            ValueError: If ``chunk_size < 1``.
        """
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be ≥ 1, got {chunk_size}")
        theta = _to_tensor(theta)
        N = self.N
        theta_out = theta[:N]
        theta_in = theta[N:]

        s_out_hat = torch.zeros(N, dtype=torch.float64)
        s_in_hat = torch.zeros(N, dtype=torch.float64)

        for i_start in range(0, N, chunk_size):
            i_end = min(i_start + chunk_size, N)
            chunk_len = i_end - i_start
            z = theta_out[i_start:i_end, None] + theta_in[None, :]  # (chunk, N)
            z_safe = z.clamp(min=1e-15)
            w_chunk = 1.0 / torch.expm1(z_safe)

            # Zero out diagonal entries (i == j)
            local_idx = torch.arange(chunk_len, dtype=torch.long)
            global_idx = torch.arange(i_start, i_end, dtype=torch.long)
            w_chunk[local_idx, global_idx] = 0.0

            # Apply zero-strength masks
            if self.zero_out[i_start:i_end].any():
                w_chunk[self.zero_out[i_start:i_end]] = 0.0
            if self.zero_in.any():
                w_chunk[:, self.zero_in] = 0.0

            s_out_hat[i_start:i_end] = w_chunk.sum(dim=1)
            s_in_hat += w_chunk.sum(dim=0)

        F = torch.empty(2 * N, dtype=torch.float64)
        F[:N] = s_out_hat - self.s_out
        F[N:] = s_in_hat - self.s_in
        return F

    # ------------------------------------------------------------------
    # Gradient of the log-likelihood (= +F(θ))
    # ------------------------------------------------------------------

    def gradient(self, theta: _ArrayLike) -> torch.Tensor:
        """Return ∇L(θ) = +F(θ).

        The DWCM log-likelihood is L(θ) = −Σ_i θ_out_i·s_out_i
        − Σ_i θ_in_i·s_in_i + Σ_{i≠j} log(1 − exp(−θ_out_i − θ_in_j)),
        so ∂L/∂θ_out_i = −s_out_i + Σ_{j≠i} w_ij = F_i(θ).

        Args:
            theta: Parameter vector, shape (2N,).

        Returns:
            Gradient vector ∇L = F(θ), shape (2N,).
        """
        return self.residual(theta)

    # ------------------------------------------------------------------
    # Diagonal Hessian of the log-likelihood
    # ------------------------------------------------------------------

    def hessian_diag(self, theta: _ArrayLike) -> torch.Tensor:
        """Return the diagonal of the Hessian of L(θ).

        The second derivatives are:

            ∂²L/∂θ_out_i² = −Σ_{j≠i} W_ij(1 + W_ij)
            ∂²L/∂θ_in_i²  = −Σ_{j≠i} W_ji(1 + W_ji)

        Args:
            theta: Parameter vector, shape (2N,).

        Returns:
            Diagonal of the Hessian, shape (2N,).
        """
        W = self.wij_matrix(theta)
        G = W * (1.0 + W)    # G_ij = W_ij(1 + W_ij); G_ii = 0
        h_out = -G.sum(dim=1)
        h_in = -G.sum(dim=0)
        return torch.cat([h_out, h_in])

    # ------------------------------------------------------------------
    # Full Jacobian of F(θ) (= Hessian of L, used by Newton solvers)
    # ------------------------------------------------------------------

    def jacobian(self, theta: _ArrayLike) -> torch.Tensor:
        """Return the full Jacobian matrix J = ∂F/∂θ = Hess(L), shape (2N, 2N).

        Denoting G = W ⊙ (1 + W) (elementwise, diagonal zero):

            J_out,out = −diag(Σ_{j≠i} G_ij)   [diagonal, negative]
            J_out,in  = −G                      [off-diagonal]
            J_in,out  = −Gᵀ                    [off-diagonal]
            J_in,in   = −diag(Σ_{j≠i} G_ji)   [diagonal, negative]

        Args:
            theta: Parameter vector, shape (2N,).

        Returns:
            Jacobian matrix, shape (2N, 2N), dtype torch.float64.
        """
        N = self.N
        W = self.wij_matrix(theta)
        G = W * (1.0 + W)   # G[i,i] = 0 since W[i,i] = 0
        idx = torch.arange(N)

        J = torch.zeros(2 * N, 2 * N, dtype=torch.float64)
        # Top-left block: ∂F_out_i / ∂θ_out_i (diagonal, negative)
        J[idx, idx] = -G.sum(dim=1)
        # Top-right block: ∂F_out_i / ∂θ_in_j = −G_ij
        J[:N, N:] = -G
        # Bottom-left block: ∂F_in_i / ∂θ_out_j = −G_ji
        J[N:, :N] = -G.T
        # Bottom-right block: ∂F_in_i / ∂θ_in_i (diagonal, negative)
        J[N + idx, N + idx] = -G.sum(dim=0)
        return J

    # ------------------------------------------------------------------
    # Initial-guess utilities
    # ------------------------------------------------------------------

    def initial_theta(self, method: str = "strengths") -> torch.Tensor:
        """Return a sensible starting point θ₀ for the solvers.

        Several initialisation strategies are provided:

        * ``"strengths"`` (default): β_i ≈ sqrt(s_i / (s_i + N − 1)), derived
          by inverting the symmetric mean-field estimate s ≈ (N−1)β²/(1−β²).
          Works well when the strength distribution is not too heterogeneous.

        * ``"normalized"`` (Squartini & Garlaschelli 2011, analytical approx.):
          β_i^{out} = s_i^{out} / Σ_j s_j^{out}, β_i^{in} = s_i^{in} / Σ_j s_j^{in}.
          Each β is proportional to the node's fractional share of total weight.

        * ``"uniform"`` : all betas set to the geometric mean of the
          "strengths" approximation — gives a flat starting point useful as a
          multi-start alternative.

        * ``"random"`` : uniform random values in θ ∈ [0.1, 2.0].

        Zero-strength nodes always have β_i = 0 exactly (θ_i = _ETA_MAX)
        regardless of method.  All θ values are clamped to [_ETA_MIN, _ETA_MAX].

        Args:
            method: One of ``"strengths"``, ``"normalized"``,
                    ``"uniform"``, ``"random"``.

        Returns:
            Initial parameter vector θ₀, shape (2N,).
        """
        N = self.N
        if method == "strengths":
            # Mean-field approximation: s ≈ (N-1) β² / (1-β²) (assuming β_out≈β_in≈β)
            # → β = sqrt(s/(s+N-1)), θ = −log(β) = ½ log((s+N-1)/s)
            s_out_safe = self.s_out.clamp(min=1e-15)
            s_in_safe = self.s_in.clamp(min=1e-15)
            beta_out = torch.sqrt(s_out_safe / (s_out_safe + (N - 1)))
            beta_in = torch.sqrt(s_in_safe / (s_in_safe + (N - 1)))
        elif method == "normalized":
            # Squartini & Garlaschelli (2011) analytical approximation:
            # β_i^{out} = s_out_i / Σ_j s_out_j  (fraction of total out-weight)
            # β_i^{in}  = s_in_i  / Σ_j s_in_j
            S_out = self.s_out.sum().clamp(min=1e-15)
            S_in = self.s_in.sum().clamp(min=1e-15)
            beta_out = self.s_out.clamp(min=1e-15) / S_out
            beta_in = self.s_in.clamp(min=1e-15) / S_in
        elif method == "uniform":
            # Flat initialisation: all betas equal to the geometric-mean of the
            # "strengths" approximation.  Useful as a diversity-inducing restart.
            s_out_safe = self.s_out.clamp(min=1e-15)
            s_in_safe = self.s_in.clamp(min=1e-15)
            beta_ref_out = torch.sqrt(s_out_safe / (s_out_safe + (N - 1)))
            beta_ref_in = torch.sqrt(s_in_safe / (s_in_safe + (N - 1)))
            # Use median of positive entries as the common value
            pos_out = beta_ref_out[~self.zero_out]
            pos_in = beta_ref_in[~self.zero_in]
            med_out = pos_out.median().item() if pos_out.numel() > 0 else 0.5
            med_in = pos_in.median().item() if pos_in.numel() > 0 else 0.5
            beta_out = torch.full((N,), med_out, dtype=torch.float64)
            beta_in = torch.full((N,), med_in, dtype=torch.float64)
        elif method == "random":
            # No fixed seed — genuinely random each call
            theta_out = torch.empty(N, dtype=torch.float64).uniform_(0.1, 2.0)
            theta_in = torch.empty(N, dtype=torch.float64).uniform_(0.1, 2.0)
            # Skip the β→θ conversion below
            theta_out = torch.where(
                self.zero_out, torch.full_like(theta_out, _ETA_MAX), theta_out
            )
            theta_in = torch.where(
                self.zero_in, torch.full_like(theta_in, _ETA_MAX), theta_in
            )
            return torch.cat([theta_out.clamp(_ETA_MIN, _ETA_MAX),
                              theta_in.clamp(_ETA_MIN, _ETA_MAX)])
        else:
            raise ValueError(f"Unknown initial-guess method: {method!r}")

        # Clamp β to (0, 1) before taking log (shared by strengths/normalized/uniform)
        beta_out = beta_out.clamp(1e-15, 1.0 - 1e-15)
        beta_in = beta_in.clamp(1e-15, 1.0 - 1e-15)
        theta_out = (-torch.log(beta_out)).clamp(_ETA_MIN, _ETA_MAX)
        theta_in = (-torch.log(beta_in)).clamp(_ETA_MIN, _ETA_MAX)

        # Zero-strength nodes: β = 0 exactly ↔ θ → +∞ (clamped to +_ETA_MAX)
        theta_out = torch.where(
            self.zero_out, torch.full_like(theta_out, _ETA_MAX), theta_out
        )
        theta_in = torch.where(
            self.zero_in, torch.full_like(theta_in, _ETA_MAX), theta_in
        )
        return torch.cat([theta_out, theta_in])

    # ------------------------------------------------------------------
    # Negative log-likelihood (objective for L-BFGS minimisation)
    # ------------------------------------------------------------------

    def neg_log_likelihood(self, theta: _ArrayLike) -> float:
        """Return −L(θ), the convex quantity to be *minimised* by L-BFGS.

        The DWCM log-likelihood (Squartini & Garlaschelli 2011) is:

            L(θ) = −Σ_i θ_out_i·s_out_i − Σ_i θ_in_i·s_in_i
                   + Σ_{i≠j} log(1 − exp(−θ_out_i − θ_in_j))

        so

            −L = Σ_i θ_out_i·s_out_i + Σ_i θ_in_i·s_in_i
                 − Σ_{i≠j} log(1 − exp(−θ_out_i − θ_in_j))

        The term −log(1 − exp(−z)) for z > 0 is computed via
        ``-torch.log1p(-torch.exp(-z))`` for numerical stability.

        For N > ``_LARGE_N_THRESHOLD`` the sum is computed in row chunks.

        Args:
            theta: Parameter vector [θ_out | θ_in], shape (2N,).

        Returns:
            Scalar −L(θ) (convex, to be minimised).
        """
        if self.N > _LARGE_N_THRESHOLD:
            return self._neg_log_likelihood_chunked(theta)
        theta = _to_tensor(theta)
        N = self.N
        theta_out = theta[:N]
        theta_in = theta[N:]
        z = theta_out[:, None] + theta_in[None, :]  # (N, N)
        z_safe = z.clamp(min=1e-15)
        # -log(1 - exp(-z)) = -log1p(-exp(-z))
        log_term = -torch.log1p(-torch.exp(-z_safe))
        log_term.fill_diagonal_(0.0)  # exclude self-loops
        dot_term = theta_out @ self.s_out + theta_in @ self.s_in
        return (dot_term + log_term.sum()).item()

    def _neg_log_likelihood_chunked(
        self, theta: _ArrayLike, chunk_size: int = _DEFAULT_CHUNK
    ) -> float:
        """Compute −L(θ) without materialising the full N×N matrix.

        Args:
            theta:      Parameter vector [θ_out | θ_in], shape (2N,).
            chunk_size: Number of rows per processing chunk (must be ≥ 1).

        Returns:
            Scalar −L(θ).

        Raises:
            ValueError: If ``chunk_size < 1``.
        """
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be ≥ 1, got {chunk_size}")
        theta = _to_tensor(theta)
        N = self.N
        theta_out = theta[:N]
        theta_in = theta[N:]

        dot_term = (theta_out @ self.s_out + theta_in @ self.s_in).item()
        log_total = 0.0

        for i_start in range(0, N, chunk_size):
            i_end = min(i_start + chunk_size, N)
            chunk_len = i_end - i_start
            z = theta_out[i_start:i_end, None] + theta_in[None, :]  # (chunk, N)
            z_safe = z.clamp(min=1e-15)
            log_chunk = -torch.log1p(-torch.exp(-z_safe))
            # Zero out diagonal
            local_idx = torch.arange(chunk_len, dtype=torch.long)
            global_idx = torch.arange(i_start, i_end, dtype=torch.long)
            log_chunk[local_idx, global_idx] = 0.0
            log_total += log_chunk.sum().item()

        return dot_term + log_total

    # ------------------------------------------------------------------
    # Evaluation of constraint satisfaction
    # ------------------------------------------------------------------

    def constraint_error(self, theta: _ArrayLike) -> float:
        """Return the maximum absolute error on all constraints.

        Args:
            theta: Parameter vector, shape (2N,).

        Returns:
            Max-abs constraint error (scalar).
        """
        return self.residual(theta).abs().max().item()

    def max_relative_error(self, theta: _ArrayLike) -> float:
        """Return the maximum relative error on all non-zero constraints.

        Computes max_i |F_i(θ)| / s_i over nodes with s_i > 0.

        Args:
            theta: Parameter vector, shape (2N,).

        Returns:
            Max relative constraint error (scalar), or 0.0 if all strengths
            are zero.
        """
        F = self.residual(theta).abs()
        targets = torch.cat([self.s_out, self.s_in])
        nonzero = targets > 0
        if not nonzero.any():
            return 0.0
        return (F[nonzero] / targets[nonzero]).max().item()
