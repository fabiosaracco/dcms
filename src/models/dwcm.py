"""Directed Weighted Configuration Model (DWCM) — integer weights.

The DWCM fixes the out- and in-strength sequences (s_out, s_in).  Each
pair (i, j) with i≠j is assigned a geometric weight distribution:

    P(w_ij = w) = (1 − z_ij) · z_ij^w   for w = 0, 1, 2, …

where  z_ij = β_out_i · β_in_j  and  β_out_i = exp(−θ_out_i),
β_in_i = exp(−θ_in_i)  are the Lagrange multipliers in exponential
parametrisation.

The expected weight is:

    E[w_ij] = z_ij / (1 − z_ij)

so the system of equations to solve is F(θ) = 0, where:

    F_i(θ)     = Σ_{j≠i} z_ij / (1 − z_ij)  − s_out_i    for i = 0 … N-1
    F_{N+i}(θ) = Σ_{j≠i} z_ji / (1 − z_ji)  − s_in_i     for i = 0 … N-1

**Feasibility constraint**: z_ij = exp(−θ_out_i − θ_in_j) < 1 for all i≠j,
i.e. θ_out_i + θ_in_j > 0.  The parametrisation automatically enforces
positivity of β; the feasibility constraint is enforced by clamping θ so
that θ_out_i + θ_in_j ≥ _THETA_PAIR_MIN.

**Zero-strength nodes**: if s_out_i = 0 then β_out_i = 0 (θ_out_i → +∞);
no weight flows out of node i.  Similarly for s_in_i = 0.

Reference:
    Squartini & Garlaschelli, New J. Phys. 13 (2011) 083001.
    Garlaschelli & Loffredo, Phys. Rev. Lett. 93, 188701 (2004).
"""
from __future__ import annotations

from typing import Union

import torch


# Type alias for inputs: accept both numpy arrays and torch tensors.
_ArrayLike = Union[torch.Tensor, "numpy.ndarray"]  # type: ignore[name-defined]

# θ is clamped to this bound; exp(-50) ≈ 2e-22, essentially zero β.
_THETA_MAX: float = 50.0

# Minimum value for θ_out_i + θ_in_j to keep z_ij < 1.
# exp(-_THETA_PAIR_MIN) ≈ 1 − 1e-4.
_THETA_PAIR_MIN: float = 1e-4

# For N > this threshold, residual() automatically uses chunked computation.
_LARGE_N_THRESHOLD: int = 5_000

# Number of rows processed per chunk in memory-efficient mode.
_DEFAULT_CHUNK: int = 512


def _to_tensor(x: _ArrayLike, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Convert *x* to a float64 CPU torch.Tensor (no-copy if already correct)."""
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype)
    return torch.tensor(x, dtype=dtype)


class DWCMModel:
    """Encapsulates all DWCM equations for a network of *N* nodes.

    Internally all quantities are stored and computed as ``torch.float64``
    tensors.

    Args:
        s_out: Observed out-strength sequence, shape (N,).
        s_in:  Observed in-strength sequence, shape (N,).
    """

    def __init__(self, s_out: _ArrayLike, s_in: _ArrayLike) -> None:
        self.s_out: torch.Tensor = _to_tensor(s_out, dtype=torch.float64)
        self.s_in: torch.Tensor = _to_tensor(s_in, dtype=torch.float64)
        self.N: int = int(self.s_out.shape[0])
        if self.s_in.shape[0] != self.N:
            raise ValueError("s_out and s_in must have the same length.")
        # Zero-strength nodes: β = 0 exactly ↔ θ → +∞.
        self.zero_out: torch.Tensor = (self.s_out == 0)
        self.zero_in: torch.Tensor = (self.s_in == 0)

    # ------------------------------------------------------------------
    # Core weight-expectation matrix
    # ------------------------------------------------------------------

    def wij_matrix(self, theta: _ArrayLike) -> torch.Tensor:
        """Compute the N×N matrix of expected weights E[w_ij].

        E[w_ij] = z_ij / (1 − z_ij)  where  z_ij = exp(−θ_out_i − θ_in_j).

        Diagonal entries are 0 (no self-loops).  Entries where z_ij ≥ 1
        are clamped to a large finite value to prevent inf/NaN.

        Args:
            theta: Parameter vector [θ_out | θ_in], shape (2N,).

        Returns:
            Expected-weight matrix W, shape (N, N), dtype torch.float64.
        """
        theta = _to_tensor(theta)
        N = self.N
        theta_out = theta[:N]
        theta_in = theta[N:]
        # log(z_ij) = −θ_out_i − θ_in_j
        log_z = -theta_out[:, None] - theta_in[None, :]  # (N, N)
        z = torch.exp(log_z)
        # Clamp denominator away from 0: 1 − z ≥ _THETA_PAIR_MIN
        denom = (1.0 - z).clamp(min=1e-15)
        W = z / denom
        W.fill_diagonal_(0.0)
        # Zero-strength nodes contribute exactly zero expected weight.
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

        For N > ``_LARGE_N_THRESHOLD`` the computation uses row chunks.

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

        Args:
            theta:      Parameter vector [θ_out | θ_in], shape (2N,).
            chunk_size: Number of rows per processing chunk (≥ 1).

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
            log_z = -theta_out[i_start:i_end, None] - theta_in[None, :]  # (chunk, N)
            z = torch.exp(log_z)
            denom = (1.0 - z).clamp(min=1e-15)
            w_chunk = z / denom  # (chunk, N)

            # Zero out diagonal entries
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
    # Full Jacobian of F(θ)
    # ------------------------------------------------------------------

    def jacobian(self, theta: _ArrayLike) -> torch.Tensor:
        """Return the full Jacobian matrix J = ∂F/∂θ, shape (2N, 2N).

        Denoting Q[i,j] = z_ij / (1−z_ij)²  (with Q[i,i] = 0):

            J_out,out = −diag(Σ_{j≠i} Q[i,j])   [diagonal]
            J_out,in  = −Q                         [off-diagonal only]
            J_in,out  = −Qᵀ                        [off-diagonal only]
            J_in,in   = −diag(Σ_{j≠i} Q[j,i])    [diagonal, col sums of Q]

        Args:
            theta: Parameter vector, shape (2N,).

        Returns:
            Jacobian matrix, shape (2N, 2N), dtype torch.float64.
        """
        N = self.N
        theta = _to_tensor(theta)
        theta_out = theta[:N]
        theta_in = theta[N:]
        log_z = -theta_out[:, None] - theta_in[None, :]
        z = torch.exp(log_z)
        denom = (1.0 - z).clamp(min=1e-15)
        Q = z / (denom ** 2)
        Q.fill_diagonal_(0.0)
        # Apply zero-strength masks
        if self.zero_out.any():
            Q[self.zero_out] = 0.0
        if self.zero_in.any():
            Q[:, self.zero_in] = 0.0

        idx = torch.arange(N)
        J = torch.zeros(2 * N, 2 * N, dtype=torch.float64)
        # Top-left: ∂F_out_i / ∂θ_out_i (diagonal)
        J[idx, idx] = -Q.sum(dim=1)
        # Top-right: ∂F_out_i / ∂θ_in_j = −Q[i,j] (off-diagonal)
        J[:N, N:] = -Q
        # Bottom-left: ∂F_in_i / ∂θ_out_k = −Q[k,i] = −Qᵀ[i,k]
        J[N:, :N] = -Q.T
        # Bottom-right: ∂F_in_i / ∂θ_in_i (diagonal, col sums of Q)
        J[N + idx, N + idx] = -Q.sum(dim=0)
        return J

    # ------------------------------------------------------------------
    # Negative log-likelihood (objective for L-BFGS minimisation)
    # ------------------------------------------------------------------

    def neg_log_likelihood(self, theta: _ArrayLike) -> float:
        """Return −L(θ), the convex quantity to be *minimised* by L-BFGS.

        The DWCM log-likelihood is:

            L(θ) = −Σ_i θ_out_i·s_out_i − Σ_i θ_in_i·s_in_i
                   + Σ_{i≠j} log(1 − z_ij)

        so −L(θ) = Σ_i θ_out_i·s_out_i + Σ_i θ_in_i·s_in_i
                   − Σ_{i≠j} log(1 − z_ij).

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
        log_z = -theta_out[:, None] - theta_in[None, :]  # (N, N)
        z = torch.exp(log_z)
        # log(1 - z); clamp z away from 1 to avoid log(0).
        log1mz = torch.log((1.0 - z).clamp(min=1e-15))
        log1mz.fill_diagonal_(0.0)  # exclude self-loops
        # Apply zero-strength masks
        if self.zero_out.any():
            log1mz[self.zero_out] = 0.0
        if self.zero_in.any():
            log1mz[:, self.zero_in] = 0.0
        dot = theta_out @ self.s_out + theta_in @ self.s_in
        return (dot - log1mz.sum()).item()

    def _neg_log_likelihood_chunked(
        self, theta: _ArrayLike, chunk_size: int = _DEFAULT_CHUNK
    ) -> float:
        """Compute −L(θ) without materialising the full N×N matrix.

        Args:
            theta:      Parameter vector [θ_out | θ_in], shape (2N,).
            chunk_size: Number of rows per processing chunk (≥ 1).

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

        dot = (theta_out @ self.s_out + theta_in @ self.s_in).item()
        log1mz_total = 0.0

        for i_start in range(0, N, chunk_size):
            i_end = min(i_start + chunk_size, N)
            chunk_len = i_end - i_start
            log_z = -theta_out[i_start:i_end, None] - theta_in[None, :]  # (chunk, N)
            z = torch.exp(log_z)
            log1mz = torch.log((1.0 - z).clamp(min=1e-15))
            # Zero out diagonal and zero-strength entries
            local_idx = torch.arange(chunk_len, dtype=torch.long)
            global_idx = torch.arange(i_start, i_end, dtype=torch.long)
            log1mz[local_idx, global_idx] = 0.0
            if self.zero_out[i_start:i_end].any():
                log1mz[self.zero_out[i_start:i_end]] = 0.0
            if self.zero_in.any():
                log1mz[:, self.zero_in] = 0.0
            log1mz_total += log1mz.sum().item()

        return dot - log1mz_total

    # ------------------------------------------------------------------
    # Initial-guess utilities
    # ------------------------------------------------------------------

    def initial_theta(self, method: str = "strengths") -> torch.Tensor:
        """Return a sensible starting point θ₀ for the solvers.

        For DWCM the mean-field approximation gives:
            s_out_i ≈ (N−1) · β_out_i · β_in_avg / (1 − β_out_i · β_in_avg)

        A simple approximation: β_i ≈ sqrt(s_i / (N−1 + s_i)).

        Zero-strength nodes (s_i = 0) have β = 0 ↔ θ = +_THETA_MAX.

        Args:
            method: ``"strengths"`` — closed-form approximation;
                    ``"random"``   — uniform random values in [0.5, 3.0].

        Returns:
            Initial parameter vector θ₀, shape (2N,).
        """
        N = self.N
        if method == "strengths":
            # β_out_i ≈ sqrt(s_out_i / (N-1 + s_out_i))
            s_out_f = self.s_out.double()
            s_in_f = self.s_in.double()
            beta_out = torch.sqrt(
                s_out_f / (N - 1 + s_out_f).clamp(min=1e-15)
            ).clamp(min=1e-15, max=1.0 - 1e-6)
            beta_in = torch.sqrt(
                s_in_f / (N - 1 + s_in_f).clamp(min=1e-15)
            ).clamp(min=1e-15, max=1.0 - 1e-6)
            theta_out = -torch.log(beta_out)
            theta_in = -torch.log(beta_in)
        elif method == "random":
            theta_out = torch.empty(N, dtype=torch.float64).uniform_(0.5, 3.0)
            theta_in = torch.empty(N, dtype=torch.float64).uniform_(0.5, 3.0)
        else:
            raise ValueError(f"Unknown initial-guess method: {method!r}")

        # Zero-strength nodes: β = 0 exactly ↔ θ = +_THETA_MAX.
        theta_out = torch.where(
            self.zero_out, torch.full_like(theta_out, _THETA_MAX), theta_out
        )
        theta_in = torch.where(
            self.zero_in, torch.full_like(theta_in, _THETA_MAX), theta_in
        )
        return torch.cat([theta_out, theta_in])

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

    def mean_relative_error(self, theta: _ArrayLike) -> float:
        """Return the mean relative error (MRE) across all constraints.

        MRE = mean over i of |F_i| / max(|c_i|, 1), where c_i is the
        observed strength (s_out or s_in).  The denominator is clamped to
        at least 1 to avoid division by zero for zero-strength nodes.

        Args:
            theta: Parameter vector, shape (2N,).

        Returns:
            Mean relative error (scalar).
        """
        F = self.residual(theta).abs()
        constraints = torch.cat([self.s_out, self.s_in]).double()
        denom = constraints.clamp(min=1.0)
        return (F / denom).mean().item()
