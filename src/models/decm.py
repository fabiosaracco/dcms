"""Directed Enhanced Configuration Model (DECM) — binary + weighted, fully coupled.

The DECM fixes *four* sequences per node: out-degree, in-degree, out-strength
and in-strength (k_out, k_in, s_out, s_in) with 4N unknowns:

    θ = [θ_out | θ_in | η_out | η_in]

The physical multipliers are:
    x_i      = exp(-θ_out_i),   y_j = exp(-θ_in_j)    (topology)
    β_out_i  = exp(-η_out_i),   β_in_j = exp(-η_in_j) (weight)

Unlike the DaECM (where p_ij uses only the topology parameters), the DECM
connection probability **couples** topology and weight parameters:

    q_ij     = 1 / expm1(η_out_i + η_in_j)
    p_ij     = x_i · y_j · q_ij / (1 + x_i · y_j · q_ij)
             = sigmoid(−θ_out_i − θ_in_j − log(expm1(η_out_i + η_in_j)))

The expected weight of arc i→j is:

    G_ij     = −1 / expm1(−(η_out_i + η_in_j))    (weight factor)
    E[w_ij]  = p_ij · G_ij

The 4N equations to solve are:

    F_k_out_i = Σ_{j≠i} p_ij  − k_out_i = 0
    F_k_in_j  = Σ_{i≠j} p_ij  − k_in_j  = 0
    F_s_out_i = Σ_{j≠i} p_ij · G_ij − s_out_i = 0
    F_s_in_j  = Σ_{i≠j} p_ij · G_ij − s_in_j  = 0

Reference:
    Squartini, T. & Garlaschelli, D. (2011).  Analytical maximum-likelihood
    method to detect patterns in real networks.  *New J. Phys.* 13, 083001.
    https://doi.org/10.1088/1367-2630/13/8/083001
"""
from __future__ import annotations

import functools
import math
from typing import Union

import torch

from src.models.parameters import DaECM_LARGE_N_THRESHOLD as _LARGE_N_THRESHOLD
from src.models.parameters import _DEFAULT_CHUNK, _ETA_MAX, _ETA_MIN
from src.solvers.base import SolverResult

# Type alias: accept both numpy arrays and torch tensors.
_ArrayLike = Union[torch.Tensor, "numpy.ndarray"]  # type: ignore[name-defined]

# θ clamping bound for topology multipliers.
_THETA_MAX: float = 50.0

# Maximum allowed β_out_i · β_in_j product (pairs closer to 1 cause G → ∞).
_Q_MAX: float = 0.9999

# Minimum η = θ_β_out + θ_β_in used in G = −1/expm1(−η).
_Z_G_CLAMP: float = 1e-8


def _to_tensor(x: _ArrayLike, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Convert *x* to a float64 CPU torch.Tensor (no-copy if already correct)."""
    if isinstance(x, torch.Tensor):
        return x.to(device="cpu", dtype=dtype)
    return torch.tensor(x, dtype=dtype, device="cpu")


class DECMModel:
    """Encapsulates the DECM equations for a directed network of *N* nodes.

    The 4N unknowns are stored as a single vector:
    ``theta = [θ_out | θ_in | η_out | η_in]``

    Attributes:
        N:          Number of nodes.
        k_out:      Observed out-degree sequence, shape (N,).
        k_in:       Observed in-degree sequence, shape (N,).
        s_out:      Observed out-strength sequence, shape (N,).
        s_in:       Observed in-strength sequence, shape (N,).
        zero_k_out: Boolean mask of zero out-degree nodes.
        zero_k_in:  Boolean mask of zero in-degree nodes.
        zero_s_out: Boolean mask of zero out-strength nodes.
        zero_s_in:  Boolean mask of zero in-strength nodes.

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

        if any(t.shape[0] != self.N for t in [self.k_in, self.s_out, self.s_in]):
            raise ValueError(
                "k_out, k_in, s_out and s_in must all have the same length."
            )

        self.zero_k_out: torch.Tensor = (self.k_out == 0)
        self.zero_k_in: torch.Tensor = (self.k_in == 0)
        self.zero_s_out: torch.Tensor = (self.s_out == 0)
        self.zero_s_in: torch.Tensor = (self.s_in == 0)

    # ------------------------------------------------------------------
    # Core matrices
    # ------------------------------------------------------------------

    def pij_matrix(self, theta: _ArrayLike) -> torch.Tensor:
        """Return the N×N DECM connection probability matrix p_ij.

        The probability is coupled to both topology and weight parameters:

            p_ij = sigmoid(−θ_out_i − θ_in_j + log_q_ij)

        where log_q_ij = −log(expm1(η_out_i + η_in_j)).

        Args:
            theta: Full parameter vector [θ_out|θ_in|η_out|η_in], shape (4N,).

        Returns:
            Probability matrix P, shape (N, N), diagonal zero.
        """
        theta = _to_tensor(theta)
        N = self.N
        theta_out = theta[:N]
        theta_in = theta[N : 2 * N]
        eta_out = theta[2 * N : 3 * N]
        eta_in = theta[3 * N :]

        eta = eta_out[:, None] + eta_in[None, :]
        eta_safe = eta.clamp(min=_Z_G_CLAMP)
        log_q = -torch.log(torch.expm1(eta_safe))
        logit_p = -theta_out[:, None] - theta_in[None, :] + log_q
        P = torch.sigmoid(logit_p)
        P.fill_diagonal_(0.0)

        if self.zero_k_out.any():
            P[self.zero_k_out] = 0.0
        if self.zero_k_in.any():
            P[:, self.zero_k_in] = 0.0
        return P

    def wij_matrix(self, theta: _ArrayLike) -> torch.Tensor:
        """Return the N×N expected weight matrix W_ij = p_ij · G_ij.

        Args:
            theta: Full parameter vector [θ_out|θ_in|η_out|η_in], shape (4N,).

        Returns:
            Expected weight matrix W, shape (N, N), diagonal zero.
        """
        theta = _to_tensor(theta)
        N = self.N
        eta_out = theta[2 * N : 3 * N]
        eta_in = theta[3 * N :]

        eta = eta_out[:, None] + eta_in[None, :]
        eta_safe = eta.clamp(min=_Z_G_CLAMP)
        G = -1.0 / torch.expm1(-eta_safe)
        G.fill_diagonal_(0.0)

        if self.zero_s_out.any():
            G[self.zero_s_out] = 0.0
        if self.zero_s_in.any():
            G[:, self.zero_s_in] = 0.0

        P = self.pij_matrix(theta)
        W = P * G
        return W

    # ------------------------------------------------------------------
    # Residual (system of 4N equations)
    # ------------------------------------------------------------------

    def residual(self, theta: _ArrayLike) -> torch.Tensor:
        """Return the 4N residual F(θ,η).

        F = [k_out_hat − k_out | k_in_hat − k_in | s_out_hat − s_out | s_in_hat − s_in]

        For N > ``_LARGE_N_THRESHOLD`` uses chunked computation to avoid
        materialising the full N×N matrices.

        Args:
            theta: Full parameter vector [θ_out|θ_in|η_out|η_in], shape (4N,).

        Returns:
            Residual vector F, shape (4N,).
        """
        if self.N > _LARGE_N_THRESHOLD:
            return self._residual_chunked(theta)

        theta = _to_tensor(theta)
        N = self.N
        theta_out = theta[:N]
        theta_in = theta[N : 2 * N]
        eta_out = theta[2 * N : 3 * N]
        eta_in = theta[3 * N :]

        eta = eta_out[:, None] + eta_in[None, :]
        eta_safe = eta.clamp(min=_Z_G_CLAMP)
        G = -1.0 / torch.expm1(-eta_safe)
        log_q = -torch.log(torch.expm1(eta_safe))
        logit_p = -theta_out[:, None] - theta_in[None, :] + log_q
        P = torch.sigmoid(logit_p)

        P.fill_diagonal_(0.0)
        G.fill_diagonal_(0.0)
        if self.zero_k_out.any():
            P[self.zero_k_out] = 0.0
        if self.zero_k_in.any():
            P[:, self.zero_k_in] = 0.0
        if self.zero_s_out.any():
            G[self.zero_s_out] = 0.0
        if self.zero_s_in.any():
            G[:, self.zero_s_in] = 0.0

        W = P * G
        k_out_hat = P.sum(1)
        k_in_hat = P.sum(0)
        s_out_hat = W.sum(1)
        s_in_hat = W.sum(0)

        return torch.cat(
            [k_out_hat - self.k_out, k_in_hat - self.k_in,
             s_out_hat - self.s_out, s_in_hat - self.s_in]
        )

    def _residual_chunked(
        self,
        theta: _ArrayLike,
        chunk_size: int = _DEFAULT_CHUNK,
    ) -> torch.Tensor:
        """Compute residual without materialising the full N×N matrix.

        Args:
            theta:      Full parameter vector, shape (4N,).
            chunk_size: Rows per processing chunk.

        Returns:
            Residual vector F, shape (4N,).
        """
        theta = _to_tensor(theta)
        N = self.N
        theta_out = theta[:N]
        theta_in = theta[N : 2 * N]
        eta_out = theta[2 * N : 3 * N]
        eta_in = theta[3 * N :]

        k_out_hat = torch.zeros(N, dtype=torch.float64)
        k_in_hat = torch.zeros(N, dtype=torch.float64)
        s_out_hat = torch.zeros(N, dtype=torch.float64)
        s_in_hat = torch.zeros(N, dtype=torch.float64)

        for i_start in range(0, N, chunk_size):
            i_end = min(i_start + chunk_size, N)
            chunk_len = i_end - i_start
            local_i = torch.arange(chunk_len, dtype=torch.long)
            global_j = torch.arange(i_start, i_end, dtype=torch.long)

            eta_chunk = eta_out[i_start:i_end, None] + eta_in[None, :]
            eta_safe = eta_chunk.clamp(min=_Z_G_CLAMP)
            G_chunk = -1.0 / torch.expm1(-eta_safe)
            log_q_chunk = -torch.log(torch.expm1(eta_safe))
            logit_p_chunk = (
                -theta_out[i_start:i_end, None]
                - theta_in[None, :]
                + log_q_chunk
            )
            p_chunk = torch.sigmoid(logit_p_chunk)
            w_chunk = p_chunk * G_chunk

            p_chunk[local_i, global_j] = 0.0
            w_chunk[local_i, global_j] = 0.0

            if self.zero_k_out[i_start:i_end].any():
                p_chunk[self.zero_k_out[i_start:i_end]] = 0.0
            if self.zero_k_in.any():
                p_chunk[:, self.zero_k_in] = 0.0
            if self.zero_s_out[i_start:i_end].any():
                w_chunk[self.zero_s_out[i_start:i_end]] = 0.0
            if self.zero_s_in.any():
                w_chunk[:, self.zero_s_in] = 0.0

            k_out_hat[i_start:i_end] = p_chunk.sum(1)
            k_in_hat += p_chunk.sum(0)
            s_out_hat[i_start:i_end] = w_chunk.sum(1)
            s_in_hat += w_chunk.sum(0)

        return torch.cat(
            [k_out_hat - self.k_out, k_in_hat - self.k_in,
             s_out_hat - self.s_out, s_in_hat - self.s_in]
        )

    # ------------------------------------------------------------------
    # Negative log-likelihood
    # ------------------------------------------------------------------

    def neg_log_likelihood(self, theta: _ArrayLike) -> float:
        """Return −L(θ,η), the convex quantity to be minimised.

        The DECM log-likelihood is:

            L = −Σ_i θ_out_i·k_out_i − Σ_i θ_in_i·k_in_i
                − Σ_i η_out_i·s_out_i − Σ_i η_in_i·s_in_i
                − Σ_{i≠j} log(1 + x_i·y_j·q_ij)

        where log(1 + x_i·y_j·q_ij) = softplus(logit_p_ij).

        Args:
            theta: Full parameter vector [θ_out|θ_in|η_out|η_in], shape (4N,).

        Returns:
            Scalar −L(θ,η).
        """
        if self.N > _LARGE_N_THRESHOLD:
            return self._neg_log_likelihood_chunked(theta)

        theta = _to_tensor(theta)
        N = self.N
        theta_out = theta[:N]
        theta_in = theta[N : 2 * N]
        eta_out = theta[2 * N : 3 * N]
        eta_in = theta[3 * N :]

        eta = eta_out[:, None] + eta_in[None, :]
        eta_safe = eta.clamp(min=_Z_G_CLAMP)
        log_q = -torch.log(torch.expm1(eta_safe))
        logit_p = -theta_out[:, None] - theta_in[None, :] + log_q
        sp = torch.nn.functional.softplus(logit_p)
        sp.fill_diagonal_(0.0)

        dot = (
            theta_out @ self.k_out
            + theta_in @ self.k_in
            + eta_out @ self.s_out
            + eta_in @ self.s_in
        )
        return (dot + sp.sum()).item()

    def _neg_log_likelihood_chunked(
        self,
        theta: _ArrayLike,
        chunk_size: int = _DEFAULT_CHUNK,
    ) -> float:
        """Chunked computation of −L(θ,η)."""
        theta = _to_tensor(theta)
        N = self.N
        theta_out = theta[:N]
        theta_in = theta[N : 2 * N]
        eta_out = theta[2 * N : 3 * N]
        eta_in = theta[3 * N :]

        sp_total = 0.0
        for i_start in range(0, N, chunk_size):
            i_end = min(i_start + chunk_size, N)
            chunk_len = i_end - i_start
            local_i = torch.arange(chunk_len, dtype=torch.long)
            global_j = torch.arange(i_start, i_end, dtype=torch.long)

            eta_chunk = eta_out[i_start:i_end, None] + eta_in[None, :]
            eta_safe = eta_chunk.clamp(min=_Z_G_CLAMP)
            log_q_chunk = -torch.log(torch.expm1(eta_safe))
            logit_p_chunk = (
                -theta_out[i_start:i_end, None]
                - theta_in[None, :]
                + log_q_chunk
            )
            sp_chunk = torch.nn.functional.softplus(logit_p_chunk)
            sp_chunk[local_i, global_j] = 0.0
            sp_total += sp_chunk.sum().item()

        dot = (
            theta_out @ self.k_out
            + theta_in @ self.k_in
            + eta_out @ self.s_out
            + eta_in @ self.s_in
        ).item()
        return dot + sp_total

    # ------------------------------------------------------------------
    # Diagonal Jacobian (for Newton step computation)
    # ------------------------------------------------------------------

    def hessian_diag(self, theta: _ArrayLike) -> torch.Tensor:
        """Return the 4N diagonal Jacobian elements ∂F_i/∂θ_i (negative).

        Uses the formulas:
            ∂F_k_out_i/∂θ_out_i = −Σ_{j≠i} p_ij(1−p_ij)
            ∂F_k_in_j/∂θ_in_j  = −Σ_{i≠j} p_ij(1−p_ij)
            ∂F_s_out_i/∂η_out_i = −Σ_{j≠i} p_ij·G_ij²·(1−p_ij+z_ij)
            ∂F_s_in_j/∂η_in_j  = −Σ_{i≠j} p_ij·G_ij²·(1−p_ij+z_ij)

        where z_ij = exp(−η_out_i−η_in_j) = 1 − 1/G_ij.

        Args:
            theta: Full parameter vector [θ_out|θ_in|η_out|η_in], shape (4N,).

        Returns:
            Diagonal Jacobian, shape (4N,), all entries ≤ 0.
        """
        theta = _to_tensor(theta)
        N = self.N
        theta_out = theta[:N]
        theta_in = theta[N : 2 * N]
        eta_out = theta[2 * N : 3 * N]
        eta_in = theta[3 * N :]

        eta = eta_out[:, None] + eta_in[None, :]
        eta_safe = eta.clamp(min=_Z_G_CLAMP)
        G = -1.0 / torch.expm1(-eta_safe)
        log_q = -torch.log(torch.expm1(eta_safe))
        logit_p = -theta_out[:, None] - theta_in[None, :] + log_q
        P = torch.sigmoid(logit_p)

        P.fill_diagonal_(0.0)
        G.fill_diagonal_(0.0)

        pq = P * (1.0 - P)
        pq.fill_diagonal_(0.0)
        PGG1 = P * G * (G - 1.0)
        PGG1.fill_diagonal_(0.0)
        CORR = pq * G.pow(2)
        CORR.fill_diagonal_(0.0)

        h_k_out = -pq.sum(1)
        h_k_in = -pq.sum(0)
        h_s_out = -(PGG1 + CORR).sum(1)
        h_s_in = -(PGG1 + CORR).sum(0)
        return torch.cat([h_k_out, h_k_in, h_s_out, h_s_in])

    # ------------------------------------------------------------------
    # Initial guess
    # ------------------------------------------------------------------

    def initial_theta(self, method: str = "degrees") -> torch.Tensor:
        """Return a 4N initial guess for the DECM solver.

        Methods:
            * ``"degrees"`` (default): DCM heuristic for topology + mean-field
              for weights.
            * ``"random"``: uniform random in [0.1, 2.0] for all components.
            * ``"uniform"``: fixed value 1.0 for all components.

        Zero-degree nodes have θ = +_THETA_MAX; zero-strength nodes have
        η = +_ETA_MAX.

        Args:
            method: Initialisation strategy.

        Returns:
            Initial parameter vector θ₀, shape (4N,).
        """
        N = self.N

        if method in ("degrees", "random", "uniform"):
            if method == "degrees":
                p_out = self.k_out.clamp(1e-6, N - 1 - 1e-6) / (N - 1)
                p_in = self.k_in.clamp(1e-6, N - 1 - 1e-6) / (N - 1)
                theta_out = -torch.log(torch.sqrt(p_out))
                theta_in = -torch.log(torch.sqrt(p_in))
            elif method == "random":
                theta_out = torch.empty(N, dtype=torch.float64).uniform_(0.1, 2.0)
                theta_in = torch.empty(N, dtype=torch.float64).uniform_(0.1, 2.0)
            else:  # uniform
                theta_out = torch.ones(N, dtype=torch.float64)
                theta_in = torch.ones(N, dtype=torch.float64)

            # Clamp topology initialisation
            theta_out = torch.where(self.zero_k_out, torch.full_like(theta_out, _THETA_MAX), theta_out)
            theta_in = torch.where(self.zero_k_in, torch.full_like(theta_in, _THETA_MAX), theta_in)
            sat_out = self.k_out >= (N - 1)
            sat_in = self.k_in >= (N - 1)
            theta_out = torch.where(sat_out, torch.full_like(theta_out, -_THETA_MAX), theta_out)
            theta_in = torch.where(sat_in, torch.full_like(theta_in, -_THETA_MAX), theta_in)

            if method in ("degrees", "uniform"):
                # Mean-field inversion: for a node i, s_i/k_i ≈ 1/(1−β_i²) in the
                # homogeneous approximation (all pairs have the same β).  Solving
                # for β gives β = sqrt(1 − k_i/s_i), which we use as a warm-start.
                k_out_safe = self.k_out.clamp(min=1.0)
                k_in_safe = self.k_in.clamp(min=1.0)
                s_out_safe = self.s_out.clamp(min=1e-15)
                s_in_safe = self.s_in.clamp(min=1e-15)
                ratio_out = (k_out_safe / s_out_safe).clamp(max=1.0 - 1e-9)
                ratio_in = (k_in_safe / s_in_safe).clamp(max=1.0 - 1e-9)
                beta_out = torch.sqrt(1.0 - ratio_out).clamp(min=1e-15)
                beta_in = torch.sqrt(1.0 - ratio_in).clamp(min=1e-15)
                eta_out = (-torch.log(beta_out)).clamp(min=1e-10, max=_ETA_MAX)
                eta_in = (-torch.log(beta_in)).clamp(min=1e-10, max=_ETA_MAX)
            else:  # random
                eta_out = torch.empty(N, dtype=torch.float64).uniform_(0.1, 2.0)
                eta_in = torch.empty(N, dtype=torch.float64).uniform_(0.1, 2.0)

            eta_out = torch.where(self.zero_s_out, torch.full_like(eta_out, _ETA_MAX), eta_out)
            eta_in = torch.where(self.zero_s_in, torch.full_like(eta_in, _ETA_MAX), eta_in)
            return torch.cat([theta_out, theta_in, eta_out, eta_in])

        raise ValueError(f"Unknown initial-guess method: {method!r}")

    # ------------------------------------------------------------------
    # Constraint evaluation
    # ------------------------------------------------------------------

    def constraint_error(self, theta: _ArrayLike) -> float:
        """Return max|F(θ,η)| over all 4N constraints.

        Args:
            theta: Full parameter vector, shape (4N,).

        Returns:
            Max-abs constraint error (scalar).
        """
        return self.residual(theta).abs().max().item()

    def max_relative_error(self, theta: _ArrayLike) -> float:
        """Return the maximum relative error over all non-zero constraints.

        MRE = max over {i : constraint_i > 0} of |F_i| / constraint_i.

        Args:
            theta: Full parameter vector, shape (4N,).

        Returns:
            Maximum relative constraint error (scalar).
        """
        F = self.residual(theta).abs()
        targets = torch.cat([self.k_out, self.k_in, self.s_out, self.s_in]).to(torch.float64)
        nz = targets > 0
        if not nz.any():
            return 0.0
        return (F[nz] / targets[nz]).max().item()

    # ------------------------------------------------------------------
    # Solver interface
    # ------------------------------------------------------------------

    def solve_tool(
        self,
        ic: str = "degrees",
        tol: float = 1e-6,
        max_iter: int = 2000,
        max_time: float = 0,
        variant: str = "theta-newton",
        anderson_depth: int = 10,
    ) -> SolverResult:
        """Solve the DECM equations with the alternating GS-Newton solver.

        Args:
            ic:            Initial condition method (``"degrees"``, ``"random"``,
                           ``"uniform"``).
            tol:           Convergence tolerance on the ℓ∞ residual.
            max_iter:      Maximum iterations.
            max_time:      Wall-clock time limit in seconds (0 = no limit).
            variant:       Solver variant (only ``"theta-newton"`` is supported).
            anderson_depth: Anderson acceleration depth.

        Returns:
            :class:`~src.solvers.base.SolverResult` instance.
        """
        from src.solvers.fixed_point_decm import solve_fixed_point_decm

        theta0 = self.initial_theta(ic)
        result = solve_fixed_point_decm(
            residual_fn=self.residual,
            theta0=theta0,
            k_out=self.k_out,
            k_in=self.k_in,
            s_out=self.s_out,
            s_in=self.s_in,
            tol=tol,
            max_iter=max_iter,
            variant=variant,
            chunk_size=0,
            anderson_depth=anderson_depth,
            max_time=max_time,
        )
        return result
