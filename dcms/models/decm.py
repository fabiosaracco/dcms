"""Directed Enhanced Configuration Model (DECM) — binary + weighted, fully coupled.

The DECM fixes *four* sequences per node: out-degree, in-degree, out-strength
and in-strength (k_out, k_in, s_out, s_in) with 4N unknowns:

    θ = [θ_out | θ_in | η_out | η_in]

The physical multipliers are:
    x_i      = exp(-θ_out_i),   y_j = exp(-θ_in_j)    (topology)
    β_out_i  = exp(-η_out_i),   β_in_j = exp(-η_in_j) (weight)

Unlike the qDECM (where p_ij uses only the topology parameters), the DECM
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
import time as _time
from typing import Union

import torch

from dcms.models.parameters import qDECM_LARGE_N_THRESHOLD as _LARGE_N_THRESHOLD
from dcms.models.parameters import _DEFAULT_CHUNK, _ETA_MAX, _ETA_MIN
from dcms.solvers.base import SolverResult

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

    where ``θ_out, θ_in`` are the topology (degree) multipliers and
    ``η_out, η_in`` are the weight (strength) multipliers.

    After calling :meth:`solve_tool`, the result is stored as ``self.sol``
    (:class:`~src.solvers.base.SolverResult`, ``theta`` shape 4N).

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
        self.sol: SolverResult | None = None

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

    def bic(self, theta: _ArrayLike) -> float:
        """Return the Bayesian Information Criterion, BIC = k·ln(n) − 2·ln L.

        k = 4N (the θ_out_i, θ_in_i, η_out_i, η_in_i Lagrange multipliers)
        and n = N(N−1) (the directed dyads i≠j summed over by
        :meth:`neg_log_likelihood`, each contributing one joint
        existence+weight observation).  Lower is better.

        Args:
            theta: Full parameter vector [θ_out|θ_in|η_out|η_in], shape (4N,).

        Returns:
            Scalar BIC(θ,η).
        """
        N = self.N
        return 4 * N * math.log(N * (N - 1)) + 2 * self.neg_log_likelihood(theta)

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
            * ``"qdecm"``: solve qDECM first (DCM topology + conditioned DWCM
              weights) and use the concatenated 4N solution as warm-start.
              More expensive to compute but lands in a much better basin for
              networks with extreme s/k ratios or heterogeneous hubs.

        Zero-degree nodes have θ = +_THETA_MAX; zero-strength nodes have
        η = +_ETA_MAX.

        Args:
            method: Initialisation strategy.

        Returns:
            Initial parameter vector θ₀, shape (4N,).
        """
        N = self.N

        if method == "qdecm":
            # Late import to avoid circular dependency at module level.
            import io, contextlib
            from dcms.models.qdecm import qDECMModel
            qdecm = qDECMModel(self.k_out, self.k_in, self.s_out, self.s_in)
            # Suppress solver progress messages — this is an IC computation,
            # not a user-facing solve.
            with contextlib.redirect_stdout(io.StringIO()):
                qdecm.solve_tool(ic_topo="degrees", ic_wei="topology")
            theta_topo = torch.as_tensor(qdecm.sol.best_theta[:2 * N], dtype=torch.float64)
            theta_weight = torch.as_tensor(qdecm.sol.best_theta[2 * N:], dtype=torch.float64)
            return torch.cat([theta_topo, theta_weight])

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
        multi_start: bool = True,
        backend: str = "auto",
        num_threads: int = 0,
        verbose: bool = False,
        monitor: bool = False,
        topo_weig: bool = False,
        hub_sk_threshold: float = 0.0,
        leaf_k_threshold: float = 0.0,
        backtracking_gamma: float = 0.0,
        z_clamp: float = 1e-8,
        reduce_degeneracy: bool = True,
        blowup_factor: float | None = None,
        patience: int = 750,
        noise_base: float = 1e-2,
        noise_cap_mult: float = 16.0,
        noise_growth: float = 2.0,
        max_stalls: int = 5,
        seed: int | None = None,
    ) -> bool:
        """Solve the DECM equations with the alternating GS-Newton solver.

        If the primary IC does not converge and ``multi_start=True``,
        automatically retries with ``"qdecm"`` and ``"random"`` warm-starts.
        Each fallback attempt uses the same ``max_iter`` budget.  The result
        stored in ``self.sol`` is always the iterate with the lowest residual
        found across all attempts.

        Results are stored on the model instance as ``self.sol``
        (:class:`~src.solvers.base.SolverResult`, full 4N parameter vector
        ``theta = [θ_out | θ_in | η_out | η_in]``).

        Args:
            ic:            Primary initial condition for the solver.  Pass a
                           string (``"degrees"``, ``"random"``, ``"uniform"``,
                           ``"qdecm"``) to use a built-in initializer, or pass
                           a numpy array / list / :class:`torch.Tensor` of
                           shape ``(4N,)`` to warm-start from a custom vector
                           ``[θ_out | θ_in | η_out | η_in]`` (e.g. a previous
                           ``sol.best_theta``).  When an array is supplied,
                           ``multi_start`` is ignored and a single run is
                           performed.
            tol:           Convergence tolerance on the ℓ∞ residual.
            max_iter:      Maximum iterations per attempt.
            max_time:      Wall-clock time limit in seconds per attempt
                           (0 = no limit).
            variant:       Solver variant: only ``"theta-newton"`` is supported.
            anderson_depth: Anderson acceleration depth.
            multi_start:   If ``True`` (default), retry with fallback ICs when
                           the primary IC does not converge.  Set to ``False``
                           for clean per-IC benchmarking.  Ignored when
                           ``ic`` is an array.
            backend:       Compute backend: ``"auto"`` (default), ``"pytorch"``,
                           or ``"numba"``.  ``"auto"`` uses PyTorch for
                           N ≤ 5 000 and Numba for larger networks.
            num_threads:   Number of Numba parallel threads. 0 (default)
                           means *auto*: uses all CPUs available to the current
                           process (respects ``taskset``/cgroup quotas on
                           Linux). Positive values are clamped to the available
                           CPU count to avoid thread-creation errors on
                           resource-limited servers. Only has effect when Numba
                           is selected as the backend.
            verbose:       If ``True``, print a progress line at every iteration
                           (timestamp, iteration count, elapsed time, MRE).
                           Default=False.
            monitor:       If ``True`` (and ``verbose=True``), overwrite the
                           same terminal line at each iteration (``end='\\r'``)
                           so only the latest status is visible.  Default=False.
            topo_weig:     If ``True`` (and ``verbose=True``), the per-iteration
                           progress line splits the MRE into ``MRE_topo``
                           (degree part) and ``MRE_weights`` (strength part).
                           If ``False`` (default), it prints a single global
                           ``MRE`` instead.
            hub_sk_threshold: When > 0, nodes with observed s_i/k_hat_i above
                           this value are treated as hub nodes and solved
                           exactly each iteration via 1D bisection over η
                           rather than the global Newton step.  Use for
                           networks with nodes having very high s/k ratios
                           (e.g. ``s/k > 5``) that cause stagnation.
                           Default=0.0 (disabled).
            leaf_k_threshold: When > 0, nodes with observed degree ``k_out``
                           (resp. ``k_in``) ``<= leaf_k_threshold`` are
                           treated as leaf nodes and have ``θ_out`` (resp.
                           ``θ_in``) solved exactly each iteration via 1D
                           bisection over the degree equation -- the θ-side
                           analogue of ``hub_sk_threshold``. Use for networks
                           that stagnate just above ``tol`` with the worst
                           residuals concentrated at low-degree nodes (e.g.
                           ``k=1``), where the same relative tolerance
                           demands very tight absolute precision that the
                           global step can struggle to hit. Default=0.0
                           (disabled). Applied once per iteration to the
                           final theta_next (after mixing/restart/rollback),
                           vectorized across all leaf nodes at once -- see
                           the detailed note on this parameter in
                           :func:`~dcms.solvers.fixed_point_decm.solve_fixed_point_decm`.
            backtracking_gamma: When > 0, enables a per-step backtracking
                           line search.  After each Newton step the residual
                           at the proposed iterate is evaluated; if it exceeds
                           ``backtracking_gamma`` times the current residual,
                           the step is halved (up to 5 times).  Typical value:
                           ``2.0``.  Default=0.0 (disabled).
            z_clamp:       Floor on the raw z=eta_out+eta_in sum, tied
                           between the G-curvature damping and the
                           per-step trust-region floor (both roles must
                           stay coupled -- decoupling them was tested and
                           made things worse, reintroducing an
                           unrecoverable plateau). Default 1e-8 (original
                           value, unchanged behavior). Raise it (e.g. to
                           1e-6) if the solver stagnates on a network with
                           extreme s/k hub nodes -- see
                           :func:`~dcms.solvers.fixed_point_decm.solve_fixed_point_decm`'s
                           ``z_clamp`` docs for the full mechanism and
                           trade-off.
            reduce_degeneracy: If ``True`` (default), nodes sharing the exact
                           same ``(k_out, k_in, s_out, s_in)`` 4-tuple are
                           collapsed into a single group before solving (see
                           README §2.5) — solves M ≤ N group unknowns instead
                           of N per-node ones, often 10-25x faster on real
                           networks with heavy-tailed degree/strength
                           distributions. Mathematically exact (not an
                           approximation). Only supported with
                           ``variant="theta-newton"``, ``backend != "numba"``,
                           and ``backtracking_gamma == 0.0``; silently falls
                           back to the full (unreduced) solver otherwise,
                           with a printed note.
            blowup_factor: Multiplicative threshold for the Anderson blowup
                           guard: a rollback triggers when the current
                           iteration's residual exceeds ``blowup_factor``
                           times the best residual ever seen this call (a
                           running minimum, so this also catches a slow
                           drift away from the record over many iterations,
                           not just a sudden spike). ``None`` (default) uses
                           the built-in scale-adaptive formula
                           ``max(200, min(5000, 200_000/N))``. Lower this
                           (e.g. 20-50) on instances that visibly wander far
                           from their best point without any single jump
                           large enough to trip the default floor -- trades
                           more frequent, cheap (noise-free) rollbacks for
                           less wasted exploration of bad regions.
            patience:      If ``best_theta_res`` has not improved for this
                           many consecutive iterations, restart from
                           ``model.sol.best_theta`` plus escalating Gaussian
                           noise instead of continuing from the stuck
                           iterate -- this is what makes hard instances
                           (hub-heavy networks that would otherwise stall
                           at a quasi-fixed point) converge unattended. The
                           same recovery kicks in when the Anderson blowup
                           guard trips repeatedly with no improvement in
                           between; an isolated, self-recovering blowup is
                           left alone. Set ``patience <= 0`` to disable and
                           get the old fail-fast behaviour (stop immediately
                           on stagnation). Default 750 (validated end-to-end
                           on a real N=15168 instance -- see README
                           "Checkpointed multi-chunk runs"). Inert (zero
                           effect on the result) for instances that never
                           stagnate or blow up.
            noise_base:    Scale of the multiplicative (log-scale) noise on
                           the first perturbed restart: every component of
                           theta is scaled as ``x_i *= exp(N(0, noise_base))``.
                           See :func:`~dcms.solvers.fixed_point_decm.solve_fixed_point_decm`
                           for why this is multiplicative rather than
                           additive. Default 1e-2.
            noise_cap_mult: Noise scale grows by ``noise_growth`` on each
                           consecutive restart that fails to improve the
                           record, capped at ``noise_base * noise_cap_mult``;
                           resets to ``noise_base`` on the next genuine
                           improvement. Default 16.0.
            noise_growth:  Per-restart growth rate of the noise scale.
                           Default 2.0 (doubling). See
                           :func:`~dcms.solvers.fixed_point_decm.solve_fixed_point_decm`
                           for when to lower this.
            max_stalls:    Give up (``converged=False``) after this many
                           restarts *at the noise cap* in a row without
                           improving the record. Default 5.
            seed:          Seed for the perturbation RNG (private, does not
                           touch global RNG state). ``None`` (default) is
                           unseeded/non-reproducible -- only relevant for
                           instances that actually hit a perturbed restart.

        Returns:
            ``True`` if any attempt converged, ``False`` otherwise.
        """
        import torch as _torch
        from dcms.solvers.fixed_point_decm import solve_fixed_point_decm, solve_fixed_point_decm_degenerate

        _FALLBACK_ICS = ["qdecm", "random"]

        _use_reduced = (
            reduce_degeneracy and variant == "theta-newton"
            and backend != "numba" and backtracking_gamma == 0.0
        )
        if reduce_degeneracy and not _use_reduced:
            print(
                "reduce_degeneracy=True requires variant='theta-newton', "
                "backend!='numba', and backtracking_gamma==0.0 -- falling "
                "back to the full (unreduced) solver."
            )

        t_global_start = _time.perf_counter()

        def _remaining() -> float:
            """Return remaining wall-clock budget (0 = no limit)."""
            if max_time <= 0:
                return 0.0
            remaining = max_time - (_time.perf_counter() - t_global_start)
            return max(remaining, 0.0)

        def _run_once(ic_name: str, theta0_override=None):
            theta0 = (
                _torch.as_tensor(theta0_override, dtype=_torch.float64)
                if theta0_override is not None
                else self.initial_theta(ic_name)
            )
            if _use_reduced:
                return solve_fixed_point_decm_degenerate(
                    theta0=theta0,
                    k_out=self.k_out,
                    k_in=self.k_in,
                    s_out=self.s_out,
                    s_in=self.s_in,
                    tol=tol,
                    max_iter=max_iter,
                    anderson_depth=anderson_depth,
                    max_time=_remaining(),
                    backend=backend,
                    num_threads=num_threads,
                    verbose=verbose,
                    monitor=monitor,
                    topo_weig=topo_weig,
                    hub_sk_threshold=hub_sk_threshold,
                    leaf_k_threshold=leaf_k_threshold,
                    z_clamp=z_clamp,
                    blowup_factor=blowup_factor,
                    patience=patience,
                    noise_base=noise_base,
                    noise_cap_mult=noise_cap_mult,
                    noise_growth=noise_growth,
                    max_stalls=max_stalls,
                    seed=seed,
                )
            return solve_fixed_point_decm(
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
                max_time=_remaining(),
                backend=backend,
                num_threads=num_threads,
                verbose=verbose,
                monitor=monitor,
                topo_weig=topo_weig,
                hub_sk_threshold=hub_sk_threshold,
                leaf_k_threshold=leaf_k_threshold,
                backtracking_gamma=backtracking_gamma,
                z_clamp=z_clamp,
                blowup_factor=blowup_factor,
                patience=patience,
                noise_base=noise_base,
                noise_cap_mult=noise_cap_mult,
                noise_growth=noise_growth,
                max_stalls=max_stalls,
                seed=seed,
            )

        if not isinstance(ic, str):
            self.ic = _torch.as_tensor(ic, dtype=_torch.float64)
            self.sol = _run_once("degrees", theta0_override=ic)
            if self.sol.message:
                print(self.sol.message+" "*50)
            return self.sol.converged

        self.ic = self.initial_theta(ic)
        self.sol = _run_once(ic)
        if self.sol.message:
            print(self.sol.message+" "*50) # the +" "*50 is necessary to avoid the output to be badly overwritten in the case of monitor=True

        if multi_start and not self.sol.converged:
            for fallback in _FALLBACK_ICS:
                if fallback == ic:
                    continue
                # If the global time budget is exhausted, stop trying fallbacks.
                if max_time > 0 and _remaining() <= 0:
                    break
                fb_result = _run_once(fallback)
                if fb_result.message:
                    print(fb_result.message+" "*50) # the +" "*50 is necessary to avoid the output to be badly overwritten in the case of monitor=True
                if fb_result.residuals[-1] < self.sol.residuals[-1]:
                    self.sol = fb_result
                if self.sol.converged:
                    break

        return self.sol.converged

    def _sample_raw(self, seed: int | None = None,
                    chunk_size: int = 512) -> "np.ndarray":
        """Internal sampler — returns shape ``(L, 3)`` NumPy array (no tolist).

        DECM link probabilities are not rank-1 in the fitnesses (the
        weight-multiplier term ``log_q_ij`` depends on both ``i`` and ``j``),
        so the O(L) Poisson sparse trick cannot be applied.  The exact
        O(N²) chunk sampler is used instead.
        """
        import numpy as np
        if self.sol is None:
            raise RuntimeError("Call solve_tool() first.")
        rng = np.random.default_rng(seed)
        N = self.N
        theta = np.asarray(self.sol.best_theta, dtype=np.float64)
        theta_out = theta[:N]
        theta_in  = theta[N : 2 * N]
        eta_out   = theta[2 * N : 3 * N]
        eta_in    = theta[3 * N :]
        beta_out  = np.exp(-eta_out)
        beta_in   = np.exp(-eta_in)
        q_in  = 1.0 / np.expm1(eta_in).clip(min=1e-300)
        q_out = 1.0 / np.expm1(eta_out).clip(min=1e-300)
        parts: list = []
        for i_start in range(0, N, chunk_size):
            i_end = min(i_start + chunk_size, N)
            q_out_chunk = q_out[i_start:i_end]
            eta_chunk = eta_out[i_start:i_end, None] + eta_in[None, :]
            log_q = -np.log(np.expm1(eta_chunk).clip(min=1e-300))
            logit_p = -theta_out[i_start:i_end, None] - theta_in[None, :] + log_q
            p = 1.0 / (1.0 + np.exp(-logit_p))
            chunk_range = np.arange(i_end - i_start)
            p[chunk_range, i_start + chunk_range] = 0.0
            rows, cols = np.where(rng.random(p.shape) < p)
            if len(rows):
                b_vals = (beta_out[i_start + rows] * beta_in[cols]).clip(0.0, 1.0 - 1e-12)
                weights = rng.geometric(1.0 - b_vals)
                parts.append(np.column_stack([i_start + rows, cols, weights]))
        if not parts:
            return np.empty((0, 3), dtype=int)
        return np.vstack(parts)

    def sample(self, seed: int | None = None,
               chunk_size: int = 512) -> "np.ndarray":
        """Sample a weighted directed network from the fitted DECM.

        Two-step procedure using the exact DECM distribution:

        1. **Topology** — draw ``A_ij ~ Bernoulli(p_ij)`` using the full
           DECM link-probability formula (weight multipliers enter ``p_ij``)::

               p_ij = σ(−θ_out_i − θ_in_j − log(exp(η_out_i + η_in_j) − 1))

        2. **Weights** — for each present link draw ``w_ij`` from a geometric
           distribution starting at 1 (conditional on the link existing)::

               P(w_ij = k | A_ij = 1) = (1 − β_ij) β_ij^{k−1},   k = 1, 2, …
               β_ij = β_out_i β_in_j = exp(−η_out_i − η_in_j)

        Args:
            seed: Random seed for reproducibility.
            chunk_size: Number of source rows processed at a time.

        Returns:
            ``np.ndarray`` of shape ``(L, 3)`` with integer columns
            ``[source, target, weight]``.  Iterating ``for i, j, w in edges``
            works identically to the old list-of-lists format.

        Raises:
            RuntimeError: if :meth:`solve_tool` has not been called yet.
        """
        return self._sample_raw(seed=seed, chunk_size=chunk_size)

    def sample_many(self, n: int, seed: int | None = None,
                    n_jobs: int = -1) -> list:
        """Generate ``n`` independent samples in parallel.

        Each sample is a ``np.ndarray`` of shape ``(L_k, 3)`` with columns
        ``[source, target, weight]``.  Iterating ``for i, j, w in sample``
        works the same as with the list returned by :meth:`sample`.

        Uses :class:`~concurrent.futures.ThreadPoolExecutor`.  NumPy releases
        the GIL during random-number generation and array operations, giving
        near-linear speedup up to the number of physical cores.  Note that
        DECM uses an O(N²) sampler (link probabilities are not rank-1), so the
        per-sample cost is higher than for DCM/DWCM/qDECM.

        Args:
            n: Number of independent samples to draw.
            seed: Master seed from which per-sample seeds are derived.
            n_jobs: Number of worker threads. ``-1`` (default) uses all CPUs.

        Returns:
            List of ``n`` NumPy arrays, one per sample.

        Raises:
            RuntimeError: if :meth:`solve_tool` has not been called yet.
        """
        import numpy as np
        from concurrent.futures import ThreadPoolExecutor
        seeds = np.random.default_rng(seed).integers(0, 2**31, n).tolist()
        if n_jobs == 1:
            return [self._sample_raw(seed=s) for s in seeds]
        max_workers = None if n_jobs < 0 else n_jobs
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            return list(ex.map(self._sample_raw, seeds))
