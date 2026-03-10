"""Directed Configuration Model (DCM) — binary, directed connections.

The DCM fixes the out- and in-degree sequences (k_out, k_in).  The maximum-
entropy probability of an arc i→j is

    p_ij = x_i * y_j / (1 + x_i * y_j)        (i ≠ j)

where x_i = exp(-θ_out_i) and y_i = exp(-θ_in_i) are the Lagrange
multipliers in exponential parametrisation.

The system of equations to solve is F(θ) = 0, where

    F_i(θ)     = Σ_{j≠i} p_ij  − k_out_i      for i = 0 … N-1
    F_{N+i}(θ) = Σ_{j≠i} p_ji  − k_in_i       for i = 0 … N-1

(residual = expected − observed)

Reference:
    Squartini & Garlaschelli, New J. Phys. 13 (2011) 083001.
"""
from __future__ import annotations

from typing import Union

import torch


# Type alias for inputs: accept both numpy arrays and torch tensors.
_ArrayLike = Union[torch.Tensor, "numpy.ndarray"]  # type: ignore[name-defined]


def _to_tensor(x: _ArrayLike, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Convert *x* to a float64 CPU torch.Tensor (no-copy if already correct)."""
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype)
    return torch.tensor(x, dtype=dtype)


class DCMModel:
    """Encapsulates all DCM equations for a network of *N* nodes.

    Internally all quantities are stored and computed as ``torch.float64``
    tensors, enabling seamless GPU acceleration in the future.

    Args:
        k_out: Observed out-degree sequence, shape (N,).
        k_in:  Observed in-degree sequence, shape (N,).
    """

    def __init__(self, k_out: _ArrayLike, k_in: _ArrayLike) -> None:
        self.k_out: torch.Tensor = _to_tensor(k_out)
        self.k_in: torch.Tensor = _to_tensor(k_in)
        self.N: int = int(self.k_out.shape[0])
        if self.k_in.shape[0] != self.N:
            raise ValueError("k_out and k_in must have the same length.")

    # ------------------------------------------------------------------
    # Core probability matrix
    # ------------------------------------------------------------------

    def pij_matrix(self, theta: _ArrayLike) -> torch.Tensor:
        """Compute the N×N matrix of link probabilities p_ij.

        Uses ``torch.sigmoid`` for numerical stability (avoids double
        evaluation of exp and overflow for large |θ|).  Diagonal entries
        are set to 0 (no self-loops).

        Args:
            theta: Parameter vector [θ_out | θ_in], shape (2N,).

        Returns:
            Probability matrix P, shape (N, N), dtype torch.float64.
        """
        theta = _to_tensor(theta)
        N = self.N
        theta_out = theta[:N]
        theta_in = theta[N:]
        # log(x_i * y_j) = -θ_out_i - θ_in_j
        log_xy = -theta_out[:, None] - theta_in[None, :]  # (N, N)
        # p_ij = sigmoid(log_xy) — numerically stable for all magnitudes
        P = torch.sigmoid(log_xy)
        P.fill_diagonal_(0.0)
        return P

    # ------------------------------------------------------------------
    # Residual (system of equations)
    # ------------------------------------------------------------------

    def residual(self, theta: _ArrayLike) -> torch.Tensor:
        """Return F(θ) = [k_out_expected − k_out_obs | k_in_expected − k_in_obs].

        Args:
            theta: Parameter vector [θ_out | θ_in], shape (2N,).

        Returns:
            Residual vector F(θ), shape (2N,).
        """
        P = self.pij_matrix(theta)
        k_out_hat = P.sum(dim=1)   # row sums = expected out-degrees
        k_in_hat = P.sum(dim=0)    # col sums = expected in-degrees
        F = torch.empty(2 * self.N, dtype=torch.float64)
        F[: self.N] = k_out_hat - self.k_out
        F[self.N :] = k_in_hat - self.k_in
        return F

    # ------------------------------------------------------------------
    # Gradient of the log-likelihood (= +F(θ))
    # ------------------------------------------------------------------

    def gradient(self, theta: _ArrayLike) -> torch.Tensor:
        """Return ∇L(θ) = +F(θ), i.e., the residual vector.

        The log-likelihood of the DCM is

            L(θ) = −Σ_i θ_out_i·k_out_i − Σ_i θ_in_i·k_in_i
                   − Σ_{i≠j} log(1 + exp(−θ_out_i − θ_in_j))

        so ∂L/∂θ_out_i = −k_out_i + Σ_{j≠i} p_ij = F_i(θ).

        The gradient of −L is −F(θ).

        Args:
            theta: Parameter vector, shape (2N,).

        Returns:
            Gradient vector ∇L = F(θ), shape (2N,).
        """
        return self.residual(theta)

    # ------------------------------------------------------------------
    # Diagonal Hessian of the log-likelihood (≈ Newton step denominator)
    # ------------------------------------------------------------------

    def hessian_diag(self, theta: _ArrayLike) -> torch.Tensor:
        """Return the diagonal of the Hessian of L(θ).

        The second derivatives are:

            ∂²L/∂θ_out_i² = −Σ_{j≠i} p_ij(1 − p_ij)
            ∂²L/∂θ_in_i²  = −Σ_{j≠i} p_ji(1 − p_ji)

        Args:
            theta: Parameter vector, shape (2N,).

        Returns:
            Diagonal of the Hessian, shape (2N,).
        """
        P = self.pij_matrix(theta)
        Q = P * (1.0 - P)           # elementwise q_ij = p_ij(1−p_ij)
        h_out = -Q.sum(dim=1)       # row sums (out-degree contributions)
        h_in = -Q.sum(dim=0)        # col sums (in-degree contributions)
        return torch.cat([h_out, h_in])

    # ------------------------------------------------------------------
    # Full Jacobian of F(θ) (= Hessian of L, used by Newton solvers)
    # ------------------------------------------------------------------

    def jacobian(self, theta: _ArrayLike) -> torch.Tensor:
        """Return the full Jacobian matrix J = ∂F/∂θ = Hess(L), shape (2N, 2N).

        L is concave so J is negative semi-definite.  Denoting Q = P⊙(1−P):

            J_out,out = −diag(Σ_{j≠i} Q_ij)   [diagonal, negative]
            J_out,in  = −Q                      [zero on diagonal]
            J_in,out  = −Qᵀ                    [zero on diagonal]
            J_in,in   = −diag(Σ_{j≠i} Q_ji)   [diagonal, negative]

        Args:
            theta: Parameter vector, shape (2N,).

        Returns:
            Jacobian matrix, shape (2N, 2N), dtype torch.float64.
        """
        N = self.N
        P = self.pij_matrix(theta)
        Q = P * (1.0 - P)  # Q[i,i] = 0 since P[i,i] = 0
        idx = torch.arange(N)

        J = torch.zeros(2 * N, 2 * N, dtype=torch.float64)
        # Top-left block: ∂F_out_i / ∂θ_out_i (diagonal, negative)
        J[idx, idx] = -Q.sum(dim=1)
        # Top-right block: ∂F_out_i / ∂θ_in_j = −Q_ij  (diagonal zero since Q[i,i]=0)
        J[:N, N:] = -Q
        # Bottom-left block: ∂F_in_i / ∂θ_out_j = −Q_ji
        J[N:, :N] = -Q.T
        # Bottom-right block: ∂F_in_i / ∂θ_in_i (diagonal, negative)
        J[N + idx, N + idx] = -Q.sum(dim=0)
        return J

    # ------------------------------------------------------------------
    # Initial-guess utilities
    # ------------------------------------------------------------------

    def initial_theta(self, method: str = "degrees") -> torch.Tensor:
        """Return a sensible starting point θ₀ for the solvers.

        Args:
            method: ``"degrees"`` — use k/(N-1) as initial probability;
                    ``"random"``  — uniform random values in [0.1, 2.0].

        Returns:
            Initial parameter vector θ₀, shape (2N,).
        """
        N = self.N
        if method == "degrees":
            p_out = torch.clamp(self.k_out / (N - 1), 1e-6, 1 - 1e-6)
            p_in = torch.clamp(self.k_in / (N - 1), 1e-6, 1 - 1e-6)
            # p ≈ x·y/(1+x·y); use geometric mean heuristic x₀ = y₀ = √p
            x0 = torch.sqrt(p_out)
            y0 = torch.sqrt(p_in)
            theta_out = -torch.log(x0)
            theta_in = -torch.log(y0)
        elif method == "random":
            # No fixed seed — genuinely random each call
            theta_out = torch.empty(N, dtype=torch.float64).uniform_(0.1, 2.0)
            theta_in = torch.empty(N, dtype=torch.float64).uniform_(0.1, 2.0)
        else:
            raise ValueError(f"Unknown initial-guess method: {method!r}")
        return torch.cat([theta_out, theta_in])

    # ------------------------------------------------------------------
    # Negative log-likelihood (objective for L-BFGS minimisation)
    # ------------------------------------------------------------------

    def neg_log_likelihood(self, theta: _ArrayLike) -> float:
        """Return −L(θ), the convex quantity to be *minimised* by L-BFGS.

        The DCM log-likelihood (Squartini & Garlaschelli 2011, eq. 13) is:

            L(θ) = −Σ_i θ_out_i·k_out_i − Σ_i θ_in_i·k_in_i
                   − Σ_{i≠j} log(1 + exp(−θ_out_i − θ_in_j))

        Its gradient satisfies ∂L/∂θ_out_i = −k_out_i + Σ_{j≠i} p_ij = F_i,
        so ∇L = F(θ) and therefore ∇(−L) = −F(θ).

        The quantity returned is:
            −L = +Σ_i θ_out_i·k_out_i + Σ_i θ_in_i·k_in_i
                 + Σ_{i≠j} log(1 + exp(−θ_out_i − θ_in_j))

        which is convex (L is concave) and has its minimum at the solution.

        Args:
            theta: Parameter vector [θ_out | θ_in], shape (2N,).

        Returns:
            Scalar −L(θ) (convex, to be minimised).
        """
        theta = _to_tensor(theta)
        N = self.N
        theta_out = theta[:N]
        theta_in = theta[N:]
        log_xy = -theta_out[:, None] - theta_in[None, :]  # (N, N)
        log1p = torch.logaddexp(torch.zeros_like(log_xy), log_xy)
        log1p.fill_diagonal_(0.0)  # exclude self-loops
        return (theta_out @ self.k_out + theta_in @ self.k_in + log1p.sum()).item()

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
