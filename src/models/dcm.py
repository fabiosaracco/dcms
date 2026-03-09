"""Directed Configuration Model (DCM) — binary, directed connections.

The DCM fixes the out- and in-degree sequences (k_out, k_in).  The maximum-
entropy probability of an arc i→j is

    p_ij = x_i * y_j / (1 + x_i * y_j)        (i ≠ j)

where x_i = exp(-θ_out_i) and y_i = exp(-θ_in_i) are the Lagrange
multipliers in exponential parametrisation.

The system of equations to solve is F(θ) = 0, where

    F_i(θ)     = k_out_i  − Σ_{j≠i} p_ij      for i = 0 … N-1
    F_{N+i}(θ) = k_in_i   − Σ_{j≠i} p_ji      for i = 0 … N-1

Reference:
    Squartini & Garlaschelli, New J. Phys. 13 (2011) 083001.
"""
from __future__ import annotations

import numpy as np


class DCMModel:
    """Encapsulates all DCM equations for a network of *N* nodes.

    Args:
        k_out: Observed out-degree sequence, shape (N,).
        k_in:  Observed in-degree sequence, shape (N,).
    """

    def __init__(self, k_out: np.ndarray, k_in: np.ndarray) -> None:
        self.k_out = np.asarray(k_out, dtype=np.float64)
        self.k_in = np.asarray(k_in, dtype=np.float64)
        self.N: int = len(self.k_out)
        if len(self.k_in) != self.N:
            raise ValueError("k_out and k_in must have the same length.")

    # ------------------------------------------------------------------
    # Core probability matrix
    # ------------------------------------------------------------------

    def pij_matrix(self, theta: np.ndarray) -> np.ndarray:
        """Compute the N×N matrix of link probabilities p_ij.

        Diagonal entries are set to 0 (no self-loops).

        Args:
            theta: Parameter vector [θ_out | θ_in], shape (2N,).

        Returns:
            Probability matrix P, shape (N, N), dtype float64.
        """
        N = self.N
        theta_out = theta[:N]
        theta_in = theta[N:]
        # log p_ij = -θ_out_i - θ_in_j; exponent trick for stability
        log_xy = -theta_out[:, None] - theta_in[None, :]  # (N, N)
        # p_ij = exp(log_xy) / (1 + exp(log_xy)) = sigmoid(log_xy)
        P = np.exp(log_xy) / (1.0 + np.exp(log_xy))
        np.fill_diagonal(P, 0.0)
        return P

    # ------------------------------------------------------------------
    # Residual (system of equations)
    # ------------------------------------------------------------------

    def residual(self, theta: np.ndarray) -> np.ndarray:
        """Return F(θ) = [k_out_expected − k_out_obs | k_in_expected − k_in_obs].

        Args:
            theta: Parameter vector [θ_out | θ_in], shape (2N,).

        Returns:
            Residual vector F(θ), shape (2N,).
        """
        P = self.pij_matrix(theta)
        k_out_hat = P.sum(axis=1)   # row sums = expected out-degrees
        k_in_hat = P.sum(axis=0)    # col sums = expected in-degrees
        F = np.empty(2 * self.N, dtype=np.float64)
        F[: self.N] = k_out_hat - self.k_out
        F[self.N :] = k_in_hat - self.k_in
        return F

    # ------------------------------------------------------------------
    # Gradient of the log-likelihood (= −F(θ))
    # ------------------------------------------------------------------

    def gradient(self, theta: np.ndarray) -> np.ndarray:
        """Return ∇L(θ) = −F(θ), i.e., the negative residual.

        The log-likelihood of the DCM is

            L(θ) = Σ_i θ_out_i·k_out_i + Σ_i θ_in_i·k_in_i
                   − Σ_{i≠j} log(1 + exp(−θ_out_i − θ_in_j))

        so ∂L/∂θ_out_i = k_out_i − Σ_{j≠i} p_ij  = −F_i(θ).

        Args:
            theta: Parameter vector, shape (2N,).

        Returns:
            Gradient vector, shape (2N,).
        """
        return -self.residual(theta)

    # ------------------------------------------------------------------
    # Diagonal Hessian of the log-likelihood (≈ Newton step denominator)
    # ------------------------------------------------------------------

    def hessian_diag(self, theta: np.ndarray) -> np.ndarray:
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
        h_out = -Q.sum(axis=1)      # row sums (out-degree contributions)
        h_in = -Q.sum(axis=0)       # col sums (in-degree contributions)
        return np.concatenate([h_out, h_in])

    # ------------------------------------------------------------------
    # Full Jacobian of F(θ) (= −Hessian of L, used by Newton solvers)
    # ------------------------------------------------------------------

    def jacobian(self, theta: np.ndarray) -> np.ndarray:
        """Return the full Jacobian matrix J = ∂F/∂θ, shape (2N, 2N).

        The log-likelihood L is concave, so its Hessian Hess(L) = J is
        negative semi-definite.  Denoting Q = P ⊙ (1−P):

            J_out,out = −diag(Σ_{j≠i} Q_ij)   [diagonal, negative]
            J_out,in  = −Q                      [zero on diagonal]
            J_in,out  = −Q^T                    [zero on diagonal]
            J_in,in   = −diag(Σ_{j≠i} Q_ji)   [diagonal, negative]

        Args:
            theta: Parameter vector, shape (2N,).

        Returns:
            Jacobian matrix, shape (2N, 2N), dtype float64.
        """
        N = self.N
        P = self.pij_matrix(theta)
        Q = P * (1.0 - P)  # Q[i,i] = 0 since P[i,i] = 0

        J = np.zeros((2 * N, 2 * N), dtype=np.float64)
        # Top-left block: ∂F_out_i / ∂θ_out_i (diagonal, negative)
        np.fill_diagonal(J[:N, :N], -Q.sum(axis=1))
        # Top-right block: ∂F_out_i / ∂θ_in_j = −Q_ij  (zero on diagonal already)
        J[:N, N:] = -Q
        # Bottom-left block: ∂F_in_i / ∂θ_out_j = −Q_ji  (zero on diagonal already)
        J[N:, :N] = -Q.T
        # Bottom-right block: ∂F_in_i / ∂θ_in_i (diagonal, negative)
        np.fill_diagonal(J[N:, N:], -Q.sum(axis=0))
        return J

    # ------------------------------------------------------------------
    # Initial-guess utilities
    # ------------------------------------------------------------------

    def initial_theta(self, method: str = "degrees") -> np.ndarray:
        """Return a sensible starting point θ₀ for the solvers.

        Args:
            method: ``"degrees"`` — use k/(N-1) as initial probability;
                    ``"random"``  — small random perturbation around 0.5.

        Returns:
            Initial parameter vector θ₀, shape (2N,).
        """
        N = self.N
        if method == "degrees":
            p_out = np.clip(self.k_out / (N - 1), 1e-6, 1 - 1e-6)
            p_in = np.clip(self.k_in / (N - 1), 1e-6, 1 - 1e-6)
            # p = x*y/(1+x*y) ≈ x when x,y small; use sqrt heuristic
            x0 = np.sqrt(p_out)
            y0 = np.sqrt(p_in)
            theta_out = -np.log(x0)
            theta_in = -np.log(y0)
        elif method == "random":
            rng = np.random.default_rng(42)
            theta_out = rng.uniform(0.1, 2.0, size=N)
            theta_in = rng.uniform(0.1, 2.0, size=N)
        else:
            raise ValueError(f"Unknown initial-guess method: {method!r}")
        return np.concatenate([theta_out, theta_in])

    # ------------------------------------------------------------------
    # Negative log-likelihood (objective for L-BFGS minimisation)
    # ------------------------------------------------------------------

    def neg_log_likelihood(self, theta: np.ndarray) -> float:
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
        N = self.N
        theta_out = theta[:N]
        theta_in = theta[N:]
        log_xy = -theta_out[:, None] - theta_in[None, :]  # (N, N)
        log1p = np.logaddexp(0.0, log_xy)
        np.fill_diagonal(log1p, 0.0)  # exclude self-loops
        # -L = +θ_out·k_out + θ_in·k_in + Σ_{i≠j} log(1+exp(-θ_out_i-θ_in_j))
        return (
            float(theta_out @ self.k_out)
            + float(theta_in @ self.k_in)
            + float(log1p.sum())
        )

    # ------------------------------------------------------------------
    # Evaluation of constraint satisfaction
    # ------------------------------------------------------------------

    def constraint_error(self, theta: np.ndarray) -> float:
        """Return the maximum absolute error on all constraints.

        Args:
            theta: Parameter vector, shape (2N,).

        Returns:
            Max-abs constraint error (scalar).
        """
        return float(np.max(np.abs(self.residual(theta))))
