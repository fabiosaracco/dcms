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

"""
from __future__ import annotations

import math
from typing import Union

import torch


# Type alias for inputs: accept both numpy arrays and torch tensors.
_ArrayLike = Union[torch.Tensor, "numpy.ndarray"]  # type: ignore[name-defined]

from dcms.models.parameters import DWCM_LARGE_N_THRESHOLD as _LARGE_N_THRESHOLD
from dcms.models.parameters import _DEFAULT_CHUNK, _ETA_MIN, _ETA_MAX

from dcms.solvers.base import SolverResult

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
        self.sol: SolverResult | None = None

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

    def pij_matrix(self, theta: _ArrayLike) -> torch.Tensor:
        """Compute the N×N matrix of link-existence probabilities p_ij = P(w_ij > 0).

        The DWCM's geometric weight distribution has P(w_ij = 0) = 1 − β_out_i·β_in_j,
        so the probability of a nonzero arc is simply

            p_ij = β_out_i · β_in_j = exp(−(θ_out_i + θ_in_j))     (i ≠ j)

        Equivalently, p_ij = W_ij / (1 + W_ij) in terms of :meth:`wij_matrix`.
        Diagonal entries are 0 (no self-loops); since θ ≥ _ETA_MIN > 0, z is
        always positive and exp(−z) needs no numerical safeguarding.

        Args:
            theta: Parameter vector [θ_out | θ_in], shape (2N,).
                   All entries should be strictly positive.

        Returns:
            Probability matrix P, shape (N, N), dtype torch.float64.
        """
        theta = _to_tensor(theta)
        N = self.N
        theta_out = theta[:N]
        theta_in = theta[N:]
        z = theta_out[:, None] + theta_in[None, :]  # (N, N)
        P = torch.exp(-z)
        P.fill_diagonal_(0.0)
        # Zero-strength nodes contribute exactly zero probability.
        if self.zero_out.any():
            P[self.zero_out] = 0.0
        if self.zero_in.any():
            P[:, self.zero_in] = 0.0
        return P

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
    # Initial-guess utilities
    # ------------------------------------------------------------------

    def initial_theta(self, method: str = "strengths") -> torch.Tensor:
        """Return a sensible starting point θ₀ for the solvers.

        Several initialisation strategies are provided:

        * ``"strengths"`` (default): β_i ≈ sqrt(s_i / (s_i + N − 1)), derived
          by inverting the symmetric mean-field estimate s ≈ (N−1)β²/(1−β²).
          Works well when the strength distribution is not too heterogeneous.

        * ``"normalized"``: β_i^{out} = s_i^{out} / Σ_j s_j^{out}.
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

        The DWCM log-likelihood is:

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

    def bic(self, theta: _ArrayLike) -> float:
        """Return the Bayesian Information Criterion, BIC = k·ln(n) − 2·ln L.

        k = 2N (the θ_out_i, θ_in_i Lagrange multipliers) and n = N(N−1)
        (the directed dyads i≠j summed over by :meth:`neg_log_likelihood`,
        each contributing one weighted-arc observation).  Lower is better.

        Args:
            theta: Parameter vector [θ_out | θ_in], shape (2N,).

        Returns:
            Scalar BIC(θ).
        """
        N = self.N
        return 2 * N * math.log(N * (N - 1)) + 2 * self.neg_log_likelihood(theta)

    # ------------------------------------------------------------------
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
        
    # ------------------------------------------------------------------
    # Using the solve function
    # ------------------------------------------------------------------

    def solve_tool(self, ic='strengths', tol:float=1e-6, max_iter:int=2000, max_time:int=0, variant:str='theta-newton', anderson_depth:int=10, backend:str='auto', num_threads:int=0, verbose:bool=False, monitor:bool=False, reduce_degeneracy:bool=True)-> SolverResult:
        """Select an initial condition on thetas and solve the equation, using the fixed-point solvers.

        Args:
            ic (str or array-like): Initial condition on theta, shape ``(2N,)``.
                Pass a string (``"strengths"`` or ``"random"``) to use a built-in
                initializer, or pass a numpy array / list / :class:`torch.Tensor`
                to warm-start from a custom vector (e.g. a previous
                ``sol.best_theta``).
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
                iteration (timestamp, iteration count, elapsed time, MRE).
                Default=False.
            monitor (bool): If ``True`` (and ``verbose=True``), overwrite the
                same terminal line at each iteration (``end='\\r'``) so only
                the latest status is visible.  Default=False.
            reduce_degeneracy (bool): If ``True`` (default), nodes sharing the
                exact same ``(s_out, s_in)`` pair are collapsed into a single
                group before solving (see README §2.5) — solves M ≤ N group
                unknowns instead of N per-node ones, often 10-70x faster on
                real networks with heavy-tailed strength distributions.
                Mathematically exact (not an approximation); the returned
                ``theta``/``best_theta`` are expanded back to shape ``(2N,)``.
                Only supported with ``variant="theta-newton"`` and
                ``backend != "numba"``; silently falls back to the full
                (unreduced) solver otherwise, with a printed note.

        Returns:
            :class:`~src.solvers.base.SolverResult` instance.
        """
        if isinstance(ic, str):
            self.ic = self.initial_theta(ic)
        else:
            self.ic = torch.as_tensor(ic, dtype=torch.float64)

        _use_reduced = reduce_degeneracy and variant == "theta-newton" and backend != "numba"
        if reduce_degeneracy and not _use_reduced:
            print(
                "reduce_degeneracy=True requires variant='theta-newton' and "
                "backend!='numba' -- falling back to the full (unreduced) solver."
            )

        if _use_reduced:
            from dcms.solvers.fixed_point_dwcm import solve_fixed_point_dwcm_degenerate  # lazy import to avoid circular dependency
            self.sol = solve_fixed_point_dwcm_degenerate(self.ic, self.s_out, self.s_in, tol=tol, max_iter=max_iter, max_time=max_time, anderson_depth=anderson_depth, backend=backend, num_threads=num_threads, verbose=verbose, monitor=monitor)
        else:
            from dcms.solvers.fixed_point_dwcm import solve_fixed_point_dwcm  # lazy import to avoid circular dependency
            self.sol = solve_fixed_point_dwcm(self.residual, self.ic, self.s_out, self.s_in, tol=tol, max_iter=max_iter, max_time=max_time, variant=variant, anderson_depth=anderson_depth, backend=backend, num_threads=num_threads, verbose=verbose, monitor=monitor)
        if len(self.sol.message)>0:
            print(self.sol.message+" "*50) # the +" "*50 is necessary to avoid the output to be badly overwritten in the case of monitor=True
            
        return self.sol.converged

    def _sample_raw(self, seed: int | None = None,
                    chunk_size: int = 512) -> "np.ndarray":
        """Internal sampler — returns shape ``(L, 3)`` NumPy array (no tolist)."""
        import numpy as np
        if self.sol is None:
            raise RuntimeError("Call solve_tool() first.")
        rng = np.random.default_rng(seed)
        theta = np.asarray(self.sol.theta, dtype=np.float64)
        N = self.N
        beta_out = np.exp(-theta[:N])
        beta_in  = np.exp(-theta[N:])

        mean_b = beta_out.sum() * beta_in.sum() / (N * (N - 1))
        if mean_b < 0.05:
            S_bo = beta_out.sum();  S_bi = beta_in.sum()
            n_cand = rng.poisson(S_bo * S_bi)
            if n_cand == 0:
                return np.empty((0, 3), dtype=int)
            src_c = rng.choice(N, size=n_cand, p=beta_out / S_bo)
            dst_c = rng.choice(N, size=n_cand, p=beta_in  / S_bi)
            mask = src_c != dst_c
            src_c, dst_c = src_c[mask], dst_c[mask]
            if len(src_c) == 0:
                return np.empty((0, 3), dtype=int)
            uid = np.unique(src_c.astype(np.int64) * N + dst_c)
            rows = (uid // N).astype(int)
            cols = (uid  % N).astype(int)
            b_vals = (beta_out[rows] * beta_in[cols]).clip(0.0, 1.0 - 1e-12)
            weights = rng.geometric(1.0 - b_vals)
            return np.column_stack([rows, cols, weights])

        # Dense fallback
        parts: list = []
        for i_start in range(0, N, chunk_size):
            i_end = min(i_start + chunk_size, N)
            b = (beta_out[i_start:i_end, None] * beta_in[None, :]).clip(0.0, 1.0 - 1e-12)
            chunk_range = np.arange(i_end - i_start)
            b[chunk_range, i_start + chunk_range] = 0.0
            rows, cols = np.where(rng.random(b.shape) < b)
            if len(rows):
                b_vals = b[rows, cols]
                weights = rng.geometric(1.0 - b_vals)
                parts.append(np.column_stack([i_start + rows, cols, weights]))
        if not parts:
            return np.empty((0, 3), dtype=int)
        return np.vstack(parts)

    def sample(self, seed: int | None = None,
               chunk_size: int = 512) -> "np.ndarray":
        """Sample a weighted directed network from the fitted DWCM.

        For each ordered pair ``(i, j)`` with ``i ≠ j``, the weight is drawn
        from a geometric distribution starting at 0::

            P(w_ij = k) = (1 − β_ij) β_ij^k,   k = 0, 1, 2, …
            β_ij = β_out_i β_in_j = exp(−η_out_i − η_in_j)

        Pairs with ``w_ij = 0`` (no link) are omitted.

        Uses a Poisson-intensity sparse sampler: O(L) random draws instead of
        O(N²), yielding P(w>0)=1−exp(−β_ij) which equals β_ij up to O(β²)
        error (negligible for sparse networks).  Falls back to the exact O(N²)
        chunk sampler when the network is dense (mean β > 5 %).

        Args:
            seed: Random seed for reproducibility.
            chunk_size: Number of source rows processed at a time for the exact
                fallback path.

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
        near-linear speedup up to the number of physical cores.

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

    def p_value_calculator(self, edgelist: _ArrayLike) -> "np.ndarray":
        """Compute the DWCM p-value of each observed weight in an edge list.

        The DWCM weight distribution is geometric starting at 0 (see
        :meth:`sample`)::

            P(w_ij = k) = (1 - beta_ij) * beta_ij**k,  k = 0, 1, 2, ...
            beta_ij = exp(-(theta_out_i + theta_in_j))

        so the p-value (probability of observing a weight at least as large
        as the one seen) is the survival probability::

            p_value(w) = P(W_ij >= w) = beta_ij**w = z_ij**-w,
            z_ij = exp(theta_out_i + theta_in_j) = 1 / beta_ij

        Computed directly on the given pairs (no dense N×N matrix is
        built), so this scales to large networks.

        Args:
            edgelist: Array-like of shape (L, 3), each row
                ``[source_id, target_id, weight]`` — 0-based node indices
                into ``s_out``/``s_in``, and observed weight
                (``weight >= 0``).

        Returns:
            Structured NumPy array of length L with fields
            ``("source_id", int64)``, ``("target_id", int64)``,
            ``("p_value", float64)``. Also stored as :attr:`p_value`.

        Raises:
            RuntimeError: if :meth:`solve_tool` has not been called yet.
        """
        import numpy as np
        if self.sol is None:
            raise RuntimeError("Call solve_tool() first.")
        edgelist = np.asarray(edgelist)
        source_id = edgelist[:, 0].astype(np.int64)
        target_id = edgelist[:, 1].astype(np.int64)
        weight = edgelist[:, 2].astype(np.float64)

        N = self.N
        theta = np.asarray(self.sol.best_theta, dtype=np.float64)
        theta_out = theta[:N]
        theta_in = theta[N:]

        z_edge = theta_out[source_id] + theta_in[target_id]
        z_edge = np.clip(z_edge, 1e-15, None)
        z_ij = np.exp(z_edge)
        p_value = z_ij ** (-weight)

        dtype = np.dtype(
            [("source_id", np.int64), ("target_id", np.int64), ("p_value", np.float64)]
        )
        result = np.empty(len(source_id), dtype=dtype)
        result["source_id"] = source_id
        result["target_id"] = target_id
        result["p_value"] = p_value
        self.p_value = result
        return self.p_value
