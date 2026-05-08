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

**Zero-degree nodes**: if k_out_i = 0 then x_i = 0 exactly (θ_out_i → +∞),
so p_ij = 0 for all j and the constraint is trivially satisfied.  Analogously,
if k_in_i = 0 then y_i = 0 (θ_in_i → +∞) and p_ji = 0 for all j.  These
nodes are identified in ``__init__`` and handled explicitly so that the
residual is *exactly* zero and the Jacobian columns/rows are zero.

"""
from __future__ import annotations

from typing import Union

import torch

from dcms.solvers.base import SolverResult



# Type alias for inputs: accept both numpy arrays and torch tensors.
_ArrayLike = Union[torch.Tensor, "numpy.ndarray"]  # type: ignore[name-defined]

# θ is clamped to this bound; exp(-50) ≈ 2e-22, essentially zero probability.
_THETA_MAX: float = 50.0

# For N > this threshold, residual() and neg_log_likelihood() automatically
# use chunked computation to avoid materialising the full N×N matrix.
from dcms.models.parameters import DCM_LARGE_N_THRESHOLD as _LARGE_N_THRESHOLD

# Number of rows processed per chunk when using memory-efficient mode.
from dcms.models.parameters import _DEFAULT_CHUNK


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
        # Nodes with degree 0: the Lagrange multiplier x (or y) is exactly 0,
        # i.e., θ → +∞.  Track them so pij_matrix can return exact zeros.
        self.zero_out: torch.Tensor = (self.k_out == 0)  # shape (N,)
        self.zero_in: torch.Tensor = (self.k_in == 0)    # shape (N,)
        self.sol: SolverResult | None = None

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
        # Zero-degree nodes contribute exactly zero probability:
        # if k_out_i=0 then x_i=0, so p_ij=0 for all j (zero the row).
        # if k_in_j=0 then y_j=0, so p_ij=0 for all i (zero the column).
        if self.zero_out.any():
            P[self.zero_out] = 0.0
        if self.zero_in.any():
            P[:, self.zero_in] = 0.0
        return P

    # ------------------------------------------------------------------
    # Residual (system of equations)
    # ------------------------------------------------------------------

    def residual(self, theta: _ArrayLike) -> torch.Tensor:
        """Return F(θ) = [k_out_expected − k_out_obs | k_in_expected − k_in_obs].

        For N > ``_LARGE_N_THRESHOLD`` the computation is automatically
        done in row chunks to avoid materialising the full N×N matrix.

        Args:
            theta: Parameter vector [θ_out | θ_in], shape (2N,).

        Returns:
            Residual vector F(θ), shape (2N,).
        """
        if self.N > _LARGE_N_THRESHOLD:
            return self._residual_chunked(theta)
        P = self.pij_matrix(theta)
        k_out_hat = P.sum(dim=1)   # row sums = expected out-degrees
        k_in_hat = P.sum(dim=0)    # col sums = expected in-degrees
        F = torch.empty(2 * self.N, dtype=torch.float64)
        F[: self.N] = k_out_hat - self.k_out
        F[self.N :] = k_in_hat - self.k_in
        return F

    def _residual_chunked(
        self, theta: _ArrayLike, chunk_size: int = _DEFAULT_CHUNK
    ) -> torch.Tensor:
        """Compute F(θ) without materialising the full N×N matrix.

        Processes ``chunk_size`` rows at a time, requiring only
        O(chunk_size × N) RAM instead of O(N²).  Produces results
        identical to :meth:`residual` for small N.

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

        k_out_hat = torch.zeros(N, dtype=torch.float64)
        k_in_hat = torch.zeros(N, dtype=torch.float64)

        for i_start in range(0, N, chunk_size):
            i_end = min(i_start + chunk_size, N)
            chunk_len = i_end - i_start
            # p_chunk[i, j] = sigmoid(-θ_out_i - θ_in_j) = x_i*y_j/(1+x_i*y_j)
            # Apply sigmoid in-place to avoid allocating a second (chunk, N) buffer.
            p_chunk = -theta_out[i_start:i_end, None] - theta_in[None, :]  # (chunk, N)
            p_chunk.sigmoid_()

            # Zero out diagonal entries p_ii = 0
            local_idx = torch.arange(chunk_len, dtype=torch.long)
            global_idx = torch.arange(i_start, i_end, dtype=torch.long)
            p_chunk[local_idx, global_idx] = 0.0

            # Apply zero-degree masks
            if self.zero_out[i_start:i_end].any():
                p_chunk[self.zero_out[i_start:i_end]] = 0.0
            if self.zero_in.any():
                p_chunk[:, self.zero_in] = 0.0

            k_out_hat[i_start:i_end] = p_chunk.sum(dim=1)
            k_in_hat += p_chunk.sum(dim=0)

        F = torch.empty(2 * N, dtype=torch.float64)
        F[:N] = k_out_hat - self.k_out
        F[N:] = k_in_hat - self.k_in
        return F

    # ------------------------------------------------------------------
    # Initial-guess utilities
    # ------------------------------------------------------------------

    def initial_theta(self, method: str = "degrees") -> torch.Tensor:
        """Return a sensible starting point θ₀ for the solvers.

        Zero-degree nodes (k_out_i = 0 or k_in_i = 0) correspond to
        x_i = 0 or y_i = 0 exactly, so their θ is set to ``+_THETA_MAX``
        (≈ +∞ in practice).  Saturated nodes (k_out_i = N-1 or k_in_i = N-1)
        correspond to x_i → ∞ or y_i → ∞, so their θ is set to ``-_THETA_MAX``
        (≈ −∞ in practice).  Both assignments are applied regardless of the
        chosen initialisation method.

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
        # Zero-degree nodes: x = 0 exactly ↔ θ → +∞ (clamped to +_THETA_MAX).
        theta_out = torch.where(self.zero_out, torch.full_like(theta_out, _THETA_MAX), theta_out)
        theta_in = torch.where(self.zero_in, torch.full_like(theta_in, _THETA_MAX), theta_in)
        # Saturated nodes: k = N-1 → x (or y) → +∞ ↔ θ → -∞ (clamped to -_THETA_MAX).
        sat_out = self.k_out >= (N - 1)
        sat_in = self.k_in >= (N - 1)
        theta_out = torch.where(sat_out, torch.full_like(theta_out, -_THETA_MAX), theta_out)
        theta_in = torch.where(sat_in, torch.full_like(theta_in, -_THETA_MAX), theta_in)
        return torch.cat([theta_out, theta_in])

    # ------------------------------------------------------------------
    # Negative log-likelihood (objective for L-BFGS minimisation)
    # ------------------------------------------------------------------

    def neg_log_likelihood(self, theta: _ArrayLike) -> float:
        """Return −L(θ), the convex quantity to be *minimised* by L-BFGS.

        The DCM log-likelihood is:

            L(θ) = −Σ_i θ_out_i·k_out_i − Σ_i θ_in_i·k_in_i
                   − Σ_{i≠j} log(1 + exp(−θ_out_i − θ_in_j))

        Its gradient satisfies ∂L/∂θ_out_i = −k_out_i + Σ_{j≠i} p_ij = F_i,
        so ∇L = F(θ) and therefore ∇(−L) = −F(θ).

        The quantity returned is:
            −L = +Σ_i θ_out_i·k_out_i + Σ_i θ_in_i·k_in_i
                 + Σ_{i≠j} log(1 + exp(−θ_out_i − θ_in_j))

        which is convex (L is concave) and has its minimum at the solution.

        For N > ``_LARGE_N_THRESHOLD`` the sum is computed in row chunks to
        avoid materialising the full N×N matrix.

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
        log_xy = -theta_out[:, None] - theta_in[None, :]  # (N, N)
        log1p = torch.logaddexp(torch.zeros_like(log_xy), log_xy)
        log1p.fill_diagonal_(0.0)  # exclude self-loops
        return (theta_out @ self.k_out + theta_in @ self.k_in + log1p.sum()).item()

    def _neg_log_likelihood_chunked(
        self, theta: _ArrayLike, chunk_size: int = _DEFAULT_CHUNK
    ) -> float:
        """Compute −L(θ) without materialising the full N×N matrix.

        Uses O(chunk_size × N) RAM instead of O(N²).  Identical result to
        :meth:`neg_log_likelihood` for any N.

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

        dot_term = (theta_out @ self.k_out + theta_in @ self.k_in).item()
        log1p_total = 0.0
        zeros_chunk = torch.zeros(1, dtype=torch.float64)

        for i_start in range(0, N, chunk_size):
            i_end = min(i_start + chunk_size, N)
            chunk_len = i_end - i_start
            log_xy = -theta_out[i_start:i_end, None] - theta_in[None, :]  # (chunk, N)
            log1p = torch.logaddexp(zeros_chunk.expand_as(log_xy), log_xy)
            # Zero out diagonal entries (no self-loops)
            local_idx = torch.arange(chunk_len, dtype=torch.long)
            global_idx = torch.arange(i_start, i_end, dtype=torch.long)
            log1p[local_idx, global_idx] = 0.0
            log1p_total += log1p.sum().item()

        return dot_term + log1p_total

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

        Computes max_i |F_i(θ)| / k_i over nodes with k_i > 0.

        Args:
            theta: Parameter vector, shape (2N,).

        Returns:
            Max relative constraint error (MRE = max|F_i|/target_i), or 0.0
            if all degrees are zero.
        """
        F = self.residual(theta).abs()
        targets = torch.cat([self.k_out, self.k_in])
        nonzero = targets > 0
        if not nonzero.any():
            return 0.0
        return (F[nonzero] / targets[nonzero]).max().item()

    # ------------------------------------------------------------------
    # Using the solve function
    # ------------------------------------------------------------------

    def solve_tool(self, ic='degrees', tol:float=1e-6, max_iter:int=2000, max_time:int=0, variant:str='theta-newton', anderson_depth:int=10, backend:str='auto', num_threads:int=0, verbose:bool=False, monitor:bool=False)-> SolverResult:
        """Select an initial condition on thetas and solve the equation, using the fixed-point solvers.

        Args:
            ic (str or array-like): Initial condition on theta, shape ``(2N,)``.
                Pass a string (``"degrees"`` or ``"random"``) to use a built-in
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

        Returns:
            :class:`~src.solvers.base.SolverResult` instance.
        """
        if isinstance(ic, str):
            self.ic = self.initial_theta(ic)
        else:
            self.ic = torch.as_tensor(ic, dtype=torch.float64)
        from dcms.solvers.fixed_point_dcm import solve_fixed_point_dcm  # lazy import to avoid circular dependency
        self.sol = solve_fixed_point_dcm(self.residual, self.ic, self.k_out, self.k_in, tol=tol, max_iter=max_iter, max_time=max_time, variant=variant, anderson_depth=anderson_depth, backend=backend, num_threads=num_threads, verbose=verbose, monitor=monitor)
        if len(self.sol.message)>0:
            print(self.sol.message+" "*50) # the +" "*50 is necessary to avoid the output to be badly overwritten in the case of monitor=True
            
        return self.sol.converged

    def _sample_raw(self, seed: int | None = None,
                    chunk_size: int = 512) -> "np.ndarray":
        """Internal sampler — returns a NumPy array, not a Python list.

        Shape ``(L, 2)`` with dtype ``int``.  Called by both :meth:`sample`
        and :meth:`sample_many` to avoid the GIL-holding ``.tolist()`` cost
        in the parallel path.
        """
        import numpy as np
        if self.sol is None:
            raise RuntimeError("Call solve_tool() first.")
        rng = np.random.default_rng(seed)
        theta = np.asarray(self.sol.theta, dtype=np.float64)
        N = self.N
        x = np.exp(-theta[:N])
        y = np.exp(-theta[N:])
        mean_p = x.sum() * y.sum() / (N * (N - 1))
        if mean_p < 0.05:
            return self._sample_sparse(x, y, rng, N)
        # Dense fallback — build ndarray directly
        parts: list = []
        for i_start in range(0, N, chunk_size):
            i_end = min(i_start + chunk_size, N)
            xy = x[i_start:i_end, None] * y[None, :]
            p = xy / (1.0 + xy)
            chunk_range = np.arange(i_end - i_start)
            p[chunk_range, i_start + chunk_range] = 0.0
            rows, cols = np.where(rng.random(p.shape) < p)
            if len(rows):
                parts.append(np.column_stack([i_start + rows, cols]))
        if not parts:
            return np.empty((0, 2), dtype=int)
        return np.vstack(parts)

    def sample(self, seed: int | None = None, chunk_size: int = 512) -> list:
        """Sample a binary directed network from the fitted DCM.

        For each ordered pair ``(i, j)`` with ``i ≠ j``, a directed edge is
        drawn independently with probability::

            p_ij = x_i y_j / (1 + x_i y_j),   x_i = exp(-θ_out_i), y_j = exp(-θ_in_j)

        Uses a Poisson-intensity sparse sampler: O(L) random draws instead of
        O(N²), yielding P(edge)=1−exp(−p_ij) which equals p_ij up to O(p²) error
        (negligible for sparse networks).  Falls back to the exact O(N²) chunk
        sampler when the network is dense (mean p > 5 %).

        Args:
            seed: Random seed for reproducibility.
            chunk_size: Number of source rows processed at a time for the exact
                fallback path (controls peak RAM).

        Returns:
            Edge list: list of ``[source, target]`` integer pairs.

        Raises:
            RuntimeError: if :meth:`solve_tool` has not been called yet.
        """
        return self._sample_raw(seed=seed, chunk_size=chunk_size).tolist()

    def sample_many(self, n: int, seed: int | None = None,
                    n_jobs: int = -1) -> list:
        """Generate ``n`` independent samples in parallel.

        Each sample is a ``np.ndarray`` of shape ``(L_k, 2)`` with columns
        ``[source, target]``.  Iterating ``for i, j in sample`` works the same
        as with the list returned by :meth:`sample`.

        Uses :class:`~concurrent.futures.ThreadPoolExecutor` — NumPy releases
        the GIL during random-number generation and array operations, so
        threading gives near-linear speedup up to the number of physical cores.

        Args:
            n: Number of independent samples to draw.
            seed: Master seed from which per-sample seeds are derived.
            n_jobs: Number of worker threads.  ``-1`` (default) uses all
                available logical CPUs.

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

    @staticmethod
    def _sample_sparse(x: "np.ndarray", y: "np.ndarray",
                       rng: "np.random.Generator", N: int) -> "np.ndarray":
        """O(L) Poisson-intensity sampler for sparse DCM graphs.

        Returns ``np.ndarray`` of shape ``(L, 2)`` — no ``.tolist()`` call,
        so the GIL is barely held (only during ``rng.choice`` alias setup).
        """
        import numpy as np
        S_x = x.sum();  S_y = y.sum()
        n_cand = rng.poisson(S_x * S_y)
        if n_cand == 0:
            return np.empty((0, 2), dtype=int)
        src = rng.choice(N, size=n_cand, p=x / S_x)
        dst = rng.choice(N, size=n_cand, p=y / S_y)
        xy_vals = x[src] * y[dst]
        keep = (rng.random(n_cand) * (1.0 + xy_vals) < 1.0) & (src != dst)
        src, dst = src[keep], dst[keep]
        if len(src) == 0:
            return np.empty((0, 2), dtype=int)
        uid = np.unique(src.astype(np.int64) * N + dst)
        rows_u = (uid // N).astype(int)
        cols_u = (uid  % N).astype(int)
        return np.column_stack([rows_u, cols_u])