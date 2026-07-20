"""Base dataclass for solver results, shared by all numerical methods."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class SolverResult:
    """Result returned by every solver in this package.

    Attributes:
        theta: Parameter vector θ at the *last* iteration step.
            Shape (2N,) for DCM/DWCM, (4N,) for DECM and qDECM.
            For qDECM the layout is [θ_out_topo, θ_in_topo, θ_β_out, θ_β_in].
        best_theta: Parameter vector θ that achieved the lowest ℓ∞ relative
            residual (MRE) during iteration (same shape as ``theta``).
            Equals ``theta`` when the solver converged normally; may differ
            when the solver stopped early (stagnation, timeout, max_iter).
        converged: True if the solver reached the requested tolerance.
        iterations: Total number of update steps performed (for qDECM: sum of
            topology and weight iterations).
        residuals: History of the joint ℓ∞ relative residual
            (MRE = max|F_i|/target_i), one entry per accepted step.
            Empty for qDECM (use ``residuals_topo`` and ``residuals_weights``
            instead); use the ``mre`` and ``last_mre`` properties for
            convenient access.
        residuals_topo: Residual history for the topology sub-solver (qDECM
            only).  Empty for all other models.
        residuals_weights: Residual history for the weight sub-solver (qDECM
            only).  Empty for all other models.
        elapsed_time: Wall-clock time in seconds.
        peak_ram_bytes: Peak RSS increase in bytes during the solver run
                        (measured via psutil, OS-level resident set size).
        message: Human-readable convergence message.
    """

    theta: np.ndarray
    best_theta: np.ndarray
    converged: bool
    iterations: int
    residuals: List[float] = field(default_factory=list)
    residuals_topo: List[float] = field(default_factory=list)
    residuals_weights: List[float] = field(default_factory=list)
    elapsed_time: float = 0.0
    peak_ram_bytes: int = 0
    message: str = ""
    best_mre: float | None = None

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def mre(self) -> float:
        """MRE at the best iterate (``best_theta``).

        Uses ``best_mre`` when the solver set it explicitly -- needed for
        solvers that accept an externally-seeded record (e.g. DECM's
        ``init_best_theta``/``init_best_res``): if that seeded record is
        never beaten during this call, ``best_theta`` correctly stays equal
        to the seed, but ``residuals`` only ever contains *this call's own*
        iteration history and would silently disagree. Falls back to the
        historical ``min(residuals)`` definition otherwise.

        For single-phase solvers (DCM, DECM, DWCM): ``min(residuals)``.
        For qDECM (two-phase): ``max(min(residuals_topo), min(residuals_weights))``.
        """
        if self.best_mre is not None:
            return self.best_mre
        if self.residuals:
            return min(self.residuals)
        if self.residuals_topo and self.residuals_weights:
            return max(min(self.residuals_topo), min(self.residuals_weights))
        return float("nan")

    @property
    def last_mre(self) -> float:
        """MRE at the last iterate (``theta``).

        For single-phase solvers: ``residuals[-1]``.
        For qDECM: ``max(residuals_topo[-1], residuals_weights[-1])``.
        """
        if self.residuals:
            return self.residuals[-1]
        if self.residuals_topo and self.residuals_weights:
            return max(self.residuals_topo[-1], self.residuals_weights[-1])
        return float("nan")

    @property
    def x(self) -> np.ndarray:
        """Physical out-degree multipliers x_i = exp(-θ_i) at best_theta."""
        n = len(self.best_theta) // 2
        return np.exp(-self.best_theta[:n])

    @property
    def y(self) -> np.ndarray:
        """Physical in-degree multipliers y_i = exp(-θ_{N+i}) at best_theta."""
        n = len(self.best_theta) // 2
        return np.exp(-self.best_theta[n:])

    def __repr__(self) -> str:
        status = "CONVERGED" if self.converged else "NOT CONVERGED"
        return (
            f"SolverResult({status}, iters={self.iterations}, "
            f"best_mre={self.mre:.3e}, last_mre={self.last_mre:.3e}, "
            f"time={self.elapsed_time:.3f}s, "
            f"peak_ram={self.peak_ram_bytes / 1024:.1f} KB)"
        )
