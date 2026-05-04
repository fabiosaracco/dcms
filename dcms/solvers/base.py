"""Base dataclass for solver results, shared by all numerical methods."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class SolverResult:
    """Result returned by every solver in this package.

    Attributes:
        theta: Parameter vector θ at the *last* iteration step, shape (2N,).
        best_theta: Parameter vector θ that achieved the lowest ℓ∞ relative
            residual (MRE) during iteration, shape (2N,).  Equals ``theta``
            when the solver converged normally; may differ when the solver
            stopped early (stagnation, timeout, max_iter).
        converged: True if the solver reached the requested tolerance.
        iterations: Number of update steps performed.
        residuals: History of the ℓ∞ relative residual (MRE = max|F_i|/target_i),
            one entry per accepted step.  ``residuals[-1]`` is the MRE at the
            last iterate (``theta``); ``min(residuals)`` is the MRE at the
            best iterate (``best_theta``).
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
    elapsed_time: float = 0.0
    peak_ram_bytes: int = 0
    message: str = ""

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

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
        last_res = self.residuals[-1] if self.residuals else float("nan")
        best_res = min(self.residuals) if self.residuals else float("nan")
        return (
            f"SolverResult({status}, iters={self.iterations}, "
            f"best_residual={best_res:.3e}, last_residual={last_res:.3e}, "
            f"time={self.elapsed_time:.3f}s, "
            f"peak_ram={self.peak_ram_bytes / 1024:.1f} KB)"
        )
