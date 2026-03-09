"""Base dataclass for solver results, shared by all numerical methods."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class SolverResult:
    """Result returned by every solver in this package.

    Attributes:
        theta: Final parameter vector θ (log-space), shape (2N,) for DCM.
        converged: True if the solver reached the requested tolerance.
        iterations: Number of iterations performed.
        residuals: History of the ℓ∞ residual norm at each iteration.
        elapsed_time: Wall-clock time in seconds.
        peak_ram_bytes: Peak RAM usage in bytes (measured via tracemalloc).
        message: Human-readable convergence message.
    """

    theta: np.ndarray
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
        """Physical out-degree multipliers x_i = exp(-θ_i)."""
        n = len(self.theta) // 2
        return np.exp(-self.theta[:n])

    @property
    def y(self) -> np.ndarray:
        """Physical in-degree multipliers y_i = exp(-θ_{N+i})."""
        n = len(self.theta) // 2
        return np.exp(-self.theta[n:])

    def __repr__(self) -> str:
        status = "CONVERGED" if self.converged else "NOT CONVERGED"
        final_res = self.residuals[-1] if self.residuals else float("nan")
        return (
            f"SolverResult({status}, iters={self.iterations}, "
            f"final_residual={final_res:.3e}, "
            f"time={self.elapsed_time:.3f}s, "
            f"peak_ram={self.peak_ram_bytes / 1024:.1f} KB)"
        )
