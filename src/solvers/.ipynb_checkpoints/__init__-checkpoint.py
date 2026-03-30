"""Numerical solvers for MaxEnt network models.

Two methods are implemented for each model:

* **FP-GS Anderson(10)** — Gauss-Seidel fixed-point with Anderson(10) acceleration
* **θ-Newton Anderson(10)** — coordinate-wise Newton in θ-space with Anderson(10) acceleration
"""
from .base import SolverResult
from .fixed_point_dcm import solve_fixed_point_dcm
from .fixed_point_dwcm import solve_fixed_point_dwcm
from .fixed_point_daecm import solve_fixed_point_daecm

__all__ = [
    "SolverResult",
    "solve_fixed_point_dcm",
    "solve_fixed_point_dwcm",
    "solve_fixed_point_daecm",
]
