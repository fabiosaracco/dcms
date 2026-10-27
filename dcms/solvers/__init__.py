"""Numerical solvers for MaxEnt network models.

Each solver is a θ-space Newton iteration with Anderson acceleration on the degeneracy-reduced system
(nodes with identical constraints share their parameters), expanded back to the N nodes on return.
"""
from .base import SolverResult
from .fixed_point_dcm import solve_fixed_point_dcm
from .fixed_point_dwcm import solve_fixed_point_dwcm
from .fixed_point_qdecm import solve_fixed_point_qdecm
from .fixed_point_decm import solve_fixed_point_decm

__all__ = [
    "SolverResult",
    "solve_fixed_point_dcm",
    "solve_fixed_point_dwcm",
    "solve_fixed_point_qdecm",
    "solve_fixed_point_decm",
]
