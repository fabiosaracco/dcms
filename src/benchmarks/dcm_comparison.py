"""DCM solver comparison benchmark.

Generates a test network using the Chung-Lu power-law model, then runs all
five solvers and prints a comparison table.

Usage::

    python -m src.benchmarks.dcm_comparison          # default N=50
    python -m src.benchmarks.dcm_comparison --n 100  # custom size
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a script from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from src.models.dcm import DCMModel
from src.solvers.base import SolverResult
from src.solvers.fixed_point import solve_fixed_point
from src.solvers.quasi_newton import solve_lbfgs
from src.solvers.newton import solve_newton
from src.solvers.broyden import solve_broyden
from src.solvers.levenberg_marquardt import solve_lm
from src.utils.wng import k_s_generator_pl


def run_comparison(N: int = 50, seed: Optional[int] = None, tol: float = 1e-8) -> None:
    """Run all DCM solvers on a random network and print a comparison table.

    Args:
        N:    Number of nodes.
        seed: Random seed for reproducibility.
        tol:  Convergence tolerance for all solvers.
    """
    print(f"\n{'='*70}")
    print(f"DCM Solver Comparison  |  N={N} nodes  |  seed={seed}  |  tol={tol:.0e}")
    print(f"{'='*70}")

    # --- Generate test network -------------------------------------------
    k, _ = k_s_generator_pl(N, rho=0.3, seed=seed)
    k_out = k[:N].numpy().astype(float)
    k_in = k[N:].numpy().astype(float)

    # Ensure at least one edge per node (skip isolated nodes)
    valid = (k_out > 0) & (k_in > 0)
    if not valid.all():
        print(f"  Warning: {(~valid).sum()} isolated nodes; replacing with 1.")
        k_out = np.where(k_out > 0, k_out, 1.0)
        k_in = np.where(k_in > 0, k_in, 1.0)

    model = DCMModel(k_out, k_in)
    theta0 = model.initial_theta("degrees")

    print(f"  k_out: min={k_out.min():.0f}  max={k_out.max():.0f}  "
          f"mean={k_out.mean():.1f}")
    print(f"  k_in:  min={k_in.min():.0f}  max={k_in.max():.0f}  "
          f"mean={k_in.mean():.1f}")
    print()

    # --- Define solvers --------------------------------------------------
    solvers = [
        (
            "Fixed-point (GS, α=1.0)",
            lambda: solve_fixed_point(
                model.residual, theta0, k_out, k_in,
                tol=tol, damping=1.0, variant="gauss-seidel",
            ),
        ),
        (
            "Fixed-point (GS, α=0.5)",
            lambda: solve_fixed_point(
                model.residual, theta0, k_out, k_in,
                tol=tol, damping=0.5, variant="gauss-seidel",
            ),
        ),
        (
            "Fixed-point (Jacobi)",
            lambda: solve_fixed_point(
                model.residual, theta0, k_out, k_in,
                tol=tol, damping=1.0, variant="jacobi",
            ),
        ),
        (
            "L-BFGS (m=20)",
            lambda: solve_lbfgs(
                model.residual, theta0, tol=tol, m=20,
                neg_loglik_fn=model.neg_log_likelihood,
            ),
        ),
        (
            "Newton (exact J)",
            lambda: solve_newton(
                model.residual, model.jacobian, theta0, tol=tol,
            ),
        ),
        (
            "Broyden (rank-1 J)",
            lambda: solve_broyden(
                model.residual, model.jacobian, theta0, tol=tol,
            ),
        ),
        (
            "Levenberg-Marquardt",
            lambda: solve_lm(
                model.residual, model.jacobian, theta0, tol=tol,
            ),
        ),
    ]

    # --- Header ----------------------------------------------------------
    col = [20, 10, 10, 14, 12, 14]
    header = (
        f"{'Method':<{col[0]}} {'Conv?':>{col[1]}} {'Iters':>{col[2]}} "
        f"{'MaxErr':>{col[3]}} {'Time(s)':>{col[4]}} {'RAM(KB)':>{col[5]}}"
    )
    print(header)
    print("-" * sum(col))

    # --- Run each solver -------------------------------------------------
    for name, fn in solvers:
        result: SolverResult = fn()
        max_err = model.constraint_error(result.theta)
        conv_str = "YES" if result.converged else "NO"
        print(
            f"{name:<{col[0]}} {conv_str:>{col[1]}} {result.iterations:>{col[2]}} "
            f"{max_err:>{col[3]}.3e} {result.elapsed_time:>{col[4]}.3f} "
            f"{result.peak_ram_bytes/1024:>{col[5]}.1f}"
        )

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DCM solver comparison")
    parser.add_argument("--n", type=int, default=50, help="Number of nodes")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--tol", type=float, default=1e-8, help="Solver tolerance")
    args = parser.parse_args()
    run_comparison(N=args.n, seed=args.seed, tol=args.tol)
