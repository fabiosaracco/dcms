"""DCM solver comparison benchmark.

Generates test networks using the Chung-Lu power-law model, then runs all
five solvers and prints a comparison table.  The multi-seed variant runs
*n_seeds* independent network realisations per node count and reports
aggregate convergence statistics.

Usage::

    python -m src.benchmarks.dcm_comparison               # 10 seeds, N=50
    python -m src.benchmarks.dcm_comparison --n 100        # custom size
    python -m src.benchmarks.dcm_comparison --seed 42      # single seed
    python -m src.benchmarks.dcm_comparison --n 50 --n_seeds 5  # fewer seeds
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Optional

# Allow running as a script from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from src.models.dcm import DCMModel
from src.solvers.base import SolverResult
from src.solvers.fixed_point import solve_fixed_point
from src.solvers.quasi_newton import solve_lbfgs
from src.solvers.newton import solve_newton
from src.solvers.broyden import solve_broyden
from src.solvers.levenberg_marquardt import solve_lm
from src.utils.wng import k_s_generator_pl


def _is_feasible(k_out: np.ndarray, k_in: np.ndarray) -> bool:
    """Return True if the degree sequence is feasible for the DCM.

    A degree sequence is feasible when no node has degree ≥ N (since
    self-loops are forbidden and the maximum possible degree is N−1).

    Args:
        k_out: Out-degree sequence, shape (N,).
        k_in:  In-degree sequence, shape (N,).

    Returns:
        True when the sequence is valid for DCM fitting.
    """
    N = len(k_out)
    return bool(k_out.max() < N and k_in.max() < N)


def _make_solvers(
    model: DCMModel, theta0: torch.Tensor, tol: float
) -> list[tuple[str, Callable[[], SolverResult]]]:
    """Return the list of (name, callable) solver pairs for *model*."""
    return [
        (
            "Fixed-point (GS, α=1.0)",
            lambda: solve_fixed_point(
                model.residual, theta0, model.k_out, model.k_in,
                tol=tol, damping=1.0, variant="gauss-seidel",
            ),
        ),
        (
            "Fixed-point (GS, α=0.5)",
            lambda: solve_fixed_point(
                model.residual, theta0, model.k_out, model.k_in,
                tol=tol, damping=0.5, variant="gauss-seidel",
            ),
        ),
        (
            "Fixed-point (Jacobi)",
            lambda: solve_fixed_point(
                model.residual, theta0, model.k_out, model.k_in,
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


def run_comparison(N: int = 50, seed: Optional[int] = None, tol: float = 1e-8) -> None:
    """Run all DCM solvers on a single random network and print a comparison table.

    Args:
        N:    Number of nodes.
        seed: Random seed for reproducibility.  ``None`` picks a random seed.
        tol:  Convergence tolerance for all solvers.
    """
    print(f"\n{'='*70}")
    print(f"DCM Solver Comparison  |  N={N} nodes  |  seed={seed}  |  tol={tol:.0e}")
    print(f"{'='*70}")

    # --- Generate test network -------------------------------------------
    k, _ = k_s_generator_pl(N, rho=0.3, seed=seed)
    k_out = k[:N].numpy().astype(float)
    k_in = k[N:].numpy().astype(float)

    if not _is_feasible(k_out, k_in):
        print("  SKIPPED: infeasible degree sequence (max degree ≥ N).")
        return

    model = DCMModel(k_out, k_in)
    theta0 = model.initial_theta("degrees")

    print(f"  k_out: min={k_out.min():.0f}  max={k_out.max():.0f}  "
          f"mean={k_out.mean():.1f}")
    print(f"  k_in:  min={k_in.min():.0f}  max={k_in.max():.0f}  "
          f"mean={k_in.mean():.1f}")
    print()

    # --- Header ----------------------------------------------------------
    col = [24, 8, 8, 14, 10, 12]
    header = (
        f"{'Method':<{col[0]}} {'Conv?':>{col[1]}} {'Iters':>{col[2]}} "
        f"{'MaxErr':>{col[3]}} {'Time(s)':>{col[4]}} {'RAM(KB)':>{col[5]}}"
    )
    print(header)
    print("-" * sum(col))

    # --- Run each solver -------------------------------------------------
    for name, fn in _make_solvers(model, theta0, tol):
        result: SolverResult = fn()
        max_err = model.constraint_error(result.theta)
        conv_str = "YES" if result.converged else "NO"
        print(
            f"{name:<{col[0]}} {conv_str:>{col[1]}} {result.iterations:>{col[2]}} "
            f"{max_err:>{col[3]}.3e} {result.elapsed_time:>{col[4]}.3f} "
            f"{result.peak_ram_bytes/1024:>{col[5]}.1f}"
        )

    print()


def run_comparison_multi_seed(
    N: int = 50,
    n_seeds: int = 10,
    tol: float = 1e-8,
    start_seed: int = 0,
) -> dict[str, list[bool]]:
    """Run all DCM solvers on *n_seeds* independent network realisations.

    Iterates over candidate seeds starting from *start_seed*, skipping any
    that produce infeasible degree sequences (max degree ≥ N), until
    *n_seeds* valid realisations have been collected.  Raises ``RuntimeError``
    if more than ``n_seeds * 20`` candidates are exhausted without finding
    enough valid networks.

    Args:
        N:          Number of nodes.
        n_seeds:    Number of valid network realisations to test.
        tol:        Convergence tolerance for all solvers.
        start_seed: First seed to try.

    Returns:
        Dictionary mapping solver name → list of convergence booleans
        (one entry per valid realisation).
    """
    print(f"\n{'='*70}")
    print(
        f"DCM Multi-Seed Comparison  |  N={N} nodes  |  "
        f"{n_seeds} runs  |  tol={tol:.0e}"
    )
    print(f"{'='*70}")

    solver_names: list[str] = []
    conv_history: dict[str, list[bool]] = {}
    valid_count = 0
    skipped_count = 0
    candidate_seed = start_seed
    max_attempts = n_seeds * 20

    while valid_count < n_seeds:
        if (candidate_seed - start_seed) >= max_attempts:
            raise RuntimeError(
                f"Could not find {n_seeds} feasible networks in "
                f"{max_attempts} attempts (N={N})."
            )

        k, _ = k_s_generator_pl(N, rho=0.3, seed=candidate_seed)
        k_out = k[:N].numpy().astype(float)
        k_in = k[N:].numpy().astype(float)

        if not _is_feasible(k_out, k_in):
            skipped_count += 1
            candidate_seed += 1
            continue

        model = DCMModel(k_out, k_in)
        theta0 = model.initial_theta("degrees")
        solvers = _make_solvers(model, theta0, tol)

        if not solver_names:
            solver_names = [name for name, _ in solvers]
            conv_history = {name: [] for name in solver_names}

        print(f"\n  Seed {candidate_seed} "
              f"(k_out max={k_out.max():.0f}, k_in max={k_in.max():.0f})")
        for name, fn in solvers:
            result: SolverResult = fn()
            max_err = model.constraint_error(result.theta)
            conv_history[name].append(result.converged)
            tag = "✓" if result.converged else "✗"
            print(f"    {tag} {name:<26} err={max_err:.2e}  iters={result.iterations}")

        valid_count += 1
        candidate_seed += 1

    if skipped_count:
        print(f"\n  (Skipped {skipped_count} infeasible seed(s).)")

    # --- Aggregate summary -----------------------------------------------
    print(f"\n{'─'*70}")
    print(f"{'Summary':^70}")
    print(f"{'─'*70}")
    col = [26, 14, 14]
    print(f"{'Method':<{col[0]}} {'Conv rate':>{col[1]}} {'Conv count':>{col[2]}}")
    print("-" * sum(col))
    for name in solver_names:
        cv = conv_history[name]
        rate = sum(cv) / len(cv)
        print(f"{name:<{col[0]}} {rate:>{col[1]}.1%} {sum(cv):>{col[2]}}/{len(cv)}")
    print()

    return conv_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DCM solver comparison")
    parser.add_argument("--n", type=int, default=50, help="Number of nodes")
    parser.add_argument("--seed", type=int, default=None,
                        help="Single random seed (omit to run multi-seed mode)")
    parser.add_argument("--n_seeds", type=int, default=10,
                        help="Number of realisations for multi-seed mode")
    parser.add_argument("--tol", type=float, default=1e-8, help="Solver tolerance")
    args = parser.parse_args()

    if args.seed is not None:
        run_comparison(N=args.n, seed=args.seed, tol=args.tol)
    else:
        run_comparison_multi_seed(N=args.n, n_seeds=args.n_seeds, tol=args.tol)
