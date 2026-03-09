"""Degree reduction: collapse nodes with identical constraint tuples.

For the DCM every node is characterised by the pair (k_out_i, k_in_i).
Nodes that share the same pair also share the same Lagrange multipliers at
the solution.  This allows solving a smaller system of size equal to the
number of *distinct* constraint pairs rather than 2 N equations.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def degree_reduce(
    k_out: np.ndarray,
    k_in: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collapse nodes with identical (k_out, k_in) pairs.

    Args:
        k_out: Out-degree sequence, shape (N,).
        k_in:  In-degree sequence, shape (N,).

    Returns:
        Tuple ``(k_out_uniq, k_in_uniq, multiplicity, node_class)`` where:

        - ``k_out_uniq``: Unique out-degrees, shape (C,).
        - ``k_in_uniq``:  Unique in-degrees, shape (C,).
        - ``multiplicity``: Number of nodes in each class, shape (C,).
        - ``node_class``:   Class index for every original node, shape (N,).
    """
    k_out = np.asarray(k_out, dtype=np.float64)
    k_in = np.asarray(k_in, dtype=np.float64)

    pairs = np.stack([k_out, k_in], axis=1)  # (N, 2)

    # Use a structured view to find unique rows
    pairs_view = np.ascontiguousarray(pairs).view(
        np.dtype((np.void, pairs.dtype.itemsize * 2))
    ).ravel()
    _, first_occ, node_class, multiplicity = np.unique(
        pairs_view, return_index=True, return_inverse=True, return_counts=True
    )

    k_out_uniq = k_out[first_occ]
    k_in_uniq = k_in[first_occ]

    return k_out_uniq, k_in_uniq, multiplicity, node_class


def degree_expand(
    theta_reduced: np.ndarray,
    node_class: np.ndarray,
    N: int,
) -> np.ndarray:
    """Expand a reduced θ vector back to the full N-node parametrisation.

    Args:
        theta_reduced: Parameter vector for the C unique classes, shape (2C,).
        node_class:    Class index for every node, shape (N,).
        N:             Total number of nodes.

    Returns:
        Full parameter vector θ of shape (2N,).
    """
    C = len(theta_reduced) // 2
    theta_out_full = theta_reduced[:C][node_class]
    theta_in_full = theta_reduced[C:][node_class]
    return np.concatenate([theta_out_full, theta_in_full])
