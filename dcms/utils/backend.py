"""Backend selection and availability detection for compute kernels.

Three backends are supported:

* ``"pytorch"`` — dense or chunked PyTorch tensor operations (always available).
* ``"numba"``   — JIT-compiled scalar loops via Numba (optional dependency).
* ``"auto"``    — automatic selection: PyTorch chunked for N ≤ 50 000, Numba scalar
                  for N > 50 000.  Falls back transparently if the preferred
                  backend is not installed.  At N = 30 000 the chunked PyTorch
                  path uses ≈ 0.7 GB peak RAM (chunk × N × 8 bytes) and is
                  ≈ 3.5× faster than Numba, so Numba is only needed for very
                  large networks (N ≳ 50 000) where physical RAM is a constraint.

The :func:`resolve_backend` function is the single entry-point used by every
solver to decide which kernel set to use at runtime.
"""
from __future__ import annotations

import logging
import warnings
from typing import Literal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Availability probes (cached at module level)
# ---------------------------------------------------------------------------

_PYTORCH_AVAILABLE: bool = True  # torch is a hard dependency

try:
    import numba  # noqa: F401
    _NUMBA_AVAILABLE: bool = True
except ImportError:
    _NUMBA_AVAILABLE: bool = False


def _has_pytorch() -> bool:
    """Return ``True`` if PyTorch is importable."""
    return _PYTORCH_AVAILABLE


def _has_numba() -> bool:
    """Return ``True`` if Numba is importable."""
    return _NUMBA_AVAILABLE


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

#: Default N threshold above which ``"auto"`` prefers the Numba backend.
#: PyTorch chunked uses ≈ chunk × N × 8 bytes RAM (about 0.7 GB at N=30 000,
#: 2.5 GB at N=100 000) and is ≈ 3.5× faster than Numba, so we keep PyTorch
#: up to N=50 000.  Only truly large networks (N > 50 000) benefit from Numba.
AUTO_NUMBA_THRESHOLD: int = 50_000

BackendStr = Literal["auto", "pytorch", "numba"]


def resolve_backend(
    backend: BackendStr = "auto",
    N: int = 0,
    *,
    threshold: int = AUTO_NUMBA_THRESHOLD,
) -> str:
    """Choose the concrete compute backend for a solver call.

    Args:
        backend:   User-requested backend (``"auto"``, ``"pytorch"``, or
                   ``"numba"``).
        N:         Problem size (number of nodes).  Used only when
                   ``backend="auto"`` to decide the crossover (default
                   threshold: N > 50 000 → Numba).
        threshold: N threshold for the ``"auto"`` crossover.  Defaults to
                   :data:`AUTO_NUMBA_THRESHOLD` (5 000).

    Returns:
        One of ``"pytorch"`` or ``"numba"`` — the *concrete* backend to use.

    Raises:
        RuntimeError: If neither PyTorch nor Numba is available (should not
            happen since PyTorch is a hard dependency, but guarded for safety).
    """
    if backend not in ("auto", "pytorch", "numba"):
        raise ValueError(
            f"Unknown backend {backend!r}. Choose 'auto', 'pytorch', or 'numba'."
        )

    if backend == "auto":
        if N > threshold and _has_numba():
            return "numba"
        # N ≤ threshold or numba not available → PyTorch dense
        return "pytorch"

    if backend == "pytorch":
        if not _has_pytorch():
            if _has_numba():
                msg = (
                    "PyTorch is not available; falling back to the Numba backend."
                )
                warnings.warn(msg, stacklevel=2)
                logger.warning(msg)
                return "numba"
            raise RuntimeError(
                "Neither PyTorch nor Numba is available. "
                "Install at least one: pip install torch   OR   pip install numba"
            )
        return "pytorch"

    # backend == "numba"
    if not _has_numba():
        if _has_pytorch():
            msg = (
                "Numba is not available; falling back to the PyTorch backend."
            )
            warnings.warn(msg, stacklevel=2)
            logger.warning(msg)
            return "pytorch"
        raise RuntimeError(
            "Neither Numba nor PyTorch is available. "
            "Install at least one: pip install torch   OR   pip install numba"
        )
    return "numba"
