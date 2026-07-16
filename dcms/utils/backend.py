"""Backend selection and availability detection for compute kernels.

Three backends are supported:

* ``"pytorch"`` — dense or chunked PyTorch tensor operations (always available).
* ``"numba"``   — JIT-compiled scalar loops via Numba (optional dependency).
* ``"auto"``    — automatic selection: PyTorch chunked for N ≤ 100 000, Numba
                  scalar for N > 100 000.  Falls back transparently if the
                  preferred backend is not installed.  Benchmarked on DECM
                  (N up to 200 000): peak RAM is essentially identical between
                  the two backends at every scale tested (Numba is *not* a
                  RAM-saving option in practice), while PyTorch is faster up
                  to N ≈ 50 000 and Numba becomes ≈ 13–14% faster only at
                  N ≥ 100 000.

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
#: Benchmarked on DECM at N=50k/100k/200k (2026-07-13, stella): peak RAM is
#: essentially identical between PyTorch (chunked) and Numba at every scale
#: (≈4.5 GB at N=50k, ≈7.2 GB at N=100k, ≈16.9 GB at N=200k for both) — Numba
#: offers *no* RAM advantage in practice.  Speed: PyTorch is faster up to
#: N ≈ 50 000 (e.g. ≈307 s/iter vs ≈373 s/iter at N=50k), while Numba becomes
#: ≈13–14% faster at N ≥ 100 000 (e.g. 818 s/iter vs 955 s/iter at N=100k).
#: The threshold is set at the confirmed crossover point.
AUTO_NUMBA_THRESHOLD: int = 100_000

BackendStr = Literal["auto", "pytorch", "numba"]


def get_available_cpu_count() -> int:
    """Return the number of CPUs available to the current process.

    Uses :func:`os.sched_getaffinity` on Linux (respects ``taskset`` /
    ``cgroups`` / container CPU quotas).  Falls back to
    :func:`os.cpu_count` on platforms where ``sched_getaffinity`` is not
    available (macOS, Windows).

    Returns:
        Number of CPUs available to this process, at least 1.
    """
    import os
    try:
        return len(os.sched_getaffinity(0)) or 1
    except AttributeError:
        return os.cpu_count() or 1


def resolve_num_threads(num_threads: int) -> int:
    """Resolve a ``num_threads`` request to a safe value.

    Args:
        num_threads: Requested number of threads.  ``0`` means *auto*
            (use all CPUs available to the process).  Positive values are
            clamped to :func:`get_available_cpu_count` to avoid
            ``libgomp: Thread creation failed`` errors on shared / resource-
            limited servers.

    Returns:
        A safe thread count in ``[1, get_available_cpu_count()]``.
    """
    avail = get_available_cpu_count()
    if num_threads <= 0:
        return avail
    return min(num_threads, avail)


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
                   threshold: N > 100 000 → Numba).
        threshold: N threshold for the ``"auto"`` crossover.  Defaults to
                   :data:`AUTO_NUMBA_THRESHOLD` (100 000).

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
