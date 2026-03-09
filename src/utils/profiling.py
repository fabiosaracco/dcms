"""Profiling utilities: time and peak RAM measurement."""
from __future__ import annotations

import time
import tracemalloc
from typing import Any, Callable, Tuple


def profile_solver(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Tuple[Any, float, int]:
    """Run *fn* with *args*/*kwargs* and measure wall-clock time and peak RAM.

    Args:
        fn: Callable to profile (typically a solver function).
        *args: Positional arguments forwarded to *fn*.
        **kwargs: Keyword arguments forwarded to *fn*.

    Returns:
        Tuple of ``(result, elapsed_seconds, peak_ram_bytes)`` where *result*
        is whatever *fn* returned.
    """
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed, peak
