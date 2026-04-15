"""Profiling utilities: wall-clock time and peak RSS memory measurement.

Peak RAM is measured as the maximum **Resident Set Size** (RSS) of the
process recorded during the profiled call, minus the RSS baseline taken
just before the call starts.  This correctly captures PyTorch tensor
allocations (which live in the C++ allocator and are invisible to
``tracemalloc``) as well as NumPy/Numba allocations.

A lightweight background thread samples RSS every ``_POLL_INTERVAL``
seconds; on a typical solver iteration (tens of ms each) the overhead
is negligible.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any, Callable, Tuple

import psutil


_POLL_INTERVAL: float = 0.05  # seconds between RSS samples


class _PeakRAMMonitor:
    """Context manager that tracks peak RSS during a block of code.

    Usage::

        with _PeakRAMMonitor() as mon:
            ... heavy computation ...
        print(mon.peak_bytes)   # peak RSS *increase* during the block
    """

    def __init__(self, poll_interval: float = _POLL_INTERVAL) -> None:
        self._proc = psutil.Process(os.getpid())
        self._interval = poll_interval
        self._baseline: int = 0
        self._peak: int = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> "_PeakRAMMonitor":
        self._baseline = self._proc.memory_info().rss
        self._peak = self._baseline
        self._stop.clear()
        self._thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self._stop.set()
        self._thread.join()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                rss = self._proc.memory_info().rss
                if rss > self._peak:
                    self._peak = rss
            except psutil.NoSuchProcess:  # pragma: no cover
                break
            self._stop.wait(self._interval)

    @property
    def peak_bytes(self) -> int:
        """Peak RSS *increase* above baseline (bytes), always ≥ 0."""
        return max(0, self._peak - self._baseline)


def profile_solver(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Tuple[Any, float, int]:
    """Run *fn* with *args*/*kwargs* and measure wall-clock time and peak RSS.

    Args:
        fn: Callable to profile (typically a solver function).
        *args: Positional arguments forwarded to *fn*.
        **kwargs: Keyword arguments forwarded to *fn*.

    Returns:
        Tuple of ``(result, elapsed_seconds, peak_ram_bytes)`` where
        *peak_ram_bytes* is the peak RSS increase (bytes) during the call.
    """
    with _PeakRAMMonitor() as mon:
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
    return result, elapsed, mon.peak_bytes

