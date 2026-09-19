"""Single-flight skip flag for the expensive pose-optimization phase.

The SLAM server handles one request at a time (global _PROGRESS / _PHASES),
so a process-wide boolean with a lock is sufficient. It is reset at the
start of every POST /api/slam and set by POST /api/slam/skip-optimization.
Both the API layer (main.py) and the pipeline (optimization.py) poll it,
so it must live in src/ and have zero heavy imports.
"""
from __future__ import annotations

import threading

_LOCK = threading.Lock()
_SKIP_REQUESTED = False


def request_skip() -> None:
    global _SKIP_REQUESTED
    with _LOCK:
        _SKIP_REQUESTED = True


def should_skip() -> bool:
    with _LOCK:
        return _SKIP_REQUESTED


def reset() -> None:
    global _SKIP_REQUESTED
    with _LOCK:
        _SKIP_REQUESTED = False


class SkipOptimization(Exception):
    """Raised inside residuals/jacobian to cancel least_squares cooperatively."""
    pass
