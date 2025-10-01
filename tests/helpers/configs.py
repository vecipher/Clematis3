

from __future__ import annotations

from typing import Dict, Any

__all__ = [
    "make_cfg_seq",
    "make_cfg_par",
]


def _base() -> Dict[str, Any]:
    """Minimal config skeleton used by identity tests.

    Keep this intentionally small to avoid side-effects from unrelated gates.
    Metrics that could change logs are disabled by default.
    """
    return {
        "perf": {
            "enabled": True,  # global perf features available
            "metrics": {
                # Keep log/metrics emissions quiet for identity comparisons
                "report_memory": False,
            },
            "parallel": {
                # Defaults are sequential; helpers below override as needed
                "enabled": False,
                "max_workers": 1,
                "t1": False,
                "t2": False,
                "agents": False,
            },
        }
    }


def make_cfg_seq() -> Dict[str, Any]:
    """Sequential/disabled-path configuration (identity baseline)."""
    return _base()


def make_cfg_par(workers: int = 2) -> Dict[str, Any]:
    """Parallel ON configuration used for identity testing.

    Enables T1, T2, and agent-level drivers with a deterministic worker cap.
    """
    if workers <= 1:
        workers = 2
    cfg = _base()
    cfg["perf"]["parallel"].update({
        "enabled": True,
        "max_workers": int(workers),
        "t1": True,
        "t2": True,
        "agents": True,
    })
    return cfg
