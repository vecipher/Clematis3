

from __future__ import annotations
from typing import Tuple, Dict, Any
import numpy as np

try:
    # Reuse the same extension module as T1
    from . import _t1_rs as _rs  # type: ignore
    _HAVE_RS = hasattr(_rs, "gel_tick_decay")
except Exception:  # pragma: no cover
    _rs = None  # type: ignore[assignment]
    _HAVE_RS = False

__all__ = ["available", "tick_decay"]


def available() -> bool:
    """Return True if the native GEL function is available."""
    return bool(_HAVE_RS)


def _tick_decay_py(weights: np.ndarray, *, rate: float, floor: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Pure-Python fallback: apply decay with stable float32 math.

    Returns (out_weights[np.float32], metrics{edges_processed, decay_applied}).
    """
    w = weights.astype(np.float32, copy=False)
    mul = np.float32(rate if rate > floor else floor)
    out = (w * mul).astype(np.float32, copy=False)
    changed = int(np.count_nonzero(out != w))
    return out, {"edges_processed": int(w.size), "decay_applied": changed}


def tick_decay(weights: np.ndarray, *, rate: float, floor: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply GEL tick/decay to edge weights.

    If the native implementation is present, call it and return its counters.
    Otherwise, use the Python reference implementation.
    """
    w = weights.astype(np.float32, copy=False)
    if _HAVE_RS:
        arr, met = _rs.gel_tick_decay(w, float(rate), float(floor))
        return np.asarray(arr, dtype=np.float32), dict(met)
    return _tick_decay_py(w, rate=rate, floor=floor)
