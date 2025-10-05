from __future__ import annotations
from typing import Any, Tuple, Sequence, Dict
import numpy as np
import inspect

import logging
import threading

_logger = logging.getLogger("clematis.native.t1")
_LOG_ONCE_KEYS = set()
_LOG_ONCE_LOCK = threading.Lock()

def _log_once(key: str, level: int, msg: str) -> None:
    with _LOG_ONCE_LOCK:
        if key in _LOG_ONCE_KEYS:
            return
        _LOG_ONCE_KEYS.add(key)
    _logger.log(level, msg)

try:
    from . import _t1_rs as _rs  # PyO3 extension (PR99)
    _HAVE_RS = True
except Exception:
    _HAVE_RS = False

__all__ = ["available", "propagate_one_graph_rs"]


def available() -> bool:
    """Native is available if the extension imports and exposes any known kernel entrypoint."""
    if not _HAVE_RS:
        return False
    return bool(
        hasattr(_rs, "propagate_one_graph_rs") or hasattr(_rs, "t1_propagate_one_graph")
    )


# ---- dtype helpers ---------------------------------------------------------

def _as_np_int32(x) -> np.ndarray:
    a = np.asarray(x)
    if a.dtype != np.int32:
        a = a.astype(np.int32, copy=False)
    return a


def _as_np_float32(x) -> np.ndarray:
    a = np.asarray(x)
    if a.dtype != np.float32:
        a = a.astype(np.float32, copy=False)
    return a


# --- relation multiplier expansion ------------------------------------------

def _per_edge_rel_mult(indices: np.ndarray,
                       rel_code: np.ndarray | None,
                       rel_mult: np.ndarray | None) -> np.ndarray:
    """Return a per-edge rel multiplier array (float32, len == indices.size).

    Accepts either an already per-edge array or a lookup table with rel_code.
    If rel_mult is None, returns all-ones of length n_edges.
    """
    n_edges = int(np.asarray(indices).size)
    if rel_mult is None:
        return np.ones(n_edges, dtype=np.float32)

    rel_mult = np.asarray(rel_mult, dtype=np.float32)

    # Already per-edge
    if int(rel_mult.size) == n_edges:
        return rel_mult

    # Table lookup path requires rel_code
    if rel_code is None:
        raise ValueError("rel_code is required when rel_mult is a lookup table")

    codes = np.asarray(rel_code, dtype=np.intp)
    if int(codes.size) != n_edges:
        raise ValueError("rel_code length must match number of edges")

    # Safe gather with clipping to avoid OOB in synthetic tests
    max_idx = int(rel_mult.size) - 1
    if max_idx < 0:
        return np.ones(n_edges, dtype=np.float32)
    codes = np.clip(codes, 0, max_idx)
    return rel_mult[codes].astype(np.float32, copy=False)


def propagate_one_graph_rs(
    indptr: Sequence[int],
    indices: Sequence[int],
    weights: Sequence[float],
    rel_code: Sequence[int],
    rel_mult: Sequence[float],
    seed_nodes: Sequence[int],
    seed_weights: Sequence[float],
    key_rank: Sequence[int],
    rate: float,
    floor: float,
    radius_cap: int,
    iter_cap_layers: int,
    node_budget: float,
    queue_cap: int = 0,
    dedupe_window: int = 0,
    visited_cap: int = 0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Python stub for the coming Rust kernel.

    Returns:
      d_nodes: int32 [K]    -- node ids with non-zero deltas
      d_vals:  float32 [K]  -- corresponding delta values
      metrics: dict         -- deterministic counters/timings

    The call shape matches the planned Rust FFI, ltr will internally call
    refactored Python inner loop (PR98) to establish strict-parity harness.

    Caps (perf-ON parity):
      queue_cap: int       -- max frontier size (0 disables)
      dedupe_window: int   -- recent-enqueue ring size for duplicate suppression (0 disables)
      visited_cap: int     -- max expanded nodes before switching to drain-only (0 disables)
    """
    # pref Rust kernel if the extension is present (PR99)
    if _HAVE_RS:
        indptr = _as_np_int32(indptr)
        indices = _as_np_int32(indices)
        weights = _as_np_float32(weights)
        rel_code = _as_np_int32(rel_code)
        rel_mult = _as_np_float32(rel_mult)
        seed_nodes = _as_np_int32(seed_nodes)
        seed_weights = _as_np_float32(seed_weights)
        key_rank = _as_np_int32(key_rank)

        # Prefer the RS API that accepts (rel_code, rel_mult LUT) and expands internally.
        try:
            if hasattr(_rs, "propagate_one_graph_rs"):
                d_nodes, d_vals, metrics = _rs.propagate_one_graph_rs(
                    indptr,
                    indices,
                    weights,
                    rel_code,
                    rel_mult,
                    seed_nodes,
                    seed_weights,
                    key_rank,
                    float(rate),
                    float(floor),
                    int(radius_cap),
                    int(iter_cap_layers),
                    float(node_budget),
                    int(queue_cap),
                    int(dedupe_window),
                    int(visited_cap),
                )
            else:
                # Legacy symbol expects per-edge rel multipliers; be ABI-tolerant on args.
                rel_mult_edges = _per_edge_rel_mult(indices, rel_code, rel_mult)
                try:
                    # Newer legacy build that includes visited_cap
                    d_nodes, d_vals, metrics = _rs.t1_propagate_one_graph(
                        indptr,
                        indices,
                        weights,
                        rel_mult_edges,
                        seed_nodes,
                        seed_weights,
                        key_rank,
                        float(rate),
                        float(floor),
                        int(radius_cap),
                        int(iter_cap_layers),
                        float(node_budget),
                        int(queue_cap),
                        int(dedupe_window),
                        int(visited_cap),
                    )
                except TypeError as te:
                    # Older build without visited_cap: retry without it
                    if "visited_cap" in str(te):
                        d_nodes, d_vals, metrics = _rs.t1_propagate_one_graph(
                            indptr,
                            indices,
                            weights,
                            rel_mult_edges,
                            seed_nodes,
                            seed_weights,
                            key_rank,
                            float(rate),
                            float(floor),
                            int(radius_cap),
                            int(iter_cap_layers),
                            float(node_budget),
                            int(queue_cap),
                            int(dedupe_window),
                        )
                    else:
                        raise
        except (MemoryError, TypeError, ValueError, OverflowError) as e:
            _log_once("pyo3_runtime_exc", logging.ERROR, f"native_t1 PyO3 raised {type(e).__name__}: {e}")
            # Re-raise so the dispatcher can decide whether to fall back or fail (strict parity)
            raise

        # Enforce ABI dtypes
        return _as_np_int32(d_nodes), _as_np_float32(d_vals), metrics

    # Lazy import to avoid import cycles if the engine isn't fully loaded yet.
    try:
        from clematis.engine.stages.t1 import _t1_one_graph_python_inner  # type: ignore
    except Exception as e:  # pragma: no cover - makes misuse obvious in tests
        raise RuntimeError(
            "native T1 Python stub requires _t1_one_graph_python_inner (PR98).\n"
            "Please move the legacy inner loop into that symbol before using this stub."
        ) from e

    # inputs normalization to canon "dtypes" as expected by this fuckass kernel.
    indptr = _as_np_int32(indptr)
    indices = _as_np_int32(indices)
    weights = _as_np_float32(weights)
    rel_code = _as_np_int32(rel_code)   # accepted for forward-compat
    rel_mult = _as_np_float32(rel_mult)
    seed_nodes = _as_np_int32(seed_nodes)
    seed_weights = _as_np_float32(seed_weights)
    key_rank = _as_np_int32(key_rank)

    # parameters packing into the structure the inner loop want
    params: Dict[str, Any] = {
        "decay": {"mode": "exp_floor", "rate": float(rate), "floor": float(floor)},
        "radius_cap": int(radius_cap),
        "iter_cap_layers": int(iter_cap_layers),
        "node_budget": float(node_budget),
    }

    # perf-ON caps parity with Python inner
    _q = int(queue_cap) if queue_cap else 0
    _d = int(dedupe_window) if dedupe_window else 0
    _v = int(visited_cap) if visited_cap else 0
    caps: Dict[str, Any] = {}
    if _q > 0:
        caps["queue_cap"] = _q
    if _v > 0:
        caps["visited_cap"] = _v
    if caps:
        params["caps"] = caps
    if _d > 0:
        params["dedupe_window"] = _d

    # kwargs for the inner function, adding optional keys only if supported.
    inner_sig = inspect.signature(_t1_one_graph_python_inner)
    kw: Dict[str, Any] = {
        "indptr": indptr,
        "indices": indices,
        "weights": weights,
        "rel_mult": rel_mult,
        "seeds": seed_nodes,
        "params": params,
        "key_rank": key_rank,
    }
    if "seed_weights" in inner_sig.parameters:
        kw["seed_weights"] = seed_weights
    if "rel_code" in inner_sig.parameters:
        kw["rel_code"] = rel_code

    d_nodes, d_vals, metrics = _t1_one_graph_python_inner(**kw)

    # output encorments of dtypes for parity and ABI stability
    d_nodes = _as_np_int32(d_nodes)
    d_vals = _as_np_float32(d_vals)
    if not isinstance(metrics, dict):
        raise TypeError("metrics must be a dict")

    return d_nodes, d_vals, metrics
