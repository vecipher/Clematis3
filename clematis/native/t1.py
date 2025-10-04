from __future__ import annotations
from typing import Any, Tuple, Sequence, Dict
import numpy as np
import inspect

__all__ = ["available", "propagate_one_graph_rs"]


def available() -> bool:
    """PR98: keep inert by default. The native kernel is not importable yet.
    Tests may monkeypatch this to True. A future PR will probe the compiled ext.
    """
    return False


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
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Python stub for the future Rust kernel.

    Returns:
      d_nodes: int32 [K]    -- node ids with non-zero deltas
      d_vals:  float32 [K]  -- corresponding delta values
      metrics: dict         -- deterministic counters/timings

    The call shape matches the planned Rust FFI. Internally we call the
    factored Python inner loop (PR98) to establish a strict-parity harness.
    """
    # Lazy import to avoid import cycles if the engine isn't fully loaded yet.
    try:
        from clematis.engine.stages.t1 import _t1_one_graph_python_inner  # type: ignore
    except Exception as e:  # pragma: no cover - makes misuse obvious in tests
        raise RuntimeError(
            "native T1 Python stub requires _t1_one_graph_python_inner (PR98).\n"
            "Please move the legacy inner loop into that symbol before using this stub."
        ) from e

    # Normalize inputs to canonical dtypes expected by the kernel.
    indptr = _as_np_int32(indptr)
    indices = _as_np_int32(indices)
    weights = _as_np_float32(weights)
    rel_code = _as_np_int32(rel_code)   # accepted for forward-compatibility
    rel_mult = _as_np_float32(rel_mult)
    seed_nodes = _as_np_int32(seed_nodes)
    seed_weights = _as_np_float32(seed_weights)
    key_rank = _as_np_int32(key_rank)

    # Pack parameters into the structure the inner loop expects.
    params: Dict[str, Any] = {
        "decay": {"mode": "exp_floor", "rate": float(rate), "floor": float(floor)},
        "radius_cap": int(radius_cap),
        "iter_cap_layers": int(iter_cap_layers),
        "node_budget": float(node_budget),
    }

    # Build kwargs for the inner function, adding optional keys only if supported.
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

    # Enforce output dtypes for parity and ABI stability.
    d_nodes = _as_np_int32(d_nodes)
    d_vals = _as_np_float32(d_vals)
    if not isinstance(metrics, dict):
        raise TypeError("metrics must be a dict")

    return d_nodes, d_vals, metrics
