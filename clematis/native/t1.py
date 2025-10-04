from __future__ import annotations
from typing import Any, Tuple

__all__ = ["available", "propagate_one_graph_rs"]


def available() -> bool:
    """PR97: keep inert. The native kernel is not importable yet.
    Later PRs will flip this to probe the compiled extension.
    """
    return False


def propagate_one_graph_rs(
    indptr,
    indices,
    weights,
    rel_code,
    rel_mult,
    seed_nodes,
    seed_weights,
    key_rank,
    rate: float,
    floor: float,
    radius_cap: int,
    iter_cap_layers: int,
    node_budget: float,
) -> Tuple[list[int], list[float], dict[str, Any]]:
    """PR97 stub for the native T1 kernel.

    The real implementation (PR99) will be provided by a PyO3 extension and
    return (d_nodes, d_vals, metrics) with identical semantics to the Python
    T1 inner loop under the default perf-OFF path.

    This stub intentionally raises to make misuse obvious during tests.
    """
    raise RuntimeError("native T1 kernel not available (PR97 stub)")
