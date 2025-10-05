
"""
Native (optional) acceleration backends for Clematis.

Phase 3 — bindings & availability
This module exposes a stable, deterministic ABI surface for the T1 native kernel:

  • t1.available() → bool
      Returns True iff the compiled extension is importable AND exports
      the symbol we depend on (t1_propagate_one_graph).

  • t1.propagate_one_graph(...)
      Thin, validated wrapper over the Rust kernel. Enforces dtypes/shapes
      so Python and Rust agree bit‑for‑bit.

Notes
  • No side‑effects on import: we import the extension lazily inside
    functions to keep package import cheap and deterministic.
  • Dtypes are enforced exactly: int32 for CSR indices/indptr/rel_code/seeds/key_rank;
    float32 for weights/rel_mult_edges/seed_weights. Arrays are made contiguous.
  • rel_code vs rel_mult_edges: we pass both through — the Rust kernel
    selects either per‑edge rel_mult (len == n_edges) OR table lookup via rel_code.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional, Tuple

import numpy as np

__all__ = ["t1"]


# ---- helpers ---------------------------------------------------------------

def _as_i32_1d(a: Any, name: str) -> np.ndarray:
    arr = np.asarray(a, dtype=np.int32)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D int32, got shape={arr.shape}")
    return np.ascontiguousarray(arr)


def _as_f32_1d(a: Any, name: str) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D float32, got shape={arr.shape}")
    return np.ascontiguousarray(arr)


# ---- public ABI: t1 --------------------------------------------------------
class _T1Bindings(SimpleNamespace):
    def available(self) -> bool:
        try:
            from . import _t1_rs as _rs  # lazy import
        except Exception:
            return False
        return hasattr(_rs, "t1_propagate_one_graph")

    def propagate_one_graph(
        self,
        *,
        indptr: np.ndarray,
        indices: np.ndarray,
        weights: np.ndarray,
        rel_code: Optional[np.ndarray],
        rel_mult_edges: np.ndarray,
        seed_nodes: np.ndarray,
        seed_weights: np.ndarray,
        key_rank: np.ndarray,
        rate: float,
        floor: float,
        radius_cap: int,
        iter_cap_layers: int,
        node_budget: float,
        queue_cap: int,
        dedupe_window: int,
        visited_cap: int,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Validated call into the native kernel.

        Shapes/dtypes:
          • indptr: int32, len = n_nodes+1
          • indices: int32, len = n_edges
          • weights: float32, len = n_edges
          • rel_code: int32, len = n_edges  (optional) — used when rel_mult_edges is a LUT
          • rel_mult_edges: float32, len = n_edges (per-edge) OR len = table_size (LUT)
          • seed_nodes: int32, len = S; seed_weights: float32, len = S
          • key_rank: int32, len = n_nodes
        """
        # dtype / shape enforcement
        indptr = _as_i32_1d(indptr, "indptr")
        if indptr.size < 2:
            raise ValueError("indptr must have at least 2 entries (n_nodes+1)")
        n_nodes = int(indptr.size - 1)

        indices = _as_i32_1d(indices, "indices")
        weights = _as_f32_1d(weights, "weights")
        if indices.size != weights.size:
            raise ValueError("indices and weights must have the same length (n_edges)")
        n_edges = int(indices.size)

        if rel_code is None:
            rel_code_arr = np.empty(0, dtype=np.int32)
        else:
            rel_code_arr = _as_i32_1d(rel_code, "rel_code")

        rel_mult_edges = _as_f32_1d(rel_mult_edges, "rel_mult_edges")
        # Note: we allow either per-edge multipliers (len == n_edges) OR a LUT.
        if not (rel_mult_edges.size == n_edges or rel_code_arr.size == n_edges):
            raise ValueError(
                "Provide per-edge rel_mult_edges (len == n_edges) OR rel_code with len == n_edges"
            )

        seed_nodes = _as_i32_1d(seed_nodes, "seed_nodes")
        seed_weights = _as_f32_1d(seed_weights, "seed_weights")
        if seed_nodes.size != seed_weights.size:
            raise ValueError("seed_nodes and seed_weights must match in length")

        key_rank = _as_i32_1d(key_rank, "key_rank")
        if key_rank.size != n_nodes:
            raise ValueError("key_rank length must equal n_nodes (len(indptr)-1)")

        # scalars (ensure exact types expected by Rust)
        rate = np.float32(rate).item()
        floor = np.float32(floor).item()
        radius_cap = int(radius_cap)
        iter_cap_layers = int(iter_cap_layers)
        node_budget = np.float32(node_budget).item()
        queue_cap = int(queue_cap)
        dedupe_window = int(dedupe_window)
        visited_cap = int(visited_cap)

        # Call into the extension
        try:
            from . import _t1_rs as _rs
        except Exception as e:  # pragma: no cover
            raise RuntimeError("native T1 extension not available") from e

        nodes, vals, metrics = _rs.t1_propagate_one_graph(
            indptr,
            indices,
            weights,
            rel_code_arr,
            rel_mult_edges,
            seed_nodes,
            seed_weights,
            key_rank,
            rate,
            floor,
            radius_cap,
            iter_cap_layers,
            node_budget,
            queue_cap,
            dedupe_window,
            visited_cap,
        )

        # Ensure exact dtypes
        nodes = np.asarray(nodes, dtype=np.int32)
        vals = np.asarray(vals, dtype=np.float32)
        return nodes, vals, dict(metrics)

    # Backwards-compat alias expected by older dispatcher/tests
    def propagate_one_graph_rs(
        self,
        *,
        indptr: np.ndarray,
        indices: np.ndarray,
        weights: np.ndarray,
        rel_code: Optional[np.ndarray],
        rel_mult_edges: np.ndarray,
        seed_nodes: np.ndarray,
        seed_weights: np.ndarray,
        key_rank: np.ndarray,
        rate: float,
        floor: float,
        radius_cap: int,
        iter_cap_layers: int,
        node_budget: float,
        queue_cap: int,
        dedupe_window: int,
        visited_cap: int,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Compatibility wrapper → calls propagate_one_graph()."""
        return self.propagate_one_graph(
            indptr=indptr,
            indices=indices,
            weights=weights,
            rel_code=rel_code,
            rel_mult_edges=rel_mult_edges,
            seed_nodes=seed_nodes,
            seed_weights=seed_weights,
            key_rank=key_rank,
            rate=rate,
            floor=floor,
            radius_cap=radius_cap,
            iter_cap_layers=iter_cap_layers,
            node_budget=node_budget,
            queue_cap=queue_cap,
            dedupe_window=dedupe_window,
            visited_cap=visited_cap,
        )


# Expose the stable namespace used by the dispatcher
t1 = _T1Bindings()
