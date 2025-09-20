

import pytest

from clematis.engine.gel import (
    merge_candidates,
    apply_merge,
    split_candidates,
    apply_split,
)


# ------------------------------
# Helpers
# ------------------------------

def _edge(a: str, b: str, w: float, rel: str = "coact"):
    # Canonical undirected key; store endpoints explicitly
    s, d = (a, b) if a <= b else (b, a)
    return f"{s}→{d}", {"id": f"{s}→{d}", "src": s, "dst": d, "rel": rel, "weight": float(w), "attrs": {}}


class _State:
    def __init__(self, edges=None):
        self.graph = {
            "nodes": {},
            "edges": dict(edges or {}),
            "meta": {"schema": "v1.1", "merges": [], "splits": [], "promotions": [], "concept_nodes_count": 0},
        }


def _ctx_ms(merge=None, split=None):
    # Minimal config surface for PR24 merge/split APIs
    return {
        "graph": {
            "enabled": True,
            "merge": ({
                "enabled": True,
                "min_size": 2,
                "min_avg_w": 0.5,
                "max_diameter": 2,
                "cap_per_turn": 4,
            } | (merge or {})),
            "split": ({
                "enabled": True,
                "weak_edge_thresh": 0.05,
                "min_component_size": 2,
                "cap_per_turn": 4,
            } | (split or {})),
        }
    }


# ------------------------------
# Tests
# ------------------------------

def test_merge_candidates_order_and_determinism():
    # Graph: a—b (0.9), b—c (0.8) → one 3-node strong component
    #        f—g (0.6)             → one 2-node strong component
    #        d—e (0.3)             → below threshold
    edges = {}
    for a, b, w in [("a", "b", 0.9), ("b", "c", 0.8), ("f", "g", 0.6), ("d", "e", 0.3)]:
        k, rec = _edge(a, b, w)
        edges[k] = rec
    state = _State(edges)
    ctx = _ctx_ms(merge={"min_size": 2, "min_avg_w": 0.5, "max_diameter": 2})

    cands1 = merge_candidates(ctx, state)
    cands2 = merge_candidates(ctx, state)

    # Deterministic ordering: highest avg_w first, then size, then lexicographic signature
    sigs1 = [c["signature"] for c in cands1]
    sigs2 = [c["signature"] for c in cands2]
    assert sigs1 == sigs2
    assert sigs1[0] == "a|b|c"
    assert sigs1[1] == "f|g"


def test_merge_apply_respects_cap_and_is_non_destructive():
    edges = {}
    for a, b, w in [("a", "b", 0.9), ("b", "c", 0.8), ("f", "g", 0.6)]:
        k, rec = _edge(a, b, w)
        edges[k] = rec
    state = _State(edges)
    ctx = _ctx_ms(merge={"cap_per_turn": 1, "min_avg_w": 0.5})

    cands = merge_candidates(ctx, state)
    assert len(cands) >= 1

    # Apply only up to cap
    before_edges = dict(state.graph["edges"])  # snapshot
    applied = 0
    for m in cands[: ctx["graph"]["merge"]["cap_per_turn"]]:
        apply_merge(ctx, state, m)
        applied += 1

    assert applied == 1
    # Edges unchanged (metadata-only in HS1)
    assert state.graph["edges"] == before_edges
    # Meta recorded
    assert len(state.graph["meta"].get("merges", [])) == 1


def test_split_candidates_threshold_and_sizes():
    # Component a-b-c-d with b-c weak; removing weak edge yields [a,b] and [c,d]
    edges = {}
    for a, b, w in [("a", "b", 0.9), ("b", "c", 0.01), ("c", "d", 0.9)]:
        k, rec = _edge(a, b, w)
        edges[k] = rec
    state = _State(edges)
    ctx = _ctx_ms(split={"weak_edge_thresh": 0.05, "min_component_size": 2})

    splits = split_candidates(ctx, state)
    assert splits, "expected a split candidate"

    parts = splits[0]["parts"]
    # Normalize for comparison
    parts_norm = [tuple(p) for p in parts]
    assert tuple(sorted(parts_norm)) == (tuple(["a", "b"]), tuple(["c", "d"]))


def test_split_apply_is_non_destructive():
    edges = {}
    for a, b, w in [("a", "b", 0.9), ("b", "c", 0.01), ("c", "d", 0.9)]:
        k, rec = _edge(a, b, w)
        edges[k] = rec
    state = _State(edges)
    ctx = _ctx_ms()

    before_edges = dict(state.graph["edges"])  # snapshot
    splits = split_candidates(ctx, state)
    assert splits

    apply_split(ctx, state, splits[0])

    # Edges unchanged
    assert state.graph["edges"] == before_edges
    # Meta recorded
    assert len(state.graph["meta"].get("splits", [])) == 1


def test_merge_excludes_large_diameter_components():
    # Path a-b-c-d has diameter 3 (> max_diameter=2) → excluded
    edges = {}
    for a, b, w in [("a", "b", 0.9), ("b", "c", 0.9), ("c", "d", 0.9)]:
        k, rec = _edge(a, b, w)
        edges[k] = rec
    state = _State(edges)
    ctx = _ctx_ms(merge={"min_size": 2, "min_avg_w": 0.5, "max_diameter": 2})

    cands = merge_candidates(ctx, state)
    assert cands == []