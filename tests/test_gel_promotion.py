

import pytest

from clematis.engine.gel import (
    promote_clusters,
    apply_promotion,
)


# ------------------------------
# Helpers
# ------------------------------

class _State:
    def __init__(self, edges=None, nodes=None):
        self.graph = {
            "nodes": dict(nodes or {}),
            "edges": dict(edges or {}),
            "meta": {"schema": "v1.1", "merges": [], "splits": [], "promotions": [], "concept_nodes_count": 0},
        }


def _ctx_promotion(**overrides):
    base = {
        "graph": {
            "enabled": True,
            "promotion": {
                "enabled": True,
                "label_mode": "lexmin",
                "topk_label_ids": 3,
                "attach_weight": 0.5,
                "cap_per_turn": 10,
            },
        }
    }
    # shallow-merge for simplicity in tests
    base["graph"]["promotion"].update(overrides)
    return base


def _edge_key_in_map(edges, u, v):
    # Edge ids are canonicalized with sorted endpoints; check presence regardless of order
    u = str(u)
    v = str(v)
    key1 = f"{u}→{v}" if u <= v else f"{v}→{u}"
    key2 = f"{v}→{u}" if v <= u else f"{u}→{v}"
    if key1 in edges:
        return key1
    if key2 in edges:
        return key2
    # exhaustive scan safeguard (in case ids contain unicode variants)
    for k, rec in edges.items():
        if isinstance(rec, dict) and {rec.get("src"), rec.get("dst")} == {u, v}:
            return k
    return None


# ------------------------------
# Tests
# ------------------------------

def test_promote_clusters_lexmin_and_order():
    ctx = _ctx_promotion(label_mode="lexmin")
    state = _State()

    clusters = [
        {"nodes": ["d", "f", "e"]},
        {"nodes": ["a", "b"]},
    ]

    promos = promote_clusters(ctx, state, clusters)
    # Deterministic concept ids and ordering (sorted by concept id)
    ids = [p["concept_id"] for p in promos]
    assert ids == ["c::a", "c::d"]
    # Labels = lexicographically smallest member
    labels = [p["label"] for p in promos]
    assert labels == ["a", "d"]


def test_promote_clusters_concat_k_labels():
    ctx = _ctx_promotion(label_mode="concat_k", topk_label_ids=2)
    state = _State()

    clusters = [
        {"nodes": ["c", "a", "b"]},  # sorted members: a, b, c
    ]

    promos = promote_clusters(ctx, state, clusters)
    assert len(promos) == 1
    p = promos[0]
    assert p["concept_id"] == "c::a"  # lexmin determines id
    assert p["label"] == "a+b"        # top-2 joined by '+'


def test_apply_promotion_attaches_edges_and_counts():
    ctx = _ctx_promotion(attach_weight=0.75)
    state = _State()

    promo = {
        "concept_id": "c::a",
        "label": "a",
        "members": ["a", "b", "c"],
        "attach_weight": 0.75,
    }

    res = apply_promotion(ctx, state, promo)
    assert res["event"] == "promotion_applied"

    nodes = state.graph["nodes"]
    edges = state.graph["edges"]
    meta = state.graph["meta"]

    # Concept node exists and counted once
    assert nodes.get("c::a", {}).get("id") == "c::a"
    assert meta.get("concept_nodes_count") == 1

    # All member edges exist with the requested weight and rel="concept"
    for m in ["a", "b", "c"]:
        k = _edge_key_in_map(edges, "c::a", m)
        assert k is not None
        rec = edges[k]
        assert rec.get("rel") == "concept"
        assert abs(float(rec.get("weight")) - 0.75) < 1e-9


def test_attach_weight_is_clamped_to_bounds():
    # attach_weight > 1 should clamp to 1.0 via promote_clusters path
    ctx = _ctx_promotion(attach_weight=2.5, label_mode="lexmin")
    state = _State()

    promos = promote_clusters(ctx, state, [{"nodes": ["x", "y"]}])
    p = promos[0]
    assert p["attach_weight"] == 1.0

    apply_promotion(ctx, state, p)
    k = _edge_key_in_map(state.graph["edges"], p["concept_id"], "x") or _edge_key_in_map(state.graph["edges"], p["concept_id"], "y")
    assert k is not None
    assert abs(float(state.graph["edges"][k]["weight"]) - 1.0) < 1e-9


def test_apply_promotion_is_idempotent():
    # Applying the same promotion twice should not duplicate nodes and should keep weights stable
    ctx = _ctx_promotion(attach_weight=0.4)
    state = _State()
    promo = {"concept_id": "c::k", "label": "k", "members": ["k", "m"], "attach_weight": 0.4}

    apply_promotion(ctx, state, promo)
    nodes_before = dict(state.graph["nodes"])  # shallow copy OK
    edges_before = dict(state.graph["edges"])  # shallow copy OK
    concept_count_before = state.graph["meta"].get("concept_nodes_count")

    apply_promotion(ctx, state, promo)
    assert state.graph["nodes"] == nodes_before
    assert state.graph["edges"] == edges_before
    assert state.graph["meta"].get("concept_nodes_count") == concept_count_before