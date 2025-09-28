import math
import pytest

_gel = pytest.importorskip("clematis.engine.gel")
observe_retrieval = _gel.observe_retrieval
tick = _gel.tick


class _Ctx:
    def __init__(self, graph_cfg):
        self.config = {"graph": graph_cfg}
        self.cfg = self.config


class _State:
    pass


def _base_cfg(**over):
    cfg = {
        "enabled": True,  # orchestrator normally gates; unit tests call directly
        "coactivation_threshold": 0.2,
        "observe_top_k": 64,
        "pair_cap_per_obs": 2048,
        "update": {"mode": "additive", "alpha": 0.02, "clamp_min": -1.0, "clamp_max": 1.0},
        "decay": {"half_life_turns": 200, "floor": 0.0},
    }
    # shallow merge for convenience
    for k, v in over.items():
        if k in ("update", "decay"):
            cfg[k].update(v)
        else:
            cfg[k] = v
    return cfg


def _edge(state, key):
    return getattr(state, "graph", {}).get("edges", {}).get(key)


def test_observe_additive_clamps():
    # alpha 0.05, clamp at 0.1; two observes should clamp
    ctx = _Ctx(_base_cfg(update={"alpha": 0.05, "clamp_min": -0.1, "clamp_max": 0.1}))
    s = _State()
    items = [("a", 0.9), ("b", 0.8)]
    m1 = observe_retrieval(ctx, s, items, turn=1, agent="A")
    key = "a→b"
    e = _edge(s, key)
    assert e is not None
    assert math.isclose(e["weight"], 0.05, rel_tol=0, abs_tol=1e-12)
    assert e["attrs"]["coact"] == 1
    assert e["attrs"]["last_seen_turn"] == 1

    m2 = observe_retrieval(ctx, s, items, turn=2, agent="A")
    e2 = _edge(s, key)
    assert math.isclose(e2["weight"], 0.1, rel_tol=0, abs_tol=1e-12)  # clamped
    assert e2["attrs"]["coact"] == 2
    assert e2["attrs"]["last_seen_turn"] == 2
    # metrics sanity
    assert m1["event"] == "observe_retrieval"
    assert m2["pairs_updated"] >= 1


def test_observe_threshold_topk_paircap():
    # Only top3 >= threshold; 3 pairs possible but cap=2 should update exactly 2 pairs
    ctx = _Ctx(_base_cfg(coactivation_threshold=0.65, observe_top_k=3, pair_cap_per_obs=2))
    s = _State()
    items = [("n1", 0.90), ("n2", 0.80), ("n3", 0.70), ("n4", 0.60), ("n5", 0.50)]

    m = observe_retrieval(ctx, s, items, turn=10, agent="A")
    edges = getattr(s, "graph")["edges"]
    # Expect prefix pairs from nested loop: (n1,n2) and (n1,n3)
    assert set(edges.keys()) == {"n1→n2", "n1→n3"}
    assert m["pairs_updated"] == 2


def test_proportional_mode_diminishing_increments():
    # proportional alpha=0.5: w0=0 -> w1=0.5 -> w2=0.75 (smaller increment)
    ctx = _Ctx(_base_cfg(update={"mode": "proportional", "alpha": 0.5}))
    s = _State()
    items = [("x", 0.9), ("y", 0.9)]

    observe_retrieval(ctx, s, items, turn=1, agent="A")
    w1 = _edge(s, "x→y")["weight"]
    assert math.isclose(w1, 0.5, abs_tol=1e-12)

    observe_retrieval(ctx, s, items, turn=2, agent="A")
    w2 = _edge(s, "x→y")["weight"]
    assert math.isclose(w2, 0.75, abs_tol=1e-12)  # 0.5 + 0.5*(1-0.5) = 0.75


def test_edge_key_canonicalization_and_order_independence():
    ctx = _Ctx(_base_cfg())
    s = _State()
    # Input items reversed lexically; key should still be a→b (not b→a)
    items = [("b", 0.9), ("a", 0.85)]
    observe_retrieval(ctx, s, items, turn=3, agent="A")
    assert _edge(s, "a→b") is not None
    assert _edge(s, "b→a") is None


def test_decay_half_life_and_floor_drop():
    # Create two edges; one should be decayed but kept, the other dropped by floor
    ctx = _Ctx(_base_cfg(decay={"half_life_turns": 2, "floor": 0.05}))
    s = _State()
    # Seed state with two edges
    observe_retrieval(ctx, s, [("u", 1.0), ("v", 1.0)], turn=0, agent="A")  # weight += 0.02 default
    # Manually set distinct weights for deterministic check
    s.graph["edges"]["u→v"]["weight"] = 0.20
    s.graph["edges"]["u→v"]["attrs"]["last_seen_turn"] = 0
    # Add a second edge
    observe_retrieval(ctx, s, [("p", 1.0), ("q", 1.0)], turn=0, agent="A")
    s.graph["edges"]["p→q"]["weight"] = 0.08
    s.graph["edges"]["p→q"]["attrs"]["last_seen_turn"] = 0

    m = tick(ctx, s, decay_dt=2, turn=5, agent="A")
    # u→v: 0.20 * 0.5**(2/2) = 0.10 (kept)
    assert math.isclose(s.graph["edges"]["u→v"]["weight"], 0.10, abs_tol=1e-12)
    # p→q: 0.08 -> 0.04 < floor 0.05, dropped
    assert "p→q" not in s.graph["edges"]
    assert m["event"] == "edge_decay"
    assert m["dropped_edges"] >= 1
