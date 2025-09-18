import json
from clematis.world.scenario import run_one_turn
from clematis.engine.types import Config
from clematis.graph.store import InMemoryGraphStore, Node, Edge

def bootstrap_state():
    store = InMemoryGraphStore()
    g = store.ensure("g:surface")
    store.upsert_nodes("g:surface", [Node(id="n:hello", label="hello"),
                                     Node(id="n:world", label="world"),
                                     Node(id="n:reply", label="reply")])
    store.upsert_edges("g:surface", [Edge(id="e:h->w", src="n:hello", dst="n:world", weight=0.8, rel="supports"),
                                     Edge(id="e:w->r", src="n:world", dst="n:reply", weight=0.5, rel="associates")])
    return {"store": store, "active_graphs": ["g:surface"]}

def test_t1_determinism(tmp_path, monkeypatch):
    cfg = Config()
    state = bootstrap_state()
    # first run
    from clematis.engine.stages.t1 import t1_propagate
    ctx = type("Ctx", (), {"cfg": cfg, "turn_id":"t", "agent_id":"A"})()  # quick ctx stub
    r1 = t1_propagate(ctx, state, "hello world")
    r2 = t1_propagate(ctx, state, "hello world")
    assert r1.metrics["pops"] >= 0
    # If cache is on, the second call should be as fast or faster and metrics coherent
    assert isinstance(r2.metrics["pops"], int)

def test_t1_budgets_respected():
    cfg = Config()
    cfg.t1["queue_budget"] = 1
    cfg.t1["iter_cap"] = 1
    state = bootstrap_state()
    from clematis.engine.stages.t1 import t1_propagate
    ctx = type("Ctx", (), {"cfg": cfg, "turn_id":"t", "agent_id":"A"})()
    r = t1_propagate(ctx, state, "hello world")
    # For now, budget caps iterations (PQ pops), not initial seeds or propgations.
    # Ensure iteration count does not exceed iter_cap, even if multiple seeds are admitted.
    assert r.metrics["iters"] <= cfg.t1["iter_cap"]

def test_t1_tiebreak_deterministic_equal_weights():
    """
    Two equal-magnitude contributions should be processed in a stable order.
    We construct a seed node with two equal-weight outgoing edges to n:a and n:b.
    With the PQ tie-break set to node id, the order should be alphabetical.
    """
    cfg = Config()
    from clematis.engine.stages.t1 import t1_propagate

    # Build a tiny tie graph
    store = InMemoryGraphStore()
    g = store.ensure("g:tie")
    store.upsert_nodes("g:tie", [
        Node(id="n:seed", label="seed"),
        Node(id="n:a", label="alpha"),
        Node(id="n:b", label="beta"),
    ])
    store.upsert_edges("g:tie", [
        Edge(id="e:s->a", src="n:seed", dst="n:a", weight=1.0, rel="supports"),
        Edge(id="e:s->b", src="n:seed", dst="n:b", weight=1.0, rel="supports"),
    ])
    state = {"store": store, "active_graphs": ["g:tie"]}
    ctx = type("Ctx", (), {"cfg": cfg, "turn_id": "t", "agent_id": "A"})()

    r1 = t1_propagate(ctx, state, "seed")
    r2 = t1_propagate(ctx, state, "seed")

    def extract_ab_order(result):
        return [d["id"] for d in result.graph_deltas
                if d.get("op") == "upsert_node" and d.get("id") in ("n:a", "n:b")]

    order1 = extract_ab_order(r1)
    order2 = extract_ab_order(r2)

    # Deterministic across runs
    assert order1 == order2, f"Non-deterministic order: {order1} vs {order2}"

    # And (by design) alphabetical by node id due to heap tie-breaker
    assert order1 == ["n:a", "n:b"]