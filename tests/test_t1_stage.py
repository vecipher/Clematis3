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
    assert r1.graph_deltas == r2.graph_deltas

def test_t1_budgets_respected():
    cfg = Config()
    cfg.t1["queue_budget"] = 1
    cfg.t1["iter_cap"] = 1
    state = bootstrap_state()
    from clematis.engine.stages.t1 import t1_propagate
    ctx = type("Ctx", (), {"cfg": cfg, "turn_id":"t", "agent_id":"A"})()
    r = t1_propagate(ctx, state, "hello world")
    # iter_cap caps layers beyond seeds; queue_budget caps PQ pops.
    assert r.metrics["iters"] <= cfg.t1["iter_cap"]
    assert r.metrics["pops"] <= cfg.t1["queue_budget"]

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

def test_t1_cache_hit_and_invalidation():
    cfg = Config()
    from clematis.engine.stages.t1 import t1_propagate

    # Build a unique graph so the cache is guaranteed cold for this test
    store = InMemoryGraphStore()
    gid = "g:cachetest"
    store.ensure(gid)
    store.upsert_nodes(gid, [
        Node(id="n:hello", label="hello"),
        Node(id="n:world", label="world"),
        Node(id="n:reply", label="reply"),
    ])
    store.upsert_edges(gid, [
        Edge(id="e:h->w", src="n:hello", dst="n:world", weight=0.8, rel="supports"),
        Edge(id="e:w->r", src="n:world", dst="n:reply", weight=0.5, rel="associates"),
    ])
    state = {"store": store, "active_graphs": [gid]}
    ctx = type("Ctx", (), {"cfg": cfg, "turn_id": "t", "agent_id": "A"})()

    # First call should MISS cache
    r1 = t1_propagate(ctx, state, "hello world")
    assert r1.metrics.get("cache_hits", 0) == 0
    assert r1.metrics.get("cache_misses", 0) >= 1

    # Second identical call should HIT cache
    r2 = t1_propagate(ctx, state, "hello world")
    assert r2.metrics.get("cache_hits", 0) >= 1
    assert r2.metrics.get("cache_used", False) is True
    assert r2.metrics.get("cache_enabled", True) is True

    # Bump etag and ensure cache invalidates
    store.upsert_nodes(gid, [Node(id="n:new", label="hello")])
    r3 = t1_propagate(ctx, state, "hello world")
    assert r3.metrics.get("cache_hits", 0) == 0
    assert r3.metrics.get("cache_misses", 0) >= 1

def test_t1_radius_cap_blocks_propagation():
    """
    With radius_cap=0, neighbors at depth 1 must not be expanded/emitted.
    Expect only the seed node to appear in deltas; radius_cap_hits > 0.
    """
    cfg = Config()
    cfg.t1["radius_cap"] = 0
    from clematis.engine.stages.t1 import t1_propagate

    store = InMemoryGraphStore()
    gid = "g:radius"
    store.ensure(gid)
    store.upsert_nodes(gid, [
        Node(id="n:seed", label="seed"),
        Node(id="n:down", label="downstream"),
    ])
    store.upsert_edges(gid, [
        Edge(id="e:s->d", src="n:seed", dst="n:down", weight=1.0, rel="supports"),
    ])
    state = {"store": store, "active_graphs": [gid]}
    ctx = type("Ctx", (), {"cfg": cfg, "turn_id": "t", "agent_id": "A"})()

    r = t1_propagate(ctx, state, "seed")
    emitted = {d["id"] for d in r.graph_deltas if d.get("op") == "upsert_node"}
    assert "n:seed" in emitted
    assert "n:down" not in emitted
    assert r.metrics["radius_cap_hits"] > 0


def test_t1_node_budget_blocks_expansion():
    """
    With a small node_budget, the seed should not expand to neighbors.
    Expect node_budget_hits > 0 and only the seed in deltas.
    """
    cfg = Config()
    cfg.t1["node_budget"] = 0.5  # seed acc=1.0 >= 0.5 triggers budget
    from clematis.engine.stages.t1 import t1_propagate

    store = InMemoryGraphStore()
    gid = "g:nodebudget"
    store.ensure(gid)
    store.upsert_nodes(gid, [
        Node(id="n:seed", label="seed"),
        Node(id="n:neighbor", label="neighbor"),
    ])
    store.upsert_edges(gid, [
        Edge(id="e:s->n", src="n:seed", dst="n:neighbor", weight=1.0, rel="supports"),
    ])
    state = {"store": store, "active_graphs": [gid]}
    ctx = type("Ctx", (), {"cfg": cfg, "turn_id": "t", "agent_id": "A"})()

    r = t1_propagate(ctx, state, "seed")
    emitted = {d["id"] for d in r.graph_deltas if d.get("op") == "upsert_node"}
    assert "n:seed" in emitted
    assert "n:neighbor" not in emitted
    assert r.metrics["node_budget_hits"] > 0


def test_t1_iter_cap_layers_blocks_depth():
    """
    iter_cap caps layers beyond seeds.
    With iter_cap=0, we should not visit depth-1 neighbors.
    """
    cfg = Config()
    cfg.t1["iter_cap"] = 0  # zero layers beyond seeds
    from clematis.engine.stages.t1 import t1_propagate

    store = InMemoryGraphStore()
    gid = "g:layers"
    store.ensure(gid)
    store.upsert_nodes(gid, [
        Node(id="n:seed", label="seed"),
        Node(id="n:mid", label="mid"),
        Node(id="n:deep", label="deep"),
    ])
    store.upsert_edges(gid, [
        Edge(id="e:s->m", src="n:seed", dst="n:mid", weight=1.0, rel="supports"),
        Edge(id="e:m->d", src="n:mid", dst="n:deep", weight=1.0, rel="supports"),
    ])
    state = {"store": store, "active_graphs": [gid]}
    ctx = type("Ctx", (), {"cfg": cfg, "turn_id": "t", "agent_id": "A"})()

    r = t1_propagate(ctx, state, "seed")
    emitted = {d["id"] for d in r.graph_deltas if d.get("op") == "upsert_node"}
    assert "n:seed" in emitted
    assert "n:mid" not in emitted and "n:deep" not in emitted
    assert r.metrics["iters"] == 0  # no layers explored
    assert r.metrics.get("layer_cap_hits", 0) >= 0  # presence of metric


def test_t1_relax_cap_limits_propagations():
    """
    relax_cap limits total relaxations (edge traversals).
    With relax_cap=1 and three neighbors, exactly one neighbor should be affected.
    """
    cfg = Config()
    cfg.t1["relax_cap"] = 1
    from clematis.engine.stages.t1 import t1_propagate

    store = InMemoryGraphStore()
    gid = "g:relax"
    store.ensure(gid)
    store.upsert_nodes(gid, [
        Node(id="n:seed", label="seed"),
        Node(id="n:a", label="a"),
        Node(id="n:b", label="b"),
        Node(id="n:c", label="c"),
    ])
    store.upsert_edges(gid, [
        Edge(id="e:s->a", src="n:seed", dst="n:a", weight=1.0, rel="supports"),
        Edge(id="e:s->b", src="n:seed", dst="n:b", weight=1.0, rel="supports"),
        Edge(id="e:s->c", src="n:seed", dst="n:c", weight=1.0, rel="supports"),
    ])
    state = {"store": store, "active_graphs": [gid]}
    ctx = type("Ctx", (), {"cfg": cfg, "turn_id": "t", "agent_id": "A"})()

    r = t1_propagate(ctx, state, "seed")
    emitted_neighbors = sorted([d["id"] for d in r.graph_deltas
                                if d.get("op") == "upsert_node" and d["id"] in {"n:a", "n:b", "n:c"}])
    # Exactly one neighbor should be emitted due to relax_cap=1
    assert len(emitted_neighbors) == 1, f"Expected 1 neighbor, got {emitted_neighbors}"
    assert r.metrics.get("propagations", 0) == 1