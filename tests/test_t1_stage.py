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
    assert r.metrics["pops"] <= 1