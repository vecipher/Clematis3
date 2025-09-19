

import numpy as np
from clematis.engine.types import Config, T1Result
from clematis.engine.stages.t2 import t2_semantic
from clematis.graph.store import InMemoryGraphStore, Node, Edge
from clematis.memory.index import InMemoryIndex


def _bootstrap_graph(state=None, gid="g:surface"):
    store = InMemoryGraphStore()
    store.ensure(gid)
    # Minimal labels for residual matching
    store.upsert_nodes(gid, [
        Node(id="n:apple", label="apple"),
        Node(id="n:banana", label="banana"),
        Node(id="n:fruit", label="fruit"),
    ])
    store.upsert_edges(gid, [
        Edge(id="e:a->b", src="n:apple", dst="n:banana", weight=1.0, rel="supports"),
    ])
    st = {"store": store, "active_graphs": [gid]}
    return st


def _add_ep(idx: InMemoryIndex, *, eid: str, text: str, ts: str, owner: str = "A", importance: float = 0.5, cluster_id: str | None = None):
    # InMemoryIndex expects a vec_full; use deterministic adapter via t2 (BGEAdapter) indirectly.
    # For the index itself we can just re-use the adapter in the stage by letting t2 embed queries;
    # episodes carry precomputed vecs here for ranking.
    from clematis.adapters.embeddings import BGEAdapter
    enc = BGEAdapter(dim=32)
    vec = enc.encode([text])[0]
    idx.add({
        "id": eid,
        "owner": owner,
        "text": text,
        "tags": [],
        "ts": ts,
        "vec_full": vec.astype(np.float32),
        "aux": {"importance": float(importance), **({"cluster_id": cluster_id} if cluster_id else {})},
    })


def test_t2_exact_semantic_recent_and_threshold():
    cfg = Config()
    # Make tiers only exact for this test
    cfg.t2["tiers"] = ["exact_semantic"]
    cfg.t2["k_retrieval"] = 10
    cfg.t2["sim_threshold"] = -1.0  # include all similarities; make test deterministic with the hash-based embeddings
    cfg.t2["exact_recent_days"] = 30

    state = _bootstrap_graph()
    idx = InMemoryIndex()
    state["mem_index"] = idx

    # One old apple (beyond 30d relative to current date), one recent apple, one recent banana
    _add_ep(idx, eid="ep_old",   text="about apple and trees", ts="2025-07-01T00:00:00Z")
    _add_ep(idx, eid="ep_apple", text="fresh apple pie story",  ts="2025-08-25T00:00:00Z")
    _add_ep(idx, eid="ep_banana",text="banana split tale",      ts="2025-08-26T00:00:00Z")

    # Seed T1 touching the apple node to include label in query
    t1 = T1Result(graph_deltas=[{"op":"upsert_node","id":"n:apple"}], metrics={})

    r = t2_semantic(type("Ctx", (), {"cfg": cfg})(), state, "tell me about", t1)

    ids = {h.id for h in r.retrieved}
    # Old one filtered by recent window
    assert "ep_old" not in ids
    # Recent items are present
    assert {"ep_apple", "ep_banana"}.issubset(ids)

    # Residual should include apple (label appears in episode text); banana may or may not
    touched = {d["id"] for d in r.graph_deltas_residual if d.get("op") == "upsert_node"}
    assert "n:apple" in touched
    assert len(touched) <= cfg.t2.get("residual_cap_per_turn", 32)


def test_t2_cluster_semantic_top_m():
    cfg = Config()
    cfg.t2["tiers"] = ["cluster_semantic"]
    cfg.t2["clusters_top_m"] = 1
    cfg.t2["sim_threshold"] = 0.0

    state = _bootstrap_graph()
    idx = InMemoryIndex()
    state["mem_index"] = idx

    # Two clusters: c1 (fruit domain), c2 (vehicles)
    _add_ep(idx, eid="c1_1", text="fruit apple orchard", ts="2025-08-10T00:00:00Z", cluster_id="c1")
    _add_ep(idx, eid="c1_2", text="fruit pear harvest", ts="2025-08-11T00:00:00Z", cluster_id="c1")
    _add_ep(idx, eid="c2_1", text="vehicle fast car",  ts="2025-08-12T00:00:00Z", cluster_id="c2")
    _add_ep(idx, eid="c2_2", text="vehicle mountain bike", ts="2025-08-13T00:00:00Z", cluster_id="c2")

    # T1 touches 'fruit' node so it appears in query
    t1 = T1Result(graph_deltas=[{"op":"upsert_node","id":"n:fruit"}], metrics={})

    r = t2_semantic(type("Ctx", (), {"cfg": cfg})(), state, "discuss", t1)

    ids = [h.id for h in r.retrieved]
    assert ids, "expected hits"
    # With top_m=1 and query biased towards 'fruit', all should be from c1_*
    assert all(i.startswith("c1_") for i in ids)


def test_t2_residual_cap_and_determinism():
    cfg = Config()
    cfg.t2["tiers"] = ["exact_semantic"]
    cfg.t2["sim_threshold"] = 0.0
    cfg.t2["residual_cap_per_turn"] = 3

    state = _bootstrap_graph()
    # Add more labels to exceed residual cap if naive
    store = state["store"]
    store.upsert_nodes("g:surface", [
        Node(id="n:cherry", label="cherry"),
        Node(id="n:grape", label="grape"),
        Node(id="n:mango", label="mango"),
    ])

    idx = InMemoryIndex()
    state["mem_index"] = idx
    # Single episode whose text contains many labels
    text = "this story mentions apple banana cherry grape mango and fruit"
    _add_ep(idx, eid="ep_all", text=text, ts="2025-08-28T00:00:00Z")

    t1 = T1Result(graph_deltas=[{"op":"upsert_node","id":"n:apple"}], metrics={})

    r1 = t2_semantic(type("Ctx", (), {"cfg": cfg})(), state, "tell me", t1)
    r2 = t2_semantic(type("Ctx", (), {"cfg": cfg})(), state, "tell me", t1)

    # Cap respected
    assert len([d for d in r1.graph_deltas_residual if d.get("op") == "upsert_node"]) <= 3
    # Deterministic emission
    assert r1.graph_deltas_residual == r2.graph_deltas_residual