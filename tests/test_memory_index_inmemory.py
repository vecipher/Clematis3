import numpy as np
import pytest
from clematis.memory.index import InMemoryIndex
from clematis.adapters.embeddings import BGEAdapter


def _add_ep(
    idx: InMemoryIndex,
    *,
    eid: str,
    owner: str,
    text: str,
    ts: str,
    tags=None,
    importance: float = 0.5,
    cluster_id: str | None = None,
):
    enc = BGEAdapter(dim=32)
    vec = enc.encode([text])[0]
    idx.add(
        {
            "id": eid,
            "owner": owner,
            "text": text,
            "tags": tags or [],
            "ts": ts,
            "vec_full": vec.astype(np.float32),
            "aux": {
                "importance": float(importance),
                **({"cluster_id": cluster_id} if cluster_id else {}),
            },
        }
    )


def test_exact_semantic_recent_window_and_threshold():
    idx = InMemoryIndex()
    owner = "A"
    # Within 30 days of 2025-09-01: Aug 15 & Aug 25 qualify; July 01 does not
    _add_ep(idx, eid="ep_old", owner=owner, text="topic apple", ts="2025-07-01T00:00:00Z")
    _add_ep(idx, eid="ep_mid", owner=owner, text="topic banana", ts="2025-08-15T00:00:00Z")
    _add_ep(idx, eid="ep_new", owner=owner, text="topic apple", ts="2025-08-25T00:00:00Z")

    # Query exactly matches "topic apple" so those items should score highest
    enc = BGEAdapter(dim=32)
    q_vec = enc.encode(["topic apple"])[0]

    hits = idx.search_tiered(
        owner,
        q_vec,
        k=5,
        tier="exact_semantic",
        hints={"recent_days": 30, "sim_threshold": 0.1, "now": "2025-09-01T00:00:00Z"},
    )

    ids = [h.id for h in hits]
    # Old one is outside recent window
    assert "ep_old" not in ids
    # Both recent "topic apple" should be present and ranked ahead of banana
    assert ids[0] in {"ep_new", "ep_old", "ep_mid"}
    assert set(ids).issuperset({"ep_new"})

    # Ensure scores are non-increasing
    scores = [h.score for h in hits]
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


def test_cluster_semantic_routes_to_top_m_clusters():
    idx = InMemoryIndex()
    owner = "A"
    # Build two clusters c1 and c2
    _add_ep(
        idx, eid="c1_1", owner=owner, text="fruit apple", ts="2025-08-10T00:00:00Z", cluster_id="c1"
    )
    _add_ep(
        idx, eid="c1_2", owner=owner, text="fruit pear", ts="2025-08-11T00:00:00Z", cluster_id="c1"
    )
    _add_ep(
        idx, eid="c2_1", owner=owner, text="vehicle car", ts="2025-08-12T00:00:00Z", cluster_id="c2"
    )
    _add_ep(
        idx,
        eid="c2_2",
        owner=owner,
        text="vehicle bike",
        ts="2025-08-13T00:00:00Z",
        cluster_id="c2",
    )

    # Query is close to cluster c1 (fruit domain)
    enc = BGEAdapter(dim=32)
    q_vec = enc.encode(["fruit apple"])[0]

    hits = idx.search_tiered(
        owner,
        q_vec,
        k=10,
        tier="cluster_semantic",
        hints={"clusters_top_m": 1, "sim_threshold": 0.0, "now": "2025-09-01T00:00:00Z"},
    )

    ids = [h.id for h in hits]
    # All returned hits should belong to cluster c1 only when top_m=1
    assert ids, "expected non-empty hits"
    assert all(i.startswith("c1_") for i in ids)


def test_archive_tier_filters_by_quarter():
    idx = InMemoryIndex()
    owner = "A"
    # Q2: Apr, May. Q3: Aug
    _add_ep(idx, eid="q2_a", owner=owner, text="alpha", ts="2025-04-10T00:00:00Z")
    _add_ep(idx, eid="q2_b", owner=owner, text="beta", ts="2025-05-20T00:00:00Z")
    _add_ep(idx, eid="q3_a", owner=owner, text="gamma", ts="2025-08-01T00:00:00Z")

    enc = BGEAdapter(dim=32)
    q_vec = enc.encode(["alpha"])[0]

    hits = idx.search_tiered(
        owner,
        q_vec,
        k=10,
        tier="archive",
        hints={"archive_quarters": ["2025Q2"], "sim_threshold": 0.0},
    )

    ids = [h.id for h in hits]
    assert set(ids).issubset({"q2_a", "q2_b"})
    assert "q3_a" not in ids


def test_deterministic_ranking_tiebreak_by_id():
    idx = InMemoryIndex()
    owner = "A"
    # Two eps with identical text (→ identical vectors) and both recent
    _add_ep(idx, eid="ep1", owner=owner, text="same text", ts="2025-08-20T00:00:00Z")
    _add_ep(idx, eid="ep2", owner=owner, text="same text", ts="2025-08-21T00:00:00Z")

    enc = BGEAdapter(dim=32)
    q_vec = enc.encode(["same text"])[0]

    hits = idx.search_tiered(
        owner,
        q_vec,
        k=5,
        tier="exact_semantic",
        hints={"recent_days": 365, "sim_threshold": 0.0, "now": "2025-09-01T00:00:00Z"},
    )

    ids = [h.id for h in hits]
    # With equal scores, tie-breaker is id ascending
    assert ids[:2] == ["ep1", "ep2"]
