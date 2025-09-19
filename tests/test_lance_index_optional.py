

import pytest
import numpy as np
from datetime import datetime, timezone

# Skip this whole file if LanceDB (and pyarrow) aren't available.
pytest.importorskip("lancedb")
pytest.importorskip("pyarrow")

from clematis.memory.lance_index import LanceIndex


def mk_ep(id, vec, owner="A", ts="2025-09-01T00:00:00Z", text=None, tags=None, aux=None):
    return {
        "id": id,
        "owner": owner,
        "text": text or f"episode {id}",
        "tags": tags or [],
        "ts": ts,
        "vec_full": vec,
        "aux": aux or {},
    }


def _ids(hits):
    out = []
    for h in hits:
        if isinstance(h, dict):
            out.append(str(h.get("id")))
        else:
            out.append(str(getattr(h, "id")))
    return out


def test_version_increments_on_add(tmp_path):
    idx = LanceIndex(uri=str(tmp_path))
    v0 = idx.index_version()
    idx.add(mk_ep("e1", [1.0, 0.0, 0.0, 0.0]))
    v1 = idx.index_version()
    assert v1 == v0 + 1
    idx.add(mk_ep("e2", [0.0, 1.0, 0.0, 0.0]))
    assert idx.index_version() == v1 + 1


def test_exact_semantic_owner_and_recent_window(tmp_path):
    idx = LanceIndex(uri=str(tmp_path / "db2"))
    now_iso = "2025-09-19T00:00:00Z"
    recent_ts = "2025-09-10T00:00:00Z"  # within 30 days
    old_ts = "2025-06-01T00:00:00Z"     # outside 30-day window

    # Owner A (recent + old), Owner B (recent)
    idx.add(mk_ep("r1", [1, 0, 0, 0], owner="A", ts=recent_ts))
    idx.add(mk_ep("r2", [1, 0, 0, 0], owner="B", ts=recent_ts))
    idx.add(mk_ep("o1", [1, 0, 0, 0], owner="A", ts=old_ts))

    q = np.array([1, 0, 0, 0], dtype=np.float32)
    hits = idx.search_tiered(
        owner="A",
        q_vec=q,
        k=10,
        tier="exact_semantic",
        hints={"now": now_iso, "recent_days": 30, "sim_threshold": -1.0},
    )
    assert _ids(hits) == ["r1"], f"unexpected hits: {_ids(hits)}"


def test_tie_breaks_by_id_on_equal_scores(tmp_path):
    idx = LanceIndex(uri=str(tmp_path / "db3"))
    # Two items with identical vectors → identical cosine; must sort by id asc
    idx.add(mk_ep("a1", [1, 0, 0, 0]))
    idx.add(mk_ep("a2", [1, 0, 0, 0]))

    q = np.array([1, 0, 0, 0], dtype=np.float32)
    hits = idx.search_tiered(owner=None, q_vec=q, k=2, tier="exact_semantic", hints={"sim_threshold": -1.0})
    assert _ids(hits) == ["a1", "a2"], f"tie-break failed: {_ids(hits)}"


def test_cluster_semantic_top_m(tmp_path):
    idx = LanceIndex(uri=str(tmp_path / "db4"))
    # Cluster c1 near the query, cluster c2 orthogonal
    idx.add(mk_ep("c1a", [1.0, 0.0, 0.0, 0.0], aux={"cluster_id": "c1"}))
    idx.add(mk_ep("c1b", [0.9, 0.1, 0.0, 0.0], aux={"cluster_id": "c1"}))
    idx.add(mk_ep("c2a", [0.0, 1.0, 0.0, 0.0], aux={"cluster_id": "c2"}))
    idx.add(mk_ep("c2b", [0.0, 0.9, 0.1, 0.0], aux={"cluster_id": "c2"}))

    q = np.array([1, 0, 0, 0], dtype=np.float32)
    hits = idx.search_tiered(
        owner=None,
        q_vec=q,
        k=10,
        tier="cluster_semantic",
        hints={"clusters_top_m": 1, "sim_threshold": -1.0},
    )
    ids = set(_ids(hits))
    assert ids <= {"c1a", "c1b"}, f"expected only c1 items, got {ids}"


def test_archive_quarter_filter(tmp_path):
    idx = LanceIndex(uri=str(tmp_path / "db5"))
    # Q3 2025 (Aug) vs Q2 2025 (May)
    idx.add(mk_ep("q3", [1, 0, 0, 0], ts="2025-08-05T00:00:00Z"))
    idx.add(mk_ep("q2", [1, 0, 0, 0], ts="2025-05-10T00:00:00Z"))

    q = np.array([1, 0, 0, 0], dtype=np.float32)
    hits = idx.search_tiered(
        owner=None,
        q_vec=q,
        k=10,
        tier="archive",
        hints={"archive_quarters": {"2025Q3"}, "sim_threshold": -1.0},
    )
    assert _ids(hits) == ["q3"], f"archive filter failed: {_ids(hits)}"