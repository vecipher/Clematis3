import pytest
from types import SimpleNamespace as NS

# Unit-target the private helpers we added in PR68.
from clematis.engine.stages.t2 import (
    _t2_parallel_enabled,
    _merge_tier_hits_across_shards,
    _collect_shard_hits,
    _qscore,
)


# -------------------------
# Helpers / test doubles
# -------------------------

class DummyIndex:
    """Minimal index stub exposing the private shard iterator attribute."""
    def __init__(self, has_shards=True):
        if has_shards:
            # Only attribute presence matters for the gate; callable not invoked here.
            self._iter_shards_for_t2 = lambda *a, **kw: iter([object(), object()])

class Ep:
    """Episode-like object for _collect_shard_hits normalization tests."""
    def __init__(self, id, score, text=""):
        self.id = id
        self.score = score
        self.text = text

class DummyShard:
    """Shard with deterministic search_tiered behavior used by _collect_shard_hits."""
    def __init__(self, tier_to_hits):
        # tier_to_hits: dict[tier -> list[dict|Ep]]
        self.tier_to_hits = tier_to_hits

    def search_tiered(self, owner, q_vec, k, tier, hints):
        hits = list(self.tier_to_hits.get(tier, []))
        return hits[:k]


# -------------------------
# Tests
# -------------------------

def test_gate_semantics_variants():
    # default disabled -> False
    cfg = NS(perf=NS(parallel=NS(enabled=False, t2=False, max_workers=4)))
    idx = DummyIndex(has_shards=True)
    assert _t2_parallel_enabled(cfg, "inmemory", idx) is False

    # enable flags for subsequent checks
    cfg = NS(perf=NS(parallel=NS(enabled=True, t2=True, max_workers=4)))

    # enabled; backend lancedb is now supported (PR69) when shards>1 -> True
    assert _t2_parallel_enabled(cfg, "lancedb", idx) is True

    # enabled, t2 true, >1 worker, in-memory with shard affordance -> True
    assert _t2_parallel_enabled(cfg, "inmemory", idx) is True

    # max_workers <= 1 -> False
    cfg = NS(perf=NS(parallel=NS(enabled=True, t2=True, max_workers=1)))
    assert _t2_parallel_enabled(cfg, "inmemory", idx) is False

    # missing shard affordance -> False
    idx2 = DummyIndex(has_shards=False)
    cfg = NS(perf=NS(parallel=NS(enabled=True, t2=True, max_workers=8)))
    assert _t2_parallel_enabled(cfg, "inmemory", idx2) is False


def test_merge_tie_break_equal_scores_dedup_by_id():
    # Two shards, same score for 'a' and 'b'; duplicate 'a' across shards.
    sh1 = {"exact_semantic": [{"id": "b", "score": 0.500}, {"id": "a", "score": 0.500}]}
    sh2 = {"exact_semantic": [{"id": "a", "score": 0.500}]}
    merged, used = _merge_tier_hits_across_shards([sh1, sh2], ["exact_semantic"], 10)
    assert used == ["exact_semantic"]
    # Tie-break is id ascending when scores are equal: 'a' before 'b'
    assert [h["id"] for h in merged] == ["a", "b"]
    # qscore monotonicity sanity
    assert _qscore(0.5) == _qscore(0.500000000)


def test_merge_respects_k_after_tier_walk():
    # Tier-ordered walk: exact tier yields x,y,z after de-dup; K=3 is satisfied within exact; cluster is not reached.
    sh1 = {
        "exact_semantic": [{"id": "x", "score": 0.90}, {"id": "y", "score": 0.80}],
        "cluster_semantic": [{"id": "c1", "score": 0.95}],
    }
    sh2 = {
        "exact_semantic": [{"id": "y", "score": 0.80}, {"id": "z", "score": 0.70}],
        "cluster_semantic": [{"id": "c2", "score": 0.70}],
    }
    tiers = ["exact_semantic", "cluster_semantic", "archive"]
    merged, _ = _merge_tier_hits_across_shards([sh1, sh2], tiers, 3)
    # We expect: x, y, z from exact (dedup y); cluster is not reached
    assert [h["id"] for h in merged] == ["x", "y", "z"]


def _sequential_emulation(shard_dicts, tiers, k):
    """Emulate the sequential tier-walk on the union of shards (for oracle comparison)."""
    out, seen = [], set()
    for tier in tiers:
        bucket = []
        for d in shard_dicts:
            bucket.extend(d.get(tier, []))
        bucket.sort(key=lambda h: (-_qscore(h.get("score", h.get("_score", 0.0))), str(h.get("id"))))
        for h in bucket:
            hid = str(h.get("id"))
            if hid in seen:
                continue
            out.append(h)
            seen.add(hid)
            if len(out) >= k:
                return out
    return out


def test_merge_equals_sequential_emulation_on_synthetic_data():
    sh1 = {
        "exact_semantic": [{"id": "a", "score": 0.91}, {"id": "c", "score": 0.72}],
        "cluster_semantic": [{"id": "d", "score": 0.66}],
    }
    sh2 = {
        "exact_semantic": [{"id": "b", "score": 0.88}, {"id": "a", "score": 0.89}],
        "cluster_semantic": [{"id": "e", "score": 0.65}, {"id": "d", "score": 0.66}],
    }
    tiers = ["exact_semantic", "cluster_semantic", "archive"]
    K = 3
    merged, _ = _merge_tier_hits_across_shards([sh1, sh2], tiers, K)
    oracle = _sequential_emulation([sh1, sh2], tiers, K)
    assert [h["id"] for h in merged] == [h["id"] for h in oracle]


def test_collect_shard_hits_normalizes_episode_objects():
    tiers = ["exact_semantic", "cluster_semantic"]
    shard = DummyShard(
        {
            "exact_semantic": [Ep("aa", 0.42, "foo"), Ep("bb", 0.41, "bar")],
            "cluster_semantic": [],
        }
    )
    out = _collect_shard_hits(
        shard=shard,
        tiers=tiers,
        owner_query="any",
        q_vec=None,
        k_retrieval=5,
        now_str=None,
        sim_threshold=0.0,
        clusters_top_m=2,
    )
    assert set(out.keys()) == set(tiers)
    assert isinstance(out["exact_semantic"], list)
    assert out["exact_semantic"] and isinstance(out["exact_semantic"][0], dict)
    first = out["exact_semantic"][0]
    assert "id" in first and "score" in first
    assert isinstance(first["id"], str)
    assert isinstance(first["score"], float)
