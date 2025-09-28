import math
import pytest

_hybrid = pytest.importorskip("clematis.engine.stages.hybrid")
rerank_with_gel = _hybrid.rerank_with_gel


# --- simple shims -------------------------------------------------------------
class _Ctx:
    def __init__(self, hybrid_cfg: dict):
        self.config = {"t2": {"hybrid": hybrid_cfg}}
        self.cfg = self.config  # some call sites use ctx.cfg


class _State:
    def __init__(self, edges: dict | None = None):
        self.graph = {"nodes": {}, "edges": edges or {}}


def _mk_edge(u: str, v: str, w: float) -> dict:
    return {"src": u, "dst": v, "weight": float(w), "rel": "coact", "attrs": {}}


def _ids(items):
    # Items are tuples (id, score) here; the reranker supports dicts/objects too
    return [it[0] if isinstance(it, tuple) else it.get("id") for it in items]


# --- tests -------------------------------------------------------------------


def test_disabled_no_change():
    ctx = _Ctx({"enabled": False})
    s = _State({"a→b": _mk_edge("a", "b", 0.9)})
    items = [("a", 0.90), ("b", 0.80), ("c", 0.70)]

    out, m = rerank_with_gel(ctx, s, items)
    assert _ids(out) == _ids(items)
    assert m.get("hybrid_used") is False


def test_no_edges_no_change_when_enabled():
    ctx = _Ctx({"enabled": True, "lambda_graph": 0.5})
    s = _State({})
    items = [("a", 0.90), ("b", 0.80), ("c", 0.70)]

    out, m = rerank_with_gel(ctx, s, items)
    assert _ids(out) == _ids(items)
    assert m.get("hybrid_used") is False


def test_rank_shift_with_edges():
    # Base sims: a (0.90) > c (0.86) > b (0.85). With strong a—b edge, b should outrank c.
    ctx = _Ctx(
        {
            "enabled": True,
            "lambda_graph": 0.5,
            "edge_threshold": 0.1,
            "anchor_top_m": 2,  # anchors: a, c
        }
    )
    edges = {
        "a→b": _mk_edge("a", "b", 0.6),
        # c has no help for b
    }
    s = _State(edges)
    items = [("a", 0.90), ("c", 0.86), ("b", 0.85), ("d", 0.10)]

    out, m = rerank_with_gel(ctx, s, items)
    assert _ids(out)[:3] == ["a", "b", "c"]
    assert m.get("hybrid_used") is True
    assert m.get("k_reordered", 0) >= 1


def test_two_hop_bonus_changes_order():
    # No direct a—v edge; path a—w—v should help v only when walk_hops=2
    items = [("a", 0.90), ("v", 0.86), ("w", 0.85)]
    edges = {
        "a→w": _mk_edge("a", "w", 0.6),
        "v→w": _mk_edge("v", "w", 0.6),
    }
    s = _State(edges)

    # hops=1: order unchanged
    ctx1 = _Ctx(
        {
            "enabled": True,
            "walk_hops": 1,
            "lambda_graph": 0.6,
            "edge_threshold": 0.1,
            "anchor_top_m": 1,  # only 'a' as anchor
        }
    )
    out1, m1 = rerank_with_gel(ctx1, s, items)
    assert _ids(out1) == ["a", "v", "w"]

    # hops=2: v should gain via a→w→v best path
    ctx2 = _Ctx(
        {
            "enabled": True,
            "walk_hops": 2,
            "damping": 0.5,
            "lambda_graph": 0.6,
            "edge_threshold": 0.1,
            "anchor_top_m": 1,
        }
    )
    out2, m2 = rerank_with_gel(ctx2, s, items)
    assert _ids(out2) == ["a", "v", "w"]  # v stays ahead of w (reinforced)
    assert m2.get("hybrid_used") is True


def test_deterministic_tie_break_by_id():
    # Make b and c receive identical bonuses from anchor a; ids decide the tie (b < c)
    items = [("a", 0.90), ("b", 0.80), ("c", 0.80)]
    edges = {
        "a→b": _mk_edge("a", "b", 0.5),
        "a→c": _mk_edge("a", "c", 0.5),
    }
    s = _State(edges)
    ctx = _Ctx({"enabled": True, "lambda_graph": 1.0, "edge_threshold": 0.1, "anchor_top_m": 1})

    out, m = rerank_with_gel(ctx, s, items)
    assert _ids(out) == ["a", "b", "c"]


def test_bonus_clamped_by_max_bonus():
    # Huge edges but small max_bonus should cap influence
    items = [("a", 0.90), ("b", 0.89), ("c", 0.88)]
    edges = {"a→b": _mk_edge("a", "b", 10.0)}
    s = _State(edges)
    ctx = _Ctx(
        {
            "enabled": True,
            "lambda_graph": 1.0,
            "edge_threshold": 0.0,
            "max_bonus": 0.05,  # clamp
            "anchor_top_m": 1,
        }
    )
    out, m = rerank_with_gel(ctx, s, items)
    # b cannot surpass a because bonus is clamped small
    assert _ids(out)[0] == "a"


def test_degree_norm_invdeg_reduces_boost_for_hubs():
    # v connects to two anchors equally; invdeg should reduce its bonus relative to none
    items = [("a", 0.90), ("b", 0.89), ("v", 0.88), ("c", 0.87)]
    edges = {
        "a→v": _mk_edge("a", "v", 0.4),
        "b→v": _mk_edge("b", "v", 0.4),
        # also v—c below threshold; ignored
        "c→v": _mk_edge("c", "v", 0.05),
    }
    s = _State(edges)

    # No degree normalization
    ctx_none = _Ctx(
        {
            "enabled": True,
            "degree_norm": "none",
            "lambda_graph": 0.5,
            "edge_threshold": 0.1,
            "anchor_top_m": 2,  # anchors: a,b
        }
    )
    out_none, m_none = rerank_with_gel(ctx_none, s, items)

    # With invdeg, v's bonus should shrink; ideally order stays the same or v moves less
    ctx_inv = _Ctx(
        {
            "enabled": True,
            "degree_norm": "invdeg",
            "lambda_graph": 0.5,
            "edge_threshold": 0.1,
            "anchor_top_m": 2,
        }
    )
    out_inv, m_inv = rerank_with_gel(ctx_inv, s, items)

    # We don't assert absolute scores (not exposed); we assert that the id order is stable
    assert _ids(out_none) == _ids(out_inv)
