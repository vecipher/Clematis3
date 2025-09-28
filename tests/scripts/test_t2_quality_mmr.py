import pytest

from clematis.engine.stages.t2_quality_mmr import (
    MMRItem,
    mmr_select,
    mmr_reorder_full,
    jaccard_distance,
    avg_pairwise_distance,
)


def _mk(items):
    """Helper: build a list of MMRItem from dicts."""
    return [MMRItem(id=i["id"], rel=float(i["rel"]), toks=frozenset(i["toks"])) for i in items]


def ids(seq):
    return [x.id if isinstance(x, MMRItem) else x for x in seq]


def test_mmr_extremes_lambda_relevance_vs_diversity():
    items = _mk(
        [
            {"id": "A", "rel": 0.90, "toks": {"apple", "pie"}},
            {"id": "B", "rel": 0.85, "toks": {"apple", "tart"}},
            {"id": "C", "rel": 0.70, "toks": {"zebra"}},
        ]
    )

    # lam=0 → relevance only: A, B, C
    order0 = mmr_reorder_full(items, k=None, lam=0.0)
    assert [items[i].id for i in order0[:3]] == ["A", "B", "C"]

    # lam=1 → diversity heavy: first A (by rel), then C (max div from A), then B
    order1 = mmr_reorder_full(items, k=None, lam=1.0)
    assert [items[i].id for i in order1[:3]] == ["A", "C", "B"]


def test_mmr_k_cap_and_tie_breaking_by_lex_id():
    items = _mk(
        [
            {"id": "A", "rel": 0.50, "toks": {"x"}},
            {"id": "B", "rel": 0.50, "toks": {"x"}},  # same rel & toks as A → tie ⇒ lex(id)
            {"id": "C", "rel": 0.40, "toks": {"y"}},
        ]
    )

    sel = mmr_select(items, k=2, lam=0.5)
    assert [items[i].id for i in sel] == ["A", "C"]


def test_jaccard_distance_edge_cases_and_avg_pairwise():
    assert jaccard_distance(frozenset(), frozenset()) == 0.0
    assert jaccard_distance(frozenset({"a"}), frozenset({"a"})) == 0.0
    assert jaccard_distance(frozenset({"a"}), frozenset({"b"})) == 1.0

    items = _mk(
        [
            {"id": "A", "rel": 0.9, "toks": {"a"}},
            {"id": "B", "rel": 0.8, "toks": {"b"}},
        ]
    )
    assert avg_pairwise_distance(items) == pytest.approx(1.0, abs=1e-9)


def test_full_permutation_head_then_tail_and_no_duplicates():
    # Initial order by (-rel, id): A(0.90), B(0.88), C(0.87), D(0.50)
    # With lam=1 and k=2: head should be A then B (both far from A, tie by id), tail C, D
    items = _mk(
        [
            {"id": "A", "rel": 0.90, "toks": {"a"}},
            {"id": "B", "rel": 0.88, "toks": {"b"}},
            {"id": "C", "rel": 0.87, "toks": {"a"}},
            {"id": "D", "rel": 0.50, "toks": {"c"}},
        ]
    )
    order = mmr_reorder_full(items, k=2, lam=1.0)
    seq = [items[i].id for i in order]
    assert seq == ["A", "B", "C", "D"]
    assert len(seq) == len(set(seq)), "Permutation must not duplicate ids"


def test_invalid_lambda_raises():
    items = _mk([{"id": "A", "rel": 1.0, "toks": {"x"}}])
    with pytest.raises(ValueError):
        mmr_select(items, k=1, lam=-0.01)
    with pytest.raises(ValueError):
        mmr_select(items, k=1, lam=1.01)
