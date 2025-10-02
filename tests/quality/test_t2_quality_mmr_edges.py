import pytest
from typing import Any, Dict, List

from clematis.engine.stages.t2.quality_ops import maybe_apply_mmr


def _mk(i: str, fused: float, text: str = "", toks=None) -> Dict[str, Any]:
    return {"id": i, "score_fused": fused, "text": text, "tokens": toks}


@pytest.mark.parametrize("lam", [0.0, 1.0])
def test_mmr_k1_returns_top_item_regardless_of_lambda(lam: float):
    # Degenerate k=1: selection should always be the top fused item irrespective of lambda
    fused: List[Dict[str, Any]] = [
        _mk("A", 0.90, toks={"a"}),
        _mk("B", 0.80, toks={"b"}),
        _mk("C", 0.70, toks={"c"}),
    ]
    qcfg = {
        "enabled": True,
        "fusion": {"mode": "rank", "alpha_semantic": 0.6},
        "lexical": {"stopwords": "en-basic"},
        "mmr": {"enabled": True, "lambda": lam, "k": 1},
    }
    out = maybe_apply_mmr(fused, qcfg)
    assert [x["id"] for x in out[:1]] == ["A"]


def test_mmr_lambda_extremes_tie_break_and_diversity():
    # Two near-duplicates (A,B) and one dissimilar (C)
    fused: List[Dict[str, Any]] = [
        _mk("A", 0.90, text="apple pie", toks={"apple", "pie"}),
        _mk("B", 0.90, text="apple tart", toks={"apple", "tart"}),
        _mk("C", 0.85, text="zebra", toks={"zebra"}),
    ]
    # Pure relevance: keep fused order; tie broken lex(id)
    qcfg_rel = {
        "enabled": True,
        "fusion": {"mode": "rank", "alpha_semantic": 0.6},
        "lexical": {"stopwords": "en-basic"},
        "mmr": {"enabled": True, "lambda": 0.0, "k": 3},
    }
    out_rel = maybe_apply_mmr(fused, qcfg_rel)
    ids_rel = [x["id"] for x in out_rel]
    assert ids_rel[:2] == ["A", "B"]  # lex(id) tie-break on equal relevance/similarity

    # Pure diversity: top-2 should include C (most dissimilar) and one of A/B
    qcfg_div = {
        "enabled": True,
        "fusion": {"mode": "rank", "alpha_semantic": 0.6},
        "lexical": {"stopwords": "en-basic"},
        "mmr": {"enabled": True, "lambda": 1.0, "k": 2},
    }
    out_div = maybe_apply_mmr(fused, qcfg_div)
    ids_div = [x["id"] for x in out_div[:2]]
    assert "C" in ids_div
    # If both A and B appear, order must be lex(id)
    if set(ids_div) == {"A", "B"}:
        assert ids_div == ["A", "B"]
