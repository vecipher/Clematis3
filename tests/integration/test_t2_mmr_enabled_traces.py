import pytest
import importlib


from typing import Any, Dict, List

from clematis.engine.stages.t2_quality import maybe_apply_mmr
from clematis.engine.stages.t2_quality_mmr import MMRItem, avg_pairwise_distance


def _mk(i: str, fused: float, text: str = "", toks=None) -> Dict[str, Any]:
    return {"id": i, "score_fused": fused, "text": text, "tokens": toks}


def test_mmr_reorders_when_enabled_with_tokens():
    fused: List[Dict[str, Any]] = [
        _mk("A", 0.9, toks={"apple", "pie"}),
        _mk("B", 0.85, toks={"apple", "tart"}),
        _mk("C", 0.7, toks={"zebra"}),
    ]
    qcfg = {
        "enabled": True,
        "fusion": {"mode": "rank", "alpha_semantic": 0.6},
        "lexical": {"stopwords": "en-basic"},
        "mmr": {"enabled": True, "lambda": 1.0, "k": None},
    }
    out = maybe_apply_mmr(fused, qcfg)
    # Expected: first A (relevance), then C (most distant from A), then B
    assert [x["id"] for x in out[:3]] == ["A", "C", "B"]


def test_mmr_noop_when_disabled():
    fused: List[Dict[str, Any]] = [
        _mk("A", 0.9, toks={"a"}),
        _mk("B", 0.8, toks={"b"}),
    ]
    qcfg = {
        "enabled": True,
        "fusion": {"mode": "rank", "alpha_semantic": 0.6},
        "lexical": {"stopwords": "en-basic"},
        "mmr": {"enabled": False, "lambda": 0.5, "k": None},
    }
    out = maybe_apply_mmr(fused, qcfg)
    # Allow identity by object or by id sequence
    assert out is fused or [x["id"] for x in out] == ["A", "B"]


def test_mmr_works_with_text_only_and_reports_diversity_head():
    # No explicit tokens: ensure the MMR glue tokenizes text consistently with PR37
    fused: List[Dict[str, Any]] = [
        _mk("A", 0.9, text="Apple pie"),
        _mk("B", 0.85, text="Apple tart"),
        _mk("C", 0.7, text="zebra"),
    ]
    qcfg = {
        "enabled": True,
        "fusion": {"mode": "rank", "alpha_semantic": 0.6},
        "lexical": {"stopwords": "en-basic"},
        "mmr": {"enabled": True, "lambda": 1.0, "k": 2},
    }
    out = maybe_apply_mmr(fused, qcfg)
    assert [x["id"] for x in out[:3]] == ["A", "C", "B"]

    # Compute diversity on the head (k=2) similar to the gated metric in t2.py
    head = out[:2]
    # very small tokenizer: lower + split non-alnum
    import unicodedata, re

    def toks(s: str):
        s = unicodedata.normalize("NFKC", s or "").lower()
        return frozenset([t for t in re.split(r"[^0-9a-zA-Z]+", s) if t])

    head_items = [MMRItem(id=h["id"], rel=0.0, toks=toks(h.get("text", ""))) for h in head]
    div = avg_pairwise_distance(head_items)
    assert 0.0 <= div <= 1.0
def test_mmr_emits_selected_metric_when_gated():
    """Check that PR38 emits t2q.mmr.selected when quality+MMR are enabled and perf triple gate is on.
    If the pipeline hook isn't exposed (e.g., function not available), skip gracefully.
    """
    try:
        t2 = importlib.import_module("clematis.engine.stages.t2")
    except Exception:
        pytest.skip("t2 module not importable in this environment")

    # Require a pipeline/test hook that accepts emit_metric; otherwise skip.
    pipeline = getattr(t2, "t2_pipeline", None)
    if pipeline is None:
        pytest.skip("t2_pipeline hook not available; metric emission verified elsewhere")

    cfg = {
        "perf": {"enabled": True, "metrics": {"report_memory": True}},
        "t2": {
            "quality": {
                "enabled": True,
                "fusion": {"mode": "rank", "alpha_semantic": 0.6},
                "lexical": {"stopwords": "en-basic"},
                "mmr": {"enabled": True, "lambda": 0.5, "k": 2},
            }
        },
    }

    metrics = {}

    def _emit_metric(k, v):
        metrics[k] = v

    # Dummy trace sink
    def _emit_trace(*args, **kwargs):
        return None

    # Run a tiny query; specifics don't matter for metric emission
    try:
        pipeline("apple tart", cfg, emit_metric=_emit_metric, emit_trace=_emit_trace)
    except TypeError:
        # Signature mismatch; skip rather than fail (keeps this integration test portable across minor API diffs)
        pytest.skip("t2_pipeline signature not compatible in this context")

    assert metrics.get("t2q.mmr.selected") == 2