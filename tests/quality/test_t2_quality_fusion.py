from typing import List, Dict, Any, Tuple
from clematis.engine.stages.t2.quality_ops import fuse


def _items() -> List[Dict[str, Any]]:
    # Semantic baseline scores: b > a > c
    # Texts chosen so that query="foo" matches only 'c'
    return [
        {"id": "b", "score": 3.0, "text": "y y"},
        {"id": "a", "score": 2.0, "text": "x"},
        {"id": "c", "score": 1.0, "text": "foo foo"},
    ]


def _cfg_enabled(alpha: float = 0.6) -> Dict[str, Any]:
    return {
        "t2": {
            "quality": {
                "enabled": True,
                "shadow": False,
                "trace_dir": "logs/quality",
                "redact": True,
                "lexical": {"bm25_k1": 1.2, "bm25_b": 0.75, "stopwords": "en-basic"},
                "fusion": {"mode": "score_interp", "alpha_semantic": float(alpha)},
            }
        }
    }


def _cfg_disabled() -> Dict[str, Any]:
    return {
        "t2": {
            "quality": {
                "enabled": False,
                "shadow": False,
                "trace_dir": "logs/quality",
                "redact": True,
            }
        }
    }


def _ids(seq: List[Dict[str, Any]]) -> List[str]:
    return [d["id"] for d in seq]


def _scores_fused(seq: List[Dict[str, Any]]) -> Dict[str, float]:
    return {d["id"]: float(d.get("score_fused", 0.0)) for d in seq}


def test_disabled_path_identity_preserved():
    items = _items()
    out, meta = fuse("foo", items, cfg=_cfg_disabled())
    # Identity path: items unchanged, meta empty
    assert _ids(out) == _ids(items)
    assert meta == {}
    # No score_fused should be required when disabled; if present, ignore


def test_alpha_one_preserves_semantic_ordering():
    items = _items()
    out, meta = fuse("foo", items, cfg=_cfg_enabled(alpha=1.0))
    # With alpha=1.0, fusion should preserve semantic rank fully
    assert _ids(out) == ["b", "a", "c"]
    assert isinstance(meta, dict)


def test_alpha_zero_prioritizes_lexical_hits_and_tie_breaks_lex():
    items = _items()
    out, meta = fuse("foo", items, cfg=_cfg_enabled(alpha=0.0))
    # Only 'c' contains "foo" -> it must rank first; 'a' and 'b' tie and break by lex(id): a before b
    assert _ids(out) == ["c", "a", "b"]
    # Meta should include total unique term hits across candidates (here, only 'c' matches -> 1)
    assert int(meta.get("t2q.lex_hits", -1)) == 1


def test_fusion_is_deterministic_across_runs():
    items = _items()
    cfg = _cfg_enabled(alpha=0.6)
    out1, meta1 = fuse("foo", items, cfg=cfg)
    out2, meta2 = fuse("foo", items, cfg=cfg)
    assert _ids(out1) == _ids(out2)
    # Ensure per-id fused scores are equal across runs
    assert _scores_fused(out1) == _scores_fused(out2)
    # Meta stable
    assert meta1 == meta2
