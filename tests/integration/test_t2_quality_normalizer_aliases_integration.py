# New file: tests/integration/test_t2_quality_normalizer_aliases_integration.py
import os
import tempfile
from typing import Any, Dict, List

import pytest

from clematis.engine.stages.t2.quality_ops import maybe_apply_mmr


def _mk(i: str, fused: float, text: str = "", toks=None) -> Dict[str, Any]:
    return {"id": i, "score_fused": fused, "text": text, "tokens": toks}


def test_alias_map_changes_mmr_order_deterministically():
    """Alias expansion affects MMR token sets and thus ordering, deterministically.

    Setup: A and B describe the same concept using an alias pair; C is unrelated.
      - Without alias map: A vs B have max diversity (no shared tokens) ⇒ with λ=1.0 and k=2, order is A, B.
      - With alias map (llm → "large language model"): A tokens expand to match B ⇒ diversity(A,B)=0 ⇒ order is A, C.
    """
    fused: List[Dict[str, Any]] = [
        _mk("A", 0.90, text="Intro to LLM"),
        _mk("B", 0.88, text="large language model guide"),
        _mk("C", 0.80, text="zebra"),
    ]

    base_qcfg = {
        "enabled": True,
        "fusion": {"mode": "rank", "alpha_semantic": 0.6},
        "lexical": {"stopwords": "en-basic"},
        "mmr": {"enabled": True, "lambda": 1.0, "k": 2},
        # normalizer defaults apply (enabled: true)
    }

    # No alias map: expect A, then B (ties by diversity → lex(id) picks B over C)
    out_no_alias = maybe_apply_mmr(fused, base_qcfg)
    assert [x["id"] for x in out_no_alias[:2]] == ["A", "B"]

    # With alias map: llm → "large language model"; expect A, then C (B has 0 distance from A)
    fd, path = tempfile.mkstemp(prefix="aliases_", suffix=".yaml")
    os.close(fd)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write("llm: large language model\n")
        qcfg_with_alias = {**base_qcfg, "aliasing": {"map_path": path}}
        out_with_alias = maybe_apply_mmr(fused, qcfg_with_alias)
        assert [x["id"] for x in out_with_alias[:2]] == ["A", "C"]
        # Determinism: repeated call yields identical order
        out_with_alias_2 = maybe_apply_mmr(fused, qcfg_with_alias)
        assert [x["id"] for x in out_with_alias_2[:3]] == [x["id"] for x in out_with_alias[:3]]
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
