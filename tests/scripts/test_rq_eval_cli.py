import importlib
import importlib.util
import json
import pathlib
import sys

import pytest


def _import_rq_eval_module():
    root = pathlib.Path(__file__).parents[2]
    path = root / "scripts" / "rq_eval.py"
    spec = importlib.util.spec_from_file_location("rq_eval_module", path)
    if spec is None or spec.loader is None:
        pytest.skip("rq_eval.py not found or not importable")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # Register module so dataclasses can resolve __module__
    spec.loader.exec_module(mod)  # type: ignore[arg-type]
    return mod


def test_metric_functions_on_toy_set():
    rq = _import_rq_eval_module()

    truth = ["d1", "d2"]
    ranked = ["d3", "d2", "d1"]
    k = 3

    r = rq.recall_at_k(truth, ranked, k)
    m = rq.mrr_at_k(truth, ranked, k)

    # Expected recall: both relevant in top-3
    assert r == pytest.approx(1.0)
    # Expected MRR: first relevant at rank 2 → 1/2
    assert m == pytest.approx(0.5)

    gains = {"d1": 2.0, "d2": 1.0}
    n = rq.ndcg_at_k(gains, ranked, k)
    # Hand-computed ~0.5872 for this toy ordering
    assert n == pytest.approx(0.5872, rel=1e-3)


def test_cli_produces_json_and_csv(tmp_path):
    rq = _import_rq_eval_module()

    # If the runtime pipeline is unavailable in this environment, skip gracefully
    try:
        t2 = importlib.import_module("clematis.engine.stages.t2")
    except Exception:
        pytest.skip("t2 module not importable")
    if not hasattr(t2, "run_t2"):
        pytest.skip("run_t2 entrypoint not available")

    root = pathlib.Path(__file__).parents[2]
    queries = root / "tests" / "fixtures" / "rq" / "queries.tsv"
    truth = root / "tests" / "fixtures" / "rq" / "qrels.tsv"
    corpus = root / "tests" / "fixtures" / "rq" / "corpus.jsonl"

    # Use existing example configs from PR37/PR39
    configA = root / "examples" / "quality" / "lexical_fusion.yaml"
    configB = root / "tests" / "examples" / "quality" / "normalizer_aliases.yaml"

    out_json = tmp_path / "rq_eval.json"
    out_csv = tmp_path / "rq_eval.csv"

    rc = rq.main(
        [
            "--corpus",
            str(corpus),
            "--queries",
            str(queries),
            "--truth",
            str(truth),
            "--configA",
            str(configA),
            "--configB",
            str(configB),
            "--k",
            "5",
            "--out",
            str(out_json),
            "--csv",
            str(out_csv),
        ]
    )
    assert rc == 0

    # JSON exists and has expected top-level keys
    assert out_json.exists()
    data = json.loads(out_json.read_text(encoding="utf-8"))
    for key in ("schema_version", "k", "systems", "delta", "per_query"):
        assert key in data
    assert data["schema_version"] == 1
    assert "A" in data["systems"] and "B" in data["systems"]
    assert isinstance(data["per_query"], list) and len(data["per_query"]) > 0

    # Metrics sanity: within [0,1]
    for sys_name in ("A", "B"):
        macro = data["systems"][sys_name]["metrics"]["macro"]
        for mkey in ("recall", "mrr", "ndcg"):
            assert 0.0 <= float(macro[mkey]) <= 1.0

    # CSV exists and has header
    assert out_csv.exists()
    header = out_csv.read_text(encoding="utf-8").splitlines()[0]
    assert header == "qid,system,recall,mrr,ndcg,hits"
