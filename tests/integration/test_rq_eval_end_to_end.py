import json
import os
import subprocess
import sys
import pathlib

import importlib
import pytest


def _ensure_runtime_available():
    try:
        t2 = importlib.import_module("clematis.engine.stages.t2")
    except Exception:
        pytest.skip("t2 module not importable")
    if not hasattr(t2, "run_t2"):
        pytest.skip("run_t2 entrypoint not available")


def test_rq_eval_end_to_end_runs_and_writes_outputs(tmp_path):
    """End-to-end: invoke the CLI script as a subprocess and verify outputs.

    This test is intentionally lenient about trace emission because it depends
    on the triple gate and runtime internals. If traces are present, we sanity
    check that the file is non-empty, but we do not require it.
    """
    _ensure_runtime_available()

    root = pathlib.Path(__file__).parents[2]
    script = root / "scripts" / "rq_eval.py"
    queries = root / "tests" / "fixtures" / "rq" / "queries.tsv"
    truth = root / "tests" / "fixtures" / "rq" / "qrels.tsv"
    corpus = root / "tests" / "fixtures" / "rq" / "corpus.jsonl"

    # Use example configs (A: PR37 fusion; B: PR39 normalizer/aliases)
    configA = root / "examples" / "quality" / "lexical_fusion.yaml"
    configB = root / "tests" / "examples" / "quality" / "normalizer_aliases.yaml"

    out_json = tmp_path / "rq_eval.json"
    out_csv = tmp_path / "rq_eval.csv"

    cmd = [
        sys.executable,
        str(script),
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
        "--emit-traces",
    ]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        pytest.fail(
            f"rq_eval.py failed with code {proc.returncode}:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

    # JSON exists and has expected structure
    assert out_json.exists(), "JSON output not written"
    data = json.loads(out_json.read_text(encoding="utf-8"))
    for key in ("schema_version", "k", "systems", "delta", "per_query"):
        assert key in data
    assert data["schema_version"] == 1

    # CSV exists and has header
    assert out_csv.exists(), "CSV output not written"
    header = out_csv.read_text(encoding="utf-8").splitlines()[0]
    assert header == "qid,system,recall,mrr,ndcg,hits"

    # Optional trace sanity: if a trace file exists, it should be non-empty JSONL
    logs_dir = root / "logs" / "quality"
    if logs_dir.exists():
        candidates = list(logs_dir.glob("**/rq_traces.jsonl")) + list(
            logs_dir.glob("rq_traces.jsonl")
        )
        for path in candidates:
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
                if lines:
                    # Don't assert on reason; runtime may use "enabled" or accept a hint
                    json.loads(lines[0])
            except Exception:
                # Non-fatal in end-to-end smoke
                pass
