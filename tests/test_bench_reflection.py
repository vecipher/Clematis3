"""
PR87 — Microbench tests for bench_reflection.py

These tests exercise the deterministic contract of the reflection microbench:
• Rule-based path is deterministic and CI-normalized (ms == 0.0).
• "Raw time" is still normalized in CI to avoid flakes.
• LLM fixtures path (if present) yields deterministic output with a fixture_key.

Notes:
• These tests force CI normalization internally via monkeypatch; you do NOT need to export CI=true for this file.
• CLEMATIS_NETWORK_BAN=1 is not required for these microbench tests. It remains recommended for the full suite to prevent incidental network.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest

from clematis.scripts.bench_reflection import run_bench

RULEBASED_CFG: Dict[str, Any] = {
    "t3": {
        "allow_reflection": True,
        "reflection": {
            "backend": "rulebased",
            "summary_tokens": 64,
            "embed": True,
            "log": True,
            "topk_snippets": 2,
        },
    },
    "scheduler": {
        "budgets": {
            "time_ms_reflection": 6000,
            "ops_reflection": 3,
        }
    },
}


def test_bench_rulebased_deterministic_ci_normalized(monkeypatch: pytest.MonkeyPatch) -> None:
    # CI normalization forces "ms" to 0.0 and guarantees equality across runs.
    monkeypatch.setenv("CI", "true")
    a = run_bench(RULEBASED_CFG, repeats=2, normalize_ms=True)
    b = run_bench(RULEBASED_CFG, repeats=2, normalize_ms=True)
    assert a == b
    assert a["backend"] == "rulebased"
    assert a["ms"] == 0.0
    assert isinstance(a["summary_len"], int) and a["summary_len"] >= 0
    assert 0 <= a["ops"] <= RULEBASED_CFG["scheduler"]["budgets"]["ops_reflection"]
    assert "fixture_key" not in a


def test_bench_rulebased_raw_time_still_normalized_in_ci(monkeypatch: pytest.MonkeyPatch) -> None:
    # Even if normalize_ms=False, CI=true should still zero ms to keep tests deterministic.
    monkeypatch.setenv("CI", "true")
    out = run_bench(RULEBASED_CFG, repeats=1, normalize_ms=False)
    assert out["backend"] == "rulebased"
    assert out["ms"] == 0.0


def test_bench_llm_fixture_if_available(monkeypatch: pytest.MonkeyPatch) -> None:
    # Optional: only runs if the fixture file exists. Keeps suite green without LLM fixtures.
    fixtures = Path("tests/fixtures/reflection_llm.jsonl")
    if not fixtures.exists():
        pytest.skip("LLM fixture file not present; skipping fixtures microbench test.")
    monkeypatch.setenv("CI", "true")
    cfg = {
        "t3": {
            "allow_reflection": True,
            "reflection": {
                "backend": "llm",
                "summary_tokens": 32,
                "embed": True,
                "log": True,
                "topk_snippets": 1,
            },
            "llm": {"fixtures": {"enabled": True, "path": str(fixtures)}},
        },
        "scheduler": {"budgets": {"time_ms_reflection": 6000, "ops_reflection": 2}},
    }
    out = run_bench(cfg, repeats=1, normalize_ms=True)
    assert out["backend"] == "llm"
    assert out["ms"] == 0.0
    # The fixtures adapter should surface a stable key in metrics → payload
    assert "fixture_key" in out and isinstance(out["fixture_key"], str) and len(out["fixture_key"]) > 0
