import os
import json
from clematis.world.scenario import run_one_turn
from clematis.engine.types import Config


def test_golden_path(tmp_path, monkeypatch):
    # Ensure logs go under repo .logs (relative to package path)
    cfg = Config()
    state = {}
    line = run_one_turn("AgentA", state, "hello world", cfg)
    assert line, "Dialogue line should be non-empty"

    # Check logs exist
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    logs_dir = os.path.join(repo_root, ".logs")
    names = [
        "t1.jsonl",
        "t2.jsonl",
        "t3_plan.jsonl",
        "t3_dialogue.jsonl",
        "t4.jsonl",
        "apply.jsonl",
        "health.jsonl",
    ]
    for n in names:
        p = os.path.join(logs_dir, n)
        assert os.path.exists(p), f"Expected log file missing: {p}"
        with open(p, "r", encoding="utf-8") as f:
            line = f.readline().strip()
            assert line, f"{n} should contain at least one record"
