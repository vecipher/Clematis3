import json
import pytest


def _set(cfg, path, value):
    """Best-effort nested setter supporting object or dict configs."""
    cur = cfg
    for key in path[:-1]:
        nxt = getattr(cur, key, None) if not isinstance(cur, dict) else cur.get(key)
        if nxt is None:
            nxt = {}
            if isinstance(cur, dict):
                cur[key] = nxt
            else:
                setattr(cur, key, nxt)
        cur = nxt
    last = path[-1]
    if isinstance(cur, dict):
        cur[last] = value
    else:
        setattr(cur, last, value)


def test_pick_reason_passthrough(monkeypatch, tmp_path):
    # Route logs_dir() to temp
    import clematis.io.paths as paths_module

    monkeypatch.setattr(paths_module, "logs_dir", lambda: str(tmp_path), raising=True)

    # Load config and force a boundary yield with tiny budgets
    from clematis.io.config import load_config

    cfg = load_config("configs/config.yaml")

    _set(cfg, ["scheduler"], {})
    _set(cfg, ["scheduler", "enabled"], True)
    _set(cfg, ["scheduler", "policy"], "round_robin")
    _set(cfg, ["scheduler", "quantum_ms"], 20)
    _set(
        cfg,
        ["scheduler", "budgets"],
        {
            "t1_iters": 0,  # force immediate boundary hit at T1
            "t1_pops": 0,
            "t2_k": 0,
            "t3_ops": 0,
            "wall_ms": 200,
        },
    )
    _set(cfg, ["scheduler", "fairness"], {"max_consecutive_turns": 1, "aging_ms": 200})

    # Force a boundary yield irrespective of timing/budget by monkeypatching orchestrator logic
    import clematis.engine.orchestrator as orch

    monkeypatch.setattr(orch, "_should_yield", lambda *a, **k: "QUANTUM_EXCEEDED", raising=True)

    # Ensure orchestrator sees scheduler enabled with deterministic tiny budgets
    monkeypatch.setattr(
        orch,
        "_get_cfg",
        lambda ctx: {
            "scheduler": {
                "enabled": True,
                "policy": "round_robin",
                "quantum_ms": 20,
                "budgets": {
                    "t1_iters": 0,
                    "t1_pops": 0,
                    "t2_k": 0,
                    "t3_ops": 0,
                    "wall_ms": 200,
                },
                "fairness": {"max_consecutive_turns": 1, "aging_ms": 200},
            }
        },
        raising=True,
    )

    # Run a single turn with a pick_reason
    from clematis.world.scenario import run_one_turn

    state = {}
    run_one_turn("AgentA", state, "hello world", cfg, pick_reason="ROUND_ROBIN")

    # Inspect scheduler.jsonl
    p = tmp_path / "scheduler.jsonl"
    assert p.exists(), "scheduler.jsonl should be written"
    last = p.read_text(encoding="utf-8").splitlines()[-1]
    rec = json.loads(last)

    assert rec.get("policy") == "round_robin"
    assert rec.get("pick_reason") == "ROUND_ROBIN"
    assert rec.get("enforced") is True
    assert rec.get("stage_end") in {"T1", "T2", "T3", "T4", "Apply"}
