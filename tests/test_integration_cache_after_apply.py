from types import SimpleNamespace
import os

import pytest

import clematis.engine.orchestrator as orch
from clematis.engine.types import ProposedDelta, T4Result


class FakeStore:
    """Minimal store that supports apply_deltas and holds weights in .w"""

    def __init__(self, wmin=-1.0, wmax=1.0):
        self.w = {}
        self.wmin = float(wmin)
        self.wmax = float(wmax)

    def _ckey(self, d: ProposedDelta):
        return (d.target_kind, d.target_id, d.attr)

    def apply_deltas(self, graph_id, deltas):
        edits = 0
        clamps = 0
        for d in deltas or []:
            k = self._ckey(d)
            old = self.w.get(k, 0.0)
            proposed = old + float(d.delta)
            clamped = max(self.wmin, min(self.wmax, proposed))
            if clamped != proposed:
                clamps += 1
            if clamped != old:
                self.w[k] = clamped
                edits += 1
        return {"edits": edits, "clamps": clamps}


def test_cache_miss_after_apply_invalidation(monkeypatch, tmp_path):
    calls = {"t2": 0}
    apply_logs = []

    # Swallow non-essential logs; capture apply.jsonl payloads
    def fake_append_jsonl(name, payload):
        if name == "apply.jsonl":
            apply_logs.append(payload)

    monkeypatch.setattr(orch, "append_jsonl", fake_append_jsonl, raising=True)

    # Minimal T1
    monkeypatch.setattr(
        orch,
        "t1_propagate",
        lambda ctx, state, text: SimpleNamespace(metrics={"t1": True}),
        raising=True,
    )

    # T2 we care about: count calls and return a simple object
    def fake_t2(ctx, state, text, t1):
        calls["t2"] += 1
        return SimpleNamespace(metrics={"t2": True}, retrieved=[])

    monkeypatch.setattr(orch, "t2_semantic", fake_t2, raising=True)

    # T3 helpers
    def fake_t3_deliberate(ctx, state, bundle):
        return SimpleNamespace(
            version="plan-v1", ops=[SimpleNamespace(kind="EditGraph")], deltas=[], reflection=False
        )

    def fake_t3_dialogue(dialog_bundle, plan):
        return "OK"

    monkeypatch.setattr(orch, "t3_deliberate", fake_t3_deliberate, raising=False)
    monkeypatch.setattr(orch, "t3_dialogue", fake_t3_dialogue, raising=False)

    # Force T4 to approve a delta when enabled, bypassing internal policy
    def fake_t4_filter(ctx, state, t1, t2, plan, utter):
        deltas = [
            ProposedDelta(target_kind="node", target_id="n:a", attr="weight", delta=0.2, op_idx=0)
        ]
        return T4Result(approved_deltas=deltas, rejected_ops=[], reasons=[], metrics={})

    monkeypatch.setattr(orch, "t4_filter", fake_t4_filter, raising=False)

    # Build ctx/state: cache enabled; start with T4 disabled to warm cache without bumping version.
    ctx = SimpleNamespace(
        turn_id=1,
        agent_id="Cohere",
        config=SimpleNamespace(
            t4={
                "enabled": False,  # start disabled for warmup
                "cache_bust_mode": "on-apply",
                "cache": {"enabled": True, "max_entries": 16, "ttl_sec": 600},
                "snapshot_every_n_turns": 1000,  # avoid snapshot churn in test
                "snapshot_dir": str(tmp_path / "snaps"),
            }
        ),
    )
    state = {"version_etag": None, "store": FakeStore(), "active_graphs": ["g:surface"]}

    # 1) First run: MISS -> t2 called
    orch.run_turn(ctx, state, input_text="hello")
    assert calls["t2"] == 1

    # 2) Second run, still T4 disabled and same input/version: HIT -> t2 NOT called again
    orch.run_turn(ctx, state, input_text="hello")
    assert calls["t2"] == 1

    # 3) Enable T4, which will run Apply and invalidate caches
    ctx.config.t4["enabled"] = True
    orch.run_turn(ctx, state, input_text="hello")

    # Ensure Apply logged invalidations > 0 (cache had at least one entry to clear)
    assert apply_logs, "apply.jsonl should have been logged"
    last_apply = apply_logs[-1]
    assert int(last_apply.get("cache_invalidations", 0)) >= 1

    # 4) Next run with same input must be a MISS due to version-aware key and/or invalidation
    orch.run_turn(ctx, state, input_text="hello")
    assert calls["t2"] == 2
