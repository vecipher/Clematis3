import hashlib
import json
from types import SimpleNamespace

from clematis.engine.stages.t3 import ReflectionBundle, reflect


class LLMResult:
    def __init__(self, text: str):
        self.text = text
        self.tokens = len((text or "").split())
        self.truncated = False


class FakeFixtureAdapter:
    def __init__(self, _path):
        self._path = _path

    def generate(self, prompt_json, max_tokens, temperature):  # pragma: no cover - replaced per test
        raise AssertionError("patched in test")


def test_llm_fixture_deterministic(monkeypatch):
    import importlib

    R = importlib.import_module("clematis.engine.stages.t3.reflect")

    monkeypatch.setattr(R, "FixtureLLMAdapter", FakeFixtureAdapter)

    cfg = {
        "t3": {
            "allow_reflection": True,
            "reflection": {
                "backend": "llm",
                "summary_tokens": 6,
                "embed": True,
                "topk_snippets": 2,
                "log": False,
            },
            "llm": {"fixtures": {"enabled": True, "path": "/dev/null"}},
        },
        "scheduler": {"budgets": {"time_ms_reflection": 6000, "ops_reflection": 5}},
    }

    ctx = SimpleNamespace(agent_id="AgentA", turn_id=1)
    bundle = ReflectionBundle(
        ctx=ctx,
        state_view=None,
        plan=SimpleNamespace(reflection=True),
        utter="Hello  world!",
        snippets=["One", "Two", "Three"],
    )

    norm_snippets = [R._normalize(s, keep_punct=True) for s in bundle.snippets[:2]]
    prompt_obj = {
        "agent": str(getattr(bundle.ctx, "agent_id", "unknown")),
        "plan_reflection": True,
        "snippets": norm_snippets,
        "summary_tokens": 6,
        "task": "reflect_summary",
        "turn": 1,
        "utter": R._normalize(bundle.utter, keep_punct=True),
        "version": 1,
    }
    prompt_json = json.dumps(prompt_obj, sort_keys=True, separators=(",", ":"))
    key = hashlib.sha256(prompt_json.encode("utf-8")).hexdigest()[:12]

    def patched_generate(self, prompt_json, max_tokens, temperature):
        actual = hashlib.sha256(
            json.dumps(json.loads(prompt_json), sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:12]
        text = "fixture summary deterministic" if actual == key else ""
        return LLMResult(text)

    monkeypatch.setattr(FakeFixtureAdapter, "generate", patched_generate, raising=False)

    r1 = reflect(bundle, cfg)
    r2 = reflect(bundle, cfg)

    assert r1.summary == r2.summary == "fixture summary deterministic"
    entry = r1.memory_entries[0]
    assert "vec_full" in entry and isinstance(entry["vec_full"], list) and len(entry["vec_full"]) == 32
    assert r1.metrics["backend"] == "llm-fixture"
    assert r1.metrics["summary_len"] == 3
    assert r1.metrics["fixture_key"] == key
