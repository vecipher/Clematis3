import pytest
from types import SimpleNamespace

from clematis.engine.stages.t3 import FixtureMissingError, ReflectionBundle, reflect


class EmptyAdapter:
    def __init__(self, _path):
        self._path = _path

    def generate(self, prompt_json, max_tokens, temperature):
        return None


def test_missing_fixture_raises(monkeypatch):
    import importlib

    R = importlib.import_module("clematis.engine.stages.t3.reflect")

    monkeypatch.setattr(R, "FixtureLLMAdapter", EmptyAdapter)

    cfg = {
        "t3": {
            "allow_reflection": True,
            "reflection": {
                "backend": "llm",
                "summary_tokens": 8,
                "embed": False,
                "topk_snippets": 1,
                "log": False,
            },
            "llm": {"fixtures": {"enabled": True, "path": "/dev/null"}},
        },
        "scheduler": {"budgets": {"time_ms_reflection": 6000, "ops_reflection": 5}},
    }

    bundle = ReflectionBundle(
        ctx=SimpleNamespace(agent_id="AgentZ", turn_id=1),
        state_view=None,
        plan=SimpleNamespace(reflection=True),
        utter="X",
        snippets=[],
    )

    with pytest.raises(FixtureMissingError):
        reflect(bundle, cfg)
