import pytest

from configs.validate import validate_config


def test_llm_requires_fixtures_enabled():
    cfg = {
        "t3": {
            "allow_reflection": True,
            "reflection": {
                "backend": "llm",
                "summary_tokens": 8,
                "embed": True,
                "topk_snippets": 1,
                "log": False,
            },
            "llm": {"fixtures": {"enabled": False, "path": None}},
        },
        "scheduler": {"budgets": {"time_ms_reflection": 6000, "ops_reflection": 5}},
    }
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_llm_requires_path_when_enabled():
    cfg = {
        "t3": {
            "allow_reflection": True,
            "reflection": {
                "backend": "llm",
                "summary_tokens": 8,
                "embed": True,
                "topk_snippets": 1,
                "log": False,
            },
            "llm": {"fixtures": {"enabled": True, "path": "   "}},
        },
        "scheduler": {"budgets": {"time_ms_reflection": 6000, "ops_reflection": 5}},
    }
    with pytest.raises(ValueError):
        validate_config(cfg)
