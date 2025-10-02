import json
import pytest
from clematis.adapters.llm import FixtureLLMAdapter, LLMAdapterError, _prompt_hash


def test_missing_file_raises():
    with pytest.raises(LLMAdapterError) as exc:
        FixtureLLMAdapter("this/does/not/exist.jsonl")
    assert str(exc.value) == "Fixture file not found"


def test_bad_jsonl_line_raises(tmp_path):
    p = tmp_path / "fx.jsonl"
    p.write_text("{bad json}\n", encoding="utf-8")
    with pytest.raises(LLMAdapterError) as exc:
        FixtureLLMAdapter(str(p))
    assert str(exc.value) == "Invalid fixture JSONL at line 1"


def test_missing_mapping_raises(tmp_path):
    p = tmp_path / "fx.jsonl"
    line = {"prompt_hash": "deadbeef", "completion": "ok"}
    p.write_text(json.dumps(line) + "\n", encoding="utf-8")
    fx = FixtureLLMAdapter(str(p))
    with pytest.raises(LLMAdapterError) as exc:
        fx.generate("some other prompt", max_tokens=16, temperature=0.0)
    assert str(exc.value) == "No fixture for prompt hash"


def test_canonical_newlines_match(tmp_path):
    # Ensure CRLF vs LF produce the same hash and lookup
    prompt_lf = "A\nB\n"
    prompt_crlf = "A\r\nB\r\n"
    h = _prompt_hash(prompt_lf)
    p = tmp_path / "fx.jsonl"
    p.write_text(json.dumps({"prompt_hash": h, "completion": "x y z"}) + "\n", encoding="utf-8")

    fx = FixtureLLMAdapter(str(p))
    out = fx.generate(prompt_crlf, max_tokens=10, temperature=0.0)
    assert out.text == "x y z"
