import json
from clematis.adapters.llm import FixtureLLMAdapter, _prompt_hash


def test_fixture_roundtrip(tmp_path):
    prompt = "SYSTEM: Return ONLY valid JSON.\nSTATE: {\"turn\":1}\n"
    h = _prompt_hash(prompt)
    p = tmp_path / "fx.jsonl"

    completion_obj = {"plan": ["a"], "rationale": "r"}
    line = {
        "prompt_hash": h,
        "completion": json.dumps(completion_obj)
    }
    p.write_text(json.dumps(line) + "\n", encoding="utf-8")

    fx = FixtureLLMAdapter(str(p))
    out = fx.generate(prompt, max_tokens=64, temperature=0.0)

    assert out.text == json.dumps(completion_obj)
    assert out.tokens >= 1
    assert out.truncated is False


def test_token_clipping(tmp_path):
    prompt = "SIMPLE\n"
    h = _prompt_hash(prompt)
    p = tmp_path / "fx.jsonl"

    long_text = " ".join(["w"] * 20)
    line = {"prompt_hash": h, "completion": long_text}
    p.write_text(json.dumps(line) + "\n", encoding="utf-8")

    fx = FixtureLLMAdapter(str(p))
    out = fx.generate(prompt, max_tokens=5, temperature=0.0)

    assert out.text.split() == ["w"] * 5
    assert out.tokens == 5
    assert out.truncated is True
