from pathlib import Path
import yaml
from configs.validate import validate_config_verbose


def _repo_root() -> Path:
    # tests/examples/test_examples_quality_shadow_yaml.py -> tests/examples -> tests -> repo root
    return Path(__file__).resolve().parents[2]


def _example_path() -> Path:
    return _repo_root() / "examples" / "quality" / "shadow.yaml"


def test_shadow_example_exists_and_validates():
    p = _example_path()
    assert p.exists(), "examples/quality/shadow.yaml must exist"

    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    # Validate and normalize
    norm, warnings = validate_config_verbose(raw)

    # Triple gate should be satisfied in the example config
    perf = norm.get("perf", {})
    assert perf.get("enabled") is True
    metrics = perf.get("metrics", {})
    assert metrics.get("report_memory") is True

    q = norm.get("t2", {}).get("quality", {})
    assert q.get("enabled") is False  # PR36 forbids enabling quality
    assert q.get("shadow") is True  # shadow-only in the example
    assert isinstance(q.get("trace_dir"), str) and q.get("trace_dir")
    assert q.get("redact") is True

    # No shadow warning expected when triple gate is satisfied
    assert not any("W[t2.quality.shadow]" in w for w in warnings)


def test_shadow_example_triple_gate_is_satisfied():
    p = _example_path()
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))

    assert raw["perf"]["enabled"] is True
    assert raw["perf"]["metrics"]["report_memory"] is True
    assert raw["t2"]["quality"]["shadow"] is True
    assert raw["t2"]["quality"].get("enabled", False) is False
