import json
import os
import sys
from pathlib import Path

import pytest

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

@pytest.mark.integration
def test_traces_written_to_custom_dir_and_first_line_parses(tmp_path):
    """
    Triple-gate ON + non-default trace_dir -> trace file is created and first JSON line parses.
    We use SHADOW mode to avoid relying on fusion usage; rankings remain unchanged.
    """
    # Import lazily to allow skip if module path differs in some envs
    try:
        from clematis.engine.stages.t2 import t2_pipeline  # type: ignore
    except Exception as e:
        pytest.skip(f"t2_pipeline unavailable: {e}")

    trace_dir = tmp_path / "quality_custom"

    cfg = {
        "perf": {"enabled": True, "metrics": {"report_memory": True, "trace_dir": str(trace_dir)}},
        "t2": {"quality": {"enabled": False, "shadow": True}},
    }

    # Run with a reason so we can assert it survived into the trace meta
    res = t2_pipeline(cfg, "trace write path smoke", ctx={"trace_reason": "test_write_path"})
    _ = res  # not asserted here; we only verify file+json behaviour

    # Assert file exists in the configured directory
    path = trace_dir / "rq_traces.jsonl"
    assert path.exists(), f"expected trace at {path}, but file does not exist"
    text = path.read_text(encoding="utf-8")
    assert text.strip(), "trace file is empty"

    # Parse first JSON line
    first = text.splitlines()[0]
    rec = json.loads(first)

    # Minimal schema checks
    assert rec.get("trace_schema_version") == 1
    meta = rec.get("meta", {})
    assert isinstance(meta, dict)
    # We passed a reason via ctx; shadow path should preserve it
    assert meta.get("reason") == "test_write_path"

    # Sanity: the recorded cfg is a dict and items is a list
    assert isinstance(rec.get("cfg", {}), dict)
    assert isinstance(rec.get("items", []), list)