from pathlib import Path
import importlib
import pytest

from clematis.io.log import append_jsonl, _append_jsonl_unbuffered


CASES = [
    {"a": 1, "b": "ascii"},
    {"a": "µ", "b": "雪", "c": [1, 2, 3]},
    {"a": 1.23456789, "b": 1e-9},
    {"nested": {"k": "v"}, "lst": list(range(10))},
]


def _repo_root() -> Path:
    # tests/engine/<this file> → repo root is two levels up from tests/
    return Path(__file__).resolve().parents[2]


def _log_base() -> Path:
    """
    Try to read the log base directory from clematis.io.log; fall back to <repo>/.logs.
    This mirrors the default observed in earlier runs.
    """
    logmod = importlib.import_module("clematis.io.log")
    for name in ("LOG_DIR", "LOG_BASE", "BASE_DIR", "base", "_BASE", "DEFAULT_LOG_DIR"):
        if hasattr(logmod, name):
            val = getattr(logmod, name)
            try:
                p = Path(val)
                if str(p):
                    return p
            except Exception:
                continue
    return _repo_root() / ".logs"


def _write_and_read(base: Path, fn, filename: str, obj) -> bytes:
    base.mkdir(parents=True, exist_ok=True)
    target = base / filename
    if target.exists():
        target.unlink()
    fn(filename, obj)
    assert target.exists(), f"Expected {target} to be created at {target}"
    return target.read_bytes()


@pytest.mark.parametrize("obj", CASES)
def test_json_writer_unbuffered_equivalence(obj):
    """
    Both writers must serialize to identical bytes for the same record.
    We write to distinct files under the module's .logs base to avoid cross-talk.
    """
    base = _log_base()
    b1 = _write_and_read(base, append_jsonl, "equiv_writer.jsonl", obj)
    b2 = _write_and_read(base, _append_jsonl_unbuffered, "equiv_unbuffered.jsonl", obj)
    assert b1 == b2
