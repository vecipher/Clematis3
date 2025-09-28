import os
import sys
import subprocess
from pathlib import Path


def _repo_root() -> Path:
    # tests/scripts/test_validate_noop_shadow.py -> tests/scripts -> tests -> repo root
    return Path(__file__).resolve().parents[2]


def _script_path() -> Path:
    return _repo_root() / "scripts" / "validate_noop_shadow.py"


def _run_validator(base_dir: Path, shadow_dir: Path):
    env = os.environ.copy()
    env["BASE_ARTIFACTS"] = str(base_dir)
    env["SHADOW_ARTIFACTS"] = str(shadow_dir)
    # Use the same interpreter pytest is running with
    proc = subprocess.run(
        [sys.executable, str(_script_path())],
        cwd=str(_repo_root()),
        env=env,
        capture_output=True,
        text=True,
    )
    return proc.returncode, proc.stdout + proc.stderr


def test_noop_shadow_ignores_trace_only(tmp_path):
    """Gate D: differences limited to logs/quality/rq_traces.jsonl should pass (exit 0)."""
    base = tmp_path / "base"
    shadow = tmp_path / "shadow"
    (base / "logs").mkdir(parents=True, exist_ok=True)
    (shadow / "logs").mkdir(parents=True, exist_ok=True)
    # common deterministic file
    (base / "logs" / "t2_topk.jsonl").write_text('{"ok":true}\n', encoding="utf-8")
    (shadow / "logs" / "t2_topk.jsonl").write_text('{"ok":true}\n', encoding="utf-8")
    # shadow-only trace files in different subpaths should be ignored
    (shadow / "logs" / "quality").mkdir(parents=True, exist_ok=True)
    (shadow / "logs" / "quality" / "rq_traces.jsonl").write_text('{"trace":1}\n', encoding="utf-8")

    code, out = _run_validator(base, shadow)
    assert code == 0, f"Comparator should ignore rq_traces.jsonl only diffs. Output:\n{out}"


def test_shadow_fails_on_non_trace_diff(tmp_path):
    """Gate D: any extra non-trace file in shadow should fail."""
    base = tmp_path / "base"
    shadow = tmp_path / "shadow"
    (base / "logs").mkdir(parents=True, exist_ok=True)
    (shadow / "logs").mkdir(parents=True, exist_ok=True)
    # common file
    (base / "logs" / "t2_topk.jsonl").write_text('{"ok":true}\n', encoding="utf-8")
    (shadow / "logs" / "t2_topk.jsonl").write_text('{"ok":true}\n', encoding="utf-8")
    # non-trace extra in shadow
    (shadow / "logs" / "extra.json").write_text("{}", encoding="utf-8")
    # also include a trace file which should be ignored (not relevant to failure)
    (shadow / "logs" / "quality").mkdir(parents=True, exist_ok=True)
    (shadow / "logs" / "quality" / "rq_traces.jsonl").write_text("{}", encoding="utf-8")

    code, out = _run_validator(base, shadow)
    assert code != 0, f"Comparator must fail when non-trace diffs exist. Output:\n{out}"
    assert "DIFF:" in out


def test_shadow_fails_when_base_has_extra_file(tmp_path):
    """Gate D: asymmetry in non-trace files (missing in shadow) should fail too."""
    base = tmp_path / "base"
    shadow = tmp_path / "shadow"
    (base / "logs").mkdir(parents=True, exist_ok=True)
    (shadow / "logs").mkdir(parents=True, exist_ok=True)
    # common file
    (base / "logs" / "t2_topk.jsonl").write_text('{"ok":true}\n', encoding="utf-8")
    (shadow / "logs" / "t2_topk.jsonl").write_text('{"ok":true}\n', encoding="utf-8")
    # base-only extra (non-trace)
    (base / "logs" / "only_in_base.txt").write_text("x", encoding="utf-8")

    code, out = _run_validator(base, shadow)
    assert code != 0, f"Comparator must fail when base has extra non-trace files. Output:\n{out}"
    assert "DIFF:" in out


def test_shadow_fails_on_trace_in_wrong_location(tmp_path):
    """Gate D: placing rq_traces.jsonl outside logs/quality should fail."""
    base = tmp_path / "base"
    shadow = tmp_path / "shadow"
    (base / "logs").mkdir(parents=True, exist_ok=True)
    (shadow / "logs").mkdir(parents=True, exist_ok=True)
    # common file
    (base / "logs" / "t2_topk.jsonl").write_text('{"ok":true}\n', encoding="utf-8")
    (shadow / "logs" / "t2_topk.jsonl").write_text('{"ok":true}\n', encoding="utf-8")
    # wrong-location trace (should not be ignored)
    (shadow / "alt" / "deep").mkdir(parents=True, exist_ok=True)
    (shadow / "alt" / "deep" / "rq_traces.jsonl").write_text("{}", encoding="utf-8")

    code, out = _run_validator(base, shadow)
    assert code != 0, f"Comparator must fail on non-canonical trace location. Output:\n{out}"
    assert "DIFF:" in out
