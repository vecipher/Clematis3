

import os
import io
import json
from pathlib import Path
import importlib.util


def _load_rotate_module():
    """Dynamically load scripts/rotate_logs.py as a module for testing."""
    repo_root = Path(__file__).resolve().parents[1]
    mod_path = repo_root / "scripts" / "rotate_logs.py"
    spec = importlib.util.spec_from_file_location("rotate_logs", str(mod_path))
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec and spec.loader, "could not load rotate_logs.py"
    spec.loader.exec_module(mod)  # type: ignore[assignment]
    return mod


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_rotate_one_creates_backup(tmp_path):
    mod = _load_rotate_module()
    p = tmp_path / "logs" / "a.jsonl"
    _write(p, "line1\n")

    # Act
    did = mod.rotate_one(str(p), backups=3)

    # Assert
    assert did is True
    assert not p.exists(), "original path should be moved"
    assert (tmp_path / "logs" / "a.jsonl.1").exists()


def test_rotate_one_cascade_with_cap(tmp_path):
    mod = _load_rotate_module()
    base = tmp_path / "logs" / "b.jsonl"
    _write(base, "B0\n")
    _write(Path(str(base) + ".1"), "B1\n")
    _write(Path(str(base) + ".2"), "B2\n")

    # Keep 3 backups: expect .2 -> .3, .1 -> .2, base -> .1, and base removed
    did = mod.rotate_one(str(base), backups=3)
    assert did is True

    p1 = Path(str(base) + ".1")
    p2 = Path(str(base) + ".2")
    p3 = Path(str(base) + ".3")

    assert not base.exists()
    assert p1.exists() and p1.read_text() == "B0\n"
    assert p2.exists() and p2.read_text() == "B1\n"
    assert p3.exists() and p3.read_text() == "B2\n"


def test_rotate_one_dry_run_changes_nothing(tmp_path):
    mod = _load_rotate_module()
    base = tmp_path / "logs" / "c.jsonl"
    _write(base, "C0\n")
    p1 = Path(str(base) + ".1")
    _write(p1, "C1\n")

    # Dry-run: should print planned actions but not move files
    did = mod.rotate_one(str(base), backups=2, dry_run=True)
    assert did is True  # dry-run still returns True if base exists
    assert base.exists() and base.read_text() == "C0\n"
    assert p1.exists() and p1.read_text() == "C1\n"


def test_script_main_rotates_when_over_size(tmp_path, capsys):
    mod = _load_rotate_module()
    d = tmp_path / "logs"
    d.mkdir()
    f = d / "big.log"
    _write(f, "X" * 10)

    # Use main() with a tiny max-bytes and matching pattern
    rc = mod.main(["--dir", str(d), "--pattern", "*.log", "--max-bytes", "1", "--backups", "2"])
    assert rc == 0

    # After rotation, base should be moved to .1
    assert not f.exists()
    assert (d / "big.log.1").exists()