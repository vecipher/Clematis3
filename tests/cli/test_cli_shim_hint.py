import os
import sys
import subprocess
import re
from pathlib import Path
import importlib.util

PY = sys.executable


def _has_mod(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _has_bracket_line(stderr: str) -> bool:
    """Returns True if any bracket-prefixed status line is present."""
    for ln in stderr.splitlines():
        s = ln.strip()
        if s.startswith("["):
            return True
    return False


def _has_dep_error(stderr: str) -> bool:
    return ("ModuleNotFoundError" in stderr) or ("ImportError" in stderr)


def _run_file(path, *args, **env):
    cmd = [PY, str(path), *args]
    return subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env={**os.environ, **env}
    )


def _assert_single_hint(stderr: str):
    # Expect exactly one shim hint line that starts with "[clematis]".
    lines = [ln for ln in stderr.splitlines() if ln.strip().startswith("[clematis]")]
    assert (
        len(lines) == 1
    ), f"expected single shim hint line starting with [clematis], got {len(lines)}: {stderr!r}"
    hint = lines[0]
    # Allow either the PR47/umbrella-style "delegate ->" breadcrumb or the deprecation note text.
    assert ("delegate ->" in hint) or (
        "note: direct script is deprecated" in hint
    ), f"unexpected shim hint content: {hint!r}"


def test_rotate_logs_shim_single_hint(tmp_path: Path):
    p = _run_file("scripts/rotate_logs.py", "--help")
    assert p.returncode == 0
    _assert_single_hint(p.stderr)


def test_bench_t4_shim_single_hint():
    p = _run_file("scripts/bench_t4.py", "--help")
    # When heavy deps aren't installed, the module import may fail before hint_once() runs.
    # We still require either a shim/notice line OR an explicit import error message.
    assert p.returncode in (0, 1, 2, 127)
    if _has_mod("numpy"):
        _assert_single_hint(p.stderr)
        assert p.returncode == 0
    else:
        assert _has_dep_error(p.stderr) or _has_bracket_line(p.stderr)


def test_seed_lance_demo_shim_single_hint():
    p = _run_file("scripts/seed_lance_demo.py", "--help")
    assert p.returncode in (0, 1, 2, 127)
    # If both numpy and lancedb are present, the shim hint must appear and exit 0.
    if _has_mod("numpy") and _has_mod("lancedb"):
        _assert_single_hint(p.stderr)
        assert p.returncode == 0
    else:
        # Accept either a bracketed project notice like "[seed-lance] ... required"
        # or a Python import error, since import happens before main().
        assert _has_dep_error(p.stderr) or _has_bracket_line(p.stderr)


def test_inspect_snapshot_shim_single_hint():
    p = _run_file("scripts/inspect_snapshot.py", "--help")
    assert p.returncode == 0
    _assert_single_hint(p.stderr)
