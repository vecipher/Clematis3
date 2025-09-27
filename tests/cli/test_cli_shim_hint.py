import os, sys, subprocess, re
from pathlib import Path
import importlib.util

PY = sys.executable

def _has_mod(name: str) -> bool:
    return importlib.util.find_spec(name) is not None

def _run_file(path, *args, **env):
    cmd = [PY, str(path), *args]
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env={**os.environ, **env})

def _assert_single_hint(stderr: str):
    # Expect exactly one shim hint line that starts with "[clematis]".
    lines = [ln for ln in stderr.splitlines() if ln.strip().startswith("[clematis]")]
    assert len(lines) == 1, f"expected single shim hint line starting with [clematis], got {len(lines)}: {stderr!r}"
    hint = lines[0]
    # Allow either the PR47/umbrella-style "delegate ->" breadcrumb or the deprecation note text.
    assert ("delegate ->" in hint) or ("note: direct script is deprecated" in hint), f"unexpected shim hint content: {hint!r}"

def test_rotate_logs_shim_single_hint(tmp_path: Path):
    p = _run_file("scripts/rotate_logs.py", "--help")
    assert p.returncode == 0
    _assert_single_hint(p.stderr)

def test_bench_t4_shim_single_hint():
    p = _run_file("scripts/bench_t4.py", "--help")
    assert p.returncode in (0, 1, 2, 127)  # tolerate non-zero when deps missing
    _assert_single_hint(p.stderr)
    if _has_mod("numpy"):
        assert p.returncode == 0
    else:
        assert "ModuleNotFoundError" in p.stderr or "ImportError" in p.stderr

def test_seed_lance_demo_shim_single_hint():
    p = _run_file("scripts/seed_lance_demo.py", "--help")
    assert p.returncode in (0, 1, 2, 127)  # tolerate non-zero when deps missing
    _assert_single_hint(p.stderr)
    if _has_mod("numpy") and _has_mod("lancedb"):
        assert p.returncode == 0
    else:
        assert "ModuleNotFoundError" in p.stderr or "ImportError" in p.stderr

def test_inspect_snapshot_shim_single_hint():
    p = _run_file("scripts/inspect_snapshot.py", "--help")
    assert p.returncode == 0
    _assert_single_hint(p.stderr)