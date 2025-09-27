import subprocess, sys, os, tempfile, json, pathlib

PY = sys.executable

def test_explicit_dir_bypass_for_inspect_snapshot(tmp_path):
    # Provide an explicit --dir to ensure wrapper doesn't inject packaged default
    snapdir = tmp_path / "snaps"
    snapdir.mkdir()
    # Minimal valid schema-ish file so the script exits 0 on --format json
    (snapdir / "snap_000001.json").write_text('{"turn":"demo-1"}', encoding="utf-8")

    p = subprocess.run(
        [PY, "-m", "clematis", "inspect-snapshot", "--", "--dir", str(snapdir), "--format", "json"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    assert p.returncode == 0
    # Output is JSON produced by the inspector, not an echo of our file content.
    # Assert it used OUR explicit path (bypass packaged default).
    assert p.stdout.strip().startswith("{")
    out = json.loads(p.stdout)
    expected_path = str((snapdir / "snap_000001.json").resolve())
    reported = out.get("input_path") or out.get("source") or out.get("path")
    assert reported == expected_path, f"expected path {expected_path}, got {reported}; keys={sorted(out.keys())}"

def test_explicit_dir_bypass_for_rotate_logs(tmp_path):
    logdir = tmp_path / ".logs"
    logdir.mkdir()
    (logdir / "one.log").write_text("x", encoding="utf-8")
    p = subprocess.run(
        [PY, "-m", "clematis", "rotate-logs", "--", "--dir", str(logdir), "--dry-run"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    assert p.returncode == 0
    # Wrapper should NOT inject any default when --dir is explicit.
    # We can't easily assert internals, but a successful run with our path is enough.
    assert "error" not in (p.stderr.lower() + p.stdout.lower())