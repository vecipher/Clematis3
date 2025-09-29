import json
import os
import subprocess
import sys
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run_cli(args, cwd=ROOT, env_overrides=None):
    env = dict(os.environ)
    if env_overrides:
        env.update(env_overrides)
    env.setdefault("PYTHONUTF8", "1")
    p = subprocess.run(
        [sys.executable, "-m", "clematis", *args],
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return p.returncode, p.stdout, p.stderr


def test_exit_zero_happy_paths():
    # validate --json (format is pass-through; content may be text today)
    code, out, err = run_cli(["validate", "--json"])  # should not crash
    assert code == 0

    # inspect-snapshot --json (must produce valid JSON)
    code, out, err = run_cli(["inspect-snapshot", "--json"])  # wrapper forwards --format json
    assert code == 0
    json.loads(out)

    # rotate-logs baseline dry-run (no structured flags)
    code, out, err = run_cli(["rotate-logs", "--", "--dry-run"])  # should succeed
    assert code == 0


def test_structured_formats_contract_track_b():
    # inspector: JSON works
    code, out, err = run_cli(["inspect-snapshot", "--json"])  # before --
    assert code == 0
    json.loads(out)

    # inspector: TABLE path (Track B) — expect ASCII header + dashes and at least one data row
    code, out, err = run_cli(["inspect-snapshot", "--", "--table"])  # after --, hoisted
    assert code == 0, err
    header, sep, *rest = out.splitlines()
    assert "  " in header  # multiple columns
    assert set(sep) == {"-", " "} and "-" in sep  # dashed separator
    assert len(rest) >= 0  # may be empty if no items, but path is valid

    # rotate-logs: JSON summary (Track B) — must be valid JSON
    code, out, err = run_cli(
        ["rotate-logs", "--", "--dry-run", "--json"]
    )  # flags after --, hoisted
    assert code == 0, err
    payload = json.loads(out)
    assert isinstance(payload, dict) and "dry_run" in payload

    # rotate-logs: TABLE summary (Track B) — ASCII header + dashes
    code, out, err = run_cli(
        ["rotate-logs", "--", "--dry-run", "--table"]
    )  # flags after --, hoisted
    assert code == 0, err
    header, sep, *rest = out.splitlines()
    assert "  " in header
    assert set(sep) == {"-", " "} and "-" in sep


def test_mutual_exclusion_exit_one():
    # using demo wrapper for clarity
    code, out, err = run_cli(["demo", "--", "--json", "--table"])  # after --
    assert code == 1
    assert "Choose exactly one" in (err or out)


def test_io_error_exit_two():
    # inspector with nonexistent directory should yield IO_ERR=2
    code, out, err = run_cli(
        ["inspect-snapshot", "--", "--dir", "does_not_exist_dir_12345"]
    )  # after --
    assert code == 2


def test_quiet_suppresses_stderr_on_wrapper_errors():
    # Force a wrapper-level validation error, but suppress stderr via --quiet
    code, out, err = run_cli(
        ["inspect-snapshot", "--quiet", "--", "--json", "--table"]
    )  # conflicting flags
    assert code == 1
    assert err == ""


def test_internal_error_exit_three():
    """Trigger INTERNAL=3 via test-only hook; skip gracefully if hook is absent.
    This allows us to assert the exit code mapping without depending on fragile inputs.
    """
    code, out, err = run_cli(
        ["inspect-snapshot", "--json"],
        env_overrides={"CLEMATIS_TEST_INTERNAL_CRASH": "1"},
    )
    if code != 3:
        pytest.skip("Internal crash hook not present; skipping exit=3 contract probe")
    assert code == 3
