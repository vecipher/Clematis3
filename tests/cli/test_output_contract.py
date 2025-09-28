import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run_cli(args, cwd=ROOT):
    env = dict(os.environ)
    # Keep environment clean; ensure UTF-8 output
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


def test_validate_json_identity_and_table_reject():
    # --json before and after -- must be identical
    code1, out1, err1 = run_cli(["validate", "--json"])
    code2, out2, err2 = run_cli(["validate", "--", "--json"])
    assert code1 == 0 and code2 == 0
    assert out1 == out2
    # Output may be human-readable text today; we only require identity across flag positions.

    # --table must be rejected with USER_ERR=1 and a concise message
    code3, out3, err3 = run_cli(["validate", "--", "--table"])
    assert code3 == 1
    assert "supports --json only" in (err3 or out3)


def test_rotate_logs_flags_and_hoist():
    # Track B: JSON summary works (flags after --, hoisted)
    code, out, err = run_cli(["rotate-logs", "--", "--dry-run", "--json"])
    assert code == 0, err
    payload = json.loads(out)
    assert isinstance(payload, dict) and "dry_run" in payload

    # Track B: TABLE summary works
    code, out, err = run_cli(["rotate-logs", "--", "--dry-run", "--table"])
    assert code == 0, err
    header, sep, *rest = out.splitlines()
    assert "  " in header and "-" in sep

    # Mutual exclusion guard
    code, out, err = run_cli(["rotate-logs", "--", "--json", "--table"])  # after --, both
    assert code == 1
    assert "Choose exactly one" in (err or out)

    # Baseline smoke (no structured flags) should succeed
    code, out, err = run_cli(["rotate-logs", "--", "--dry-run"])  # may print text
    assert code == 0
    assert out or err  # some message expected


def test_inspect_snapshot_json_and_table_reject():
    # --json before --
    code, out, err = run_cli(["inspect-snapshot", "--json"])  # wrapper forwards --format json
    assert code == 0
    assert out.strip().startswith("{")
    json.loads(out)  # must be valid JSON

    # --json after -- (hoisted)
    code, out, err = run_cli(["inspect-snapshot", "--", "--json"])  # same behavior
    assert code == 0
    json.loads(out)

    # Track B: --table prints an ASCII table
    code, out, err = run_cli(["inspect-snapshot", "--", "--table"])  # after --
    assert code == 0, err
    header, sep, *_ = out.splitlines()
    assert "  " in header and "-" in sep

    # Mutually exclusive flags rejected
    code, out, err = run_cli(["inspect-snapshot", "--", "--json", "--table"])  # both
    assert code == 1
    assert "Choose exactly one" in (err or out)