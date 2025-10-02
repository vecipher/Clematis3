import os
import sys
import hashlib
import subprocess
from pathlib import Path
import textwrap
import shutil

import pytest


def _repo_root() -> Path:
    # tests/integration/<this file> → repo root is two levels up from tests/
    return Path(__file__).resolve().parents[2]


def _env_base():
    env = os.environ.copy()
    # Determinism guards to mirror identity workflow
    env.setdefault("CLEMATIS_NETWORK_BAN", "1")
    env.setdefault("PYTHONHASHSEED", "0")
    env.setdefault("SOURCE_DATE_EPOCH", "0")
    env.setdefault("TZ", "UTC")
    return env


def _collect_artifacts_bytes(logs_dir: Path):
    """
    Collect bytes for the identity-relevant artifacts we care about.
    If some files are absent (depending on the tiny demo), we only compare the intersection.
    """
    candidates = [
        "t1.jsonl",
        "t2.jsonl",
        "t4.jsonl",
        "apply.jsonl",
        "turn.jsonl",
        "scheduler.jsonl",
    ]
    present = {}
    for name in candidates:
        p = logs_dir / name
        if p.exists():
            present[name] = p.read_bytes()
    # Sanity: we expect at least turn/scheduler in any demo
    assert present, f"No expected artifacts were produced in {logs_dir}"
    return present


def _cleanup_artifacts(logs_dir: Path, names):
    """Remove a set of artifacts from a logs directory to avoid cross-run append."""
    for name in names:
        p = logs_dir / name
        try:
            if p.exists():
                p.unlink()
        except Exception:
            # Best-effort: ignore if concurrently modified
            pass


def _run_demo(tmpdir: Path, cfg_text: str):
    """
    Run the demo with a provided config YAML in an isolated temp directory.
    Returns a dict of artifact name -> bytes collected from the run.
    Tries per-run logs under tmpdir/logs first; falls back to repo-level .logs.
    If we fall back to repo-level .logs, we clean up those files after reading
    to avoid cross-run append in subsequent calls.
    """
    cfg_path = tmpdir / "config.yaml"
    tmpdir.mkdir(parents=True, exist_ok=True)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(textwrap.dedent(cfg_text), encoding="utf-8")

    # Ensure no snapshot/version state carries across runs in the same base dir
    state_dir = tmpdir / ".data"
    if state_dir.exists():
        shutil.rmtree(state_dir, ignore_errors=True)

    # Prefer per-run logs dir, but don't rely on it being honored by the subprocess.
    per_run_logs = tmpdir / "logs"
    env = _env_base()
    env["CLEMATIS_LOG_DIR"] = str(per_run_logs)

    # Run the packaged demo via the script path (PR73 placed this under repo/scripts/run_demo.py)
    script_path = _repo_root() / "scripts" / "run_demo.py"
    assert script_path.exists(), f"Expected run_demo script at {script_path}"
    cmd = [sys.executable, str(script_path), "--config", str(cfg_path)]
    result = subprocess.run(
        cmd,
        cwd=str(tmpdir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"run_demo failed (rc={result.returncode})\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    # Collect artifacts: prefer per-run logs dir; otherwise repo-level .logs
    repo_logs = _repo_root() / ".logs"
    for base in (per_run_logs, repo_logs):
        try:
            present = _collect_artifacts_bytes(base)
            # If we used the repo-level logs, remove these files to avoid cross-run append.
            if base == repo_logs:
                _cleanup_artifacts(base, present.keys())
            return present
        except AssertionError:
            continue
    raise AssertionError(
        f"No expected artifacts were produced in either {per_run_logs} or {repo_logs}"
    )


@pytest.mark.fast
def test_identity_off_path_is_straight(tmp_path: Path):
    """
    With no perf.parallel block (implicit OFF) vs explicit OFF block,
    artifacts must be byte-identical.
    """
    # Use separate base dirs for each run so the config path is different and state is isolated
    base_A = tmp_path / "A"
    base_B = tmp_path / "B"

    # Case A: implicit OFF (no perf block at all)
    cfg_implicit = """
    # minimal config; rely on defaults (parallel OFF)
    # no `perf:` block present on purpose
    """
    A = _run_demo(base_A, cfg_implicit)

    # Case B: explicit OFF (all knobs OFF; ≤1 workers)
    cfg_explicit_off = """
    perf:
      parallel:
        enabled: false
        max_workers: 1
        t1: false
        t2: false
        agents: false
      metrics:
        enabled: false
        report_memory: false
    """
    B = _run_demo(base_B, cfg_explicit_off)

    # Compare on the intersection of produced files (demo may omit some)
    keys = sorted(set(A.keys()) & set(B.keys()))
    assert keys, "No overlapping artifacts to compare between runs"

    diffs = []
    for k in keys:
        if A[k] != B[k]:
            diffs.append(k)

    if diffs:
        # Produce a short hex digest to aid debugging while keeping logs small
        def digest(b: bytes) -> str:
            return hashlib.sha256(b).hexdigest()[:16]
        msg = ["Artifacts differ under OFF-path parity (implicit vs explicit):"]
        for k in diffs:
            msg.append(f"  {k}: A={digest(A[k])} B={digest(B[k])}")
        pytest.fail("\n".join(msg))
