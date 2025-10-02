import os
import hashlib
import subprocess
from pathlib import Path

EXPECT = ("apply.jsonl", "t1.jsonl", "t2.jsonl", "t4.jsonl", "turn.jsonl")


def _sha256(fp: Path) -> str:
    h = hashlib.sha256()
    with fp.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_outputs_dir(out_dir: Path) -> Path:
    # Prefer env-directed logs directory; otherwise search known defaults
    candidates = [
        out_dir,
        Path(".") / ".data" / "logs",   # older default
        Path(".") / ".logs",            # current default printed by demo
        Path("clematis") / ".logs",     # package-local fallback
    ]
    for d in candidates:
        if all((d / name).exists() for name in EXPECT):
            return d
    checked = ", ".join(str(c) for c in candidates)
    raise AssertionError(f"Expected logs not found. Checked: {checked}")


def test_disabled_path_identity(tmp_path: Path):
    """All flags OFF ⇒ byte-identical outputs vs goldens."""
    goldens_dir = Path(__file__).parent / "goldens" / "disabled_path"
    assert goldens_dir.exists(), f"Missing goldens dir: {goldens_dir}"
    for name in EXPECT:
        assert (goldens_dir / name).exists(), f"Missing golden {name}"

    out_dir = tmp_path / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure no stale logs: remove entire directories used by the demo
    import shutil as _sh
    for d in (out_dir, Path(".") / ".data" / "logs", Path(".") / ".logs", Path("clematis") / ".logs"):
        try:
            if d.exists():
                _sh.rmtree(d)
        except Exception:
            pass
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reset snapshots so version_etag starts from the same baseline
    for snapdir in (Path(".") / ".data" / "snapshots",):
        try:
            import shutil as _sh
            _sh.rmtree(snapdir)
        except FileNotFoundError:
            pass
        except Exception:
            pass

    # Environment for strict disabled path + determinism
    env = os.environ.copy()
    env.update(
        {
            "CI": "true",
            "CLEMATIS_NETWORK_BAN": "1",
            "CLEMATIS_GIT_SHA": "testsha",
            # hard-off all feature gates
            "CLEMATIS_SCHEDULER__ENABLED": "false",
            "CLEMATIS_PERF__PARALLEL__ENABLED": "false",
            "CLEMATIS_T2__QUALITY__ENABLED": "false",
            "CLEMATIS_T2__QUALITY__SHADOW": "false",
            "CLEMATIS_T2__HYBRID__ENABLED": "false",
            "CLEMATIS_GRAPH__ENABLED": "false",
            # try to direct logs into our tmp dir (paths.logs_dir() may honor either)
            "CLEMATIS_LOG_DIR": str(out_dir),
            "CLEMATIS_LOGS_DIR": str(out_dir),
        }
    )

    # Run the deterministic demo that emits standard logs
    # Keep it short and fix the "now" for determinism
    subprocess.run(
        [
            "python",
            "-m",
            "clematis.scripts.demo",
            "--steps",
            "2",
            "--text",
            "identity",
            "--fixed-now-ms",
            "0",
        ],
        check=True,
        env=env,
    )

    actual_dir = _ensure_outputs_dir(out_dir)

    # Compare exact bytes for each expected log
    for name in EXPECT:
        got = actual_dir / name
        ref = goldens_dir / name
        assert got.exists(), f"Missing output log: {name} in {actual_dir}"
        assert _sha256(got) == _sha256(ref), f"Drift detected in {name}"
