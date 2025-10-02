"""
Bless the identity goldens from the disabled path.

Usage:
  python tests/identity/_bless_goldens.py
"""

import os
import shutil
import subprocess
from pathlib import Path

EXPECT = ("apply.jsonl", "t1.jsonl", "t2.jsonl", "t4.jsonl", "turn.jsonl")


def _pick_logs_dir(preferred: Path) -> Path:
    candidates = [
        preferred,
        Path(".") / ".data" / "logs",   # older default
        Path(".") / ".logs",            # current default printed by demo
        Path("clematis") / ".logs",     # package-local fallback
    ]
    for d in candidates:
        if all((d / n).exists() for n in EXPECT):
            return d
    names = ", ".join(str(c) for c in candidates)
    raise SystemExit(f"Expected logs not found in any of: {names}")


def main():
    here = Path(__file__).parent
    goldens = here / "goldens" / "disabled_path"
    tmp = Path(os.getenv("TMPDIR", "/tmp")) / "clem_identity_golden"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "CI": "true",
            "CLEMATIS_NETWORK_BAN": "1",
            "CLEMATIS_GIT_SHA": "testsha",
            "CLEMATIS_SCHEDULER__ENABLED": "false",
            "CLEMATIS_PERF__PARALLEL__ENABLED": "false",
            "CLEMATIS_T2__QUALITY__ENABLED": "false",
            "CLEMATIS_T2__QUALITY__SHADOW": "false",
            "CLEMATIS_T2__HYBRID__ENABLED": "false",
            "CLEMATIS_GRAPH__ENABLED": "false",
            # try to route logs into tmp
            "CLEMATIS_LOG_DIR": str(tmp),
            "CLEMATIS_LOGS_DIR": str(tmp),
        }
    )

    # Ensure no stale logs: remove entire directories used by the demo
    import shutil as _sh
    for d in (tmp, Path(".") / ".data" / "logs", Path(".") / ".logs", Path("clematis") / ".logs"):
        try:
            if d.exists():
                _sh.rmtree(d)
        except Exception:
            pass
    tmp.mkdir(parents=True, exist_ok=True)

    # Also reset snapshots so version_etag starts from the same baseline
    for snapdir in (Path(".") / ".data" / "snapshots",):
        try:
            import shutil as _sh
            _sh.rmtree(snapdir)
        except FileNotFoundError:
            pass
        except Exception:
            pass

    # Produce logs with short, deterministic run
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

    src_dir = _pick_logs_dir(tmp)
    goldens.mkdir(parents=True, exist_ok=True)
    for name in EXPECT:
        src = src_dir / name
        dst = goldens / name
        shutil.copy2(src, dst)
        print(f"Blessed {dst}")

    print("✅ Goldens updated.")


if __name__ == "__main__":
    main()
