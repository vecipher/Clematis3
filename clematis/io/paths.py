import os
import tempfile
from pathlib import Path

def repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def logs_dir() -> Path:
    """
    Resolve the logs directory with the following precedence:
    1) CLEMATIS_LOG_DIR (primary env)
    2) CLEMATIS_LOGS_DIR (legacy env)
    3) ./.logs under the current working directory (repo default)
    4) {tempdir}/clematis/logs (final fallback)

    Ensures the directory exists and returns a Path.
    """
    # 1) Primary env
    v = os.environ.get("CLEMATIS_LOG_DIR")
    if v:
        p = Path(v)
        p.mkdir(parents=True, exist_ok=True)
        return p.resolve()

    # 2) Legacy env
    v = os.environ.get("CLEMATIS_LOGS_DIR")
    if v:
        p = Path(v)
        p.mkdir(parents=True, exist_ok=True)
        return p.resolve()

    # 3) Repo default: ./.logs relative to current working directory
    p = Path.cwd() / ".logs"
    try:
        p.mkdir(parents=True, exist_ok=True)
        return p.resolve()
    except Exception:
        # 4) Final fallback: OS temp
        t = temp_root() / "clematis" / "logs"
        t.mkdir(parents=True, exist_ok=True)
        return t.resolve()


def snapshots_dir() -> Path:
    """
    Resolve the snapshots directory with the following precedence:
    1) CLEMATIS_SNAPSHOT_DIR (primary env)
    2) CLEMATIS_SNAPSHOTS_DIR (legacy env)
    3) ./.data/snapshots under the repository root if it exists (dev/legacy)
    4) {tempdir}/clematis/snapshots (final fallback)

    Ensures the directory exists and returns a Path.
    """
    # 1) Primary env
    v = os.environ.get("CLEMATIS_SNAPSHOT_DIR")
    if v:
        p = Path(v)
        p.mkdir(parents=True, exist_ok=True)
        return p.resolve()

    # 2) Legacy env
    v = os.environ.get("CLEMATIS_SNAPSHOTS_DIR")
    if v:
        p = Path(v)
        p.mkdir(parents=True, exist_ok=True)
        return p.resolve()

    # 3) Repo/dev default: ./.data/snapshots under the repository root (if present)
    repo_snapshots = Path(repo_root()) / ".data" / "snapshots"
    try:
        if repo_snapshots.exists():
            repo_snapshots.mkdir(parents=True, exist_ok=True)
            return repo_snapshots.resolve()
    except Exception:
        # 4) Final fallback below
        pass

    # 4) Final fallback: OS temp
    t = temp_root() / "clematis" / "snapshots"
    t.mkdir(parents=True, exist_ok=True)
    return t.resolve()


def temp_root() -> Path:
    """
    Return the platform's temporary directory as a Path.
    Allows override via CLEMATIS_TMP for tests/CI.
    """
    env = os.environ.get("CLEMATIS_TMP")
    try:
        base = Path(env) if env else Path(tempfile.gettempdir())
    except Exception:
        # Extremely defensive: fall back to Python's tempdir
        base = Path(tempfile.gettempdir())
    return base
