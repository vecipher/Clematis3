# Ensure the repo root (which contains the 'clematis' package) is on sys.path for tests.
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Ensure the interpreter's bin directory is on PATH so subprocess calls to
# `python` resolve even on environments where only `python3` is registered.
_exe_dir = os.path.dirname(sys.executable or "")
paths = os.environ.get("PATH", "")
segments = [p for p in paths.split(os.pathsep) if p]

if _exe_dir and _exe_dir not in segments:
    segments.insert(0, _exe_dir)

if ROOT not in segments:
    segments.insert(0, ROOT)

os.environ["PATH"] = os.pathsep.join(segments)

# CI/offline guard: ban outbound sockets when CLEMATIS_NETWORK_BAN=1
import socket
import pytest

# Redirect logs away from ./.logs for non-identity suites
# Identity workflows should export IDENTITY_TESTS=1 so this fixture no-ops
@pytest.fixture(autouse=True, scope="session")
def _isolate_logs_for_non_identity(tmp_path_factory):
    # Respect explicit overrides and identity runs
    if os.environ.get("IDENTITY_TESTS") == "1":
        yield
        return
    if os.environ.get("CLEMATIS_LOG_DIR"):
        yield
        return
    # Default: write logs to a session temp dir to avoid polluting ./.logs
    tmp_logs = str(tmp_path_factory.mktemp("logs"))
    os.environ["CLEMATIS_LOG_DIR"] = tmp_logs
    try:
        yield
    finally:
        # Leave temp logs in place for debugging; test harness may clean up
        pass


@pytest.fixture(autouse=True, scope="session")
def _ban_network():
    if os.environ.get("CLEMATIS_NETWORK_BAN") != "1":
        # no-op locally; CI sets the env to enforce offline determinism
        yield
        return
    real_connect = socket.socket.connect

    def _blocked(*args, **kwargs):
        raise AssertionError("Network calls are banned in CI")

    socket.socket.connect = _blocked
    try:
        yield
    finally:
        socket.socket.connect = real_connect


# Ensure the repository's ./.logs directory exists for CLI tests that use --dir ./.logs
@pytest.fixture(autouse=True, scope="session")
def _ensure_repo_logs_dir():
    """
    Ensure the repository's ./.logs directory exists so CLI tests that
    pass --dir ./.logs don't fail due to a missing directory.
    This fixture is intentionally no-op on teardown.
    """
    repo_logs = os.path.join(ROOT, ".logs")
    try:
        os.makedirs(repo_logs, exist_ok=True)
    except Exception:
        # best-effort; tests that require this path will fail loudly otherwise
        pass
    yield
