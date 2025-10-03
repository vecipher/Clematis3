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
