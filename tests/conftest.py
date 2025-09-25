# Ensure the repo root (which contains the 'clematis' package) is on sys.path for tests.
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# CI/offline guard: ban outbound sockets when CLEMATIS_NETWORK_BAN=1
import socket, pytest

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