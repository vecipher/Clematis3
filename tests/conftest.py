# tests/conftest.py
from __future__ import annotations

import os
import socket
import ipaddress
import pytest

# Preserve the original connect so we can delegate when allowed
_ORIG_CONNECT = socket.socket.connect

# Explicit allowlist for loopback
_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


def _is_loopback_host(host: str) -> bool:
    """
    Return True if `host` is a loopback literal (IPv4/IPv6) or 'localhost' (case-insensitive).
    Avoid DNS to keep things strictly offline.
    """
    h = host.strip().lower()
    if h in _LOOPBACK_HOSTS:
        return True
    try:
        ip = ipaddress.ip_address(h)
        return ip.is_loopback
    except ValueError:
        # Not an IP literal; treat as non-loopback unless it's 'localhost'
        return False


@pytest.fixture(autouse=True)
def _ban_external_network(monkeypatch: pytest.MonkeyPatch):
    """
    Ban all outbound network connections in tests when CLEMATIS_NETWORK_BAN=1,
    but allow loopback so local event loops/Playwright can bootstrap (esp. on Windows).
    """
    if os.environ.get("CLEMATIS_NETWORK_BAN", "0") != "1":
        # No ban requested; run tests normally.
        yield
        return

    def _connect_guard(self: socket.socket, address):
        # Allow Unix domain sockets outright (local IPC)
        if isinstance(address, str):
            return _ORIG_CONNECT(self, address)

        # Expect (host, port) tuples for INET sockets
        try:
            host, port = address
        except Exception:
            raise AssertionError(f"Network calls are banned in CI (unexpected address: {address!r})")

        if _is_loopback_host(str(host)):
            return _ORIG_CONNECT(self, address)

        raise AssertionError(f"Network calls are banned in CI (attempted connect to {address!r})")

    # Patch low-level connect
    monkeypatch.setattr(socket.socket, "connect", _connect_guard, raising=True)
    try:
        yield
    finally:
        # monkeypatch fixture auto-restores, this is just explicit
        pass
