

import os
import threading
from pathlib import Path

import pytest

from clematis.io.atomic import (
    atomic_write_text,
    atomic_write_json,
    atomic_replace,
)


def test_atomic_write_text_lf_only(tmp_path: Path):
    p = tmp_path / "t.txt"
    # Intentionally include CRLF; writer must normalize to LF
    atomic_write_text(p, "a\r\nb\r\n")
    data = p.read_bytes()
    assert b"\r\n" not in data
    assert data == b"a\nb\n"


def test_atomic_write_json_is_canonical(tmp_path: Path):
    p = tmp_path / "obj.json"
    atomic_write_json(p, {"b": 1, "a": 2})
    s = p.read_text("utf-8")
    # Canonical: sorted keys, compact separators, no trailing newline required
    assert s == '{"a":2,"b":1}'


def test_atomic_replace_under_reader_contention(tmp_path: Path):
    # Simulate readers holding the file open while writers replace it repeatedly
    target = tmp_path / "state.txt"
    target.write_text("OLD\n", encoding="utf-8", newline="\n")

    stop = False
    seen = []

    def reader():
        # Continuously read; every content must be exactly old or new, never partial
        while not stop:
            try:
                data = target.read_text(encoding="utf-8")
                seen.append(data)
            except FileNotFoundError:
                # Tiny windows during replace are acceptable; just continue
                pass

    t = threading.Thread(target=reader, daemon=True)
    t.start()

    # Writer: perform several atomic swaps
    for i in range(20):
        tmp = tmp_path / f"tmp-{i}.txt"
        tmp.write_text(f"NEW-{i}\n", encoding="utf-8", newline="\n")
        atomic_replace(tmp, target)

    stop = True
    t.join(timeout=2)

    # All observed reads must be one of the complete states
    valid = {"OLD\n"} | {f"NEW-{i}\n" for i in range(20)}
    assert all(s in valid for s in seen)
    # Ensure no temp files left behind
    assert not any(p.name.startswith("tmp-") for p in tmp_path.iterdir())


def test_atomic_write_cleanup_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Force replace failure once and ensure tmp is cleaned and final is written."""
    p = tmp_path / "x.txt"

    calls = {"n": 0}
    real_replace = os.replace

    def flaky_replace(src, dst):  # noqa: ANN001 - signature must match os.replace
        calls["n"] += 1
        if calls["n"] < 2:
            raise PermissionError("simulated share violation")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", flaky_replace)

    atomic_write_text(p, "hello\n")
    assert p.read_text("utf-8") == "hello\n"

    # No stray tmp.* files in dir
    leftovers = [x for x in tmp_path.iterdir() if x.name.startswith(p.name + ".")]
    assert not leftovers
