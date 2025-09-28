import json
import logging
from pathlib import Path
import pytest

try:
    # Project API (reader)
    from clematis.engine.snapshot import read_snapshot
except Exception as e:  # pragma: no cover
    pytest.skip(f"read_snapshot not importable: {e}", allow_module_level=True)


def _call_read_snapshot(tmp_path: Path, etag_to: str):
    """Be robust to minor signature differences across branches."""
    # Try common signatures
    try:
        return read_snapshot(tmp_path, etag_to=etag_to, baseline_dir=tmp_path)
    except TypeError:
        pass
    try:
        return read_snapshot(root=tmp_path, etag_to=etag_to, baseline_dir=tmp_path)
    except TypeError:
        pass
    try:
        return read_snapshot(path=tmp_path, etag=etag_to, baseline_dir=tmp_path)
    except TypeError as e:
        pytest.skip(f"read_snapshot signature not supported: {e}")


def _is_ok_snapshot(obj) -> bool:
    if isinstance(obj, dict):
        if obj.get("ok") is True:
            return True
        for k in ("payload", "body", "data"):
            v = obj.get(k)
            if isinstance(v, dict) and v.get("ok") is True:
                return True
    return False


def test_fallback_when_baseline_missing(tmp_path, caplog):
    """
    Given a delta snapshot whose baseline is missing, the reader should:
      1) Log a deterministic warning (e.g., SNAPSHOT_BASELINE_MISSING or similar), and
      2) Fall back to loading the full snapshot for the same etag, without crashing.
    """
    etag_to = "cafebabe"
    delta_of = "deadbeef"

    # Write delta header + payload (two-line JSON file)
    (tmp_path / f"snapshot-{etag_to}.delta.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "mode": "delta",
                "etag_to": etag_to,
                "delta_of": delta_of,
                "codec": "none",
                "level": 0,
            }
        )
        + "\n"
        + json.dumps({"_adds": {}, "_mods": {}, "_dels": []})
    )

    # Also provide a full snapshot for etag_to to exercise the fallback-to-full path
    (tmp_path / f"snapshot-{etag_to}.full.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "mode": "full",
                "etag_to": etag_to,
                "codec": "none",
                "level": 0,
            }
        )
        + "\n"
        + json.dumps({"ok": True})
    )

    with caplog.at_level(logging.WARNING):
        snap = _call_read_snapshot(tmp_path, etag_to)

    # Must reconstruct to the full snapshot contents
    assert _is_ok_snapshot(snap), f"unexpected snapshot shape: {snap}"

    # Must warn about the missing baseline; accept a few canonical phrasings
    msg = caplog.text.lower()
    assert (
        "snapshot_baseline_missing" in msg
        or "baseline missing" in msg
        or "baseline mismatch" in msg
    ), f"no baseline warning captured; logs were:\n{caplog.text}"
