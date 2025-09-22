import json
from pathlib import Path
import pytest

# Robust imports: handle either `_read_header_body` (string payload) or
# `_read_header_payload` (dict payload) depending on snapshot module version.
try:
    from clematis.engine.snapshot import write_snapshot_auto  # type: ignore
except Exception as e:  # pragma: no cover
    pytest.skip(f"snapshot writer not importable: {e}", allow_module_level=True)

_READS_STRING = False

try:
    from clematis.engine.snapshot import _read_header_body as _read_helper  # type: ignore
    _READS_STRING = True
except Exception:
    try:
        from clematis.engine.snapshot import _read_header_payload as _read_helper  # type: ignore
        _READS_STRING = False
    except Exception as e:  # pragma: no cover
        pytest.skip(f"snapshot read helper not importable: {e}", allow_module_level=True)

try:
    from clematis.engine.util.snapshot_delta import apply_delta  # type: ignore
except Exception as e:  # pragma: no cover
    pytest.skip(f"delta codec not importable: {e}", allow_module_level=True)


def _load_body_obj(p: Path):
    """Return the payload object regardless of which helper variant we have."""
    hdr, payload = _read_helper(p)  # type: ignore
    if _READS_STRING:
        # payload is a JSON string body (header already JSON), parse it
        return json.loads(payload or "{}")
    else:
        # payload is already a dict
        return payload or {}


def test_writer_delta_roundtrip(tmp_path: Path):
    snapdir = tmp_path / "snaps"
    # 1) write baseline full
    base = {"store": {"nodes": {"a": 1, "b": 2}}}
    base_path, wrote_delta = write_snapshot_auto(
        snapdir,
        etag_from=None,
        etag_to="aaaa",
        payload=base,
        compression="none",
        delta_mode=False,
    )
    assert Path(base_path).exists() and not wrote_delta

    # 2) write curr as delta from baseline
    curr = {"store": {"nodes": {"a": 1, "b": 3, "c": 4}}}
    delta_path, wrote_delta2 = write_snapshot_auto(
        snapdir,
        etag_from="aaaa",
        etag_to="bbbb",
        payload=curr,
        compression="none",
        delta_mode=True,
    )
    assert Path(delta_path).exists() and wrote_delta2

    # 3) Reconstruct manually: read baseline body + apply delta
    base_obj = _load_body_obj(Path(base_path))
    delta_obj = _load_body_obj(Path(delta_path))
    recon = apply_delta(base_obj, delta_obj)  # type: ignore
    assert recon == curr