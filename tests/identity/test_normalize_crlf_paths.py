from __future__ import annotations

import json

from tests.helpers.identity import normalize_log_bytes_for_identity


def test_crlf_and_windows_paths_are_normalized():
    # Craft a Windows-style JSONL blob with CRLF endings and backslash paths
    recs = [
        {"event": "write", "file": "C:\\Users\\tom\\repo\\clematis\\.data\\logs\\t1.jsonl", "ms": 12},
        {"event": "snapshot", "path": "\\\\server\\share\\snaps\\latest.zst"},
        {"note": "not a path", "value": "foo\\bar is not a path if we say so"},
    ]
    blob = ("\r\n".join(json.dumps(r) for r in recs) + "\r\n").encode("utf-8")

    norm = normalize_log_bytes_for_identity({"t1.jsonl": blob})
    out = norm["t1.jsonl"].decode("utf-8")

    # Expect LF newlines
    assert "\r\n" not in out
    assert out.endswith("\n")

    # Expect forward slashes for path-like strings
    lines = [ln for ln in out.split("\n") if ln.strip()]
    objs = [json.loads(ln) for ln in lines]
    assert objs[0]["file"].startswith("C:/Users/")
    assert objs[1]["path"].startswith("//")  # UNC preserved
    # Non-path string remains unchanged (no forced slash rewrite)
    assert objs[2]["value"] == "foo\\bar is not a path if we say so"
