

import json
from pathlib import Path

from clematis.engine.snapshot import get_latest_snapshot_info, SCHEMA_VERSION


def _write_json(p: Path, obj: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj), encoding="utf-8")


def test_no_snapshot(tmp_path):
    d = tmp_path / "snaps"
    d.mkdir()
    assert get_latest_snapshot_info(str(d)) is None


def test_get_latest_snapshot_info_reads_fields(tmp_path):
    d = tmp_path / "snaps"
    d.mkdir()

    payload = {
        "schema_version": SCHEMA_VERSION,
        "version_etag": "G123-456",
        "graph": {
            "nodes_count": 3,
            "edges_count": 5,
            "meta": {"last_update": "2025-09-18T21:12:07Z"},
        },
        "t4_caps": {
            "delta_norm_cap_l2": 1.5,
            "novelty_cap_per_node": 0.3,
            "churn_cap_edges": 64,
            "weight_min": -1.0,
            "weight_max": 1.0,
        },
    }
    p = d / "snap_000123.json"
    _write_json(p, payload)

    info = get_latest_snapshot_info(str(d))
    assert info is not None
    assert info["schema_version"] == SCHEMA_VERSION
    assert info["version_etag"] == "G123-456"
    assert info["nodes"] == 3
    assert info["edges"] == 5
    assert info["last_update"] == "2025-09-18T21:12:07Z"
    caps = info["caps"]
    assert caps["delta_norm_cap_l2"] == 1.5
    assert caps["novelty_cap_per_node"] == 0.3
    assert caps["churn_cap_edges"] == 64
    assert caps["weight_min"] == -1.0
    assert caps["weight_max"] == 1.0


def test_pick_latest_by_numeric_suffix(tmp_path):
    d = tmp_path / "snaps"
    d.mkdir()

    for n in [1, 3, 2]:
        _write_json(d / f"snap_{n:06d}.json", {"schema_version": SCHEMA_VERSION, "graph": {}})

    info = get_latest_snapshot_info(str(d))
    assert info is not None
    assert info["path"].endswith("snap_000003.json")