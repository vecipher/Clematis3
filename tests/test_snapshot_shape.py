import json
from types import SimpleNamespace
from pathlib import Path

from clematis.engine.snapshot import write_snapshot


def test_snapshot_shape_without_store(tmp_path):
    snap_dir = tmp_path / "snaps"
    ctx = SimpleNamespace(
        turn_id=7,
        agent_id="ShapeBot",
        # Either ctx.cfg or ctx.config works; snapshot._get_cfg normalizes this
        config=SimpleNamespace(
            t4={
                "snapshot_dir": str(snap_dir),
                "snapshot_every_n_turns": 1,
            }
        ),
    )
    state = {}  # no store

    path = write_snapshot(ctx, state, version_etag="42", applied=0, deltas=[])
    assert path is not None
    p = Path(path)
    assert p.exists(), f"snapshot file not found at {p}"

    data = json.loads(p.read_text())
    # Stable top-level schema
    for key in ["turn", "agent", "version_etag", "applied", "deltas", "store"]:
        assert key in data

    assert isinstance(data["deltas"], list)
    # When no exportable store is present, 'store' must be an empty object
    assert isinstance(data["store"], dict) and data["store"] == {}
