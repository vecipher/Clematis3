

import sys
import types

import pytest

# We only import the stage module under test; do NOT import Lance code paths here.
# This test verifies that importing the T2 stage does not implicitly import `lancedb`.

def test_import_does_not_pull_in_lancedb(monkeypatch):
    before = "lancedb" in sys.modules
    # Import the module under test
    import importlib
    t2mod = importlib.import_module("clematis.engine.stages.t2")
    after = "lancedb" in sys.modules
    # If lancedb was not present before, it must remain absent after
    if not before:
        assert not after, "Importing T2 stage should not import lancedb"


def _cfg(enabled=False, t2=False, max_workers=1):
    # Minimal nested config dict satisfying _cfg_get lookups
    return {
        "perf": {
            "parallel": {
                "enabled": bool(enabled),
                "t2": bool(t2),
                "max_workers": int(max_workers),
            }
        }
    }


class DummyIndexNoAttr:
    """Index without the shard iterator attribute."""
    pass


class DummyIndex:
    def __init__(self, shards):
        self._shards = list(shards)

    # Offer both signatures that _t2_parallel_enabled may probe
    def _iter_shards_for_t2(self, tier: str, suggested: int = None):  # type: ignore[override]
        return list(self._shards)


def test_gate_off_when_parallel_disabled():
    import importlib
    t2mod = importlib.import_module("clematis.engine.stages.t2")
    idx = DummyIndex([1, 2, 3])
    assert not t2mod._t2_parallel_enabled(_cfg(enabled=False, t2=True, max_workers=4), "lancedb", idx)


def test_gate_off_when_t2_flag_disabled():
    import importlib
    t2mod = importlib.import_module("clematis.engine.stages.t2")
    idx = DummyIndex([1, 2, 3])
    assert not t2mod._t2_parallel_enabled(_cfg(enabled=True, t2=False, max_workers=4), "lancedb", idx)


def test_gate_off_when_max_workers_le_one():
    import importlib
    t2mod = importlib.import_module("clematis.engine.stages.t2")
    idx = DummyIndex([1, 2, 3])
    assert not t2mod._t2_parallel_enabled(_cfg(enabled=True, t2=True, max_workers=1), "lancedb", idx)


def test_gate_off_unknown_backend():
    import importlib
    t2mod = importlib.import_module("clematis.engine.stages.t2")
    idx = DummyIndex([1, 2, 3])
    assert not t2mod._t2_parallel_enabled(_cfg(enabled=True, t2=True, max_workers=4), "sqlite", idx)


def test_gate_off_when_no_shard_iterator():
    import importlib
    t2mod = importlib.import_module("clematis.engine.stages.t2")
    idx = DummyIndexNoAttr()
    assert not t2mod._t2_parallel_enabled(_cfg(enabled=True, t2=True, max_workers=4), "lancedb", idx)


def test_gate_off_when_only_one_shard():
    import importlib
    t2mod = importlib.import_module("clematis.engine.stages.t2")
    # One logical shard -> gate must remain off
    idx = DummyIndex([object()])
    assert not t2mod._t2_parallel_enabled(_cfg(enabled=True, t2=True, max_workers=4), "lancedb", idx)


def test_gate_on_when_multiple_shards_and_flags_set():
    import importlib
    t2mod = importlib.import_module("clematis.engine.stages.t2")
    # Multiple logical shards -> gate may turn on (backend = lancedb)
    idx = DummyIndex([object(), object(), object()])
    assert t2mod._t2_parallel_enabled(_cfg(enabled=True, t2=True, max_workers=4), "lancedb", idx)


def test_gate_on_inmemory_backend_with_multiple_shards():
    import importlib
    t2mod = importlib.import_module("clematis.engine.stages.t2")
    idx = DummyIndex([1, 2])
    assert t2mod._t2_parallel_enabled(_cfg(enabled=True, t2=True, max_workers=2), "inmemory", idx)
