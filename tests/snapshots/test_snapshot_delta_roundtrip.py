

import copy
from clematis.engine.util.snapshot_delta import compute_delta, apply_delta


def test_delta_roundtrip_identity_simple():
    base = {"a": 1, "b": {"c": 2, "d": 3}}
    curr = {"a": 1, "b": {"c": 20, "e": 5}}
    delta = compute_delta(base, curr)
    rebuilt = apply_delta(base, delta)
    assert rebuilt == curr


def test_delta_roundtrip_identity_nested_and_deletes():
    base = {
        "root": {
            "alpha": {"x": 1, "y": 2},
            "beta": {"z": 3},
            "keep": 42,
        }
    }
    curr = {
        "root": {
            "alpha": {"x": 100},  # y deleted, x modified
            "gamma": {"w": 9},     # new subtree
            "keep": 42,             # unchanged
        }
    }
    delta = compute_delta(base, curr)
    rebuilt = apply_delta(base, delta)
    assert rebuilt == curr


def test_no_changes_produces_empty_delta_and_is_idempotent():
    base = {"a": 1, "b": {"c": [1, 2, 3]}}
    curr = copy.deepcopy(base)
    delta = compute_delta(base, curr)
    assert delta["_adds"] == {}
    assert delta["_mods"] == {}
    assert delta["_dels"] == []
    rebuilt = apply_delta(base, delta)
    assert rebuilt == base


def test_lists_are_atomic_values():
    base = {"arr": [1, 2, 3]}
    curr = {"arr": [1, 2, 3, 4]}  # treated as a single value modification
    delta = compute_delta(base, curr)
    # Should be a single modification at path 'arr'
    assert "arr" in delta["_mods"] and delta["_adds"] == {} and delta["_dels"] == []
    rebuilt = apply_delta(base, delta)
    assert rebuilt == curr


def test_none_bases_and_currents():
    # base=None should behave like empty dict
    base = None
    curr = {"a": 1, "b": {"c": 2}}
    delta = compute_delta(base, curr)
    rebuilt = apply_delta(base, delta)
    assert rebuilt == curr