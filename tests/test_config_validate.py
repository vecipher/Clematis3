


import pytest

from configs.validate import validate_config, validate_config_verbose


def test_validate_happy_defaults():
    # Empty cfg should be merged with defaults and pass
    out = validate_config({})
    assert isinstance(out, dict)
    # Spot-check a few normalized fields
    assert out["t4"]["enabled"] is True
    assert out["t4"]["delta_norm_cap_l2"] > 0
    assert out["t4"]["novelty_cap_per_node"] > 0
    assert out["t4"]["cache"]["ttl_sec"] > 0  # normalized key


@pytest.mark.parametrize(
    "bad_cfg,needle",
    [
        ({"t4": {"weight_min": 1.0, "weight_max": 0.5}}, "t4.weight_min/weight_max"),
        ({"t4": {"delta_norm_cap_l2": 0}}, "t4.delta_norm_cap_l2"),
        ({"t4": {"cache_bust_mode": "sometimes"}}, "t4.cache_bust_mode"),
        ({"t2": {"k_retrieval": 0}}, "t2.k_retrieval"),
        ({"t3": {"max_rag_loops": 2}}, "t3.max_rag_loops"),
    ],
)
def test_validate_raises_on_bad_values(bad_cfg, needle):
    with pytest.raises(ValueError) as ei:
        validate_config(bad_cfg)
    msg = str(ei.value)
    assert needle in msg


def test_t4_cache_ttl_alias_and_priority():
    # ttl_s alone should normalize to ttl_sec
    out1 = validate_config({"t4": {"cache": {"ttl_s": 123}}})
    assert out1["t4"]["cache"]["ttl_sec"] == 123
    # When both provided, ttl_sec wins
    out2 = validate_config({"t4": {"cache": {"ttl_s": 50, "ttl_sec": 70}}})
    assert out2["t4"]["cache"]["ttl_sec"] == 70


def test_t4_cache_namespaces_must_be_list_of_str():
    with pytest.raises(ValueError) as ei:
        validate_config({"t4": {"cache": {"namespaces": "oops"}}})
    assert "t4.cache.namespaces" in str(ei.value)


def test_coercions_numeric_and_bool_and_ranges():
    cfg = {
        "t4": {
            "enabled": "yes",
            "churn_cap_edges": "10",
            "snapshot_every_n_turns": "2",
        },
        "t1": {"cache": {"ttl_s": "300"}},
        "t2": {"backend": "inmemory", "k_retrieval": "5", "sim_threshold": "0.5"},
        "t3": {"max_rag_loops": "1", "max_ops_per_turn": "8", "backend": "rulebased"},
    }
    out = validate_config(cfg)
    assert out["t4"]["enabled"] is True
    assert out["t4"]["churn_cap_edges"] == 10
    assert out["t4"]["snapshot_every_n_turns"] == 2
    assert out["t1"]["cache"]["ttl_s"] == 300
    assert out["t2"]["k_retrieval"] == 5
    assert out["t2"]["sim_threshold"] == 0.5
    assert out["t3"]["max_rag_loops"] == 1
    assert out["t3"]["max_ops_per_turn"] == 8


def test_t2_backend_enum_and_threshold():
    # good
    out = validate_config({"t2": {"backend": "inmemory", "sim_threshold": 0.0}})
    assert out["t2"]["backend"] == "inmemory"
    # bad backend
    with pytest.raises(ValueError) as ei:
        validate_config({"t2": {"backend": "duckdb"}})
    assert "t2.backend" in str(ei.value)


def test_t4_cooldowns_nonnegative_and_string_keys():
    # ok map
    out = validate_config({"t4": {"cooldowns": {"EditGraph": 2, "CreateGraph": 0}}})
    assert out["t4"]["cooldowns"]["EditGraph"] == 2
    # negative value
    with pytest.raises(ValueError) as ei:
        validate_config({"t4": {"cooldowns": {"EditGraph": -1}}})
    assert "t4.cooldowns[EditGraph]" in str(ei.value)


# Additional tests
def test_unknown_key_suggestion():
    # Typo should raise with a suggestion in the error message
    bad = {"t2": {"backnd": "inmemory"}}
    with pytest.raises(ValueError) as ei:
        validate_config(bad)
    msg = str(ei.value)
    assert "t2.backnd" in msg
    assert "did you mean 'backend'" in msg

def test_t1_t2_cache_ttl_alias_and_priority():
    # t1: ttl_sec alias should normalize into ttl_s
    out1 = validate_config({"t1": {"cache": {"ttl_sec": "600"}}})
    assert out1["t1"]["cache"]["ttl_s"] == 600
    # t2: ttl_sec alias should normalize into ttl_s
    out2 = validate_config({"t2": {"cache": {"ttl_sec": 420}}})
    assert out2["t2"]["cache"]["ttl_s"] == 420
    # precedence: explicit ttl_s wins over default; explicit ttl_sec should take effect if provided alone
    out3 = validate_config({"t1": {"cache": {"ttl_s": 111}}})
    assert out3["t1"]["cache"]["ttl_s"] == 111

def test_duplicate_cache_namespace_warns_in_verbose():
    cfg = {
        "t2": {"cache": {"enabled": True}},
        "t4": {"cache": {"namespaces": ["t2:semantic"]}},
    }
    _, warnings = validate_config_verbose(cfg)
    assert any("duplicate-cache-namespace" in w for w in warnings)

def test_t4_cache_namespace_membership():
    # Invalid namespace should error with allowed set indicated
    cfg = {"t4": {"cache": {"namespaces": ["t2:semantics"]}}}
    with pytest.raises(ValueError) as ei:
        validate_config(cfg)
    msg = str(ei.value)
    assert "t4.cache.namespaces[t2:semantics]" in msg
    assert "allowed: ['t2:semantic']" in msg

@pytest.mark.parametrize("bad", [1.2, -1.2])
def test_t2_sim_threshold_bounds(bad):
    cfg = {"t2": {"sim_threshold": bad}}
    with pytest.raises(ValueError) as ei:
        validate_config(cfg)
    assert "t2.sim_threshold" in str(ei.value)

def test_snapshot_every_n_turns_must_be_positive():
    cfg = {"t4": {"snapshot_every_n_turns": 0, "cache": {"namespaces": ["t2:semantic"]}}}
    with pytest.raises(ValueError) as ei:
        validate_config(cfg)
    assert "t4.snapshot_every_n_turns" in str(ei.value)