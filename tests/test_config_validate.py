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


# ------------------------------
# PR24: graph.merge/split/promotion validation
# ------------------------------


def test_graph_merge_defaults_and_bounds():
    cfg = {"graph": {"enabled": True}}
    out = validate_config(cfg)
    gm = out["graph"]["merge"]
    assert gm["enabled"] is False  # default OFF even if graph.enabled
    assert gm["min_size"] == 3
    assert 0.0 <= gm["min_avg_w"] <= 1.0
    assert gm["max_diameter"] >= 1
    assert gm["cap_per_turn"] >= 0


def test_graph_merge_invalid_values():
    # min_size < 2
    with pytest.raises(ValueError) as ei:
        validate_config({"graph": {"merge": {"min_size": 1}}})
    assert "graph.merge.min_size" in str(ei.value)
    # min_avg_w out of range
    with pytest.raises(ValueError) as ei:
        validate_config({"graph": {"merge": {"min_avg_w": 1.5}}})
    assert "graph.merge.min_avg_w" in str(ei.value)
    # max_diameter < 1
    with pytest.raises(ValueError) as ei:
        validate_config({"graph": {"merge": {"max_diameter": 0}}})
    assert "graph.merge.max_diameter" in str(ei.value)
    # cap_per_turn negative
    with pytest.raises(ValueError) as ei:
        validate_config({"graph": {"merge": {"cap_per_turn": -1}}})
    assert "graph.merge.cap_per_turn" in str(ei.value)


def test_graph_split_defaults_and_bounds():
    out = validate_config({"graph": {"split": {"enabled": True}}})
    gs = out["graph"]["split"]
    assert gs["enabled"] is True
    assert 0.0 <= gs["weak_edge_thresh"] <= 1.0
    assert gs["min_component_size"] >= 2
    assert gs["cap_per_turn"] >= 0


def test_graph_split_invalid_values():
    # weak_edge_thresh out of range
    with pytest.raises(ValueError) as ei:
        validate_config({"graph": {"split": {"weak_edge_thresh": -0.1}}})
    assert "graph.split.weak_edge_thresh" in str(ei.value)
    # min_component_size < 2
    with pytest.raises(ValueError) as ei:
        validate_config({"graph": {"split": {"min_component_size": 1}}})
    assert "graph.split.min_component_size" in str(ei.value)
    # cap_per_turn negative
    with pytest.raises(ValueError) as ei:
        validate_config({"graph": {"split": {"cap_per_turn": -2}}})
    assert "graph.split.cap_per_turn" in str(ei.value)


def test_graph_promotion_defaults_and_bounds():
    out = validate_config({"graph": {"promotion": {"enabled": True}}})
    gp = out["graph"]["promotion"]
    assert gp["enabled"] is True
    assert gp["label_mode"] in {"lexmin", "concat_k"}
    assert gp["topk_label_ids"] >= 1
    assert -1.0 <= gp["attach_weight"] <= 1.0
    assert gp["cap_per_turn"] >= 0


def test_graph_promotion_invalid_values():
    # bad label_mode
    with pytest.raises(ValueError) as ei:
        validate_config({"graph": {"promotion": {"label_mode": "weird"}}})
    assert "graph.promotion.label_mode" in str(ei.value)
    # topk_label_ids < 1
    with pytest.raises(ValueError) as ei:
        validate_config({"graph": {"promotion": {"topk_label_ids": 0}}})
    assert "graph.promotion.topk_label_ids" in str(ei.value)
    # attach_weight out of range
    with pytest.raises(ValueError) as ei:
        validate_config({"graph": {"promotion": {"attach_weight": 2}}})
    assert "graph.promotion.attach_weight" in str(ei.value)
    # cap_per_turn negative
    with pytest.raises(ValueError) as ei:
        validate_config({"graph": {"promotion": {"cap_per_turn": -1}}})
    assert "graph.promotion.cap_per_turn" in str(ei.value)


def test_graph_cross_field_consistency():
    # weak_edge_thresh should be <= merge.min_avg_w
    bad = {
        "graph": {
            "merge": {"min_avg_w": 0.20},
            "split": {"weak_edge_thresh": 0.50},
        }
    }
    with pytest.raises(ValueError) as ei:
        validate_config(bad)
    msg = str(ei.value)
    assert "graph.split.weak_edge_thresh" in msg
    assert "<= graph.merge.min_avg_w" in msg


def test_graph_unknown_keys_under_promotion():
    bad = {"graph": {"promotion": {"speed": 1}}}
    with pytest.raises(ValueError) as ei:
        validate_config(bad)
    msg = str(ei.value)
    assert "graph.promotion.speed" in msg
    assert "unknown key" in msg
