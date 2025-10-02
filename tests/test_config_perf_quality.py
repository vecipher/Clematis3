import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

# Import the validator(s) under test
from configs.validate import validate_config, validate_config_verbose


def _without_quality_and_perf(cfg_norm: dict) -> dict:
    """Return a deep copy of cfg_norm with t2.quality and perf removed for inertness comparisons."""
    x = copy.deepcopy(cfg_norm)
    t2 = x.get("t2", {})
    if "quality" in t2:
        t2 = copy.deepcopy(t2)
        t2.pop("quality", None)
        x["t2"] = t2
    x.pop("perf", None)
    return x


def test_perf_defaults_off_identity():
    """
    With no 'perf' key supplied, the normalized config should not introduce it implicitly.
    This preserves disabled-path identity guarantees.
    """
    base = {
        # keep this tiny: provide only minimal keys used elsewhere
        "t2": {"k_retrieval": 16},
    }
    norm = validate_config(base)
    assert "perf" not in norm, (
        "validator must not inject a perf section when user did not provide one"
    )
    # quality block also absent unless explicitly provided
    assert "quality" not in norm.get("t2", {})


def test_quality_prep_inert_variations():
    """
    Changing subfields inside t2.quality while enabled:false must not affect the rest of the normalized config.
    We compare normalized copies with t2.quality and perf removed.
    """
    base_cfg = {
        "t2": {
            "k_retrieval": 16,
            "quality": {
                "enabled": False,
            },
        }
    }
    # Variant with many knobs tweaked (still enabled:false)
    var_cfg = {
        "t2": {
            "k_retrieval": 16,
            "quality": {
                "enabled": False,
                "normalizer": {"stopwords": "builtin", "stemmer": "none", "min_token_len": 3},
                "aliasing": {
                    "enabled": False,
                    "map_path": "examples/quality/aliases.yaml",
                    "max_expansions_per_token": 2,
                },
                "lexical": {"enabled": True, "bm25": {"k1": 1.2, "b": 0.75, "doclen_floor": 10}},
                "fusion": {"enabled": True, "alpha_semantic": 0.7, "score_norm": "zscore"},
                "mmr": {
                    "enabled": True,
                    "lambda_relevance": 0.75,
                    "diversity_by_owner": True,
                    "diversity_by_token": True,
                    "k_final": 64,
                },
                "cache": {"salt": ""},
            },
        }
    }

    norm_base = validate_config(base_cfg)
    norm_var = validate_config(var_cfg)

    # Inertness assertion excludes the quality subtree (which is allowed to normalize) and any perf subtree
    assert _without_quality_and_perf(norm_base) == _without_quality_and_perf(norm_var)


def test_perf_validation_bounds_and_types():
    """
    perf.* accepts only bounded values; invalid settings raise ValueError with precise paths.
    """
    bads = [
        ({"perf": {"t1": {"queue_cap": 0}}}, "perf.t1.queue_cap"),
        ({"perf": {"t1": {"dedupe_window": 0}}}, "perf.t1.dedupe_window"),
        ({"perf": {"t1": {"cache": {"max_entries": -1}}}}, "perf.t1.cache.max_entries"),
        ({"perf": {"t1": {"cache": {"max_bytes": -1}}}}, "perf.t1.cache.max_bytes"),
        ({"perf": {"t2": {"embed_dtype": "float64"}}}, "perf.t2.embed_dtype"),
        ({"perf": {"t2": {"embed_store_dtype": "bfloat16"}}}, "perf.t2.embed_store_dtype"),
        ({"perf": {"snapshots": {"compression": "zip"}}}, "perf.snapshots.compression"),
        ({"perf": {"snapshots": {"level": 0}}}, "perf.snapshots.level"),
        ({"perf": {"snapshots": {"every_n_turns": 0}}}, "perf.snapshots.every_n_turns"),
    ]
    for cfg, needle in bads:
        with pytest.raises(ValueError) as ei:
            validate_config(cfg)
        assert needle in str(ei.value)


def test_warnings_fp16_without_norms_and_kfinal_gt_k():
    """
    validate_config_verbose should emit warnings for:
      - fp16 storage without precompute_norms
      - mmr.k_final > t2.k_retrieval (sanity guard for M7 wiring)
    """
    cfg = {
        "t2": {"k_retrieval": 8, "quality": {"enabled": False, "mmr": {"k_final": 64}}},
        "perf": {"t2": {"embed_store_dtype": "fp16", "precompute_norms": False}},
    }
    _, warnings = validate_config_verbose(cfg)
    joined = "\n".join(warnings)
    assert "perf.t2" in joined and "precompute_norms" in joined
    assert "t2.quality.mmr.k_final" in joined


def test_quality_value_bounds_and_types_errors():
    """
    Certain invalid values in quality subkeys should raise validation errors even if enabled:false.
    (Bounds/type checks are independent of wiring.)
    """
    bads = [
        (
            {"t2": {"quality": {"normalizer": {"stemmer": "snowball"}}}},
            "t2.quality.normalizer.stemmer",
        ),
        (
            {"t2": {"quality": {"normalizer": {"min_token_len": 0}}}},
            "t2.quality.normalizer.min_token_len",
        ),
        ({"t2": {"quality": {"aliasing": {"map_path": ""}}}}, "t2.quality.aliasing.map_path"),
        (
            {"t2": {"quality": {"lexical": {"bm25": {"doclen_floor": -5}}}}},
            "t2.quality.lexical.bm25.doclen_floor",
        ),
        ({"t2": {"quality": {"fusion": {"score_norm": "l2"}}}}, "t2.quality.fusion.score_norm"),
        ({"t2": {"quality": {"mmr": {"k_final": 0}}}}, "t2.quality.mmr.k_final"),
    ]
    for cfg, needle in bads:
        with pytest.raises(ValueError) as ei:
            validate_config(cfg)
        assert needle in str(ei.value)


@pytest.mark.slow
def test_cli_validator_smoke(tmp_path: Path, monkeypatch):
    """
    Optional smoke test: ensure scripts/validate_config.py can run in this repo env and prints OK.
    Skips if the script is not importable as a module path; uses subprocess to avoid import side-effects.
    """
    script = Path("scripts/validate_config.py")
    if not script.exists():
        pytest.skip("validate_config.py script not found")
    # Minimal valid config via STDIN
    cfg = {
        "t2": {"k_retrieval": 16, "quality": {"enabled": False}, "cache": {"enabled": False}},
        "t4": {"cache": {"namespaces": []}},
        "perf": {"enabled": False},
    }
    # Use echo -> python script - to pass via stdin
    cmd = f'python3 {script} - --strict'
    proc = subprocess.run(
        cmd, input=json.dumps(cfg).encode("utf-8"), shell=True, capture_output=True
    )
    assert proc.returncode == 0, proc.stderr.decode() or proc.stdout.decode()
    assert "OK" in proc.stdout.decode()
