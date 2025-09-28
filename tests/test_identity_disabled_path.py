import copy
from pathlib import Path

import pytest

from configs.validate import validate_config


def _strip_perf_and_quality(cfg_norm: dict) -> dict:
    """Return a deep copy of cfg_norm with 'perf' and 't2.quality' removed, for identity comparisons."""
    x = copy.deepcopy(cfg_norm)
    x.pop("perf", None)
    if "t2" in x:
        t2 = copy.deepcopy(x["t2"])
        t2.pop("quality", None)
        x["t2"] = t2
    return x


def test_disabled_perf_and_quality_identity_to_base():
    """
    Identity guarantee (unit level):
    When perf is disabled and t2.quality.enabled=false, the normalized config—after removing
    those (disabled) subtrees—must be identical to a base config that never specified them.
    This protects the disabled path from accidental behavior/log differences.
    """
    base_cfg = {
        "t2": {"k_retrieval": 16},
        # No perf and no t2.quality present here
    }
    disabled_cfg = {
        "t2": {
            "k_retrieval": 16,
            "quality": {
                "enabled": False,
                # arbitrary subfields should be inert in M6
                "fusion": {"enabled": True, "alpha_semantic": 0.7, "score_norm": "zscore"},
                "mmr": {
                    "enabled": True,
                    "lambda_relevance": 0.75,
                    "diversity_by_owner": True,
                    "diversity_by_token": True,
                    "k_final": 64,
                },
            },
        },
        "perf": {
            "enabled": False,
            "t1": {
                "queue_cap": 10000,
                "dedupe_window": 8192,
                "cache": {"max_entries": 512, "max_bytes": 64_000_000},
            },
            "t2": {
                "embed_dtype": "fp32",
                "embed_store_dtype": "fp32",
                "precompute_norms": True,
                "cache": {"max_entries": 512, "max_bytes": 128_000_000},
            },
            "snapshots": {
                "compression": "zstd",
                "level": 3,
                "delta_mode": False,
                "every_n_turns": 1,
            },
            "metrics": {"report_memory": True},
        },
    }

    norm_base = validate_config(base_cfg)
    norm_disabled = validate_config(disabled_cfg)

    assert _strip_perf_and_quality(norm_disabled) == _strip_perf_and_quality(norm_base)


def test_ci_wiring_exists_or_skip():
    """
    This test nudges the repo to include the disabled-path identity CI workflow, but won't
    fail local dev runs if it isn't present yet. The hard enforcement lives in CI itself.
    """
    wf = Path(".github/workflows/disabled_identity.yml")
    if not wf.exists():
        pytest.skip("CI workflow for disabled-path identity not present yet.")
    assert wf.read_text().strip() != ""
