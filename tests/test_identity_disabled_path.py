import os
import importlib
import json
from pathlib import Path

import pytest

from configs.validate import validate_config
from tests.helpers.identity import (
    _strip_perf_and_quality,
    _strip_perf_and_quality_and_graph,
)


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


@pytest.mark.identity
def test_disabled_path_identity_config_roundtrip_graph_subtree():
    """
    Supplying a full 'graph.*' subtree with enabled:false must NOT alter the
    validated configuration compared to a base config that never mentions 'graph'.
    Other disabled-path subtrees (perf.*, t2.quality.*) are stripped for equality.
    """
    base_cfg = {"t2": {"k_retrieval": 16}}
    disabled_cfg = {
        "t2": {
            "k_retrieval": 16,
            "quality": {
                "enabled": False,
                "fusion": {"enabled": True},
            },
        },
        "graph": {
            "enabled": False,
            "coactivation_threshold": 0.33,
            "observe_top_k": 7,
            "update": {
                "mode": "proportional",
                "alpha": 0.07,
                "clamp_min": -0.9,
                "clamp_max": 0.9,
            },
            "decay": {"half_life_turns": 123, "floor": 0.01},
            # Intentionally enabled but inert while graph.enabled==false
            "merge": {"enabled": True, "min_size": 2, "cap_per_turn": 3},
            "split": {"enabled": True, "weak_edge_thresh": 0.05},
            "promotion": {"enabled": True, "label_mode": "concat_k"},
        },
    }

    norm_base = validate_config(base_cfg)
    norm_disabled = validate_config(disabled_cfg)

    assert _strip_perf_and_quality_and_graph(norm_disabled) == _strip_perf_and_quality_and_graph(norm_base)


def _maybe_get_turn_entrypoint():
    """
    Locate the canonical smoke entrypoint. We only accept `run_smoke_turn`;
    if it's absent on this branch, we skip the runtime identity test.
    """
    try:
        core = importlib.import_module("clematis.engine.orchestrator.core")
    except Exception:
        return None
    fn = getattr(core, "run_smoke_turn", None)
    return fn if callable(fn) else None


def _read_all_logs(log_dir: Path):
    """
    Deterministically collect files under log_dir → {relative_path: bytes} for byte-equality checks.
    For *.jsonl files, normalize fields that are allowed to differ across configs but do not affect
    disabled-path behavior (e.g., config-derived etags).
    """
    def _normalize(path: Path, raw: bytes) -> bytes:
        if path.suffix == ".jsonl":
            def _canon(obj):
                if isinstance(obj, dict):
                    out = {}
                    for k, v in obj.items():
                        # normalize config/hash-like fields
                        if k in ("version_etag", "cfg_hash"):
                            out[k] = "<NORM>"
                        # normalize timing jitter (ms fields)
                        elif k == "ms" or k.endswith("_ms") or k.startswith("ms_"):
                            out[k] = 0.0
                        else:
                            out[k] = _canon(v)
                    return out
                if isinstance(obj, list):
                    return [_canon(x) for x in obj]
                return obj

            out_lines = []
            for ln in raw.splitlines():
                if not ln:
                    out_lines.append(b"")
                    continue
                try:
                    obj = json.loads(ln)
                    obj = _canon(obj)
                    out_lines.append((json.dumps(obj, sort_keys=True) + "\n").encode("utf-8"))
                except Exception:
                    # Pass through non-JSON lines as-is (preserve newline)
                    if not ln.endswith(b"\n"):
                        out_lines.append(ln + b"\n")
                    else:
                        out_lines.append(ln)
            return b"".join(out_lines)
        return raw

    results = {}
    for p in sorted(log_dir.rglob("*")):
        if p.is_file():
            rel = p.relative_to(log_dir).as_posix()
            raw = p.read_bytes()
            results[rel] = _normalize(p, raw)
    return results


@pytest.mark.identity
def test_disabled_path_runtime_no_gel_and_log_identity(tmp_path, monkeypatch):
    """
    With graph.enabled=false, a single small orchestrator turn should:
      1) NOT produce logs/gel.jsonl
      2) Produce an identical set of logs with identical bytes to a baseline config
         that never mentions 'graph'.
    If no test entrypoint is exposed by the orchestrator on this branch, we skip.
    """
    run_entry = _maybe_get_turn_entrypoint()
    if run_entry is None:
        pytest.skip("No orchestrator test entrypoint available; skipping runtime identity smoke")

    # Normalize environment for deterministic logs
    monkeypatch.setenv("CI", "true")
    monkeypatch.setenv("CLEMATIS_NETWORK_BAN", "1")

    # Ensure logs go into our temp tree
    base_logs = tmp_path / "base_logs"
    off_logs = tmp_path / "off_logs"
    base_logs.mkdir(parents=True, exist_ok=True)
    off_logs.mkdir(parents=True, exist_ok=True)

    # Minimal baseline config (no 'graph' key)
    base_cfg = {"t2": {"k_retrieval": 8}}
    # Config with inert graph subtree
    graph_off_cfg = {
        "t2": {"k_retrieval": 8},
        "graph": {
            "enabled": False,
            "coactivation_threshold": 0.20,
            "observe_top_k": 4,
            "update": {"mode": "additive", "alpha": 0.02, "clamp_min": -1.0, "clamp_max": 1.0},
            "decay": {"half_life_turns": 200, "floor": 0.0},
            "merge": {"enabled": True},       # inert
            "split": {"enabled": True},       # inert
            "promotion": {"enabled": True},   # inert
        },
    }

    # Some orchestrator entrypoints allow log_dir; otherwise rely on env.
    def _run(fn, cfg, log_dir):
        """
        Call the canonical smoke entrypoint with keyword args only.
        This avoids accidentally hitting low-level APIs with positional fallbacks.
        """
        os.environ["CLEMATIS_LOG_DIR"] = str(log_dir)
        return fn(cfg=cfg, log_dir=str(log_dir))

    _run(run_entry, base_cfg, base_logs)
    _run(run_entry, graph_off_cfg, off_logs)

    # 1) No gel.jsonl in either case
    assert not (base_logs / "gel.jsonl").exists(), "baseline unexpectedly produced gel.jsonl"
    assert not (off_logs / "gel.jsonl").exists(), "graph.enabled=false should not produce gel.jsonl"

    # 2) Byte-identical logs overall
    base_map = _read_all_logs(base_logs)
    off_map = _read_all_logs(off_logs)
    assert base_map.keys() == off_map.keys(), f"Log file sets differ: {base_map.keys()} vs {off_map.keys()}"
    for rel in base_map.keys():
        assert base_map[rel] == off_map[rel], f"Log '{rel}' differs between baseline and graph.enabled=false"


def test_ci_wiring_exists_or_skip():
    """
    This test nudges the repo to include the disabled-path identity CI workflow, but won't
    fail local dev runs if it isn't present yet. The hard enforcement lives in CI itself.
    """
    wf = Path(".github/workflows/disabled_identity.yml")
    if not wf.exists():
        pytest.skip("CI workflow for disabled-path identity not present yet.")
    assert wf.read_text().strip() != ""
