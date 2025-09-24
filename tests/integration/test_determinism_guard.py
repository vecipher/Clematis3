

import copy
import json
import os
import sys
from typing import Any, List

import pytest

# Ensure project root is on sys.path when tests are invoked from subdirs
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
  sys.path.insert(0, ROOT)

# Import the T2 runner
try:
    from clematis.engine.stages.t2 import run_t2  # type: ignore
except Exception as e:  # pragma: no cover
    pytest.skip(f"Cannot import run_t2: {e}")

# Config loading: prefer configs.validate.load_config if available, else use {} (defaults-only)
def _load_cfg() -> dict:
    try:
        import configs.validate as cfgmod  # type: ignore
    except Exception:
        return {}
    # Try common loader names
    for fn in ("load_config", "load", "load_default_config"):
        if hasattr(cfgmod, fn):
            try:
                # Most loaders accept either no args or a path; try both
                try:
                    return getattr(cfgmod, fn)("configs/config.yaml")
                except TypeError:
                    return getattr(cfgmod, fn)()
            except Exception:
                pass
    # Fallback: parse YAML directly if available
    try:
        import yaml  # type: ignore
        with open(os.path.join(ROOT, "configs", "config.yaml"), "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        return {}

def _extract_ids(res: Any) -> List[str]:
    seq = getattr(res, "items", None)
    if seq is None:
        seq = getattr(res, "retrieved", None)
    if seq is None:
        seq = []
    ids = []
    for x in seq:
        try:
            ids.append(str(getattr(x, "id", "")))
        except Exception:
            ids.append("")
    return ids

def _metric_keys(res: Any) -> List[str]:
    m = getattr(res, "metrics", {}) or {}
    try:
        return sorted(str(k) for k in m.keys())
    except Exception:
        return []

@pytest.mark.integration
def test_run_t2_is_deterministic_for_ids_and_metric_keys():
    """
    Determinism guard:
      - Same config + same query => identical ordered IDs
      - Metric keys set identical (values may vary, but keys must be stable)
    """
    base_cfg = _load_cfg()
    query = "determinism guard — fixed input"

    res1 = run_t2(copy.deepcopy(base_cfg), query=query, ctx={"trace_reason": "determinism_guard"})
    res2 = run_t2(copy.deepcopy(base_cfg), query=query, ctx={"trace_reason": "determinism_guard"})

    ids1 = _extract_ids(res1)
    ids2 = _extract_ids(res2)
    keys1 = _metric_keys(res1)
    keys2 = _metric_keys(res2)

    assert ids1 == ids2, f"retrieved ID order differs:\n{ids1}\n!=\n{ids2}"
    assert keys1 == keys2, f"metric keys differ:\n{keys1}\n!=\n{keys2}"