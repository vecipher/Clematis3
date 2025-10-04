

#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import pathlib
from typing import Any, Dict, List

try:
    import yaml  # Optional; only needed when --config is used
except Exception:
    yaml = None

# Reflection imports
from clematis.engine.stages.t3.reflect import reflect, ReflectionBundle
try:
    # Prefer the repo's deterministic JSON dumper (added in PR86)
    from clematis.engine.util.io_logging import stable_json_dumps as _stable_json_dumps
except Exception:
    def _stable_json_dumps(obj: Any) -> str:
        return json.dumps(obj, sort_keys=True, separators=(",", ":"))

# Defaults mirror PR77 config surface to keep this script self‑contained.
DEFAULT_CFG: Dict[str, Any] = {
    "t3": {
        "allow_reflection": True,
        "reflection": {
            "backend": "rulebased",   # or "llm" when fixtures are enabled
            "summary_tokens": 128,
            "embed": True,
            "log": True,
            "topk_snippets": 3,
        },
    },
    "scheduler": {
        "budgets": {
            "time_ms_reflection": 6000,
            "ops_reflection": 5,
        }
    },
}

def _load_cfg(path: str | None) -> Dict[str, Any]:
    """
    Load a YAML config and shallow‑merge onto DEFAULT_CFG to ensure
    required keys are present. If no path is provided, return defaults.
    """
    if path is None:
        return DEFAULT_CFG
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    if yaml is None:
        raise RuntimeError("PyYAML is required to load configs via --config.")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    # Shallow merges; keep nested sub‑dicts intact
    cfg = {
        "t3": {**DEFAULT_CFG["t3"], **(data.get("t3") or {})},
        "scheduler": {**DEFAULT_CFG["scheduler"], **(data.get("scheduler") or {})},
    }
    # Merge nested reflection + budgets
    cfg["t3"]["reflection"] = {
        **DEFAULT_CFG["t3"]["reflection"],
        **(cfg["t3"].get("reflection") or {}),
    }
    cfg["scheduler"]["budgets"] = {
        **DEFAULT_CFG["scheduler"]["budgets"],
        **(cfg["scheduler"].get("budgets") or {}),
    }
    return cfg

def _demo_inputs(cfg: Dict[str, Any]) -> tuple[str, List[str]]:
    """
    Provide deterministic utter/snippets independent of orchestrator.
    """
    utter = "assistant: summary: . next: question"
    k = int(cfg["t3"]["reflection"]["topk_snippets"])
    base = [
        "user asked to summarise prior step",
        "agent proposed next action planning",
        "note: constraints unchanged in this turn",
        "hint: prefer concise phrasing",
        "meta: deterministic test path",
    ]
    return utter, base[:max(0, k)]

def run_bench(cfg: Dict[str, Any], repeats: int = 1, normalize_ms: bool = True) -> Dict[str, Any]:
    """
    Execute reflect(...) on a fixed bundle. We keep this script independent of
    orchestrator and identity logs; it exercises only the reflection unit.
    """
    utter, snippets = _demo_inputs(cfg)
    bundle = ReflectionBundle(
        ctx=None,
        state_view=None,
        plan={"reflection": True},
        utter=utter,
        snippets=snippets,
    )

    last_result = None
    elapsed_ms = 0.0
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        last_result = reflect(bundle, cfg)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert last_result is not None
    backend = cfg["t3"]["reflection"]["backend"]
    payload: Dict[str, Any] = {
        "backend": backend,
        "allow_reflection": bool(cfg["t3"]["allow_reflection"]),
        "tokens_budget": int(cfg["t3"]["reflection"]["summary_tokens"]),
        "summary_len": len((last_result.summary or "").split()),
        "ops": len(last_result.memory_entries or []),
        "embed": bool(cfg["t3"]["reflection"]["embed"]),
        "ops_cap": int(cfg["scheduler"]["budgets"]["ops_reflection"]),
        "time_budget_ms": int(cfg["scheduler"]["budgets"]["time_ms_reflection"]),
        "ms": 0.0 if normalize_ms or os.getenv("CI", "").lower() == "true" else round(elapsed_ms, 3),
        "reason": (last_result.metrics or {}).get("reason"),
    }
    # Include fixture key only for LLM fixtures path, if present
    if last_result.metrics and "fixture_key" in last_result.metrics:
        payload["fixture_key"] = last_result.metrics["fixture_key"]
    return payload

def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="bench_reflection",
        description="Deterministic microbench for Clematis reflection pipeline.",
    )
    ap.add_argument(
        "--config", "-c", type=str, default=None,
        help="Path to YAML config (e.g., examples/reflection/enabled.yaml)."
    )
    ap.add_argument(
        "--repeats", "-n", type=int, default=1,
        help="Number of consecutive runs (keeps last result)."
    )
    ap.add_argument(
        "--raw-time", action="store_true",
        help="Do NOT normalize 'ms' (non‑deterministic timing; CI will still normalize)."
    )
    args = ap.parse_args(argv)

    cfg = _load_cfg(args.config)
    # Normalize by default; CI always normalizes regardless of flag.
    normalize = (not args.raw_time) or (os.getenv("CI", "").lower() == "true")
    out = run_bench(cfg, repeats=args.repeats, normalize_ms=normalize)
    sys.stdout.write(_stable_json_dumps(out) + "\n")
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
