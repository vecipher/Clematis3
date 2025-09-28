#!/usr/bin/env python3
"""
Local smoke test for the LLM planner using Ollama (Qwen3 4B Instruct).

Requirements (local only; NOT for CI):
  - Ollama running locally
  - Model pulled: `ollama pull qwen3:4b-instruct`

Usage:
  python scripts/llm_smoke.py
"""

import json
import sys
import os

# Ensure repo root is importable when running as a script from ./scripts
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from configs.validate import validate_config_api
from clematis.engine.stages.t3 import make_planner_prompt, plan_with_llm


class _Ctx:
    turn_id = 1
    agent_id = "demo"


def main() -> int:
    # Minimal inline config; defaults elsewhere remain OFF (rulebased)
    cfg = {
        "t3": {
            "backend": "llm",
            "llm": {
                "provider": "ollama",
                "model": "qwen3:4b-instruct",
                "endpoint": "http://localhost:11434/api/generate",
                "max_tokens": 256,
                "temp": 0.2,
                "timeout_ms": 15000,
            },
        }
    }

    ok, errs, cfg = validate_config_api(cfg)
    if not ok:
        print("Config error:", errs)
        return 2

    # Dummy state with log sink
    state = type("S", (), {"logs": []})()
    ctx = _Ctx()

    prompt = make_planner_prompt(ctx)
    print("PROMPT:\n" + prompt)

    out = plan_with_llm(ctx, state, cfg)
    print("\nOUTPUT:")
    print(json.dumps(out, indent=2, ensure_ascii=False))

    if state.logs:
        print("\nLOGS:", state.logs)

    # Minimal success check: must return a dict with required keys
    if not (isinstance(out, dict) and "plan" in out and "rationale" in out):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
