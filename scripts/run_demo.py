import os
import sys


# Ensure the project root is importable before loading clematis modules.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from clematis.io.config import load_config
from clematis.world.scenario import run_one_turn


# --- Begin helpers for CLI overrides (object/dict safe) ---
from typing import Any, List

def _ensure_child(obj: Any, key: str) -> Any:
    """Ensure obj[key] (for dict) or obj.key (for object) exists and is a dict; return it.
    Creates an empty dict if missing. Works for simple attribute-bearing config objects.
    """
    if isinstance(obj, dict):
        val = obj.get(key)
        if not isinstance(val, dict):
            val = {}
            obj[key] = val
        return val
    # object with attributes
    val = getattr(obj, key, None)
    if not isinstance(val, dict):
        val = {} if val is None else val
        try:
            setattr(obj, key, val)
        except Exception:
            # As a fallback, if setting attribute fails, wrap into a dict on a side-car mapping
            # but in most configs this path won't be taken.
            pass
    return val


def _get_path(root: Any, path: List[str], default: Any = None) -> Any:
    """Traverse attributes/keys by path; return default if any hop is missing."""
    cur = root
    for k in path:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(k, None)
        else:
            cur = getattr(cur, k, None)
    return default if cur is None else cur


def _maybe_override_cfg_inplace(cfg: Any, args: Any) -> Any:
    """Apply CLI overrides to cfg in-place, supporting both object and dict configs."""
    sched = _ensure_child(cfg, "scheduler")
    budgets = _ensure_child(sched, "budgets")
    fairness = _ensure_child(sched, "fairness")

    def set_in(target: Any, key: str, value: Any):
        if value is None:
            return
        if isinstance(target, dict):
            target[key] = value
        else:
            try:
                setattr(target, key, value)
            except Exception:
                # Fall back to dict semantics if attribute setting fails
                d = getattr(target, "__dict__", None)
                if isinstance(d, dict):
                    d[key] = value

    set_in(sched, "policy", getattr(args, "policy", None))
    set_in(sched, "quantum_ms", getattr(args, "quantum_ms", None))
    set_in(budgets, "wall_ms", getattr(args, "wall_ms", None))
    set_in(budgets, "t1_iters", getattr(args, "t1_iters", None))
    set_in(budgets, "t1_pops", getattr(args, "t1_pops", None))
    set_in(budgets, "t2_k", getattr(args, "t2_k", None))
    set_in(budgets, "t3_ops", getattr(args, "t3_ops", None))

    # Ensure enabled for the demo to show yields if not already set
    enabled = _get_path(sched, ["enabled"], None)
    if enabled is None:
        set_in(sched, "enabled", True)
    return cfg
# --- End helpers ---


def main():
    cfg = load_config(os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml"))
    # If you want to support CLI overrides, you'd parse args here and pass to _maybe_override_cfg_inplace.
    # For demo purposes, we use a dummy object for args (no overrides).
    class DummyArgs:
        policy = None
        quantum_ms = None
        wall_ms = None
        t1_iters = None
        t1_pops = None
        t2_k = None
        t3_ops = None
    args = DummyArgs()
    cfg = _maybe_override_cfg_inplace(cfg, args)
    state = {}
    # Demo: get policy and fairness using new helpers
    policy = _get_path(cfg, ["scheduler", "policy"], "round_robin")
    fairness = _get_path(cfg, ["scheduler", "fairness"], {}) or {}

    # PR28: Provide a simple, deterministic pick_reason for logging.
    # In a real multi-agent loop this would come from scheduler.next_turn(...).
    if isinstance(policy, str):
        pick_reason = "ROUND_ROBIN" if policy == "round_robin" else ("AGING_BOOST" if policy == "fair_queue" else "ROUND_ROBIN")
    else:
        pick_reason = "ROUND_ROBIN"

    line = run_one_turn("AgentA", state, "hello world", cfg, pick_reason=pick_reason)
    print("Utterance:", line)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    print("Logs written to:", os.path.join(repo_root, ".logs"))

if __name__ == "__main__":
    # Add repo root to sys.path to simplify running without install
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    main()
