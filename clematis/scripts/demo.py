import os
import sys
import time
import argparse
import json
from typing import Any, List, Dict
from pathlib import Path

# Ensure the project root is importable before loading clematis modules.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from clematis.io.config import load_config
from clematis.io.log import append_jsonl
import clematis.io.paths as _paths
from clematis.world.scenario import run_one_turn
from clematis.scheduler import init_scheduler_state, next_turn, on_yield


# --- Begin helpers for CLI overrides (object/dict safe) ---


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


def _set_field(target: Any, key: str, value: Any) -> None:
    """Set key on dict or attribute on object; tolerate non-slot objects."""
    if value is None:
        return
    if isinstance(target, dict):
        target[key] = value
    else:
        try:
            setattr(target, key, value)
        except Exception:
            d = getattr(target, "__dict__", None)
            if isinstance(d, dict):
                d[key] = value
                return

def _maybe_get_child(target: Any, key: str) -> Any | None:
    """Best-effort read of a child (dict or object-attr) without creating it."""
    if isinstance(target, dict):
        return target.get(key)
    try:
        return getattr(target, key)
    except Exception:
        return None

def _apply_overrides(cfg: Any, overrides: Dict[str, Any]) -> Any:
    """
    Recursively apply nested dict overrides onto a mixed dict/object config.

    Important: if an overrides subtree explicitly sets {"enabled": False},
    we DO NOT materialize or update any other nested keys in that subtree.
    This guarantees disabled features remain inert for identity.
    """
    for k, v in overrides.items():
        if isinstance(v, dict):
            # If explicitly disabled, do not create/populate the subtree.
            if v.get("enabled") is False:
                child = _maybe_get_child(cfg, k)
                if child is not None:
                    _set_field(child, "enabled", False)  # only flip enabled if present
                continue  # skip all other keys under this disabled subtree
            child = _ensure_child(cfg, k)
            _apply_overrides(child, v)
        else:
            _set_field(cfg, k, v)
    return cfg


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


class DemoCtx:
    """
    Deterministic clock by default to keep identity CI stable.
    Use --wall-clock to switch back to real time.
    """

    def __init__(self, fixed_now_ms: int):
        self._fixed_now_ms = int(fixed_now_ms)

    def now_ms(self) -> int:
        return self._fixed_now_ms


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Clematis demo: scheduler rotation + driver-authored logs"
    )
    p.add_argument(
        "--config",
        default=os.path.join(REPO_ROOT, "configs", "config.yaml"),
        help="Path to config.yaml (default: configs/config.yaml)",
    )
    p.add_argument(
        "--agents",
        default="AgentA,AgentB,AgentC",
        help="Comma-separated agent ids (default: AgentA,AgentB,AgentC)",
    )
    p.add_argument(
        "--text", default="hello world", help="Input text for each turn (default: 'hello world')"
    )
    p.add_argument(
        "--steps", type=int, default=6, help="Number of selections/turns to run (default: 6)"
    )
    p.add_argument(
        "--policy",
        choices=["round_robin", "fair_queue"],
        default=None,
        help="Override scheduler.policy (default: use config)",
    )
    # Optional budget / timing overrides
    p.add_argument("--quantum-ms", type=int, default=None, help="Override scheduler.quantum_ms")
    p.add_argument("--wall-ms", type=int, default=None, help="Override scheduler.budgets.wall_ms")
    p.add_argument("--t1-iters", type=int, default=None, help="Override scheduler.budgets.t1_iters")
    p.add_argument("--t1-pops", type=int, default=None, help="Override scheduler.budgets.t1_pops")
    p.add_argument("--t2-k", type=int, default=None, help="Override scheduler.budgets.t2_k")
    p.add_argument("--t3-ops", type=int, default=None, help="Override scheduler.budgets.t3_ops")
    p.add_argument(
        "--wall-clock",
        action="store_true",
        help="Use wall clock time for now_ms (default: deterministic fixed seed)",
    )
    p.add_argument(
        "--fixed-now-ms",
        type=int,
        default=13371337,
        help="Deterministic now_ms base when --wall-clock is not set (default: 13371337)",
    )
    p.add_argument(
        "--config-overrides",
        type=str,
        default=None,
        help="Path to a JSON file with nested config overrides (applied after --policy/--t*-* flags).",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Directory to write logs to."
    )
    return p.parse_args()


def main():
    args = _parse_args()
    cfg = load_config(args.config)
    cfg = _maybe_override_cfg_inplace(cfg, args)

    # Apply generic nested overrides from a JSON file, if provided
    if getattr(args, "config_overrides", None):
        with open(args.config_overrides, "r", encoding="utf-8") as f:
            try:
                overrides = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Invalid JSON in --config-overrides: {e}", file=sys.stderr)
                sys.exit(2)
        _apply_overrides(cfg, overrides)

    # If --out is provided, override the logs_dir() function to point there (before any logging).
    if getattr(args, "out", None):
        out_dir = os.path.abspath(str(args.out))
        os.makedirs(out_dir, exist_ok=True)
        _paths.logs_dir = lambda: out_dir  # type: ignore[assignment]

    agents = sorted([a.strip() for a in args.agents.split(",") if a.strip()])
    if not agents:
        print("No agents provided. Exiting.")
        return

    # Deterministic by default; use --wall-clock to override
    now_ms = int(time.time() * 1000) if args.wall_clock else int(args.fixed_now_ms)
    sched_state = init_scheduler_state(agents, now_ms=now_ms)
    state: Dict[str, Any] = {}

    demo_ctx = DemoCtx(now_ms)
    policy = args.policy or _get_path(cfg, ["scheduler", "policy"], "round_robin")
    fairness = _get_path(cfg, ["scheduler", "fairness"], {}) or {}
    ldir = _paths.logs_dir()

    # Emit a non-canonical perf record up-front if the hybrid reranker is enabled.
    # This satisfies PR122 shadow-log segregation and does not touch canonical logs.
    try:
        from clematis.engine.observability_perf import write_perf_jsonl
    except Exception:
        write_perf_jsonl = None  # pragma: no cover

    if write_perf_jsonl is not None:
        hybrid_enabled = bool(_get_path(cfg, ["t2", "hybrid", "enabled"], False))
        if hybrid_enabled:
            write_perf_jsonl(
                Path(ldir),
                "t2_hybrid",
                {
                    "hybrid_used": True,
                    "lambda_graph": _get_path(cfg, ["t2", "hybrid", "lambda_graph"], None),
                    "anchor_top_m": _get_path(cfg, ["t2", "hybrid", "anchor_top_m"], None),
                    "walk_hops": _get_path(cfg, ["t2", "hybrid", "walk_hops"], None),
                    "degree_norm": _get_path(cfg, ["t2", "hybrid", "degree_norm"], None),
                },
            )

    print(f"Demo: policy={policy}, agents={agents}, steps={args.steps}")
    print(f"Logs dir: {ldir}")
    print("---")
    # NOTE: In CI, identity workflow relies on deterministic now_ms for stable logs.

    for step in range(1, args.steps + 1):
        agent_id, _, pick_reason = next_turn(
            demo_ctx, sched_state, policy=policy, fairness_cfg=fairness
        )
        if not agent_id:
            print(f"[{step}] No eligible agent.")
            break

        queue_before = list(sched_state["queue"]) if policy == "round_robin" else []
        capture: Dict[str, Any] = {}

        # Run one turn; orchestrator will capture boundary event instead of writing scheduler.jsonl
        line = run_one_turn(
            agent_id,
            state,
            args.text,
            cfg,
            pick_reason=pick_reason,
            driver_logging=True,
            capture=capture,
        )

        yielded = bool(capture)

        if yielded:
            # Update fairness clocks/counters; reset on saturation
            reset = pick_reason == "RESET_CONSEC"
            on_yield(
                demo_ctx,
                sched_state,
                agent_id,
                consumed=capture.get("consumed", {}),
                reason=capture.get("reason", ""),
                fairness_cfg=fairness,
                reset=reset,
            )

            # Perform deterministic RR rotation (head -> tail) only for round_robin
            if policy == "round_robin":
                q = sched_state["queue"]
                try:
                    q.remove(agent_id)
                    q.append(agent_id)
                except ValueError:
                    pass
                queue_after = list(q)
            else:
                queue_after = []

            # Driver-authored enriched scheduler log (merge captured event)
            event = dict(capture)
            # Ensure pick_reason is present; orchestrator may have included it already
            event.setdefault("pick_reason", pick_reason)
            if policy == "round_robin":
                event["queue_before"] = queue_before
                event["queue_after"] = queue_after
            else:
                event["queue_before"] = []
                event["queue_after"] = []
            append_jsonl("scheduler.jsonl", event)

            # Print concise summary
            if policy == "round_robin":
                print(
                    f"[{step}] YIELD agent={agent_id} pick={pick_reason}  queue: {queue_before} -> {event['queue_after']}  | utter={line!r}"
                )
            else:
                print(f"[{step}] YIELD agent={agent_id} pick={pick_reason}  | utter={line!r}")
        else:
            print(f"[{step}] NO-YIELD agent={agent_id} pick={pick_reason}  | utter={line!r}")

    print("---")
    print("Logs written to:", ldir)


if __name__ == "__main__":
    # Add repo root to sys.path to simplify running without install
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    main()
