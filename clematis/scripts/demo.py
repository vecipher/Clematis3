import os
import sys
import time
import argparse
from typing import Any, List, Dict

# Ensure the project root is importable before loading clematis modules.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from clematis.io.config import load_config
from clematis.io.log import append_jsonl
from clematis.io.paths import logs_dir
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
    return p.parse_args()


def main():
    args = _parse_args()
    cfg = load_config(args.config)
    cfg = _maybe_override_cfg_inplace(cfg, args)

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

    print(f"Demo: policy={policy}, agents={agents}, steps={args.steps}")
    print(f"Logs dir: {logs_dir()}")
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

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    print("---")
    print("Logs written to:", os.path.join(repo_root, ".logs"))


if __name__ == "__main__":
    # Add repo root to sys.path to simplify running without install
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    main()
