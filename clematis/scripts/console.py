#!/usr/bin/env python3
"""
Deterministic local console for Clematis.
Usage:
  python -m clematis console -- step [--now-ms N] [--input "…"] [--out run.json]
  python -m clematis console -- reset [--snapshot PATH]
  python -m clematis console -- status
  python -m clematis console -- compare --a A.json --b B.json
Exit codes:
  0 = OK/equal; 1 = compare:differs; 2 = adapter/misuse error.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import pathlib
import tempfile
import shutil
from typing import Any, Dict

# Snapshot import deferred; see adapter_reset() for lazy import with fallbacks.

# Exporter (PR128). Optional; console can fall back to a minimal bundle.
try:
    from clematis.scripts.export_logs_for_frontend import export_state_to_bundle as _export_state_to_bundle  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    try:
        from clematis.cli.export_logs_for_frontend import export_state_to_bundle as _export_state_to_bundle  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        _export_state_to_bundle = None  # type: ignore[assignment]
try:
    from clematis.scripts.export_logs_for_frontend import build_run_bundle as _build_run_bundle  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    _build_run_bundle = None  # type: ignore[assignment]

DEFAULT_EPOCH = int(os.environ.get("SOURCE_DATE_EPOCH") or "315532800")  # 1980-01-01
DEFAULT_NOW_MS = DEFAULT_EPOCH * 1000

# -------------------------
# Deterministic env helpers
# -------------------------
REQ_ENV = {
    "TZ": "UTC",
    "PYTHONHASHSEED": "0",
    "SOURCE_DATE_EPOCH": str(DEFAULT_EPOCH),
    "CLEMATIS_NETWORK_BAN": "1",
}

def warn_nondeterminism() -> None:
    missing = [k for k,v in REQ_ENV.items() if os.environ.get(k) != v]
    if missing:
        print(f"[console] WARNING: non-deterministic env vars differ: {missing}", file=sys.stderr)

# -------------------------
# Orchestrator adapter
# -------------------------


def _state_from_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Construct a minimal engine state from a snapshot payload dict.
    Keeps both `gel` and `graph` keys to satisfy downstream consumers.
    """
    gel = payload.get("gel") or payload.get("graph") or {}
    if not isinstance(gel, dict):
        gel = {}
    state: Dict[str, Any] = {"gel": gel, "graph": gel}
    ver = payload.get("version_etag")
    if ver is not None:
        state["version_etag"] = str(ver)
    store = payload.get("store")
    if isinstance(store, dict):
        state["store"] = store
    return state


def adapter_reset(snapshot_path: str | None):
    """
    Deterministic reset using the snapshot module: explicit path -> read_snapshot;
    otherwise use load_latest_snapshot (engine decides where "latest" is).
    Falls back to scanning ./.data/snapshots when the helper is unavailable.
    """
    try:
        from clematis.engine.snapshot import read_snapshot, load_latest_snapshot  # type: ignore[attr-defined]
    except Exception as e:
        print("[console] ERROR: snapshot module not available (clematis.engine.snapshot)", file=sys.stderr)
        raise SystemExit(2) from e

    # Explicit path: read payload and synthesise a minimal state
    if snapshot_path:
        try:
            payload = read_snapshot(path=snapshot_path)  # type: ignore[call-arg]
        except Exception as e:
            print(f"[console] ERROR: failed to read snapshot: {snapshot_path}", file=sys.stderr)
            raise SystemExit(2) from e
        return _state_from_payload(payload)

    # No explicit path: try engine helper to load latest into a fresh state
    try:
        from types import SimpleNamespace
        ctx = SimpleNamespace(cfg={})
        state: Dict[str, Any] = {}
        load_latest_snapshot(ctx, state)  # type: ignore[misc]
        return state
    except Exception:
        # Fallback: pick most-recent .json manually and read it
        base = os.environ.get("CLEMATIS_SNAPSHOTS_DIR") or "./.data/snapshots"
        p = find_latest_snapshot(pathlib.Path(base))
        if not p:
            print("[console] ERROR: no snapshots found and no default loader", file=sys.stderr)
            raise SystemExit(2)
        try:
            payload = read_snapshot(path=p)  # type: ignore[call-arg]
        except Exception as e:
            print(f"[console] ERROR: failed to read snapshot: {p}", file=sys.stderr)
            raise SystemExit(2) from e
        return _state_from_payload(payload)

def _make_minimal_bundle(low_level_logs: Dict[str, Any]) -> Dict[str, Any]:
    stages = ("t1", "t2", "t4", "apply", "turn")
    logs = {k: list(low_level_logs.get(k, [])) for k in stages}
    return {
        "meta": {"tool": "clematis-console", "schema": "v1", "stages": list(stages)},
        "snapshots": [],
        "logs": logs,
    }

def adapter_step(state: dict, now_ms: int, input_text: str):
    """
    Execute exactly one deterministic turn.
    - If the orchestrator returns (state, logs) or an object with `.state`/`.logs`, use that.
    - Otherwise, assume logs were written to disk (CLEMATIS_LOG_DIR) and assemble a bundle from disk.
    Falls back to a minimal in-memory bundle if no exporter is available.
    """
    try:
        from clematis.engine.orchestrator.core import run_turn  # type: ignore[attr-defined]
    except Exception as e:
        print("[console] ERROR: orchestrator run_turn not available", file=sys.stderr)
        raise SystemExit(2) from e

    # Prepare deterministic context
    try:
        from types import SimpleNamespace
        # Build a small t3 config from environment flags so orchestrator can pick it up
        _t3_allow = (os.environ.get("CLEMATIS_T3_ALLOW") == "1")
        _t3_apply = (os.environ.get("CLEMATIS_T3_APPLY_OPS") == "1")
        _llm_mode = os.environ.get("CLEMATIS_LLM_MODE", "mock")
        _llm_cassette = os.environ.get("CLEMATIS_LLM_CASSETTE")
        _backend = "rulebased" if _llm_mode == "rulebased" else "llm"

        cfg_ns = SimpleNamespace(
            t1={},
            t2={},
            t3={
                "enabled": _t3_allow,
                "allow": _t3_allow,          # alias accepted by orchestrator
                "apply_ops": _t3_apply,
                "backend": _backend,
                "llm": {"mode": _llm_mode, "cassette": _llm_cassette},
                "max_rag_loops": 1,
            },
            scheduler={},
        )
        ctx = SimpleNamespace(now_ms=now_ms, turn_id="1", agent_id="console", cfg=cfg_ns)

        # Best-effort: attach an adapter if available for mock/replay/live; silently skip if not present
        try:
            if _backend == "llm":
                if _llm_mode == "mock":
                    try:
                        from clematis.adapters.llm import FixtureLLMAdapter  # type: ignore
                        ctx.llm_adapter = FixtureLLMAdapter()
                    except Exception:
                        pass
                elif _llm_mode == "replay" and _llm_cassette:
                    try:
                        from clematis.adapters.llm import ReplayLLMAdapter  # type: ignore
                        ctx.llm_adapter = ReplayLLMAdapter(_llm_cassette)
                    except Exception:
                        pass
                elif _llm_mode == "live":
                    try:
                        from clematis.adapters.llm import LiveOpenAIAdapter  # type: ignore
                        ctx.llm_adapter = LiveOpenAIAdapter.from_env()  # may raise; ok to skip
                    except Exception:
                        pass
        except Exception:
            # never let adapter wiring break deterministic runs
            pass
    except Exception:
        # Fallback (shouldn't happen): minimal mapping shape
        ctx = {"now_ms": now_ms, "cfg": {"t1": {}, "t2": {}, "t3": {}, "scheduler": {}}}

    # Ensure we have a logs directory; if none configured, use a temp dir
    cleanup_dir = None
    restore_env = None
    if not os.environ.get("CLEMATIS_LOG_DIR"):
        cleanup_dir = tempfile.mkdtemp(prefix="clematis-logs-")
        restore_env = os.environ.get("CLEMATIS_LOG_DIR")
        os.environ["CLEMATIS_LOG_DIR"] = cleanup_dir
        print(f"[console] using logs_dir={cleanup_dir}", file=sys.stderr)
    else:
        print(f"[console] using logs_dir={os.environ.get('CLEMATIS_LOG_DIR')}", file=sys.stderr)

    try:
        res = run_turn(ctx, state, input_text or "")

        # Derive new_state and low_level_logs from various possible return shapes
        new_state = state
        low_level_logs: Dict[str, Any] | None = None
        if isinstance(res, tuple) and len(res) >= 2:
            new_state = res[0] if res[0] is not None else state
            low_level_logs = res[1]
        else:
            # object with attributes?
            cand_state = getattr(res, "state", None)
            cand_logs = getattr(res, "logs", None)
            if cand_state is not None:
                new_state = cand_state
            if cand_logs is not None:
                low_level_logs = cand_logs

        # Prefer exporter if we have in-memory logs
        if _export_state_to_bundle and low_level_logs is not None:
            try:
                bundle = _export_state_to_bundle(new_state, logs=low_level_logs, include_perf=False)  # type: ignore[misc]
                return new_state, bundle
            except TypeError:
                # Signature mismatch; fall through to from-disk bundling
                pass
            except NotImplementedError as e:
                print("[console] ERROR: exporter not available (export_state_to_bundle)", file=sys.stderr)
                raise SystemExit(2) from e

        # Assemble from disk if possible
        if _build_run_bundle:
            logs_dir = os.environ.get("CLEMATIS_LOG_DIR") or cleanup_dir or "./.data/logs"
            snaps_dir = os.environ.get("CLEMATIS_SNAPSHOTS_DIR") or "./.data/snapshots"
            try:
                bundle, warns, rc = _build_run_bundle(
                    logs_dir=logs_dir,
                    snapshots_dir=snaps_dir,
                    include_perf=False,
                    strict=False,
                    max_stage_entries=None,
                )
                if rc == 0:
                    return new_state, bundle
            except Exception:
                # Fall through to minimal bundle
                pass

        # Last resort: minimal in-memory bundle
        return new_state, _make_minimal_bundle(low_level_logs or {})
    finally:
        if cleanup_dir:
            try:
                shutil.rmtree(cleanup_dir)
            except Exception:
                pass
            # Restore env
            if restore_env is None:
                os.environ.pop("CLEMATIS_LOG_DIR", None)
            else:
                os.environ["CLEMATIS_LOG_DIR"] = restore_env

def adapter_status(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Summarize scheduler configuration and budgets from state/config,
    and (optionally) recent scheduler event counts from scheduler.jsonl.
    """
    cfg = state.get("cfg") or state.get("config") or {}
    scfg = (cfg.get("scheduler") or {}) if isinstance(cfg, dict) else {}
    policy = scfg.get("policy", "round_robin")
    fairness = scfg.get("fairness") or {}
    # Prefer budgets from cfg.scheduler; fall back to top-level
    budgets_src = scfg.get("budgets") or state.get("budgets") or {}

    status: Dict[str, Any] = {
        "scheduler": {
            "policy": policy,
            "fairness_keys": sorted(list(fairness.keys())) if isinstance(fairness, dict) else [],
        },
        "budgets": {},
    }

    # Normalize canonical budget keys referenced in orchestrator.core
    for k in ("t1_iters", "t1_pops", "t2_k", "t3_ops", "quantum_ms", "wall_ms"):
        if isinstance(budgets_src, dict) and k in budgets_src:
            status["budgets"][k] = budgets_src[k]

    # Optionally summarize recent scheduler events from logs (best-effort)
    try:
        logs_dir = os.environ.get("CLEMATIS_LOG_DIR")
        if logs_dir:
            sched_path = os.path.join(logs_dir, "scheduler.jsonl")
            if os.path.exists(sched_path):
                counts: Dict[str, int] = {}
                with open(sched_path, "r", encoding="utf-8", errors="ignore") as fh:
                    tail = fh.readlines()[-100:]
                for line in tail:
                    try:
                        ev = json.loads(line)
                        ev_type = str(ev.get("event", "unknown"))
                        counts[ev_type] = counts.get(ev_type, 0) + 1
                    except Exception:
                        counts["parse_error"] = counts.get("parse_error", 0) + 1
                if counts:
                    status["scheduler"]["recent_event_counts"] = counts
    except Exception:
        # fail-soft: logging isn’t critical for status
        pass

    return status

# -------------------------
# Utilities
# -------------------------
def find_latest_snapshot(dir_path: pathlib.Path) -> str | None:
    if not dir_path.exists():
        return None
    files = [p for p in dir_path.glob("*.json") if p.is_file()]
    if not files:
        return None
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(files[0])

def write_json(path: str, obj: Any) -> None:
    # Canonical ordering + LF newlines to match exporter conventions.
    def canonical(o: Any) -> Any:
        if isinstance(o, dict):
            return {k: canonical(o[k]) for k in sorted(o.keys())}
        if isinstance(o, list):
            return [canonical(x) for x in o]
        return o
    data = json.dumps(canonical(obj), indent=2, ensure_ascii=False)
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True); p.write_text(data + "\n", encoding="utf-8")

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def summarize_bundle(bundle: Dict[str, Any]) -> Dict[str, Any]:
    logs = bundle.get("logs") or {}
    stages = ("t1", "t2", "t3", "t3_reflection", "t4", "apply", "turn")
    counts = {k: len(logs.get(k, [])) for k in stages}
    snaps_list = bundle.get("snapshots")
    if isinstance(snaps_list, list):
        snaps_len = len(snaps_list)
    else:
        snaps_len = 1 if "snapshot" in bundle else 0
    meta = bundle.get("meta") or {}
    return {
        "counts": counts,
        "snapshots_len": snaps_len,
        "meta_keys": sorted(list(meta.keys())),
    }

def compare_bundles(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    sa, sb = summarize_bundle(a), summarize_bundle(b)
    diff = {}
    for k in ("counts", "snapshots_len", "meta_keys"):
        if sa[k] != sb[k]:
            diff[k] = {"a": sa[k], "b": sb[k]}
    return diff

# -------------------------
# Commands
# -------------------------
def cmd_reset(args: argparse.Namespace) -> int:
    warn_nondeterminism()
    snap = args.snapshot
    if snap is None:
        base = os.environ.get("CLEMATIS_SNAPSHOTS_DIR") or "./.data/snapshots"
        snap = find_latest_snapshot(pathlib.Path(base))  # default convention
    st = adapter_reset(snap)
    out = {"ok": True, "snapshot": snap, "state_hint": list(st.keys())[:8]}
    print(json.dumps(out, indent=2))
    return 0

def cmd_status(args: argparse.Namespace) -> int:
    snap = args.snapshot
    if snap is None:
        base = os.environ.get("CLEMATIS_SNAPSHOTS_DIR") or "./.data/snapshots"
        snap = find_latest_snapshot(pathlib.Path(base))
    st = adapter_reset(snap) if snap else adapter_reset(None)
    info = adapter_status(st)
    if snap:
        info = {"snapshot": snap, **info}
    print(json.dumps(info, indent=2))
    return 0

def cmd_step(args: argparse.Namespace) -> int:
    warn_nondeterminism()
    st = adapter_reset(args.snapshot) if args.snapshot else adapter_reset(None)
    now_ms = args.now_ms if args.now_ms is not None else DEFAULT_NOW_MS
    # Propagate T3/LLM flags via env so the orchestrator gate can see them
    if getattr(args, "t3", False):
        os.environ["CLEMATIS_T3_ALLOW"] = "1"
    if getattr(args, "t3_apply_ops", False):
        os.environ["CLEMATIS_T3_APPLY_OPS"] = "1"
    if getattr(args, "llm_mode", None):
        os.environ["CLEMATIS_LLM_MODE"] = args.llm_mode
    if getattr(args, "llm_cassette", None):
        os.environ["CLEMATIS_LLM_CASSETTE"] = args.llm_cassette
    st2, logs = adapter_step(st, now_ms=now_ms, input_text=(args.input or ""))
    if args.out:
        write_json(args.out, logs)
    else:
        print(json.dumps(logs, indent=2))
    return 0

def cmd_compare(args: argparse.Namespace) -> int:
    a = load_json(args.a)
    b = load_json(args.b)
    diff = compare_bundles(a, b)
    if diff:
        print(json.dumps(diff, indent=2))
        return 1
    print(json.dumps({"equal": True}))
    return 0

# -------------------------
# Entry
# -------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="console", add_help=True)
    sub = p.add_subparsers(dest="cmd", required=True)

    p_step = sub.add_parser("step", aliases=["next"], help="run one turn deterministically")
    p_step.add_argument("--snapshot", type=str, default=None, help="snapshot .json path (default: latest)")
    p_step.add_argument("--now-ms", type=int, default=None, help=f"logical time (default: {DEFAULT_NOW_MS})")
    p_step.add_argument("--input", type=str, default="", help="input text for the turn")
    # T3 / LLM gating
    p_step.add_argument("--t3", action="store_true", help="enable T3 (planner/dialogue/ops)")
    p_step.add_argument("--t3-apply-ops", action="store_true", help="apply T3 ops to state (off by default)")
    p_step.add_argument(
        "--llm-mode",
        choices=["mock", "replay", "live", "rulebased"],
        default=os.environ.get("CLEMATIS_LLM_MODE", "mock"),
        help="LLM backend mode (default: mock). 'rulebased' forces non-LLM speak",
    )
    p_step.add_argument(
        "--llm-cassette",
        type=str,
        default=os.environ.get("CLEMATIS_LLM_CASSETTE"),
        help="Path to replay cassette when --llm-mode=replay",
    )
    p_step.add_argument("--out", type=str, default=None, help="write logs to file (default: stdout)")
    p_step.set_defaults(fn=cmd_step)

    p_reset = sub.add_parser("reset", help="load snapshot and reset state")
    p_reset.add_argument("--snapshot", type=str, default=None)
    p_reset.set_defaults(fn=cmd_reset)

    p_status = sub.add_parser("status", help="print scheduler/budgets summary")
    p_status.add_argument("--snapshot", type=str, default=None)
    p_status.set_defaults(fn=cmd_status)

    p_cmp = sub.add_parser("compare", help="diff two run_bundle.json files")
    p_cmp.add_argument("--a", type=str, required=True)
    p_cmp.add_argument("--b", type=str, required=True)
    p_cmp.set_defaults(fn=cmd_compare)

    return p

def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = build_parser()
    ns = parser.parse_args(argv)
    return ns.fn(ns)

if __name__ == "__main__":
    raise SystemExit(main())
