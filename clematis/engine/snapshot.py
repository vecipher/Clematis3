from __future__ import annotations
from typing import Any, Dict, List, Optional
from dataclasses import asdict
import os
import json


# -------------------------
# PR18: schema tag & inspector helpers
# -------------------------

SCHEMA_VERSION = "v1"  # snapshots written going forward should include this

def _safe_read_json(path: str) -> Optional[Dict[str, Any]]:
    """Read JSON file; return None on any error."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _pick_latest_snapshot_path(directory: str) -> Optional[str]:
    """
    Deterministically pick a snapshot in `directory`:
    1) Prefer files matching 'snap_*.json' by descending numeric suffix
    2) Else prefer files matching 'state_*.json' by most-recent mtime
    3) Else fall back to newest '*.json' by mtime
    4) Else None
    """
    if not os.path.isdir(directory):
        return None

    # Prefer numbered JSON snapshots like snap_000123.json
    numbered = []
    try:
        names = os.listdir(directory)
    except Exception:
        return None

    for name in names:
        if not name.endswith(".json"):
            continue
        if name.startswith("snap_"):
            stem = name[len("snap_") : -len(".json")]
            if stem.isdigit():
                numbered.append((int(stem), os.path.join(directory, name)))
    if numbered:
        numbered.sort(key=lambda t: t[0], reverse=True)
        return numbered[0][1]

    # Next: state_*.json by mtime
    state_candidates = [
        os.path.join(directory, n) for n in names if n.startswith("state_") and n.endswith(".json")
    ]
    if state_candidates:
        try:
            state_candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        except Exception:
            state_candidates.sort()
        return state_candidates[0]

    # Fallback: any *.json by mtime
    any_json = [os.path.join(directory, n) for n in names if n.endswith(".json")]
    if not any_json:
        return None
    try:
        any_json.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    except Exception:
        any_json.sort()
    return any_json[0]

def get_latest_snapshot_info(directory: str = "./.data/snapshots") -> Optional[Dict[str, Any]]:
    """
    Pure metadata probe for the latest snapshot file in `directory`.
    Returns a dict:
    {
      "path": str,
      "schema_version": str | "unknown",
      "version_etag": str | None,
      "nodes": int | None,
      "edges": int | None,
      "last_update": str | None,
      "caps": {
        "delta_norm_cap_l2": float|None,
        "novelty_cap_per_node": float|None,
        "churn_cap_edges": int|None,
        "weight_min": float|None,
        "weight_max": float|None
      }
    }
    or None if no snapshot is found/readable. Never raises.
    """
    path = _pick_latest_snapshot_path(directory)
    if not path:
        return None

    data = _safe_read_json(path)
    if not data:
        return None

    schema = data.get("schema_version", "unknown")
    graph = (data.get("graph") or {})
    caps = (data.get("t4_caps") or {})
    meta = (graph.get("meta") or {})

    # --- Legacy fallbacks for counts ---
    nodes = graph.get("nodes_count")
    if nodes is None:
        g_nodes = graph.get("nodes")
        if isinstance(g_nodes, (list, dict)):
            nodes = len(g_nodes)
        else:
            top_nodes = data.get("nodes")
            if isinstance(top_nodes, (list, dict)):
                nodes = len(top_nodes)

    edges = graph.get("edges_count")
    if edges is None:
        g_edges = graph.get("edges")
        if isinstance(g_edges, (list, dict)):
            edges = len(g_edges)
        else:
            top_edges = data.get("edges")
            if isinstance(top_edges, (list, dict)):
                edges = len(top_edges)

    # --- Legacy fallback for last_update ---
    last_update = meta.get("last_update") or data.get("last_update")

    return {
        "path": path,
        "schema_version": schema,
        "version_etag": data.get("version_etag"),
        "nodes": nodes,
        "edges": edges,
        "last_update": last_update,
        "caps": {
            "delta_norm_cap_l2": caps.get("delta_norm_cap_l2"),
            "novelty_cap_per_node": caps.get("novelty_cap_per_node"),
            "churn_cap_edges": caps.get("churn_cap_edges"),
            "weight_min": caps.get("weight_min"),
            "weight_max": caps.get("weight_max"),
        },
    }


# -------------------------
# Config & path helpers
# -------------------------

def _get_cfg(ctx) -> Dict[str, Any]:
    """
    Normalize config access across ctx.cfg / ctx.config and return a T4 dict
    merged with defaults. Values are coerced to expected types.
    """
    t4_default = {
        "snapshot_every_n_turns": 1,
        "snapshot_dir": "./.data/snapshots",
        "weight_min": -1.0,
        "weight_max": 1.0,
    }

    # Pull a dict from ctx.cfg or ctx.config (SimpleNamespace or dict)
    def _as_dict(obj):
        if obj is None:
            return {}
        if isinstance(obj, dict):
            return obj
        try:
            # SimpleNamespace or any object with __dict__
            return dict(obj.__dict__)
        except Exception:
            return {}

    full_cfg = {}
    full_cfg.update(_as_dict(getattr(ctx, "cfg", None)))
    full_cfg.update(_as_dict(getattr(ctx, "config", None)))

    t4 = _as_dict(full_cfg.get("t4"))
    out = dict(t4_default)
    out.update(t4)

    # Coerce types
    out["snapshot_every_n_turns"] = int(out.get("snapshot_every_n_turns", 1))
    out["snapshot_dir"] = str(out.get("snapshot_dir", "./.data/snapshots"))
    out["weight_min"] = float(out.get("weight_min", -1.0))
    out["weight_max"] = float(out.get("weight_max", 1.0))
    return out


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _snapshot_path(cfg: Dict[str, Any], ctx) -> str:
    agent = getattr(ctx, "agent_id", "agent")
    dir_ = cfg["snapshot_dir"]
    _ensure_dir(dir_)
    fname = f"state_{agent}.json"
    return os.path.join(dir_, fname)


# -------------------------
# Store export/import
# -------------------------

def _export_store_for_snapshot(store: Any) -> Optional[Dict[str, Any]]:
    """
    Try to export the store in a structured way, falling back to a weight map.
    Prefers store.export_state(), else inspects a `.w` dict as used by tests.
    """
    exp = getattr(store, "export_state", None)
    if callable(exp):
        try:
            return {"state": exp()}
        except Exception:
            pass

    w = getattr(store, "w", None)
    if isinstance(w, dict):
        weights: List[Dict[str, Any]] = []
        for k, v in w.items():
            try:
                target_kind, target_id, attr = k
            except Exception:
                continue
            weights.append({
                "target_kind": str(target_kind),
                "target_id": str(target_id),
                "attr": str(attr),
                "value": float(v),
            })
        return {"weights": weights}
    return None


def _import_store_from_snapshot(store: Any, snap_store: Dict[str, Any]) -> bool:
    """
    Import store state from a snapshot `store` section.
    Supports store.import_state(...) or weights fallback into `.w`.
    """
    if not isinstance(snap_store, dict):
        return False

    imp = getattr(store, "import_state", None)
    if "state" in snap_store and callable(imp):
        try:
            imp(snap_store["state"])
            return True
        except Exception:
            return False

    if "weights" in snap_store and isinstance(getattr(store, "w", None), dict):
        try:
            newmap: Dict[tuple, float] = {}
            for item in snap_store["weights"] or []:
                tk = str(item.get("target_kind", "node"))
                tid = str(item.get("target_id", ""))
                attr = str(item.get("attr", "weight"))
                val = float(item.get("value", 0.0))
                newmap[(tk, tid, attr)] = val
            store.w.clear()
            store.w.update(newmap)
            return True
        except Exception:
            return False

    return False


# -------------------------
# Serialization helpers
# -------------------------

def _serialize_deltas(deltas: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not deltas:
        return out
    for d in deltas:
        try:
            out.append({
                "target_kind": d.target_kind,
                "target_id": d.target_id,
                "attr": d.attr,
                "delta": float(d.delta),
                "op_idx": getattr(d, "op_idx", None),
                "idx": getattr(d, "idx", None),
            })
        except Exception:
            try:
                out.append(asdict(d))
            except Exception:
                # Best-effort: attempt dict-like access
                try:
                    out.append({
                        "target_kind": str(d.get("target_kind")),
                        "target_id": str(d.get("target_id")),
                        "attr": str(d.get("attr")),
                        "delta": float(d.get("delta", 0.0)),
                        "op_idx": d.get("op_idx"),
                        "idx": d.get("idx"),
                    })
                except Exception:
                    continue
    return out


# -------------------------
# Public API
# -------------------------

def write_snapshot(ctx, state, version_etag: str, applied: int = 0, deltas=None) -> str:
    """
    Write a JSON snapshot for this agent. Returns the absolute file path.
    The payload includes {turn, agent, version_etag, applied, deltas, store?}.
    """
    cfg = _get_cfg(ctx)
    path = _snapshot_path(cfg, ctx)

    try:
        turn = int(getattr(ctx, "turn_id", 0))
    except Exception:
        turn = 0

    payload: Dict[str, Any] = {
        "turn": turn,
        "agent": getattr(ctx, "agent_id", "agent"),
        "version_etag": version_etag,
        "applied": int(applied or 0),
        "deltas": _serialize_deltas(deltas),
        "schema_version": SCHEMA_VERSION,
    }

    store = getattr(state, "store", None) if not isinstance(state, dict) else state.get("store")
    store_export = _export_store_for_snapshot(store) if store is not None else None
    # Always include a 'store' key for shape stability (empty object when not available)
    payload["store"] = store_export if isinstance(store_export, dict) else {}

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    return path


def load_latest_snapshot(ctx, state) -> Dict[str, Any]:
    """
    Load the latest snapshot for this agent from t4.snapshot_dir.
    Returns {"loaded": bool, "path": str|None, "version_etag": str|None}.
    Tolerates both PR13-lite and PR14-enriched snapshot schemas.
    """
    cfg = _get_cfg(ctx)
    path = _snapshot_path(cfg, ctx)
    if not os.path.isfile(path):
        return {"loaded": False, "path": None, "version_etag": None}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {"loaded": False, "path": path, "version_etag": None}

    ver = data.get("version_etag")
    if ver is not None:
        # set version_etag on state in a dict/attr-safe way
        if isinstance(state, dict):
            state["version_etag"] = str(ver)
        else:
            setattr(state, "version_etag", str(ver))

    store = getattr(state, "store", None) if not isinstance(state, dict) else state.get("store")
    loaded_store = False
    snap_store = data.get("store")
    if store is not None and snap_store is not None:
        loaded_store = _import_store_from_snapshot(store, snap_store)

    return {"loaded": bool(loaded_store or ver is not None), "path": path, "version_etag": ver}