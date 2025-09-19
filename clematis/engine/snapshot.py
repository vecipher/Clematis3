

from __future__ import annotations
from typing import Any, Dict, List, Optional
from dataclasses import asdict
import os
import json


# -------------------------
# Config & path helpers
# -------------------------

def _get_cfg(ctx) -> Dict[str, Any]:
    t4_default = {
        "snapshot_every_n_turns": 1,
        "snapshot_dir": "./.data/snapshots",
        "weight_min": -1.0,
        "weight_max": 1.0,
    }
    cfg = getattr(getattr(ctx, "config", object()), "t4", None)
    if isinstance(cfg, dict):
        out = dict(t4_default)
        out.update(cfg)
        out["snapshot_every_n_turns"] = int(out.get("snapshot_every_n_turns", 1))
        out["snapshot_dir"] = str(out.get("snapshot_dir", "./.data/snapshots"))
        out["weight_min"] = float(out.get("weight_min", -1.0))
        out["weight_max"] = float(out.get("weight_max", 1.0))
        return out
    return t4_default


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

def write_snapshot(ctx, state, version_etag: str, applied_count: int, deltas) -> str:
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
        "applied": int(applied_count or 0),
        "deltas": _serialize_deltas(deltas),
    }

    store = getattr(state, "store", None) if not isinstance(state, dict) else state.get("store")
    store_export = _export_store_for_snapshot(store) if store is not None else None
    if store_export:
        payload["store"] = store_export

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