from __future__ import annotations
from typing import Any, Dict, List, Optional
from dataclasses import asdict
import os
import json
import math
import hashlib
import logging
import sys

try:
    import zstandard as _zstd  # optional; used for .zst snapshots
except Exception:
    _zstd = None


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
    graph_schema = data.get("graph_schema_version", None)
    gel = data.get("gel") or {}
    graph = data.get("graph") or {}
    caps = data.get("t4_caps") or {}
    meta = graph.get("meta") or {}

    # --- Prefer GEL counts if present, else fall back to legacy 'graph' or top-level ---
    nodes = None
    edges = None

    # GEL structure (preferred in newer snapshots)
    if isinstance(gel, dict):
        g_nodes = gel.get("nodes")
        if isinstance(g_nodes, dict):
            nodes = len(g_nodes)
        elif isinstance(g_nodes, list):
            nodes = len(g_nodes)
        g_edges = gel.get("edges")
        if isinstance(g_edges, dict):
            edges = len(g_edges)
        elif isinstance(g_edges, list):
            edges = len(g_edges)

    # Legacy 'graph' field
    if nodes is None:
        nodes = graph.get("nodes_count")
    if nodes is None:
        g_nodes = graph.get("nodes")
        if isinstance(g_nodes, (list, dict)):
            nodes = len(g_nodes)
        else:
            top_nodes = data.get("nodes")
            if isinstance(top_nodes, (list, dict)):
                nodes = len(top_nodes)

    if edges is None:
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
        "graph_schema_version": graph_schema,
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
            weights.append(
                {
                    "target_kind": str(target_kind),
                    "target_id": str(target_id),
                    "attr": str(attr),
                    "value": float(v),
                }
            )
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
            out.append(
                {
                    "target_kind": d.target_kind,
                    "target_id": d.target_id,
                    "attr": d.attr,
                    "delta": float(d.delta),
                    "op_idx": getattr(d, "op_idx", None),
                    "idx": getattr(d, "idx", None),
                }
            )
        except Exception:
            try:
                out.append(asdict(d))
            except Exception:
                # Best-effort: attempt dict-like access
                try:
                    out.append(
                        {
                            "target_kind": str(d.get("target_kind")),
                            "target_id": str(d.get("target_id")),
                            "attr": str(d.get("attr")),
                            "delta": float(d.get("delta", 0.0)),
                            "op_idx": d.get("op_idx"),
                            "idx": d.get("idx"),
                        }
                    )
                except Exception:
                    continue
    return out


# -------------------------
# GEL (Graph Evolution Layer) helpers
# -------------------------


def _extract_full_cfg(ctx) -> Dict[str, Any]:
    def _as_dict(obj):
        if obj is None:
            return {}
        if isinstance(obj, dict):
            return obj
        try:
            return dict(obj.__dict__)
        except Exception:
            return {}

    full = {}
    full.update(_as_dict(getattr(ctx, "cfg", None)))
    full.update(_as_dict(getattr(ctx, "config", None)))
    return full


def _graph_bounds_from_cfg(ctx) -> Dict[str, float]:
    """Get clamp/epsilon from graph.* if present, else fall back to t4.*."""
    full = _extract_full_cfg(ctx)
    g = full.get("graph") or {}
    t4 = full.get("t4") or {}
    wmin = float(g.get("weight_min", t4.get("weight_min", -1.0)))
    wmax = float(g.get("weight_max", t4.get("weight_max", 1.0)))
    decay = g.get("decay") or {}
    eps = float(decay.get("epsilon_prune", 0.0))
    if not (wmin < wmax):
        # fallback safety
        wmin, wmax = -1.0, 1.0
    if eps < 0:
        eps = 0.0
    return {"wmin": wmin, "wmax": wmax, "eps": eps}


def _round6(x: float) -> float:
    try:
        if not math.isfinite(x):
            return 0.0
        return round(float(x), 6)
    except Exception:
        return 0.0


def _clamp(x: float, lo: float, hi: float) -> float:
    try:
        if x < lo:
            return lo
        if x > hi:
            return hi
        return x
    except Exception:
        return x


def _edge_id(src: str, dst: str, rel: str) -> str:
    a, b = (str(src), str(dst))
    if a <= b:
        return f"{a}__{b}__{rel}"
    return f"{b}__{a}__{rel}"


def _sanitize_gel_for_write(gel: Any, ctx) -> Dict[str, Any]:
    bounds = _graph_bounds_from_cfg(ctx)
    wmin, wmax, eps = bounds["wmin"], bounds["wmax"], bounds["eps"]
    nodes_out: Dict[str, Any] = {}
    edges_out: Dict[str, Any] = {}

    # Nodes: accept dict or list of node dicts with an 'id'
    try:
        gnodes = (gel or {}).get("nodes", {}) if isinstance(gel, dict) else {}
        if isinstance(gnodes, dict):
            for nid, nd in gnodes.items():
                nodes_out[str(nid)] = nd
        elif isinstance(gnodes, list):
            for nd in gnodes:
                nid = str((nd or {}).get("id", ""))
                if nid:
                    nodes_out[nid] = nd
    except Exception:
        pass

    # Edges: accept dict keyed by id, or list of {src,dst,rel,weight,...}
    try:
        gedges = (gel or {}).get("edges", {}) if isinstance(gel, dict) else {}
        if isinstance(gedges, dict):
            items = gedges.items()
        elif isinstance(gedges, list):
            items = []
            for ed in gedges:
                if not isinstance(ed, dict):
                    continue
                src = str(ed.get("src", ""))
                dst = str(ed.get("dst", ""))
                rel = str(ed.get("rel", "coact"))
                eid = _edge_id(src, dst, rel)
                items.append((eid, ed))
        else:
            items = []
        for eid, ed in items:
            if not isinstance(ed, dict):
                continue
            src = str(ed.get("src", ""))
            dst = str(ed.get("dst", ""))
            rel = str(ed.get("rel", "coact"))
            w = _round6(_clamp(float(ed.get("weight", 0.0)), wmin, wmax))
            if abs(w) < eps:
                w = 0.0
            edges_out[_edge_id(src, dst, rel)] = {
                "src": src,
                "dst": dst,
                "rel": rel,
                "weight": w,
                "updated_at": ed.get("updated_at"),
                "attrs": ed.get("attrs", {}),
            }
    except Exception:
        pass

    # Meta: preserve if present; otherwise initialize deterministic containers
    meta_in = {}
    try:
        if isinstance(gel, dict):
            meta_in = gel.get("meta") or {}
    except Exception:
        meta_in = {}
    meta_out: Dict[str, Any] = {}
    # carry known fields if reasonably shaped
    for k in ("merges", "splits", "promotions"):
        v = meta_in.get(k)
        if isinstance(v, list):
            meta_out[k] = v
        else:
            meta_out[k] = []
    # counters
    try:
        ccount = int(meta_in.get("concept_nodes_count", 0))
    except Exception:
        ccount = 0
    meta_out["concept_nodes_count"] = ccount
    # always include edges_count for inspector/health
    meta_out["edges_count"] = len(edges_out)
    # bump meta schema to reflect v1.1 graph payload
    meta_out["schema"] = "v1.1"

    return {"nodes": nodes_out, "edges": edges_out, "meta": meta_out}


def _sanitize_gel_for_load(gel: Any, ctx) -> Dict[str, Any]:
    # For PR24, we preserve meta as well. Reuse write-path normalization and merge meta if provided.
    out = _sanitize_gel_for_write(gel, ctx)
    # If input had a meta dict, overlay non-list counters (e.g., last_update) without breaking our defaults
    try:
        meta_in = (gel or {}).get("meta") if isinstance(gel, dict) else None
        if isinstance(meta_in, dict):
            meta = out.get("meta", {})
            # carry forward last_update if present in legacy summaries
            if "last_update" in meta_in and meta.get("last_update") is None:
                meta["last_update"] = meta_in.get("last_update")
            out["meta"] = meta
    except Exception:
        pass
    return out


def _set_state_field(state, key: str, val: Any) -> None:
    if isinstance(state, dict):
        state[key] = val
    else:
        try:
            setattr(state, key, val)
        except Exception:
            pass


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

    # Include GEL (Graph Evolution Layer) section and schema tag; tolerate absence
    try:
        # Prefer runtime store at state.graph; fall back to state.gel for compatibility
        graph_state = None
        if isinstance(state, dict):
            graph_state = state.get("graph") or state.get("gel")
        else:
            graph_state = getattr(state, "graph", None) or getattr(state, "gel", None)
        payload["graph_schema_version"] = "v1.1"
        gel_out = _sanitize_gel_for_write(graph_state or {"nodes": {}, "edges": {}}, ctx)
        # Normalize edge keys to canonical "src→dst" order (unicode arrow), drop rel from key
        try:
            edges = gel_out.get("edges", {})
            if isinstance(edges, dict):
                new_edges = {}
                for k, rec in edges.items():
                    if not isinstance(rec, dict):
                        continue
                    src = str(rec.get("src")) if rec.get("src") is not None else None
                    dst = str(rec.get("dst")) if rec.get("dst") is not None else None
                    if src and dst:
                        key = f"{src}→{dst}" if src <= dst else f"{dst}→{src}"
                        rec = dict(rec)
                        rec["id"] = key
                        new_edges[key] = rec
                    else:
                        new_edges[k] = rec
                gel_out["edges"] = new_edges
                # keep meta edges_count in sync after rekeying
                try:
                    meta = gel_out.setdefault("meta", {})
                    meta["edges_count"] = len(new_edges)
                    if "schema" not in meta:
                        meta["schema"] = "v1.1"
                except Exception:
                    pass
        except Exception:
            pass
        payload["gel"] = gel_out

        # Back-compat summary: also include a compact `graph` meta with counts so
        # older inspectors can read sizes without duplicating the whole structure.
        try:
            nodes_cnt = len(gel_out.get("nodes", {}))
            edges_cnt = len(gel_out.get("edges", {}))
        except Exception:
            nodes_cnt = None
            edges_cnt = None
        payload["graph"] = {
            "nodes_count": nodes_cnt,
            "edges_count": edges_cnt,
            "meta": {"last_update": None},
        }
    except Exception:
        # Ensure keys exist even if sanitization fails
        payload.setdefault("graph_schema_version", "v1.1")
        payload.setdefault(
            "gel",
            {
                "nodes": {},
                "edges": {},
                "meta": {
                    "schema": "v1.1",
                    "merges": [],
                    "splits": [],
                    "promotions": [],
                    "concept_nodes_count": 0,
                    "edges_count": 0,
                },
            },
        )
        payload.setdefault(
            "graph", {"nodes_count": None, "edges_count": None, "meta": {"last_update": None}}
        )

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    return path


def load_latest_snapshot(ctx, state) -> Dict[str, Any]:
    """
    Load the latest snapshot from the configured directory.
    Returns {"loaded": bool, "path": str|None, "version_etag": str|None}.
    Tolerates both legacy and enriched schemas.
    """
    cfg = _get_cfg(ctx)
    dir_ = cfg["snapshot_dir"]
    path = _pick_latest_snapshot_path(dir_)

    # Ensure graph containers exist on state even if nothing loads
    empty_meta = {
        "schema": "v1.1",
        "merges": [],
        "splits": [],
        "promotions": [],
        "concept_nodes_count": 0,
        "edges_count": 0,
    }
    _set_state_field(state, "graph", {"nodes": {}, "edges": {}, "meta": dict(empty_meta)})
    _set_state_field(state, "gel", {"nodes": {}, "edges": {}, "meta": dict(empty_meta)})

    if not path or not os.path.isfile(path):
        return {"loaded": False, "path": None, "version_etag": None}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {"loaded": False, "path": path, "version_etag": None}

    ver = data.get("version_etag")
    if ver is not None:
        if isinstance(state, dict):
            state["version_etag"] = str(ver)
        else:
            setattr(state, "version_etag", str(ver))

    # Import store if present and compatible
    store = getattr(state, "store", None) if not isinstance(state, dict) else state.get("store")
    loaded_store = False
    snap_store = data.get("store")
    if store is not None and snap_store is not None:
        loaded_store = _import_store_from_snapshot(store, snap_store)

    # GEL restore (tolerant): prefer `gel`, fall back to `graph`; normalize keys to "src→dst"
    gel_out = {"nodes": {}, "edges": {}}
    try:
        snap_gel = data.get("gel")
        if snap_gel is None:
            snap_gel = data.get("graph")
        gel_out = _sanitize_gel_for_load(snap_gel or {"nodes": {}, "edges": {}}, ctx)
        try:
            edges = gel_out.get("edges", {})
            if isinstance(edges, dict):
                new_edges = {}
                for k, rec in edges.items():
                    if not isinstance(rec, dict):
                        continue
                    src = str(rec.get("src")) if rec.get("src") is not None else None
                    dst = str(rec.get("dst")) if rec.get("dst") is not None else None
                    if src and dst:
                        key = f"{src}→{dst}" if src <= dst else f"{dst}→{src}"
                        rec = dict(rec)
                        rec["id"] = key
                        new_edges[key] = rec
                    else:
                        new_edges[k] = rec
                gel_out["edges"] = new_edges
        except Exception:
            pass
        _set_state_field(state, "graph", gel_out)
        _set_state_field(state, "gel", gel_out)
    except Exception:
        # defaults are already set above
        pass

    loaded_flag = bool(loaded_store or ver is not None or gel_out.get("edges"))
    return {"loaded": loaded_flag, "path": path, "version_etag": ver}


# ---------------------------------------------------------------------------------
# PR34 helpers: canonical JSON, hash, and minimal delta/codec-aware reader
# ---------------------------------------------------------------------------------
def _canonical_json(obj) -> str:
    try:
        return json.dumps(obj, sort_keys=True, separators=(",", ":"))
    except Exception:
        # best-effort fallback
        return json.dumps(obj)


def _sha256_of(obj) -> str:
    return hashlib.sha256(_canonical_json(obj).encode("utf-8")).hexdigest()


def _read_text(path) -> str:
    """Read a snapshot file as text. Accepts str or Path-like.
    Transparently decompresses .zst when zstandard is available.
    """
    p = os.fspath(path)
    with open(p, "rb") as f:
        data = f.read()
    # If compressed, transparently decompress
    if p.endswith(".zst"):
        if _zstd is None:
            raise RuntimeError("zstandard module not available to read .zst snapshot")
        dctx = _zstd.ZstdDecompressor()
        data = dctx.decompress(data)
    return data.decode("utf-8")


def _read_header_payload(path):
    """
    Read 'header\npayload' JSON format used by PR34 snapshots.
    Returns (header_dict, payload_dict). If single-JSON is present, header=None, payload=that object.
    """
    raw = _read_text(path)
    # Try two-JSON format first
    parts = raw.splitlines()
    if len(parts) >= 2:
        try:
            header = json.loads(parts[0])
            payload = json.loads("\n".join(parts[1:]))
            if isinstance(header, dict):
                return header, payload
        except Exception:
            pass
    # Fallback: single JSON body
    try:
        body = json.loads(raw)
        return None, body
    except Exception:
        raise


def _find_snapshot_file(root: str, stem: str):
    """
    Return a path for either uncompressed .json or compressed .json.zst if it exists.
    """
    p_json = os.path.join(root, f"{stem}.json")
    p_zst = os.path.join(root, f"{stem}.json.zst")
    if os.path.isfile(p_json):
        return p_json
    if os.path.isfile(p_zst):
        return p_zst
    return None


# Public reader used by tests (compatible signature)
def read_snapshot(
    root: str = None, etag_to: str = None, baseline_dir: str = None, path: str = None, **kwargs
):
    """
    Minimal PR34 reader with safe fallback:
      * If 'path' points to a snapshot, read that (supports .full/.delta, .json/.json.zst).
      * Else, if 'etag_to' is given, try 'snapshot-{etag_to}.delta' first (to exercise fallback), then '...full'.
      * If delta is chosen but baseline is missing/mismatch, log a deterministic warning and fall back to full.
    Returns the payload dict (full snapshot body after reconstruction).
    """
    if path:
        # Allow direct path to either full or delta
        header, payload = _read_header_payload(path)
        if header and header.get("mode") == "delta":
            # Require baseline for reconstruction
            etag_to = header.get("etag_to")
            delta_of = header.get("delta_of")
            bdir = baseline_dir or os.path.dirname(path)
            base = _find_snapshot_file(bdir, f"snapshot-{delta_of}.full")
            if base:
                _, base_payload = _read_header_payload(base)
                from clematis.engine.util.snapshot_delta import (
                    apply_delta,
                )  # local import avoids hard dep in old paths

                return apply_delta(base_payload or {}, payload or {})
            # No baseline -> warn and return payload if it's actually full-like or empty dict
            logging.warning("SNAPSHOT_BASELINE_MISSING: delta_of=%s etag_to=%s", delta_of, etag_to)
            # Try sibling full as last resort
            sib_full = _find_snapshot_file(os.path.dirname(path), f"snapshot-{etag_to}.full")
            if sib_full:
                _, full_payload = _read_header_payload(sib_full)
                return full_payload or {}
            return {}
        # Full or legacy single JSON
        return payload or {}

    # Resolve root directory and filenames when given etag
    root = root or "."
    # Prefer delta first to exercise fallback-path deterministically (as per tests)
    delta_path = _find_snapshot_file(root, f"snapshot-{etag_to}.delta")
    if delta_path:
        header, payload = _read_header_payload(delta_path)
        delta_of = (header or {}).get("delta_of") if header else None
        bdir = baseline_dir or root
        base = _find_snapshot_file(bdir, f"snapshot-{delta_of}.full") if delta_of else None
        if base:
            _, base_payload = _read_header_payload(base)
            from clematis.engine.util.snapshot_delta import apply_delta

            return apply_delta(base_payload or {}, payload or {})
        # Baseline missing -> warn and try full fallback
        logging.warning("SNAPSHOT_BASELINE_MISSING: delta_of=%s etag_to=%s", delta_of, etag_to)
        full_path = _find_snapshot_file(root, f"snapshot-{etag_to}.full")
        if full_path:
            _, full_payload = _read_header_payload(full_path)
            return full_payload or {}
        return {}

    # No delta; try full directly
    full_path = _find_snapshot_file(root, f"snapshot-{etag_to}.full")
    if full_path:
        _, full_payload = _read_header_payload(full_path)
        return full_payload or {}
    # Nothing found
    return {}


# Writer helpers reuse _canonical_json, _find_snapshot_file, and _read_header_payload above.
def _write_lines(p: str, header: Dict[str, Any], body_json: str, *, codec: str, level: int) -> None:
    # If zstd requested but not installed, degrade deterministically to none (warn once per call site).
    if codec == "zstd" and _zstd is None:
        print("W[SNAPSHOT]: zstandard not installed; writing uncompressed", file=sys.stderr)
        codec = "none"
    payload = _canonical_json(header) + "\n" + body_json
    if codec == "zstd" and _zstd is not None:
        lvl = max(1, min(19, int(level or 3)))
        cctx = _zstd.ZstdCompressor(level=lvl)
        with open(p, "wb") as f:
            f.write(cctx.compress(payload.encode("utf-8")))
    else:
        with open(p, "w", encoding="utf-8") as f:
            f.write(payload)


def write_snapshot_auto(
    dir_path: str,
    *,
    etag_from: Optional[str],
    etag_to: str,
    payload: Dict[str, Any],
    compression: str = "none",  # "none" | "zstd"
    level: int = 3,
    delta_mode: bool = False,
) -> tuple[str, bool]:
    """
    Write a snapshot into `dir_path` with canonical header+payload lines.

    - delta_mode=True: if a matching full baseline {etag_from} is found, emit a delta file;
                       otherwise fall back to a full file and print a deterministic warning.
    - compression: "none" or "zstd"; if zstandard is not installed, falls back to "none" with a warning.

    Returns: (path, wrote_delta: bool)
    """
    os.makedirs(dir_path, exist_ok=True)
    codec = "zstd" if str(compression).lower() == "zstd" else "none"
    body_json = _canonical_json(payload)

    # Try delta when requested and possible
    if delta_mode and etag_from:
        base = _find_snapshot_file(dir_path, f"snapshot-{etag_from}.full")
        if base is not None:
            try:
                # Local import avoids hard dependency at import time
                from clematis.engine.util.snapshot_delta import compute_delta  # type: ignore
            except Exception:
                compute_delta = None  # type: ignore
            if compute_delta is not None:
                _, base_payload = _read_header_payload(base)
                base_obj = base_payload or {}
                delta_blob = compute_delta(base_obj, payload)  # type: ignore[misc]
                header = {
                    "schema": "snapshot:v1",
                    "mode": "delta",
                    "etag_from": etag_from,
                    "delta_of": etag_from,  # kept for tool compatibility
                    "etag_to": etag_to,
                    "codec": codec,
                    "level": int(level if codec == "zstd" else 0),
                }
                out_path = os.path.join(
                    dir_path, f"snapshot-{etag_to}.delta.json" + (".zst" if codec == "zstd" else "")
                )
                _write_lines(
                    out_path, header, _canonical_json(delta_blob), codec=codec, level=level
                )
                return out_path, True
        # Baseline unavailable or codec missing; fall back to full with deterministic warning
        print(
            "W[SNAPSHOT_BASELINE_MISSING]: delta requested but baseline full not found; writing full",
            file=sys.stderr,
        )

    # Fall back to full
    header = {
        "schema": "snapshot:v1",
        "mode": "full",
        "etag_to": etag_to,
        "codec": codec,
        "level": int(level if codec == "zstd" else 0),
    }
    out_path = os.path.join(
        dir_path, f"snapshot-{etag_to}.full.json" + (".zst" if codec == "zstd" else "")
    )
    _write_lines(out_path, header, body_json, codec=codec, level=level)
    return out_path, False
