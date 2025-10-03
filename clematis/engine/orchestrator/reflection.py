# -*- coding: utf-8 -*-
# Deterministic write path for reflection entries (PR80)
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    # Prefer a stable adapter path if you have one
    from clematis.memory.index import InMemoryIndex  # type: ignore
except Exception:  # pragma: no cover
    InMemoryIndex = None  # type: ignore


@dataclass(frozen=True)
class WriteReport:
    ops_attempted: int
    ops_written: int
    errors: List[str]
    reason: Optional[str]
    index_kind: Optional[str]
    ts_iso: Optional[str]


def _now_iso_from_ctx(ctx) -> str:
    """
    Deterministic timestamp from the turn clock.
    Assumes ctx.now_ms is the fixed turn clock in milliseconds.
    """
    # Avoid importing datetime to keep this path minimal; if you already use it elsewhere, feel free.
    # We derive a simple UTC ISO-8601 string deterministically.
    ms = int(getattr(ctx, "now_ms", 0))
    # Convert to seconds, then format manually to avoid locale
    # If you already have ctx.now_iso, prefer returning it directly to keep consistency.
    if hasattr(ctx, "now_iso") and isinstance(ctx.now_iso, str):
        return ctx.now_iso  # preferred if provided by orchestrator
    # Fallback deterministic ISO (epoch-based)
    # NOTE: Exact shape isn't critical as long as it is stable within a run configuration.
    return f"1970-01-01T00:00:{ms // 1000:02d}.{ms % 1000:03d}Z"


def _choose_index(state, cfg_root) -> Optional[Any]:
    """
    Picks the active memory index deterministically.
    Contract: returns an object with .add(MemoryEpisodeDict) -> None
    """
    # Prefer whatever you already attach to state (e.g., state.memory_index).
    idx = getattr(state, "memory_index", None)
    if idx is not None:
        return idx

    # Optional: build a new in-memory index if none attached and policy allows.
    # If constructing an index here violates your lifecycle, return None to fail-soft.
    return None


def _episode_id(ctx, slot: int, text: str) -> str:
    """
    Deterministic episode ID: stable across reruns with same inputs.
    """
    h = hashlib.sha256()
    # Include agent_id, turn_id, slot, and text. Do NOT include wall clock or non-deterministic fields.
    agent = str(getattr(ctx, "agent_id", "unknown"))
    turn = str(getattr(ctx, "turn_id", "0"))
    h.update(agent.encode("utf-8"))
    h.update(b"|")
    h.update(turn.encode("utf-8"))
    h.update(b"|")
    h.update(str(slot).encode("utf-8"))
    h.update(b"|")
    h.update(text.encode("utf-8"))
    # Trim to short suffix for readability
    return f"refl-{turn}-{agent}-{slot}-{h.hexdigest()[:12]}"


def _normalize_entry(
    base: Dict[str, Any],
    *,
    owner: str,
    ts_iso: str,
    episode_id: str,
    maybe_vec: Optional[List[float]],
) -> Dict[str, Any]:
    """
    Produce the exact MemoryEpisode dict shape your MemoryIndex expects.
    Keep keys stable and minimal; avoid incidental churn.
    """
    # Base (from reflect result) should at least have 'text' and 'tags'.
    text = str(base.get("text", "")).strip()
    tags = list(base.get("tags", []))
    kind = str(base.get("kind", "summary"))

    ep: Dict[str, Any] = {
        "id": episode_id,
        "owner": owner,  # e.g., "agent"
        "ts": ts_iso,    # deterministic
        "text": text,
        "tags": tags or ["reflection"],
        "kind": kind,
    }
    if maybe_vec is not None:
        ep["vec_full"] = maybe_vec  # align with your MemoryEpisode schema; fp32 list

    # Keep a canonical field order by re-creating dict (optional, but helps file-based backends)
    ordered = {
        "id": ep["id"],
        "owner": ep["owner"],
        "ts": ep["ts"],
        "kind": ep["kind"],
        "tags": ep["tags"],
        "text": ep["text"],
    }
    if "vec_full" in ep:
        ordered["vec_full"] = ep["vec_full"]
    return ordered


def write_reflection_entries(ctx, state, cfg_root, result) -> WriteReport:
    """
    Deterministically persist reflection entries produced by reflect(...).
    Never raises; on failure returns a report with errors populated.
    """
    errors: List[str] = []
    index = None
    index_kind: Optional[str] = None

    # No result or no entries → no-op
    if result is None or not getattr(result, "memory_entries", None):
        return WriteReport(
            ops_attempted=0, ops_written=0, errors=errors, reason=None,
            index_kind=None, ts_iso=None
        )

    attempted_total = len(result.memory_entries)
    entries = list(result.memory_entries)

    # Enforce ops cap one more time here (belt-and-suspenders)
    ops_cap = int(
        cfg_root.get("scheduler", {}).get("budgets", {}).get("ops_reflection", 0)
    )
    if ops_cap <= 0:
        return WriteReport(
            ops_attempted=attempted_total, ops_written=0, errors=errors,
            reason=f"ops_cap={ops_cap}", index_kind=None, ts_iso=None
        )
    if len(entries) > ops_cap:
        entries = entries[:ops_cap]

    ts_iso = _now_iso_from_ctx(ctx)

    try:
        index = _choose_index(state, cfg_root)
        if index is None:
            return WriteReport(
                ops_attempted=attempted_total, ops_written=0,
                errors=["memory_index_missing"],
                reason="index_missing",
                index_kind=None,
                ts_iso=ts_iso,
            )
        index_kind = getattr(index, "kind", None) or (
            "inmemory" if InMemoryIndex and isinstance(index, InMemoryIndex) else None
        )
    except Exception as e:  # fail-soft
        errors.append(f"index_select_error:{type(e).__name__}")
        return WriteReport(
            ops_attempted=attempted_total, ops_written=0, errors=errors,
            reason="index_select_error", index_kind=None, ts_iso=ts_iso
        )

    owner = "agent"  # prefer agent-owned reflections; adjust if you support "world"
    written = 0
    for i, base in enumerate(entries):
        try:
            # If reflect() produced an embedding vector, prefer it; else None.
            maybe_vec = base.get("vec_full") if isinstance(base, dict) else None
            ep_id = _episode_id(ctx, i, str(base.get("text", "")))
            ep = _normalize_entry(base, owner=owner, ts_iso=ts_iso, episode_id=ep_id, maybe_vec=maybe_vec)

            # Compatible call: some indexes accept dict, others kwargs; try dict first.
            try:
                index.add(ep)  # expected API per your MemoryIndex contract
            except TypeError:
                index.add(**ep)  # fallback for older signatures

            written += 1
        except Exception as e:
            errors.append(f"add_error[{i}]:{type(e).__name__}")

    return WriteReport(
        ops_attempted=attempted_total,
        ops_written=written,
        errors=errors,
        reason=None if not errors else "partial_failure",
        index_kind=index_kind,
        ts_iso=ts_iso,
    )
