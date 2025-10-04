from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from clematis.adapters.embeddings import DeterministicEmbeddingAdapter
from clematis.adapters.llm import FixtureLLMAdapter, LLMAdapterError

_PUNCT_RE = re.compile(r"[^\w\s]+", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+", flags=re.UNICODE)

Embedder = Callable[[str], List[float]]


@dataclass(frozen=True)
class ReflectionBundle:
    ctx: Any
    state_view: Any
    plan: Any
    utter: str
    snippets: List[str]


@dataclass(frozen=True)
class ReflectionResult:
    summary: str
    memory_entries: List[Dict[str, Any]]
    metrics: Dict[str, Any]


class FixtureMissingError(RuntimeError):
    """Raised when an LLM fixture lookup fails for the deterministic backend."""


_EMBED_ADAPTER = DeterministicEmbeddingAdapter(dim=32)


def _normalize(text: str, *, keep_punct: bool = False) -> str:
    if not text:
        return ""
    t = str(text)
    t = _WS_RE.sub(" ", t).strip()
    if keep_punct:
        return t
    t = t.lower()
    t = _PUNCT_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t)
    return t.strip()


def _truncate_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0 or not text:
        return ""
    toks = text.split(" ")
    if len(toks) <= max_tokens:
        return " ".join(toks)
    return " ".join(toks[:max_tokens])


def _owner_and_ts(ctx: Any) -> tuple[str, Any]:
    owner = getattr(ctx, "agent_id", None) or getattr(ctx, "agent", None) or "unknown"
    ts = (
        getattr(ctx, "now_iso", None)
        or getattr(ctx, "now", None)
        or getattr(ctx, "now_ms", None)
        or "0000-00-00T00:00:00Z"
    )
    return str(owner), ts


def _maybe_embed(summary: str, do_embed: bool, embedder: Optional[Embedder]) -> Optional[List[float]]:
    if not do_embed or not summary:
        return None
    if embedder is not None:
        try:
            vec = embedder(summary)
        except Exception:
            return None
        if vec is None:
            return None
        if hasattr(vec, "tolist"):
            vec = vec.tolist()  # type: ignore[assignment]
        return list(vec)
    try:
        arr = _EMBED_ADAPTER.encode([summary])[0]
    except Exception:
        return None
    return arr.tolist() if hasattr(arr, "tolist") else list(arr)


def reflect(
    bundle: ReflectionBundle,
    cfg_root: Dict[str, Any],
    embedder: Optional[Embedder] = None,
) -> ReflectionResult:
    cfg_root = cfg_root or {}
    t3_cfg = cfg_root.get("t3") or {}
    reflection_cfg = t3_cfg.get("reflection") or {}
    backend = str(reflection_cfg.get("backend", "rulebased")).lower()

    budgets = (cfg_root.get("scheduler") or {}).get("budgets") or {}
    try:
        ops_cap = int(budgets.get("ops_reflection", 5))
    except Exception:
        ops_cap = 5
    if ops_cap < 0:
        ops_cap = 0

    if backend == "llm":
        return _reflect_llm(bundle, cfg_root, reflection_cfg, ops_cap, embedder)
    if backend != "rulebased":
        raise ValueError(f"Unknown reflection backend: {backend}")
    return _reflect_rulebased(bundle, reflection_cfg, ops_cap, embedder)


def _reflect_rulebased(
    bundle: ReflectionBundle,
    reflection_cfg: Dict[str, Any],
    ops_cap: int,
    embedder: Optional[Embedder],
) -> ReflectionResult:
    k = int(reflection_cfg.get("topk_snippets", 3))
    limit = int(reflection_cfg.get("summary_tokens", 128))
    do_embed = bool(reflection_cfg.get("embed", True))

    snippets = (bundle.snippets or [])[:k]
    snippets = [s for s in snippets if isinstance(s, str)]

    normalized_pieces = [_normalize(bundle.utter or "")]
    normalized_pieces.extend(_normalize(s or "") for s in snippets)
    raw = " ".join(p for p in normalized_pieces if p)
    summary = _truncate_tokens(raw, limit)
    summary_len = len(summary.split()) if summary else 0

    owner, ts = _owner_and_ts(bundle.ctx)
    entry: Dict[str, Any] = {
        "owner": owner,
        "ts": ts,
        "text": summary,
        "tags": ["reflection"],
        "kind": "summary",
    }

    vec = _maybe_embed(summary, do_embed, embedder)
    if vec is not None:
        entry["vec_full"] = vec

    entries = [entry] if ops_cap > 0 else []

    metrics = {
        "backend": "rulebased",
        "summary_len": summary_len,
        "used_topk": len(snippets),
        "embed": do_embed,
        "ops_cap": ops_cap,
    }
    return ReflectionResult(summary=summary, memory_entries=entries, metrics=metrics)


def _reflect_llm(
    bundle: ReflectionBundle,
    cfg_root: Dict[str, Any],
    reflection_cfg: Dict[str, Any],
    ops_cap: int,
    embedder: Optional[Embedder],
) -> ReflectionResult:
    k = int(reflection_cfg.get("topk_snippets", 3))
    limit = int(reflection_cfg.get("summary_tokens", 128))
    do_embed = bool(reflection_cfg.get("embed", True))

    t3_cfg = cfg_root.get("t3") or {}
    llm_cfg = t3_cfg.get("llm") or {}
    fixtures_cfg = llm_cfg.get("fixtures") or {}
    if not fixtures_cfg.get("enabled", False):
        raise ValueError("LLM backend requires fixtures.enabled=true")

    fixtures_path = fixtures_cfg.get("path")
    if not isinstance(fixtures_path, str) or not fixtures_path.strip():
        raise FixtureMissingError("LLM fixtures path must be a non-empty string")

    snippets_src = (bundle.snippets or [])[:k]
    norm_snippets = [_normalize(s, keep_punct=True) for s in snippets_src]

    prompt_obj = {
        "agent": str(getattr(bundle.ctx, "agent_id", "unknown")),
        "plan_reflection": bool(getattr(bundle.plan, "reflection", False)),
        "snippets": norm_snippets,
        "summary_tokens": limit,
        "task": "reflect_summary",
        "turn": int(getattr(bundle.ctx, "turn_id", 0) or 0),
        "utter": _normalize(bundle.utter or "", keep_punct=True),
        "version": 1,
    }
    prompt_json = json.dumps(prompt_obj, sort_keys=True, separators=(",", ":"))
    fixture_key = hashlib.sha256(prompt_json.encode("utf-8")).hexdigest()[:12]

    try:
        adapter = FixtureLLMAdapter(fixtures_path)
    except LLMAdapterError as exc:
        raise FixtureMissingError(f"Fixture adapter init failed: {exc}") from exc

    max_tokens = max(0, limit)
    try:
        res = adapter.generate(prompt_json, max_tokens=max_tokens, temperature=0.0)
    except LLMAdapterError as exc:
        raise FixtureMissingError(f"Fixture lookup failed: {exc}") from exc

    text = getattr(res, "text", "") if res is not None else ""
    if not text:
        raise FixtureMissingError(f"Missing fixture for key={fixture_key}")

    summary = _truncate_tokens(text, limit)
    summary_len = len(summary.split()) if summary else 0

    owner, ts = _owner_and_ts(bundle.ctx)
    entry: Dict[str, Any] = {
        "owner": owner,
        "ts": ts,
        "text": summary,
        "tags": ["reflection"],
        "kind": "summary",
    }

    vec = _maybe_embed(summary, do_embed, embedder)
    if vec is not None:
        entry["vec_full"] = vec

    entries = [entry] if ops_cap > 0 else []

    metrics = {
        "backend": "llm-fixture",
        "summary_len": summary_len,
        "fixture_key": fixture_key,
        "used_topk": len(norm_snippets),
        "embed": do_embed,
        "ops_cap": ops_cap,
    }
    return ReflectionResult(summary=summary, memory_entries=entries, metrics=metrics)
