# clematis/engine/stages/t3/reflect.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import re

# --- Local, deterministic text ops (no RNG, no locale dependence) ---

_PUNCT_RE = re.compile(r"[^\w\s]+", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+", flags=re.UNICODE)


def _normalize(text: str) -> str:
    """
    Deterministic normalization:
    - lower-case
    - strip punctuation to spaces (keep unicode letters/digits/_)
    - collapse whitespace to single spaces
    """
    if not text:
        return ""
    t = text.lower()
    t = _PUNCT_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t).strip()
    return t


def _truncate_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0 or not text:
        return ""
    toks = text.split(" ")
    if len(toks) <= max_tokens:
        return " ".join(toks)
    return " ".join(toks[:max_tokens])


def _build_summary_rulebased(
    utter: str, snippets: List[str], max_tokens: int
) -> Tuple[str, int]:
    parts: List[str] = []
    if utter:
        parts.append(_normalize(utter))
    for s in snippets:
        if s:
            parts.append(_normalize(s))
    raw = " ".join(p for p in parts if p)
    summary = _truncate_tokens(raw, max_tokens)
    token_count = 0 if not summary else len(summary.split(" "))
    return summary, token_count


# --- Public API data classes ---

@dataclass(frozen=True)
class ReflectionBundle:
    """
    Inputs to reflection. All fields must be deterministic snapshots/values.
    - ctx: should expose deterministic fields (e.g., agent_id, now_iso).
    - state_view: read-only selection of state (opaque here).
    - plan: executed plan; may carry a .reflection flag (opaque here).
    - utter: final utterance string produced this turn.
    - snippets: retrieval snippets chosen for context (top-K already filtered).
    """
    ctx: Any
    state_view: Any
    plan: Any
    utter: str
    snippets: List[str]


@dataclass(frozen=True)
class ReflectionResult:
    """
    Pure function output from `reflect()`. No side effects performed here.
    - summary: deterministic text built from utterance + snippets.
    - memory_entries: at most ops_reflection entries to be written later (PR79/PR80).
    - metrics: minimal counters useful for logging in later PRs; stable keys.
    """
    summary: str
    memory_entries: List[Dict[str, Any]]
    metrics: Dict[str, Any]


# --- Public function ---

Embedder = Callable[[str], List[float]]


def reflect(
    bundle: ReflectionBundle,
    cfg: Dict[str, Any],
    embedder: Optional[Embedder] = None,
) -> ReflectionResult:
    """
    Deterministic rule-based reflection:
    - Concatenate normalized utterance + top-K snippets.
    - Truncate to summary_tokens (whitespace tokens).
    - Produce up to ops_reflection memory entries with tags=["reflection"], kind="summary".
    - If embed=true and an embedder is provided, attach vec; failures set vec=None.

    No logging, I/O, or wall-time sampling. All timestamps/IDs must come from bundle.ctx.
    """
    t3_cfg = (cfg.get("t3") or {})
    ref_cfg = (t3_cfg.get("reflection") or {})
    sch_cfg = (cfg.get("scheduler") or {})
    budgets = (sch_cfg.get("budgets") or {})

    summary_tokens = int(ref_cfg.get("summary_tokens", 128))
    topk_snippets = int(ref_cfg.get("topk_snippets", 3))
    do_embed = bool(ref_cfg.get("embed", True))
    ops_cap = int(budgets.get("ops_reflection", 5))

    snippets = (bundle.snippets or [])[:topk_snippets]
    summary, token_count = _build_summary_rulebased(bundle.utter or "", snippets, summary_tokens)

    # Agent/owner and timestamp come from ctx (must be deterministic; orchestrator supplies in PR79)
    owner = getattr(bundle.ctx, "agent_id", None) or getattr(bundle.ctx, "agent", None) or "unknown"
    # Prefer an ISO string stable in tests; fallback to a stable scalar if present
    ts = getattr(bundle.ctx, "now_iso", None) or getattr(bundle.ctx, "now", None) or "0000-00-00T00:00:00Z"

    entry: Dict[str, Any] = {
        "owner": owner,
        "ts": ts,
        "text": summary,
        "tags": ["reflection"],
        "kind": "summary",
    }

    if do_embed and embedder is not None and summary:
        try:
            vec = embedder(summary)
        except Exception:
            vec = None
        entry["vec"] = vec

    memory_entries = [entry] if ops_cap > 0 else []
    metrics = {
        "backend": "rulebased",
        "tokens": token_count,
        "used_topk": len(snippets),
    }
    return ReflectionResult(summary=summary, memory_entries=memory_entries[:ops_cap], metrics=metrics)

def try_get_deterministic_embedder() -> Optional[Embedder]: # unused for now, for PR79 or onwards, added in PR78
    """
    Best-effort adapter loader. Must not raise; returns None if unavailable.
    """
    try:
        # Example path only; defer to your actual adapter location in PR84.
        from clematis.adapters.embeddings import DeterministicEmbeddingAdapter  # type: ignore
        adapter = DeterministicEmbeddingAdapter(dim=32)
        return lambda text: adapter.encode([text])[0]
    except Exception:
        return None
