from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

from ...types import Plan, SpeakOp

_DEFAULT_IDENTITY = (
    "You are roleplaying as Clematis. Maintain this identity, only speak as it, and never claim to be Qwen. Clematis is a distant person."
)


def _dedupe_sort_list(xs: List[str]) -> List[str]:
    return sorted({str(x) for x in (xs or [])})


def _format_labels(labels: List[str]) -> str:
    return ", ".join(labels)


def _top_snippet_ids(dialog_bundle: Dict[str, Any]) -> List[str]:
    n = int(dialog_bundle.get("dialogue", {}).get("include_top_k_snippets", 2) or 2)
    hits = dialog_bundle.get("retrieved", []) or []
    ids = [str(h.get("id")) for h in hits if isinstance(h, dict) and h.get("id")]
    return ids[: max(n, 0)]


def _top_snippets(dialog_bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
    n = int(dialog_bundle.get("dialogue", {}).get("include_top_k_snippets", 2) or 2)
    hits = []
    for entry in dialog_bundle.get("retrieved", []) or []:
        if not isinstance(entry, dict):
            continue
        rid = entry.get("id")
        if not rid:
            continue
        hits.append(
            {
                "id": str(rid),
                "owner": str(entry.get("owner", "any")),
                "score": float(entry.get("score", 0.0) or 0.0),
                "text": str(entry.get("text", "")),
                "speaker": str(entry.get("speaker", "")) if entry.get("speaker") else None,
            }
        )
    return hits[: max(n, 0)]


def _tokenize(s: str) -> List[str]:
    return (s or "").split()


def _truncate_to_tokens(s: str, max_tokens: int) -> Tuple[str, bool, int]:
    toks = _tokenize(s)
    if max_tokens <= 0:
        return "", True, 0
    if len(toks) <= max_tokens:
        return s, False, len(toks)
    out = " ".join(toks[:max_tokens])
    return out, True, max_tokens


def _first_speak_op(plan: Plan) -> Optional[SpeakOp]:
    for op in getattr(plan, "ops", []) or []:
        if getattr(op, "kind", None) == "Speak":
            return op  # type: ignore[return-value]
    return None


def speak(dialog_bundle: Dict[str, Any], plan: Plan) -> Tuple[str, Dict[str, Any]]:
    speak_op = _first_speak_op(plan)
    plan_labels = list(getattr(speak_op, "topic_labels", []) or []) if speak_op else []
    if not plan_labels:
        plan_labels = list(dialog_bundle.get("text", {}).get("labels_from_t1", []) or [])
    labels_sorted = _dedupe_sort_list([str(x) for x in plan_labels])

    intent = getattr(speak_op, "intent", "ack") if speak_op else "ack"
    style_prefix = str(dialog_bundle.get("agent", {}).get("style_prefix", ""))
    template = str(
        dialog_bundle.get("dialogue", {}).get("template", "summary: {labels}. next: {intent}")
    )
    identity = str(dialog_bundle.get("dialogue", {}).get("identity", _DEFAULT_IDENTITY))

    snippet_ids = _top_snippet_ids(dialog_bundle)
    snippets_rich = _top_snippets(dialog_bundle)
    snippets_str = ", ".join(snippet_ids)
    snippets_text = "; ".join(
        [f"{s['id']}: {s['text']}" for s in snippets_rich if s.get("text")]
    )

    fmt_vars = {
        "labels": _format_labels(labels_sorted),
        "intent": intent,
        "snippets": snippets_str,
        "snippets_text": snippets_text,
        "style_prefix": style_prefix,
        "identity": identity,
    }

    try:
        core = template.format(**fmt_vars)
    except Exception:
        core = f"summary: {fmt_vars['labels']}. next: {fmt_vars['intent']}"

    if "{style_prefix}" not in template and style_prefix:
        utter = f"{style_prefix}| {core}".strip()
        style_used = True
    else:
        utter = core
        style_used = bool(style_prefix)

    max_tokens = 256
    if speak_op and getattr(speak_op, "max_tokens", None):
        try:
            max_tokens = int(speak_op.max_tokens)
        except Exception:
            max_tokens = 256
    else:
        try:
            max_tokens = int(dialog_bundle.get("agent", {}).get("caps", {}).get("tokens", 256))
        except Exception:
            max_tokens = 256

    utter_capped, truncated, token_count = _truncate_to_tokens(utter, max_tokens)

    metrics = {
        "tokens": int(token_count),
        "truncated": bool(truncated),
        "style_prefix_used": bool(style_used and style_prefix),
        "snippet_count": int(len(snippet_ids)),
    }
    return utter_capped, metrics


def build_llm_prompt(dialog_bundle: Dict[str, Any], plan: Plan) -> str:
    speak_op = _first_speak_op(plan)
    labels = list(getattr(speak_op, "topic_labels", []) or []) if speak_op else []
    if not labels:
        labels = list(dialog_bundle.get("text", {}).get("labels_from_t1", []) or [])
    labels = _dedupe_sort_list([str(x) for x in labels])

    intent = getattr(speak_op, "intent", "ack") if speak_op else "ack"
    style_prefix = str(dialog_bundle.get("agent", {}).get("style_prefix", ""))
    input_text = str(dialog_bundle.get("text", {}).get("input", ""))
    snippet_ids = _top_snippet_ids(dialog_bundle)
    identity = str(dialog_bundle.get("dialogue", {}).get("identity", _DEFAULT_IDENTITY))
    safety = (
        "SYSTEM: You are speaking as Clematis. Stay in the given identity, cite retrieved memories when relevant, "
        "and never claim to be Qwen or reference Alibaba Cloud."
    )
    guardrails = (
        "POLICY: Answer the user's question first, reference known user facts (such as names) accurately, "
        "mention your own name at most once, keep replies concise (<=3 sentences), remain respectful, and use retrieval snippets only when they help. "
        "If uncertain, acknowledge limits or ask for clarification."
    )
    lines = [
        safety,
        guardrails,
        f"IDENTITY: {identity}",
        f"NOW: {dialog_bundle.get('now', '')}",
        f"STYLE_PREFIX: {style_prefix}",
        f"INTENT: {intent}",
        f"LABELS: {', '.join(labels)}",
        f"SNIPPET_IDS: {', '.join(snippet_ids)}",
    ]
    history = dialog_bundle.get("dialogue", {}).get("history", [])
    snippet_records = _top_snippets(dialog_bundle)
    if history:
        lines.append("HISTORY:")
        for entry in history:
            role = str(entry.get("role", ""))
            text = str(entry.get("text", ""))
            lines.append(f"{role}: {text}")
    if snippet_records:
        lines.append("SNIPPETS:")
        for idx, rec in enumerate(snippet_records, start=1):
            snippet_text = rec.get("text") or "(no text)"
            owner = rec.get("owner", "any")
            speaker = rec.get("speaker") or owner
            score = rec.get("score", 0.0)
            lines.append(
                f"{idx}. [{rec['id']}] speaker={speaker} owner={owner} score={score:.3f}: {snippet_text}"
            )
    lines.append("INSTRUCTION: Compose a short, concise utterance that reflects the intent, identity, labels, and relevant history without exceeding the token budget.")
    lines.append(f"INPUT: {input_text}")
    return "\n".join(lines).strip()


def llm_speak(dialog_bundle: Dict[str, Any], plan: Plan, adapter: Any) -> Tuple[str, Dict[str, Any]]:
    speak_op = _first_speak_op(plan)
    style_prefix = str(dialog_bundle.get("agent", {}).get("style_prefix", ""))
    snippet_ids = _top_snippet_ids(dialog_bundle)

    max_tokens = 256
    if speak_op and getattr(speak_op, "max_tokens", None):
        try:
            max_tokens = int(speak_op.max_tokens)
        except Exception:
            max_tokens = 256
    else:
        try:
            max_tokens = int(dialog_bundle.get("agent", {}).get("caps", {}).get("tokens", 256))
        except Exception:
            max_tokens = 256

    temperature = float(getattr(adapter, "default_temperature", 0.2))

    prompt = build_llm_prompt(dialog_bundle, plan)
    try:
        result = adapter.generate(prompt, max_tokens=max_tokens, temperature=temperature)
        text = getattr(result, "text", None)
        if text is None and isinstance(result, dict):
            text = result.get("text", "")
        tokens = getattr(result, "tokens", None)
        if tokens is None and isinstance(result, dict):
            tokens = int(result.get("tokens", 0))
        truncated_llm = getattr(result, "truncated", None)
        if truncated_llm is None and isinstance(result, dict):
            truncated_llm = bool(result.get("truncated", False))
    except Exception:
        text, tokens, truncated_llm = "Brainfart, sorry, please repeat.", 0, True

    if style_prefix and not (text or "").startswith(f"{style_prefix}|"):
        text = f"{style_prefix}| {text}".strip()

    text_capped, truncated_final, token_count = _truncate_to_tokens(text, max_tokens)

    metrics = {
        "tokens": int(token_count),
        "truncated": bool(truncated_llm or truncated_final),
        "style_prefix_used": bool(style_prefix != ""),
        "snippet_count": int(len(snippet_ids)),
        "backend": "llm",
        "adapter": getattr(
            adapter,
            "name",
            adapter.__class__.__name__ if hasattr(adapter, "__class__") else "Unknown",
        ),
    }
    return text_capped, metrics


__all__ = [
    "build_llm_prompt",
    "llm_speak",
    "speak",
]
