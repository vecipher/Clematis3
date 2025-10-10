from __future__ import annotations
from typing import Dict, Any, List, Tuple, Callable, Optional

import os
import json
from pathlib import Path

from ...policy.json_schemas import PLANNER_V1
from ...policy.sanitize import parse_and_validate

from ...types import Plan, SpeakOp, EditGraphOp, RequestRetrieveOp
from . import bundle as _bundle_mod
from . import dialogue as _dialogue_mod

# PR8/M3-08: Optional LLM adapter types (duck-typed if unavailable)
try:  # runtime optional
    from ....adapters.llm import (
        LLMAdapter,
        LLMResult,
        FixtureLLMAdapter,
        QwenLLMAdapter,
        LLMAdapterError,
    )  # type: ignore
except Exception:  # pragma: no cover
    LLMAdapter = object  # type: ignore
    LLMResult = object  # type: ignore
    FixtureLLMAdapter = None  # type: ignore
    QwenLLMAdapter = None  # type: ignore

    class LLMAdapterError(Exception):  # type: ignore
        pass

# PR4: Pure T3 bundle assembly (no policy/RAG/dialogue yet)
# Deterministic: all lists sorted; explicit caps; empty defaults when absent.

BUNDLE_VERSION = _bundle_mod.BUNDLE_VERSION


def _iso_now(ctx) -> str:
    return _bundle_mod.iso_now(ctx)


def _get_cfg_caps(ctx) -> Dict[str, int]:
    return _bundle_mod.cfg_caps(ctx)


def _get_agent_info(ctx) -> Dict[str, Any]:
    return _bundle_mod.agent_info(ctx)


def _world_hot_labels(state) -> Tuple[List[str], int]:
    return _bundle_mod.world_hot_labels(state)


def _extract_t1_touched_nodes(t1, cap: int = 32) -> List[Dict[str, Any]]:
    return _bundle_mod.extract_t1_touched_nodes(t1, cap=cap)


def _extract_t1_metrics(t1) -> Dict[str, Any]:
    return _bundle_mod.extract_t1_metrics(t1)


def _extract_labels_from_t1(t1) -> List[str]:
    return _bundle_mod.extract_labels_from_t1(t1)


def _extract_t2_retrieved(t2, k_retrieval: int) -> List[Dict[str, Any]]:
    return _bundle_mod.extract_t2_retrieved(t2, k_retrieval)


def _extract_t2_metrics(t2) -> Dict[str, Any]:
    return _bundle_mod.extract_t2_metrics(t2)


def _cfg_snapshot(ctx) -> Dict[str, Any]:
    return _bundle_mod.cfg_snapshot(ctx)


def make_plan_bundle(ctx, state, t1, t2) -> Dict[str, Any]:
    """Assemble the compact, deterministic T3 bundle (facade)."""
    from .bundle import assemble_bundle

    return assemble_bundle(ctx, state, t1, t2)


# --- PR7: Dialogue bundle (deterministic, pure)
DIALOG_BUNDLE_VERSION = "t3-dialog-bundle-v1"
_DEFAULT_TEMPLATE = "{style_prefix}| summary: {labels}. next: {intent}"
_DEFAULT_IDENTITY = (
    "You are Clematis, the knowledge gardener. Maintain this persona, remember user-provided facts accurately, and never claim to be Qwen."
)
_HISTORY_WINDOW = 6


def _load_template_from_file(path_str: str) -> Optional[str]:
    try:
        path = Path(path_str)
        if not path.exists():
            return None
        suffix = path.suffix.lower()
        if suffix in {".json", ".jsonl"}:
            with path.open("r", encoding="utf-8") as fh:
                if suffix == ".jsonl":
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        rec = json.loads(line)
                        break
                    else:
                        return None
                else:
                    rec = json.load(fh)
            template = rec.get("template")
            if not template:
                return None
            return str(template)
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return None


def _resolve_dialogue_template(dialogue_cfg: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    template = str(dialogue_cfg.get("template", _DEFAULT_TEMPLATE))
    template_file_cfg = dialogue_cfg.get("template_file")
    template_file_path: Optional[str] = None
    if template_file_cfg:
        template_file_path = str(template_file_cfg)
        loaded = _load_template_from_file(template_file_path)
        if loaded:
            template = loaded
        else:
            template_file_path = None
    return template, template_file_path


def make_dialog_bundle(ctx, state, t1, t2, plan=None) -> Dict[str, Any]:
    """Assemble a deterministic dialogue bundle from PR4 bundle + Plan.
    Pure: no I/O. Uses only ctx/state/t1/t2/plan and config.
    """
    base = make_plan_bundle(ctx, state, t1, t2)
    # Dialogue config snapshot
    cfg_t3 = (
        (getattr(ctx, "cfg", {}) or {}).get("t3", {})
        if isinstance(getattr(ctx, "cfg", {}), dict)
        else {}
    )
    dialogue_cfg = {}
    if isinstance(cfg_t3, dict):
        dialogue_cfg = cfg_t3.get("dialogue", {}) or {}
    template, template_file = _resolve_dialogue_template(dialogue_cfg if isinstance(dialogue_cfg, dict) else {})
    include_top_k = int(dialogue_cfg.get("include_top_k_snippets", 2) or 2)
    identity = str(dialogue_cfg.get("identity", _DEFAULT_IDENTITY))
    if isinstance(state, dict):
        history_raw = list(state.get("_chat_history", []) or [])
    else:
        history_raw = list(getattr(state, "_chat_history", []) or [])
    history = history_raw[-_HISTORY_WINDOW:]

    # Retrieved already sorted/capped by make_plan_bundle
    retrieved = list(base.get("t2", {}).get("retrieved", []) or [])

    return {
        "version": DIALOG_BUNDLE_VERSION,
        "now": base["now"],
        "agent": base["agent"],
        "text": base["text"],
        "retrieved": retrieved,
        "dialogue": {
            "template": template,
            "include_top_k_snippets": include_top_k,
            "template_file": template_file,
            "identity": identity,
            "history": history,
        },
    }


# --- PR5: Rule-based policy (no RAG) ---
# Deterministic `deliberate(bundle) -> Plan` using explicit thresholds and caps.

def _policy_thresholds(bundle: Dict[str, Any]) -> Dict[str, float]:
    from . import policy as _policy_mod

    return _policy_mod._policy_thresholds(bundle)


def _topic_labels_from_bundle(bundle: Dict[str, Any], cap: int = 5) -> List[str]:
    from . import policy as _policy_mod

    return _policy_mod._topic_labels_from_bundle(bundle, cap)


def _edit_nodes_from_bundle(
    bundle: Dict[str, Any], eps_edit: float, cap_nodes: int
) -> List[Dict[str, Any]]:
    from . import policy as _policy_mod

    return _policy_mod._edit_nodes_from_bundle(bundle, eps_edit, cap_nodes)


def deliberate(bundle: Dict[str, Any]) -> Plan:
    from . import policy as _policy_mod

    return _policy_mod.deliberate(bundle)


# --- PR6: One-shot RAG refinement (pure) ---
# Orchestrator provides a deterministic retrieve_fn; this function remains pure.
RetrieveFn = Callable[[Dict[str, Any]], Dict[str, Any]]


def _first_request_retrieve_payload(plan: Plan) -> Optional[Dict[str, Any]]:
    for op in getattr(plan, "ops", []) or []:
        if getattr(op, "kind", None) == "RequestRetrieve":
            # Normalize to a plain dict payload
            return {
                "query": getattr(op, "query", ""),
                "owner": getattr(op, "owner", "any"),
                "k": int(getattr(op, "k", 0) or 0),
                "tier_pref": getattr(op, "tier_pref", None),
                "hints": dict(getattr(op, "hints", {}) or {}),
            }
    return None


def _normalize_rr_payload(bundle: Dict[str, Any], rr: Dict[str, Any]) -> Dict[str, Any]:
    cfg_t2 = bundle.get("cfg", {}).get("t2", {})
    hints = dict(rr.get("hints", {}) or {})
    # Ensure deterministic required hints
    hints.setdefault("now", bundle.get("now", ""))
    hints.setdefault("sim_threshold", float(cfg_t2.get("sim_threshold", 0.3)))
    owner = rr.get("owner", "any")
    owner = owner if owner in ("agent", "world", "any") else "any"
    k = int(rr.get("k", int(cfg_t2.get("k_retrieval", 1))))
    k = max(1, k)
    return {
        "query": rr.get("query", ""),
        "owner": owner,
        "k": k,
        "tier_pref": rr.get("tier_pref"),
        "hints": hints,
    }


def _normalize_retrieved_result(result: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], int, float]:
    hits: List[Dict[str, Any]] = []
    for r in result.get("retrieved", []) or []:
        if isinstance(r, dict):
            rid = str(r.get("id"))
            score = float(r.get("_score", r.get("score", 0.0)) or 0.0)
            owner = str(r.get("owner", "any"))
            quarter = str(r.get("quarter", ""))
        else:
            rid = str(getattr(r, "id", ""))
            score = float(getattr(r, "score", 0.0) or 0.0)
            owner = str(getattr(r, "owner", "any"))
            quarter = str(getattr(r, "quarter", ""))
        if not rid:
            continue
        hits.append({"id": rid, "score": score, "owner": owner, "quarter": quarter})
    # Deterministic ordering
    hits.sort(key=lambda e: (-float(e.get("score", 0.0)), str(e.get("id", ""))))
    k_ret = len(hits)
    s_max = max([h.get("score", 0.0) for h in hits], default=0.0)
    return hits, k_ret, float(s_max)


def _refined_intent(tau_high: float, tau_low: float, labels: List[str], s_max: float) -> str:
    if s_max >= tau_high:
        return "summary"
    if s_max >= tau_low:
        return "assertion" if labels else "ack"
    return "question"


def rag_once(
    bundle: Dict[str, Any],
    plan: Plan,
    retrieve_fn: RetrieveFn,
    already_used: bool = False,
) -> Tuple[Plan, Dict[str, Any]]:
    """Single retrieval refinement. Pure given a deterministic retrieve_fn.
    - If already_used: return plan unchanged with rag_blocked.
    - If no RequestRetrieve in plan.ops: return unchanged (no-op).
    - Else: call retrieve_fn with normalized payload; refine Speak intent and possibly add one EditGraph if within cap.
    Returns: (refined_plan, rag_metrics)
    """
    # Pre metrics
    sim_stats0 = bundle.get("t2", {}).get("metrics", {}).get("sim_stats", {}) or {}
    pre_s_max = float(sim_stats0.get("max", 0.0))

    rr = _first_request_retrieve_payload(plan)
    if already_used:
        return plan, {
            "rag_used": False,
            "rag_blocked": True,
            "pre_s_max": pre_s_max,
            "post_s_max": pre_s_max,
            "k_retrieved": 0,
            "owner": rr.get("owner") if rr else None,
            "tier_pref": rr.get("tier_pref") if rr else None,
            "retrieved_ids": [],
        }

    if rr is None:
        return plan, {
            "rag_used": False,
            "rag_blocked": False,
            "pre_s_max": pre_s_max,
            "post_s_max": pre_s_max,
            "k_retrieved": 0,
            "owner": None,
            "tier_pref": None,
            "retrieved_ids": [],
        }

    payload = _normalize_rr_payload(bundle, rr)
    result = retrieve_fn(payload)
    hits, k_retrieved, s_max_rag = _normalize_retrieved_result(
        result if isinstance(result, dict) else {}
    )

    # Compute refined evidence level
    thresholds = _policy_thresholds(bundle)
    tau_high, tau_low, eps_edit = (
        thresholds["tau_high"],
        thresholds["tau_low"],
        thresholds["eps_edit"],
    )
    labels = _topic_labels_from_bundle(bundle)

    post_s_max = max(pre_s_max, s_max_rag)
    new_intent = _refined_intent(tau_high, tau_low, labels, post_s_max)

    # Build new ops deterministically
    base_ops = int(bundle.get("agent", {}).get("caps", {}).get("ops", 3))
    try:
        slice_cap = int(bundle.get("slice_caps", {}).get("t3_ops", base_ops))
    except Exception:
        slice_cap = base_ops
    caps_ops = min(base_ops, slice_cap)
    tokens = int(bundle.get("cfg", {}).get("t3", {}).get("tokens", 256))

    new_ops: List[Any] = []
    has_editgraph = False
    # Replace first Speak with refined intent; keep others in order
    speak_replaced = False
    for op in plan.ops:
        k = getattr(op, "kind", None)
        if k == "Speak" and not speak_replaced:
            new_ops.append(
                SpeakOp(kind="Speak", intent=new_intent, topic_labels=labels, max_tokens=tokens)
            )
            speak_replaced = True
        else:
            new_ops.append(op)
        if k == "EditGraph":
            has_editgraph = True
    # Optionally add one EditGraph if evidence now >= tau_low and none exists yet
    if (post_s_max >= tau_low) and (not has_editgraph) and (len(new_ops) < caps_ops):
        remaining = max(caps_ops - len(new_ops), 0)
        edits = _edit_nodes_from_bundle(bundle, eps_edit=eps_edit, cap_nodes=remaining * 4)
        if edits:
            new_ops.append(
                EditGraphOp(kind="EditGraph", edits=edits, cap=min(len(edits), remaining * 4))
            )

    # Enforce ops cap deterministically
    if len(new_ops) > caps_ops:
        new_ops = new_ops[:caps_ops]

    refined = Plan(
        version="t3-plan-v1",
        reflection=getattr(plan, "reflection", False),
        ops=new_ops,
        request_retrieve=getattr(plan, "request_retrieve", None),
    )

    metrics = {
        "rag_used": True,
        "rag_blocked": False,
        "pre_s_max": pre_s_max,
        "post_s_max": post_s_max,
        "k_retrieved": int(k_retrieved),
        "owner": payload.get("owner"),
        "tier_pref": payload.get("tier_pref"),
        "retrieved_ids": [h.get("id") for h in hits],
    }
    return refined, metrics


# --- PR7: Deterministic dialogue synthesis ---


def _dedupe_sort_list(xs: List[str]) -> List[str]:
    return sorted({str(x) for x in (xs or [])})


def _format_labels(labels: List[str]) -> str:
    return ", ".join(labels)


def _top_snippet_ids(dialog_bundle: Dict[str, Any]) -> List[str]:
    n = int(dialog_bundle.get("dialogue", {}).get("include_top_k_snippets", 2) or 2)
    hits = dialog_bundle.get("retrieved", []) or []
    ids = [str(h.get("id")) for h in hits if isinstance(h, dict) and h.get("id")]
    return ids[: max(n, 0)]


def _tokenize(s: str) -> List[str]:
    # Deterministic whitespace tokenization
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
    return _dialogue_mod.speak(dialog_bundle, plan)


# --- PR8: LLM-based dialogue (optional backend) ---


def build_llm_prompt(dialog_bundle: Dict[str, Any], plan: Plan) -> str:
    return _dialogue_mod.build_llm_prompt(dialog_bundle, plan)


def llm_speak(dialog_bundle: Dict[str, Any], plan: Plan, adapter: Any) -> Tuple[str, Dict[str, Any]]:
    return _dialogue_mod.llm_speak(dialog_bundle, plan, adapter)


# --- M3-08: LLM Planner (opt-in, defaults OFF) ---


def make_planner_prompt(ctx) -> str:
    from . import policy as _policy_mod

    return _policy_mod.make_planner_prompt(ctx)


def _get_llm_adapter_from_cfg(cfg: Dict[str, Any]):
    from . import policy as _policy_mod

    return _policy_mod._get_llm_adapter_from_cfg(cfg)


def plan_with_llm(ctx, state: Any, cfg: Dict[str, Any]) -> Dict[str, Any]:
    from . import policy as _policy_mod

    return _policy_mod.plan_with_llm(ctx, state, cfg)
