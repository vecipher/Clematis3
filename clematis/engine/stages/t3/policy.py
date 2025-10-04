from __future__ import annotations
from typing import Any, Dict, List

import json as _json
import os

from ...policy.json_schemas import PLANNER_V1
from ...policy.sanitize import parse_and_validate
from ...types import Plan, SpeakOp, EditGraphOp, RequestRetrieveOp
from .types import PolicyHandle, PolicyOutput

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

_DEFAULT_TAU_HIGH = 0.8
_DEFAULT_TAU_LOW = 0.4
_DEFAULT_EPS_EDIT = 0.10


def select_policy(cfg_root: Dict[str, Any], ctx: Any) -> PolicyHandle:
    t3_cfg = (cfg_root.get("t3") if isinstance(cfg_root, dict) else None) or {}
    name = str(t3_cfg.get("backend") or t3_cfg.get("policy") or "rulebased").lower()
    if name not in ("rulebased", "llm"):
        name = "rulebased"
    return {"name": name, "meta": {}}


def _policy_thresholds(bundle: Dict[str, Any]) -> Dict[str, float]:
    t3cfg = bundle.get("cfg", {}).get("t3", {}) if isinstance(bundle.get("cfg", {}), dict) else {}
    pol = t3cfg.get("policy", {}) if isinstance(t3cfg, dict) else {}
    return {
        "tau_high": float(pol.get("tau_high", _DEFAULT_TAU_HIGH)),
        "tau_low": float(pol.get("tau_low", _DEFAULT_TAU_LOW)),
        "eps_edit": float(pol.get("epsilon_edit", _DEFAULT_EPS_EDIT)),
    }


def _topic_labels_from_bundle(bundle: Dict[str, Any], cap: int = 5) -> List[str]:
    labels = list(bundle.get("text", {}).get("labels_from_t1", []) or [])
    if not labels:
        labels = [
            str(n.get("label", n.get("id"))) for n in bundle.get("t1", {}).get("touched_nodes", [])
        ]
    labels = sorted({str(x) for x in labels})
    return labels[: max(int(cap), 0)]


def _edit_nodes_from_bundle(
    bundle: Dict[str, Any], eps_edit: float, cap_nodes: int
) -> List[Dict[str, Any]]:
    nodes = bundle.get("t1", {}).get("touched_nodes", []) or []
    selected = []
    for n in nodes:
        try:
            if abs(float(n.get("delta", 0.0))) >= float(eps_edit):
                selected.append({"id": str(n.get("id"))})
        except Exception:
            continue
    selected.sort(key=lambda e: e["id"])  # id asc
    selected = selected[: max(int(cap_nodes), 0)]
    return [{"op": "upsert_node", "id": e["id"]} for e in selected]


def deliberate(bundle: Dict[str, Any]) -> Plan:
    cfg_t3 = bundle.get("cfg", {}).get("t3", {})
    cfg_t2 = bundle.get("cfg", {}).get("t2", {})

    base_ops = int(bundle.get("agent", {}).get("caps", {}).get("ops", 3))
    try:
        slice_cap = int(bundle.get("slice_caps", {}).get("t3_ops", base_ops))
    except Exception:
        slice_cap = base_ops
    caps_ops = min(base_ops, slice_cap)
    tokens = int(cfg_t3.get("tokens", 256))

    thresholds = _policy_thresholds(bundle)
    tau_high = thresholds["tau_high"]
    tau_low = thresholds["tau_low"]
    eps_edit = thresholds["eps_edit"]

    sim_stats = bundle.get("t2", {}).get("metrics", {}).get("sim_stats", {}) or {}
    s_max = float(sim_stats.get("max", 0.0))

    labels = _topic_labels_from_bundle(bundle)
    if s_max >= tau_high:
        intent = "summary"
    elif s_max >= tau_low:
        intent = "assertion" if labels else "ack"
    else:
        intent = "question"

    ops: List[Any] = []
    ops.append(SpeakOp(kind="Speak", intent=intent, topic_labels=labels, max_tokens=tokens))

    if s_max >= tau_low and len(ops) < caps_ops:
        remaining = max(caps_ops - len(ops), 0)
        edits = _edit_nodes_from_bundle(bundle, eps_edit=eps_edit, cap_nodes=remaining * 4)
        if edits:
            ops.append(
                EditGraphOp(kind="EditGraph", edits=edits, cap=min(len(edits), remaining * 4))
            )

    if s_max < tau_low and len(ops) < caps_ops:
        owner = str(cfg_t2.get("owner_scope", "any"))
        k_all = int(cfg_t2.get("k_retrieval", 64))
        k = max(1, k_all // 2)
        ops.append(
            RequestRetrieveOp(
                kind="RequestRetrieve",
                query=bundle.get("text", {}).get("input", ""),
                owner=owner if owner in ("agent", "world", "any") else "any",
                k=k,
                tier_pref="cluster_semantic",
                hints={
                    "now": bundle.get("now", ""),
                    "sim_threshold": float(cfg_t2.get("sim_threshold", 0.3)),
                },
            )
        )

    if len(ops) > caps_ops:
        ops = ops[:caps_ops]

    return Plan(version="t3-plan-v1", reflection=False, ops=ops, request_retrieve=None)


def make_planner_prompt(ctx) -> str:
    summary = {
        "turn": getattr(ctx, "turn_id", 0),
        "agent": getattr(ctx, "agent_id", "agent"),
    }
    return (
        "SYSTEM: Return ONLY valid JSON with keys {plan: list[str], rationale: str, reflection: boolean}. "
        "No prose. No markdown. No trailing commas.\n"
        f"STATE: {_json.dumps(summary, separators=(',', ':'))}\n"
        "USER: Propose up to 4 next steps as short strings; include a brief rationale and set reflection to true only if a reflection pass is recommended."
    )


def _get_llm_adapter_from_cfg(cfg: Dict[str, Any]):
    try:
        t3 = cfg.get("t3", {}) if isinstance(cfg, dict) else {}
    except Exception:
        t3 = {}
    if str(t3.get("backend", "rulebased")) != "llm":
        return None
    llm = t3.get("llm", {}) if isinstance(t3, dict) else {}
    provider = str(llm.get("provider", "fixture"))
    try:
        ci = str(os.environ.get("CI", "")).lower() == "true"
    except Exception:
        ci = False
    if ci:
        if provider != "fixture":
            raise LLMAdapterError(f"CI requires fixture provider; got {provider}")
        fixtures = (llm.get("fixtures", {}) or {}) if isinstance(llm, dict) else {}
        enabled = fixtures.get("enabled", True)
        if not enabled:
            raise LLMAdapterError("CI requires fixtures.enabled=true")
        fx_path = fixtures.get("path") or "fixtures/llm/qwen_small.jsonl"
        if not os.path.exists(fx_path):
            raise LLMAdapterError(f"CI fixture path not found: {fx_path}")
    if provider == "fixture":
        if FixtureLLMAdapter is None:
            return None
        path = (llm.get("fixtures", {}) or {}).get("path") or "fixtures/llm/qwen_small.jsonl"
        return FixtureLLMAdapter(path)
    if provider == "ollama":
        if QwenLLMAdapter is None:
            return None
        endpoint = llm.get("endpoint", "http://localhost:11434/api/generate")
        model = llm.get("model", "qwen3:4b-instruct")
        temp = float(llm.get("temp", 0.2))
        timeout_s = max(1, int(int(llm.get("timeout_ms", 10000)) / 1000))

        def _ollama_call(
            prompt: str, *, model: str, max_tokens: int, temperature: float, timeout_s: int
        ) -> str:
            import json as _json_local
            import urllib.request as _ur

            body = _json_local.dumps(
                {
                    "model": model,
                    "prompt": prompt,
                    "options": {"temperature": float(temperature), "num_predict": int(max_tokens)},
                    "stream": False,
                    "format": "json",
                }
            ).encode("utf-8")
            req = _ur.Request(endpoint, data=body, headers={"Content-Type": "application/json"})
            with _ur.urlopen(req, timeout=timeout_s) as resp:
                payload = _json_local.loads(resp.read().decode("utf-8"))
            txt = payload.get("response")
            if not isinstance(txt, str):
                raise LLMAdapterError("Ollama returned no text response")
            return txt

        return QwenLLMAdapter(
            call_fn=_ollama_call, model=model, temperature=temp, timeout_s=timeout_s
        )
    return None


def plan_with_llm(ctx, state: Any, cfg: Dict[str, Any]) -> Dict[str, Any]:
    try:
        adapter = _get_llm_adapter_from_cfg(cfg)
    except Exception as e:
        try:
            logs = getattr(state, "logs", None)
            if isinstance(logs, list):
                prov = str(
                    (((cfg.get("t3", {}) or {}).get("llm", {}) or {}).get("provider", "unknown"))
                )
                ci = str(os.environ.get("CI", ""))
                logs.append({"llm_error": str(e), "provider": prov, "ci": ci})
        except Exception:
            pass
        return {"plan": [], "rationale": "fallback: invalid llm output"}
    if adapter is None:
        return {"plan": [], "rationale": "fallback: llm backend not active"}

    prompt = make_planner_prompt(ctx)
    try:
        llm_max = int(((cfg.get("t3", {}) or {}).get("llm", {}) or {}).get("max_tokens", 256))
        llm_temp = float(((cfg.get("t3", {}) or {}).get("llm", {}) or {}).get("temp", 0.2))
    except Exception:
        llm_max, llm_temp = 256, 0.2

    try:
        result = adapter.generate(prompt, max_tokens=llm_max, temperature=llm_temp)
        raw = getattr(result, "text", None)
        if raw is None and isinstance(result, dict):
            raw = result.get("text", "")
        ok, parsed = parse_and_validate(raw if isinstance(raw, str) else "", PLANNER_V1)
        if not ok:
            try:
                logs = getattr(state, "logs", None)
                if isinstance(logs, list):
                    prov = str(
                        (
                            ((cfg.get("t3", {}) or {}).get("llm", {}) or {}).get(
                                "provider", "unknown"
                            )
                        )
                    )
                    ci = str(os.environ.get("CI", ""))
                    logs.append({"llm_validation_failed": str(parsed), "provider": prov, "ci": ci})
            except Exception:
                pass
            return {"plan": [], "rationale": "fallback: invalid llm output"}
        return parsed
    except Exception as e:
        try:
            logs = getattr(state, "logs", None)
            if isinstance(logs, list):
                prov = str(
                    (((cfg.get("t3", {}) or {}).get("llm", {}) or {}).get("provider", "unknown"))
                )
                ci = str(os.environ.get("CI", ""))
                logs.append({"llm_error": str(e), "provider": prov, "ci": ci})
        except Exception:
            pass
        return {"plan": [], "rationale": "fallback: invalid llm output"}


def run_policy(
    handle: PolicyHandle,
    inputs: Dict[str, Any],
    cfg_root: Dict[str, Any],
    ctx: Any,
    *,
    state: Any | None = None,
) -> PolicyOutput:
    if handle["name"] == "llm":
        if state is None:
            raise ValueError("LLM policy requires state for logging/error capture.")
        out = plan_with_llm(ctx, state, cfg_root)
        plan = out.get("plan", []) if isinstance(out, dict) else []
        rationale = out.get("rationale", "") if isinstance(out, dict) else ""
        # Stash planner reflection flag for orchestrator consumption without changing return shape
        try:
            if state is not None and isinstance(out, dict):
                setattr(state, "_planner_reflection_flag", bool(out.get("reflection", False)))
        except Exception:
            pass
        return {"plan": plan, "rationale": rationale}

    plan_obj = deliberate(inputs)
    ops = getattr(plan_obj, "ops", None)
    if ops is None and hasattr(plan_obj, "get"):
        ops = plan_obj.get("ops", [])
    if ops is None:
        ops = []
    return {"plan": list(ops), "rationale": ""}


__all__ = [
    "deliberate",
    "make_planner_prompt",
    "plan_with_llm",
    "run_policy",
    "select_policy",
]
