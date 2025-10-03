"""T3 stage package façade with legacy re-exports."""

from .bundle import assemble_bundle, make_plan_bundle, validate_bundle
from .core import t3_pipeline
from .dialogue import build_llm_prompt, llm_speak, speak
from .metrics import finalize as finalize_metrics
finalize = finalize_metrics
from .policy import run_policy, select_policy
from .trace import emit_trace
from .legacy import (
    deliberate,
    make_dialog_bundle,
    make_plan_bundle as legacy_make_plan_bundle,
    make_planner_prompt,
    plan_with_llm,
    rag_once,
)

__all__ = [
    "assemble_bundle",
    "build_llm_prompt",
    "deliberate",
    "finalize",
    "finalize_metrics",
    "llm_speak",
    "make_dialog_bundle",
    "make_plan_bundle",
    "make_planner_prompt",
    "plan_with_llm",
    "rag_once",
    "run_policy",
    "select_policy",
    "speak",
    "emit_trace",
    "t3_pipeline",
    "validate_bundle",
]
