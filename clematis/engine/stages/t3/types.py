from __future__ import annotations
from typing import Any, Dict, List, TypedDict


class PolicyHandle(TypedDict):
    name: str
    meta: Dict[str, Any]


class PolicyOutput(TypedDict):
    plan: List[Dict[str, Any]]
    rationale: str


PlanBundle = Dict[str, Any]
