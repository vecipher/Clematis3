from __future__ import annotations
from typing import Any, Dict
from ..engine.types import Plan

class DummyLLMAdapter:
    def deliberate(self, plan_bundle: Dict[str, Any]) -> Plan:
        # Always propose a single Speak op
        return Plan(version="t3-plan-v1", ops=[{"kind":"Speak","payload":{"style":"neutral"}}], request_retrieve=None, reflection=False)

    def speak(self, dialog_bundle: Dict[str, Any]) -> str:
        return "Hello (demo)."
