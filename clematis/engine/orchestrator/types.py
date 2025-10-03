"""Shared orchestrator types."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List

TurnCtx = SimpleNamespace


@dataclass
class TurnResult:
    line: str
    events: List[Dict[str, Any]] = field(default_factory=list)

__all__ = ["TurnCtx", "TurnResult"]
