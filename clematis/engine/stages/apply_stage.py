from __future__ import annotations
from typing import Dict, Any
from ..types import ApplyResult

def apply_changes(ctx, state, t4) -> ApplyResult:
    # Demo applies nothing and emits a canned line.
    return ApplyResult(applied={"edits":0}, clamps=[], line="Hello (demo).", metrics={})
