from __future__ import annotations
from typing import Dict, Any
from ..types import ApplyResult
 
def apply_changes(ctx, state, t4) -> ApplyResult:
    store = state.get("store")
    edits = 0
    for d in t4.approved_deltas:
        r = store.apply_deltas("g:surface", [d])
        edits += int(r.get("edits", 0))
    return ApplyResult(applied={"edits":edits}, clamps=[], line="Hello (demo).", metrics={})