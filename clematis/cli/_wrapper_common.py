

from __future__ import annotations
import os, sys
from typing import List

def _enabled(ns) -> bool:
    try:
        if getattr(ns, "debug", False):
            return True
    except Exception:
        pass
    return os.getenv("CLEMATIS_DEBUG") == "1"

def maybe_debug(ns, *, resolved: str, argv: List[str]) -> None:
    """Emit a single-line, deterministic breadcrumb to stderr (never logs)."""
    if _enabled(ns):
        sys.stderr.write(f"[clematis] delegate -> {resolved} argv={argv}\n")
        sys.stderr.flush()