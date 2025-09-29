from __future__ import annotations
import os, sys

_PRINTED = False


def hint_once() -> None:
    """
    Deterministic single-line stderr hint. Never pollutes stdout/JSON.
    Printed exactly once per process. Controlled by CLEMATIS_NO_HINT=1.
    """
    global _PRINTED
    if _PRINTED or os.getenv("CLEMATIS_NO_HINT") == "1":
        return
    sys.stderr.write(
        "[clematis] note: direct script is deprecated; use `python -m clematis <subcommand> ...`\n"
    )
    sys.stderr.flush()
    _PRINTED = True
