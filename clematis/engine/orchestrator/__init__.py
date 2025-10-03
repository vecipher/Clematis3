"""Orchestrator package exposing the core orchestration helpers."""

from . import core as _core
from . import logging as _logging

apply_changes = _core.apply_changes
enable_staging = _logging.enable_staging
disable_staging = _logging.disable_staging
default_key_for = _logging.default_key_for
_begin_log_capture = _logging._begin_log_capture
_end_log_capture = _logging._end_log_capture
_make_readonly_snapshot = _core._make_readonly_snapshot
t1_propagate = _core._t1_propagate
t2_semantic = _core._t2_semantic
t4_filter = _core._t4_filter
append_jsonl = _core._append_jsonl


def __getattr__(name: str):
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(name)
    return getattr(_core, name)


def __setattr__(name: str, value) -> None:
    setattr(_core, name, value)


def __dir__() -> list[str]:
    names: set[str] = set()
    for module in (_core, _logging):
        for n in dir(module):
            if not (n.startswith("__") and n.endswith("__")):
                names.add(n)
    return sorted(names)


__all__ = __dir__()
