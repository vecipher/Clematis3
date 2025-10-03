"""Orchestrator package exposing the core orchestration helpers."""

from __future__ import annotations

from . import core as _core
from . import logging as _logging
from . import parallel as _parallel
from . import types as _types

# Re-export commonly patched callables/objects for tests.
apply_changes = _core.apply_changes
enable_staging = _logging.enable_staging
disable_staging = _logging.disable_staging
default_key_for = _logging.default_key_for
_begin_log_capture = _logging._begin_log_capture
_end_log_capture = _logging._end_log_capture
_make_readonly_snapshot = _parallel._make_readonly_snapshot
_run_turn_compute = _parallel._run_turn_compute
_run_agents_parallel_batch = _parallel._run_agents_parallel_batch
_get_cfg = _core._get_cfg
_should_yield = _core._should_yield
_append_jsonl = _logging._append_jsonl
_append_jsonl_unbuffered = _logging._append_unbuffered
append_jsonl = _logging._append_jsonl

# Stage functions (tests may monkeypatch these).
t1_propagate = _core._t1_propagate
t2_semantic = _core._t2_semantic
t4_filter = _core._t4_filter

# Public types
TurnCtx = _types.TurnCtx
TurnResult = _types.TurnResult


def __getattr__(name: str):
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(name)
    for module in (_core, _logging, _parallel, _types):
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(name)


def __setattr__(name: str, value) -> None:
    handled = False
    for module in (_core, _logging, _parallel, _types):
        if hasattr(module, name):
            setattr(module, name, value)
            handled = True
    if not handled:
        # Default to core namespace for new orchestration hooks so run_turn can see them.
        setattr(_core, name, value)
    globals()[name] = value


def __dir__() -> list[str]:
    names: set[str] = set()
    for module in (_core, _logging, _parallel, _types):
        for n in dir(module):
            if not (n.startswith("__") and n.endswith("__")):
                names.add(n)
    return sorted(names)


__all__ = __dir__()

## Clean up module aliases from the namespace
# del _core, _logging, _types  # intentionally kept for patching access
