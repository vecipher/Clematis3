"""Read-only state snapshot utilities (PR70 support).

These helpers produce a lightweight, *read-only* view over the engine State so
agent turns can be computed in parallel without mutating the live world.

Design goals:
- Shallow structural freezing: top-level and nested mappings/lists are immutable.
- Identity for heavy objects (e.g., vectors) is preserved to avoid copies.
- Transparent attribute access for existing code (`state.graphs`, etc.).
- Zero behavior change when unused (the orchestrator opts in explicitly).

This module does not depend on LanceDB or any optional extras.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from types import MappingProxyType, SimpleNamespace
from typing import Any, Dict, Iterable

__all__ = [
    "FrozenDict",
    "FrozenList",
    "freeze",
    "ReadOnlyState",
    "readonly_snapshot",
]


class FrozenDict(Mapping):
    """An immutable mapping wrapper that allows hashable keys and arbitrary values.
    Values are not deep-copied; use `freeze` to recursively wrap if needed.
    """

    __slots__ = ("_data",)

    def __init__(self, data: Dict[str, Any] | Mapping[str, Any]):
        self._data = dict(data)

    def __getitem__(self, k: str) -> Any:
        return self._data[k]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:  # pragma: no cover
        return f"FrozenDict({self._data!r})"


class FrozenList(Sequence):
    """An immutable sequence wrapper."""

    __slots__ = ("_data",)

    def __init__(self, data: Iterable[Any]):
        self._data = tuple(data)

    def __getitem__(self, idx: int) -> Any:
        return self._data[idx]

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:  # pragma: no cover
        return f"FrozenList(len={len(self._data)})"


def freeze(obj: Any) -> Any:
    """Recursively wrap mappings/lists/tuples in immutable views.

    - Dict/Mapping -> FrozenDict with values frozen
    - List/Tuple/Sequence -> FrozenList
    - SimpleNamespace -> FrozenDict of its __dict__
    - Other objects returned as-is (identity preserved)
    """
    if obj is None:
        return None
    if isinstance(obj, FrozenDict) or isinstance(obj, FrozenList):
        return obj
    if isinstance(obj, Mapping):
        return FrozenDict({k: freeze(v) for k, v in obj.items()})
    if isinstance(obj, (list, tuple)):
        return FrozenList(obj)
    if isinstance(obj, SimpleNamespace):
        return FrozenDict(dict(obj.__dict__))
    return obj


class ReadOnlyState:
    """A shallow, read-only facade over the engine State.

    Known attributes (`graphs`, `agent_meta`, `meta_filter_state`, `caches`, etc.)
    are frozen via `freeze`. Unknown attributes are lazily looked up and, if they
    are mapping/sequence-like, frozen on first access. Attribute assignment is
    blocked.
    """

    __slots__ = ("_orig", "_cache")

    def __init__(self, orig: Any):
        object.__setattr__(self, "_orig", orig)
        object.__setattr__(self, "_cache", {})

    def __getattr__(self, name: str) -> Any:
        cache: Dict[str, Any] = object.__getattribute__(self, "_cache")
        if name in cache:
            return cache[name]
        orig = object.__getattribute__(self, "_orig")
        val = getattr(orig, name)
        # Freeze only structural containers; preserve identity otherwise
        frozen = freeze(val)
        cache[name] = frozen
        return frozen

    # Block attribute assignment / mutation
    def __setattr__(self, name: str, value: Any) -> None:  # pragma: no cover
        raise AttributeError("ReadOnlyState does not allow attribute assignment")

    def __delattr__(self, name: str) -> None:  # pragma: no cover
        raise AttributeError("ReadOnlyState does not allow attribute deletion")

    # Convenience: expose the original for debugging (read-only usage only)
    @property
    def _orig_state(self) -> Any:
        return object.__getattribute__(self, "_orig")

    def __repr__(self) -> str:  # pragma: no cover
        return f"ReadOnlyState(orig={type(object.__getattribute__(self, '_orig')).__name__})"


def readonly_snapshot(state: Any) -> ReadOnlyState:
    """Create a read-only snapshot facade over `state`.

    This is *not* a deep copy; it freezes structural containers to guard against
    accidental mutation in parallel compute, while preserving identity of heavy
    objects to keep the operation cheap.
    """
    return ReadOnlyState(state)
