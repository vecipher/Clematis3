from __future__ import annotations

import json
import sys
from typing import Any, Iterable, Mapping, Sequence, cast

# Verbosity gates
VERBOSE = False
QUIET = False


def set_verbosity(verbose: bool = False, quiet: bool = False) -> None:
    global VERBOSE, QUIET
    VERBOSE, QUIET = bool(verbose), bool(quiet)


def eprint_once(msg: str) -> None:
    if not QUIET:
        print(msg, file=sys.stderr)


def print_json(obj: Any = None, *, preencoded: str | None = None) -> None:
    """
    If preencoded is provided it is printed as-is (preserves existing JSON).
    Otherwise dumps obj using compact, stable separators (no color).
    """
    if preencoded is not None:
        sys.stdout.write(preencoded if preencoded.endswith("\n") else preencoded + "\n")
        return
    sys.stdout.write(json.dumps(obj, separators=(",", ":")) + "\n")


def _stringify(x: Any) -> str:
    return "" if x is None else str(x)


def print_table(
    rows: Iterable[Sequence[Any]] | Iterable[Mapping[str, Any]],
    headers: Sequence[str] | None = None,
) -> None:
    """
    Plain ASCII table (no color). Accepts list-of-dicts or list-of-sequences.
    """
    # Normalize to rows + headers
    it = list(rows)
    if not it:
        return
    if isinstance(it[0], Mapping):
        rows_m = [cast(Mapping[str, Any], r) for r in it]
        if headers is None:
            headers = list(rows_m[0].keys())
        matrix = [[_stringify(r.get(h, "")) for h in headers] for r in rows_m]
    else:
        rows_s = [cast(Sequence[Any], r) for r in it]
        matrix = [[_stringify(c) for c in r] for r in rows_s]
        if headers is None:
            headers = [f"col{i+1}" for i in range(len(rows_s[0]))]

    assert headers is not None

    widths = [max(len(h), *(len(r[i]) for r in matrix)) for i, h in enumerate(headers)]

    def fmt(row):
        return "  ".join(s.ljust(w) for s, w in zip(row, widths))

    print(fmt(list(map(str, headers))))
    print("  ".join("-" * w for w in widths))
    for r in matrix:
        print(fmt(r))
