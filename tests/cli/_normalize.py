"""
Normalization utilities for CLI golden snapshots.

- normalize_help(): scrubs incidental diffs (width, dates, versions, __main__.py).
- write_or_assert(): writes on first run or when BLESS=1; otherwise asserts equality.
"""

from __future__ import annotations

import os
import re

_VER = re.compile(r"\b\d+\.\d+\.\d+(?:[ab]\d+)?\b")  # 0.8.0a3, 1.2.3b1, 1.2.3
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_USAGE_MAIN = re.compile(r"(?im)^usage:\s+__main__\.py")
# collapse runs of 2+ spaces that aren't at start of line
_RUN_SPACES = re.compile(r"(?m)(?<!^)[ ]{2,}")
_BRACE_LINE = re.compile(r"^(\s*)\{([^}]*)\}(.*)$")


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _canon_brace_list_line(line: str, *, ensure_ellipsis: bool) -> str:
    """
    If the line contains a brace-enclosed, comma-separated list like:
        "  {a,b,c} ..."
    normalize it by sorting + deduping the items and normalizing trailing ellipses.
    Returns the possibly-updated line.
    """
    m = _BRACE_LINE.match(line)
    if not m:
        return line
    indent, items, tail = m.groups()
    parts = [p.strip() for p in items.split(",") if p.strip()]
    # Sort case-insensitively and dedupe while preserving order
    seen = set()
    parts_sorted = []
    for p in sorted(parts, key=lambda s: s.lower()):
        if p not in seen:
            seen.add(p)
            parts_sorted.append(p)
    brace = "{" + ",".join(parts_sorted) + "}"
    if ensure_ellipsis:
        # Collapse any number of trailing ellipses/groups into exactly one " ..."
        tail = re.sub(r"(?:\s*\.\.\.)*\s*$", " ...", tail)
    else:
        # Do not enforce ellipsis; just strip trailing whitespace
        tail = re.sub(r"\s+$", "", tail)
    return f"{indent}{brace}{tail}"


def normalize_help(text: str) -> str:
    text = _normalize_newlines(text)
    # unify argparse program name
    text = _USAGE_MAIN.sub("usage: clematis", text)
    # mask versions/dates
    text = _VER.sub("<VER>", text)
    text = _DATE.sub("<DATE>", text)
    # collapse spacing artifacts from different line-wrap behaviors
    text = _RUN_SPACES.sub(" ", text)

    # --- Canonicalize the top-level usage brace line across Python versions ---
    # We only rewrite the line *immediately following* the first "usage: clematis"
    # (the positional-arguments brace block later is left untouched).
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("usage: clematis"):
            j = i + 1
            if j < len(lines):
                lines[j] = _canon_brace_list_line(lines[j], ensure_ellipsis=True)
            break

    # strip trailing whitespace on each line
    text = "\n".join(line.rstrip() for line in lines)
    # ensure trailing newline
    if not text.endswith("\n"):
        text += "\n"
    return text



def normalize_completion(text: str) -> str:
    # Completions are already fairly stable; just normalize newlines and strip trailing spaces
    text = _normalize_newlines(text)
    text = "\n".join(line.rstrip() for line in text.splitlines())
    if not text.endswith("\n"):
        text += "\n"
    return text


def write_or_assert(path: str, content: str, bless: bool) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if bless or not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return
    with open(path, "r", encoding="utf-8") as f:
        expected = f.read()
    assert content == expected, f"Golden mismatch for {path}. Set BLESS=1 to update."


__all__ = ["normalize_help", "normalize_completion", "write_or_assert"]
