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
_ELLIPSIS_ONLY_LINE = re.compile(r"^\s*\.{3,}\s*$")
_BRACE_SEARCH = re.compile(r"\{([^}]*)\}")


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _canon_brace_list_line(line: str, *, ensure_ellipsis: bool) -> str:
    """
    If the line contains a brace-enclosed, comma-separated list like
    "... {a,b,c} ...", normalize trailing ellipses to exactly one and
    preserve original item order.
    """
    m = _BRACE_SEARCH.search(line)
    if not m:
        return line
    # Left part (indent+prefix), the brace, and tail
    start, end = m.span()
    prefix, items, tail = line[:start], m.group(0)[1:-1], line[end:]
    parts = [p.strip() for p in items.split(",") if p.strip()]
    brace = "{" + ",".join(parts) + "}"
    if ensure_ellipsis:
        tail = re.sub(r"(?:\s*\.\.\.)*\s$", " ...", tail)
    else:
        tail = re.sub(r"\s+$", "", tail)
    # Preserve original leading whitespace/prefix
    return f"{prefix}{brace}{tail}"


def _rebuild_top_usage_block(lines: list[str]) -> list[str]:
    """
    Canonicalize the top-level usage block into exactly two lines:
        usage: clematis [-h] [--version]
          {a,b,c} ...
    followed by a single blank line. This avoids argparse wrapping
    differences across Python versions and platforms.
    """
    # find first usage: clematis
    try:
        i = next(idx for idx, l in enumerate(lines) if l.startswith("usage: clematis"))
    except StopIteration:
        return lines

    # extract the first brace list within the next few lines (if any)
    items: list[str] = []
    for j in range(i, min(i + 6, len(lines))):
        m = _BRACE_SEARCH.search(lines[j])
        if m:
            raw = m.group(1)
            items = [p.strip() for p in raw.split(",") if p.strip()]
            break

    # Build canonical two-line usage (match goldens)
    usage_head = "usage: clematis [-h] [--version]"
    new_block = [usage_head]
    if items:
        new_block.append("  {" + ",".join(items) + "} ...")

    # determine extent of the existing usage block to replace
    k = i + 1
    while k < len(lines):
        # stop when we reach a clearly non-usage line (e.g., section title or blank)
        if lines[k].strip() == "":
            k += 1
            break
        if not lines[k].startswith(" ") and "usage:" not in lines[k]:
            break
        k += 1

    # also drop any immediate ellipsis-only lines after usage
    while k < len(lines) and _ELLIPSIS_ONLY_LINE.match(lines[k]):
        k += 1

    # ensure exactly one blank line after our canonical block
    rest = lines[k:]
    while rest and (rest[0].strip() == "" or _ELLIPSIS_ONLY_LINE.match(rest[0])):
        rest.pop(0)
    lines[i:] = new_block + [""] + rest
    return lines


def normalize_help(text: str) -> str:
    text = _normalize_newlines(text)
    # unify argparse program name
    text = _USAGE_MAIN.sub("usage: clematis", text)
    # mask versions/dates
    text = _VER.sub("<VER>", text)
    text = _DATE.sub("<DATE>", text)
    # collapse spacing artifacts from different line-wrap behaviors
    text = _RUN_SPACES.sub(" ", text)

    # Canonicalize the ENTIRE top-level usage block (2 lines + one blank line)
    lines = text.splitlines()
    lines = _rebuild_top_usage_block(lines)

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
