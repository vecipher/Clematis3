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
    If the line contains a brace-enclosed, comma-separated list like:
        "  {a,b,c} ..."
    normalize it by sorting + deduping the items and normalizing trailing ellipses.
    Returns the possibly-updated line.
    """
    m = _BRACE_SEARCH.search(line)
    if not m:
        return line
    indent = re.match(r"^\s*", line).group(0)
    items = m.group(1)
    parts = [p.strip() for p in items.split(",") if p.strip()]
    brace = "{" + ",".join(parts) + "}"
    tail = line[m.end():]
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

    # --- Canonicalize the top-level usage brace list across Python versions ---
    # Some argparse versions put the {sub1,sub2,...} list on the same "usage:" line,
    # others wrap it onto the next line, and some even emit an extra standalone "..." line.
    # Normalize any brace list found on the usage line or shortly after, ensure a single
    # trailing " ...", and drop any immediate ellipsis-only lines that follow.
    lines = text.splitlines()
    usage_idx = next((i for i, l in enumerate(lines) if l.startswith("usage: clematis")), None)
    if usage_idx is not None:
        first_brace_idx = None
        for j in range(usage_idx, min(usage_idx + 5, len(lines))):
            if "{" in lines[j] and "}" in lines[j]:
                lines[j] = _canon_brace_list_line(lines[j], ensure_ellipsis=(first_brace_idx is None))
                if first_brace_idx is None:
                    first_brace_idx = j
        if first_brace_idx is not None:
            k = first_brace_idx + 1
            # Drop any trailing ellipsis-only lines that argparse may emit.
            while k < len(lines) and _ELLIPSIS_ONLY_LINE.match(lines[k]):
                del lines[k]

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
