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


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def normalize_help(text: str) -> str:
    text = _normalize_newlines(text)
    # unify argparse program name
    text = _USAGE_MAIN.sub("usage: clematis", text)
    # mask versions/dates
    text = _VER.sub("<VER>", text)
    text = _DATE.sub("<DATE>", text)
    # collapse spacing artifacts from different line-wrap behaviors
    text = _RUN_SPACES.sub(" ", text)
    # strip trailing whitespace on each line
    text = "\n".join(line.rstrip() for line in text.splitlines())
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
