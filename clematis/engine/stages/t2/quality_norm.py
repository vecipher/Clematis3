from __future__ import annotations

"""
PR39 — Normalizer & Aliasing utilities (quality paths only)

Deterministic helpers used by lexical scoring (BM25, etc.) and MMR token
sets when quality modes are enabled. No RNG, no wall‑clock.

Provided functions:
  • normalize_text(s) -> str
  • tokenize(s, stopset=None, min_token_len=1, stemmer="none") -> list[str]
  • load_alias_map(path) -> dict[str, str]         (cached; safe on failure)
  • apply_aliases(tokens, alias_map) -> list[str]  (exact token matches; 1‑pass)

Design notes
  • Normalization: NFKC → lower → collapse whitespace.
  • Tokenization: split on non‑alphanumeric; drop empties; optional stopset/min length.
  • Aliasing: exact token‑level rewrite or expansion (if canonical contains spaces).
    Single pass only ⇒ idempotent by construction even if canonical repeats an alias.
  • IO failures for alias map return {} (validator surfaces warnings; runtime is no‑op).
"""

from dataclasses import dataclass
import io
import os
import re
import unicodedata
from typing import Dict, Iterable, List, Optional, Sequence

__all__ = [
    "normalize_text",
    "tokenize",
    "load_alias_map",
    "apply_aliases",
]

# ----------------------------- Normalization -----------------------------

_WS_RE = re.compile(r"\s+")
_TOK_SPLIT_RE = re.compile(r"[^0-9A-Za-z]+")
_ALIAS_SPLIT_RE = re.compile(r"[^0-9A-Za-z_]+")


def normalize_text(s: str) -> str:
    """Deterministically normalize text: NFKC → lower → collapse whitespace.

    Returns an empty string for falsy inputs. No locale sensitivity.
    """
    if not s:
        return ""
    # Unicode NFKC fold then ASCII lower; collapse runs of whitespace to single spaces
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    s = _WS_RE.sub(" ", s).strip()
    return s


# ------------------------------ Tokenization -----------------------------


def _apply_stemmer(tok: str, stemmer: str) -> str:
    """Very small, deterministic stemmer hook.

    Supported values:
      - "none"         : no changes
      - "porter-lite"  : crude suffix stripping for demo/parity; stable
    """
    if stemmer == "porter-lite":
        # Order matters; check longest first. Deterministic, ASCII‑only.
        for suf in (
            "ingly",
            "edly",
            "ing",
            "ness",
            "ment",
            "tion",
            "es",
            "s",
            "ed",
        ):  # pragma: no branch
            if tok.endswith(suf) and len(tok) > len(suf) + 2:
                return tok[: -len(suf)]
    return tok


def tokenize(
    s: str,
    stopset: Optional[Sequence[str]] = None,
    min_token_len: int = 1,
    stemmer: str = "none",
) -> List[str]:
    """Tokenize normalized text deterministically.

    - Splits on non‑alphanumeric boundaries
    - Filters empties and tokens shorter than `min_token_len`
    - Filters tokens present in `stopset` (if provided)
    - Applies a tiny optional stemmer for parity hooks
    """
    if not s:
        return []
    s_norm = normalize_text(s)
    raw = [t for t in _TOK_SPLIT_RE.split(s_norm) if t]
    if min_token_len > 1:
        raw = [t for t in raw if len(t) >= min_token_len]
    if stopset:
        stop = set(stopset)
        raw = [t for t in raw if t not in stop]
    if stemmer and stemmer != "none":
        raw = [_apply_stemmer(t, stemmer) for t in raw]
    return raw


# ------------------------------- Aliasing --------------------------------

# Process‑lifetime cache for alias maps; simple and deterministic.
_ALIAS_MAP_CACHE: Dict[str, Dict[str, str]] = {}


def _parse_simple_kv(buf: str) -> Dict[str, str]:
    """Very small YAML/INI‑like fallback parser: lines of `key: value`.

    Ignores comments and blank lines. Whitespace around key/value is stripped.
    This is only used if PyYAML is unavailable; sufficient for tiny alias maps.
    """
    out: Dict[str, str] = {}
    for raw in buf.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k = k.strip()
        v = v.strip()
        if k:
            out[k] = v
    return out


def load_alias_map(path: str) -> Dict[str, str]:
    """Load an alias map from a YAML/JSON‑like file; cache by absolute path.

    Returns `{}` if `path` is falsy, missing, or unreadable, or if the payload
    is not a dict[str,str]. This function **never** raises; validator should
    surface warnings, runtime remains a no‑op.
    """
    if not path:
        return {}
    abspath = os.path.abspath(path)
    if abspath in _ALIAS_MAP_CACHE:
        return _ALIAS_MAP_CACHE[abspath]
    try:
        with io.open(abspath, "r", encoding="utf-8") as f:
            buf = f.read()
        # Try PyYAML first
        amap: Dict[str, str]
        try:
            import yaml  # type: ignore

            data = yaml.safe_load(buf)
            if isinstance(data, dict):
                amap = {
                    str(k): str(v)
                    for k, v in data.items()
                    if isinstance(k, (str, int)) and isinstance(v, (str, int))
                }
            else:
                amap = _parse_simple_kv(buf)
        except Exception:
            # Fallback to a tiny parser for simple `key: value` lines
            amap = _parse_simple_kv(buf)
        _ALIAS_MAP_CACHE[abspath] = amap
        return amap
    except Exception:
        _ALIAS_MAP_CACHE[abspath] = {}
        return {}


def apply_aliases(tokens: Iterable[str], alias_map: Optional[Dict[str, str]]) -> List[str]:
    """Apply exact token aliases deterministically (single pass).

    If a canonical value contains whitespace, it is split into multiple tokens
    using the same tokenization splitter; the order is preserved.

    Example:
        tokens = ["cuda", "install"]
        alias_map = {"cuda": "nvidia_cuda"}
        → ["nvidia_cuda", "install"]

        alias_map = {"llm": "large language model"}
        → ["large", "language", "model", "install"]

    This is a 1‑pass transform → idempotent even if canonical strings repeat
    alias keys.
    """
    if not tokens:
        return []
    amap = alias_map or {}
    out: List[str] = []
    for t in tokens:
        ct = amap.get(t)
        if ct is None:
            out.append(t)
            continue
        # If canonical contains whitespace or punctuation, split with alias rule (preserve underscores)
        repl = _ALIAS_SPLIT_RE.split(normalize_text(ct))
        repl = [x for x in repl if x]
        if not repl:
            # Degenerate mapping → drop token
            continue
        out.extend(repl)
    return out
