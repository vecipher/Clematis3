# clematis/engine/policy/sanitize.py
import json
from typing import Tuple, Any, Optional

from .json_schemas import (
    PLAN_MAX_ITEMS,
    PLAN_ITEM_MAX_LEN,
    RATIONALE_MAX_LEN,
)

# Raw size guards so we never try to parse megabyte-scale blobs in CI
_MAX_RAW_LEN = 20000


def _strip_triple_fences(s: str) -> Tuple[str, Optional[str]]:
    """Return (candidate_text, language) after trimming a single triple‑backtick fence.

    Accepts only a *single* fenced block (```json ... ``` or ``` ... ```).
    If there is no fence, returns (trimmed_text, None).
    """
    s = s.strip()
    if s.startswith("```") and s.endswith("```"):
        # allow ```json\n...\n``` or ```\n...\n```
        first_nl = s.find("\n")
        if first_nl != -1:
            lang = s[3:first_nl].strip().lower()
            body = s[first_nl + 1 : -3].strip()
            return (body if body else s, lang)
    return s, None


def parse_and_validate(text: str, schema: dict) -> Tuple[bool, Any]:
    """
    Returns (ok, obj_or_reason).

    Strict acceptance rules:
      • Input must be either pure JSON or a single fenced JSON block.
      • Reject any extra prose/prefix/suffix.
      • Enforce caps (max items/lengths). No additional properties.
    """
    if not isinstance(text, str):
        return False, "non-string LLM output"
    if len(text) > _MAX_RAW_LEN:
        return False, "raw output too large"

    candidate, lang = _strip_triple_fences(text)

    # If the model used a fenced block, only allow json/jsonc (or empty tag)
    if lang not in (None, "", "json", "jsonc"):
        return False, f"unsupported code fence language: {lang}"

    if len(candidate) > _MAX_RAW_LEN:
        return False, "json block too large"

    # Must be a single JSON object; no surrounding prose permitted
    try:
        obj = json.loads(candidate)
    except Exception as e:
        return False, f"non-JSON: {e}"

    # Minimal in-house validator (keep deps out of CI)
    if not isinstance(obj, dict):
        return False, "top-level must be object"

    # required keys
    for k in ("plan", "rationale"):
        if k not in obj:
            return False, f"missing key: {k}"

    # additionalProperties == False
    for k in obj.keys():
        if k not in ("plan", "rationale"):
            return False, f"unknown key: {k}"

    plan = obj["plan"]
    rat = obj["rationale"]

    if not isinstance(plan, list):
        return False, "plan must be array"

    if len(plan) > PLAN_MAX_ITEMS:
        return False, "plan too long"

    # plan items must be non-empty (after strip) strings and within length limits
    if any(
        (not isinstance(x, str))
        or (len(x) == 0)
        or (len(x.strip()) == 0)
        or (len(x) > PLAN_ITEM_MAX_LEN)
        for x in plan
    ):
        return False, "plan item length/content invalid"

    if (not isinstance(rat, str)) or (len(rat) == 0) or (len(rat) > RATIONALE_MAX_LEN):
        return False, "rationale length/type invalid"

    # Normalized pass-through
    return True, {"plan": plan, "rationale": rat}
