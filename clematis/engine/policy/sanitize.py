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
      • Optional top-level `reflection` flag is allowed; coerced to bool if given.
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
        if k not in ("plan", "rationale", "reflection"):
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

    # optional reflection flag
    ref: bool = False
    if "reflection" in obj:
        ok_b, bv = _coerce_bool(obj["reflection"])
        if not ok_b:
            return False, "reflection must be boolean (or 'true'/'false','1'/'0')"
        ref = bool(bv)

    # Normalized pass-through
    return True, {"plan": plan, "rationale": rat, "reflection": ref}

def _coerce_bool(v):
    if isinstance(v, bool):
        return True, v
    if isinstance(v, int) and v in (0, 1):
        return True, bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "t", "yes", "y", "1"):
            return True, True
        if s in ("false", "f", "no", "n", "0"):
            return True, False
    return False, None

def sanitize_plan(plan_dict, errors):
    """
    Lightweight sanitizer for unit tests and rulebased planners.

    Validates 'ops' and 'reflection' keys in the input dict, but returns only:
        {'reflection': bool}

    - 'ops' is validated (and errors appended) but NOT included in the return value
      to avoid colliding with callers that pass ops explicitly.
    - Appends human-readable strings to `errors`; does not raise.
    """
    out = {} if plan_dict is None else dict(plan_dict)

    # ops
    ops = out.get("ops", [])
    if not isinstance(ops, list):
        errors.append("ops must be an array")
        ops = []
    else:
        bad = [x for x in ops if not isinstance(x, str) or (len(x.strip()) == 0)]
        if bad:
            errors.append("ops must be non-empty strings")
            ops = [x for x in ops if isinstance(x, str) and len(x.strip()) > 0]

    # reflection (optional)
    ref = False
    if "reflection" in out:
        ok_b, bv = _coerce_bool(out["reflection"])
        if not ok_b:
            errors.append("plan.reflection must be boolean (or boolean-like 'true'/'false','1'/'0').")
        else:
            ref = bool(bv)

    return {"reflection": ref}
