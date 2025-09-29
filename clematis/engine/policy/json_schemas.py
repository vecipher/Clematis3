"""JSON Schemas used by the T3 LLM policy.

These schemas are intentionally small and dependency-free so CI stays deterministic.
Keep the limits in sync with the enforcement in `sanitize.parse_and_validate`.
"""

from typing import Final, Dict, Any

# Hard caps (mirrored in sanitizer). If you change these, update sanitize.py too.
PLAN_MAX_ITEMS: Final[int] = 16
PLAN_ITEM_MAX_LEN: Final[int] = 200
RATIONALE_MAX_LEN: Final[int] = 2000

PLANNER_V1: Final[Dict[str, Any]] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "PlannerV1",
    "type": "object",
    "required": ["plan", "rationale"],
    "additionalProperties": False,
    "properties": {
        "plan": {
            "type": "array",
            "minItems": 0,
            "maxItems": PLAN_MAX_ITEMS,
            "items": {
                "type": "string",
                "minLength": 1,
                "maxLength": PLAN_ITEM_MAX_LEN,
            },
        },
        "rationale": {
            "type": "string",
            "minLength": 1,
            "maxLength": RATIONALE_MAX_LEN,
        },
    },
}

__all__ = [
    "PLANNER_V1",
    "PLAN_MAX_ITEMS",
    "PLAN_ITEM_MAX_LEN",
    "RATIONALE_MAX_LEN",
]
