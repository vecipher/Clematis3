from typing import Any, Dict, List, Tuple

DEFAULTS = {
    # ... other defaults ...
    "t3.llm.fixtures.enabled": False,
    "t3.llm.fixtures.path": "tests/fixtures/llm/reflection.json",
}

def validate_config_api(cfg: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
    errors: List[str] = []

    def _validate_t3(cfg: Dict[str, Any]) -> None:
        t3 = cfg.get("t3", {})
        backend = t3.get("backend", "rulebased")
        llm = t3.get("llm", {})

        # Validate backend default
        if "backend" not in t3:
            t3["backend"] = "rulebased"

        # Validate llm defaults
        if "llm" not in t3:
            t3["llm"] = {}

        # Set defaults for llm keys if missing
        llm.setdefault("provider", "fixture")
        llm.setdefault("model", "default-model")
        llm.setdefault("endpoint", "default-endpoint")
        llm.setdefault("max_tokens", 256)
        llm.setdefault("temp", 0.7)
        llm.setdefault("timeout_ms", 1000)
        fixtures = llm.setdefault("fixtures", {})
        fixtures.setdefault("enabled", False)
        fixtures.setdefault("path", "tests/fixtures/llm/reflection.json")

        # Validate provider
        if llm.get("provider") not in ("fixture", "openai", "other_known_provider"):
            errors.append("t3.llm.provider: unknown provider")

        # Validate temp bounds
        temp = llm.get("temp")
        if not (isinstance(temp, (int, float)) and 0.0 <= float(temp) <= 1.0):
            errors.append("t3.llm.temp: must be between 0.0 and 1.0")

        # Validate max_tokens positive
        max_tokens = llm.get("max_tokens")
        if not (isinstance(max_tokens, int) and max_tokens >= 1):
            errors.append("t3.llm.max_tokens: must be positive integer")

        # Validate timeout_ms positive
        timeout_ms = llm.get("timeout_ms")
        if not (isinstance(timeout_ms, int) and timeout_ms >= 1):
            errors.append("t3.llm.timeout_ms: must be positive integer")

        # Validate fixtures path unconditionally
        fixtures_enabled = bool(fixtures.get("enabled", False))
        fixtures_path = fixtures.get("path")
        if not isinstance(fixtures_path, str) or not fixtures_path.strip():
            errors.append("t3.llm.fixtures.path must be a non-empty string")

        # Determinism guard
        allow_reflection = t3.get("allow_reflection", False)
        if allow_reflection and backend == "llm" and not fixtures_enabled:
            errors.append(
                "Invalid config: backend='llm' requires t3.llm.fixtures.enabled=true (fixtures-only, deterministic)."
            )

        # Check for unknown keys in llm and fixtures
        known_llm_keys = {
            "provider", "model", "endpoint", "max_tokens", "temp", "timeout_ms", "fixtures"
        }
        for key in llm:
            if key not in known_llm_keys:
                errors.append(f"t3.llm.{key}: unknown key")

        if isinstance(fixtures, dict):
            known_fixtures_keys = {"enabled", "path"}
            for key in fixtures:
                if key not in known_fixtures_keys:
                    errors.append(f"t3.llm.fixtures.{key}: unknown key")

    _validate_t3(cfg)

    ok = not errors
    return ok, errors, cfg
