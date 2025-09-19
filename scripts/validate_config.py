

#!/usr/bin/env python3
"""
Validate a Clematis3 config file.

Usage:
  python3 scripts/validate_config.py [path/to/config.yaml]
  # If omitted, defaults to configs/config.yaml
  # Use '-' to read from STDIN

Exit codes:
  0 = OK
  1 = Validation errors
  2 = Load/parse errors or bad usage
"""
from __future__ import annotations
import sys
import os
import json
from typing import Any, Dict

try:
    import yaml  # type: ignore
except Exception:  # PyYAML optional
    yaml = None  # type: ignore

# Ensure the project root (parent of scripts/) is importable when run directly
try:
    from configs.validate import validate_config  # type: ignore
except ModuleNotFoundError:
    HERE = os.path.abspath(os.path.dirname(__file__))
    ROOT = os.path.abspath(os.path.join(HERE, ".."))
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from configs.validate import validate_config  # type: ignore


def _eprint(*args: Any) -> None:
    print(*args, file=sys.stderr)


USAGE = (
    "usage: python3 scripts/validate_config.py [config.yaml | -]  \n"
    "       (defaults to configs/config.yaml)"
)


def _load_config(path: str) -> Dict[str, Any]:
    """Load YAML (preferred) or JSON; '-' reads from stdin."""
    try:
        if path == "-":
            data = sys.stdin.read()
            if not data.strip():
                return {}
            if yaml is not None:
                return yaml.safe_load(data) or {}
            # Fallback to JSON for stdin
            return json.loads(data)
        # File path
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        if yaml is not None:
            return yaml.safe_load(text) or {}
        # Fallback to JSON
        return json.loads(text)
    except json.JSONDecodeError as je:
        _eprint("error: failed to parse config as JSON; install PyYAML for YAML support (pip install pyyaml)")
        raise je
    except Exception:
        raise


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[1] in {"-h", "--help"}:
        print(USAGE)
        return 0

    path = argv[1] if len(argv) >= 2 else os.path.join("configs", "config.yaml")

    try:
        cfg = _load_config(path)
    except FileNotFoundError:
        _eprint(f"error: config file not found: {path}")
        _eprint(USAGE)
        return 2
    except Exception as ex:
        _eprint(f"error: failed to load config: {ex}")
        return 2

    try:
        normalized = validate_config(cfg)
    except ValueError as ve:
        print("CONFIG INVALID\n" + str(ve))
        return 1

    # Success summary (non-verbose)
    t4 = normalized.get("t4", {})
    cache = t4.get("cache", {})
    print("OK")
    print(
        "t4.cache: ttl_sec={ttl} namespaces={ns} cache_bust_mode={mode}".format(
            ttl=cache.get("ttl_sec"), ns=cache.get("namespaces"), mode=t4.get("cache_bust_mode")
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))