

#!/usr/bin/env python3
"""
Validate a Clematis3 config file.

Usage:
  python3 scripts/validate_config.py [--strict] [path/to/config.yaml]
  # If omitted, defaults to configs/config.yaml
  # Use '-' to read from STDIN

Exit codes:
  0 = OK
  1 = Validation errors (or warnings when --strict)
  2 = Load/parse errors or bad usage
"""
from __future__ import annotations
import sys
import os
import json
from typing import Any, Dict
import argparse

try:
    import yaml  # type: ignore
except Exception:  # PyYAML optional
    yaml = None  # type: ignore

# Ensure the project root (parent of scripts/) is importable when run directly
try:
    from configs.validate import validate_config_verbose, validate_config  # type: ignore
except ModuleNotFoundError:
    HERE = os.path.abspath(os.path.dirname(__file__))
    ROOT = os.path.abspath(os.path.join(HERE, ".."))
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    try:
        from configs.validate import validate_config_verbose, validate_config  # type: ignore
    except ImportError:
        # Older versions may not have the verbose API
        from configs.validate import validate_config  # type: ignore
        validate_config_verbose = None  # type: ignore
except ImportError:
    # Older versions may not have the verbose API
    from configs.validate import validate_config  # type: ignore
    validate_config_verbose = None  # type: ignore


def _eprint(*args: Any) -> None:
    print(*args, file=sys.stderr)


USAGE = (
    "usage: python3 scripts/validate_config.py [--strict] [config.yaml | -]\n"
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
    ap = argparse.ArgumentParser(
        prog="validate_config.py",
        description="Validate Clematis3 configuration",
    )
    ap.add_argument("path", nargs="?", default=os.path.join("configs", "config.yaml"),
                    help="Path to config file (YAML preferred). Use '-' for STDIN.")
    ap.add_argument("--strict", action="store_true",
                    help="Treat warnings as errors (non-zero exit if warnings present).")
    args = ap.parse_args(argv[1:])

    path = args.path

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
        if 'validate_config_verbose' in globals() and validate_config_verbose is not None:  # type: ignore
            normalized, warnings = validate_config_verbose(cfg)  # type: ignore
        else:
            normalized = validate_config(cfg)
            warnings = []
    except ValueError as ve:
        print("CONFIG INVALID\n" + str(ve))
        return 1

    # --strict: treat warnings as errors
    if args.strict and warnings:
        print("CONFIG WARNINGS (treated as errors due to --strict)")
        for w in sorted(warnings):
            print(w)
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
    # PR30: extra summary for perf + RQ prep (informational only)
    perf_cfg = normalized.get("perf", {}) or {}
    perf_t2 = perf_cfg.get("t2", {}) or {}
    perf_snap = perf_cfg.get("snapshots", {}) or {}
    perf_metrics = perf_cfg.get("metrics", {}) or {}
    q_cfg = (normalized.get("t2", {}) or {}).get("quality", {}) or {}

    print(
        "perf: enabled={pe} report_memory={rm} embed_store_dtype={esd} snapshots.compression={comp}".format(
            pe=perf_cfg.get("enabled"),
            rm=perf_metrics.get("report_memory"),
            esd=perf_t2.get("embed_store_dtype"),
            comp=perf_snap.get("compression"),
        )
    )
    print("t2.quality: enabled={qe}  # prep-only in M6".format(qe=q_cfg.get("enabled")))
    # Print warnings (if any) without failing the run
    for w in sorted(warnings):
        print(w)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))