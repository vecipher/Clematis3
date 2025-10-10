#!/usr/bin/env python3
"""CLI subcommand: validate — Delegates to scripts/validate_config.py.

Behavior‑neutral wrapper so `python -m clematis validate ...` works consistently.
Parses no custom flags here; captures passthrough via REMAINDER and forwards to
the packaged shim (`clematis.scripts.validate`), which locates the real script
and adapts main(argv)/main().
"""

from __future__ import annotations

import argparse
import sys
import json
import subprocess

from ._exit import OK, USER_ERR
from ._io import eprint_once, set_verbosity
from ._util import add_passthrough_subparser
from ._wrapper_common import prepare_wrapper_args

_HELP = "Delegates to scripts/"
_DESC = "Delegates to scripts/validate_config.py"


def register(subparsers: argparse._SubParsersAction) -> None:
    sp = add_passthrough_subparser(
        subparsers,
        name="validate",
        help_text=_HELP,
        description=_DESC,
    )
    sp.set_defaults(command="validate", func=_run)


def _run(ns: argparse.Namespace) -> int:
    opts = prepare_wrapper_args(ns)

    set_verbosity(opts.verbose, opts.quiet)

    if opts.help_requested:
        print(_HELP)
        return OK

    if opts.wants_json and opts.wants_table:
        if not opts.quiet:
            eprint_once("Choose exactly one of --json or --table.")
        return USER_ERR

    if opts.wants_table:
        if not opts.quiet:
            eprint_once("`validate` currently supports --json only.")
        return USER_ERR

    rest = opts.argv
    if opts.wants_json:
        # Run the delegate as a real subprocess so we capture FD-level stdout/stderr
        if "--json" not in rest:
            rest = ["--json", *rest]
        cmd = [sys.executable, "-m", "clematis.scripts.validate", *rest]
        pr = subprocess.run(cmd, capture_output=True, text=True)
        out, err = pr.stdout, pr.stderr

        def extract_json_block(s: str) -> tuple[str | None, tuple[int, int] | None]:
            i, j = s.find("{"), s.rfind("}")
            if i != -1 and j != -1 and j >= i:
                candidate = s[i : j + 1]
                json.loads(candidate)  # validate
                return candidate, (i, j + 1)
            return None, None

        json_text, span, src = None, None, None
        # Prefer JSON on stdout; fall back to stderr
        cand, sp = extract_json_block(out)
        if cand is not None:
            json_text, span, src = cand, sp, "stdout"
        else:
            cand, sp = extract_json_block(err)
            if cand is not None:
                json_text, span, src = cand, sp, "stderr"

        # Emit only JSON to stdout; prefer unwrapped config if present
        if json_text is not None:
            chosen = json_text
            try:
                parsed = json.loads(json_text)

                def pick_config(o):
                    if isinstance(o, dict):
                        # If this dict already looks like a config (has t1/t2/t3), return it
                        if any(k in o for k in ("t1", "t2", "t3")):
                            return o
                        # Otherwise, try common wrapper keys
                        for key in ("config", "effective_config", "cfg", "data", "result"):
                            v = o.get(key)
                            if isinstance(v, dict) and any(k in v for k in ("t1", "t2", "t3")):
                                return v
                    return o

                cfg_obj = pick_config(parsed)
                chosen = json.dumps(cfg_obj, ensure_ascii=False)
            except Exception:
                pass
            sys.stdout.write(chosen)
            sys.stdout.flush()

        # Forward any non-JSON output to stderr, avoiding duplication of the JSON slice
        def forward_noise(s: str, sp: tuple[int, int] | None) -> None:
            if not s:
                return
            if sp is None:
                sys.stderr.write(s)
                return
            a, b = sp
            if a > 0:
                sys.stderr.write(s[:a])
            if b < len(s):
                sys.stderr.write(s[b:])

        if src == "stdout":
            forward_noise(out, span)
            forward_noise(err, None)
        elif src == "stderr":
            forward_noise(out, None)
            forward_noise(err, span)
        else:
            forward_noise(out, None)
            forward_noise(err, None)

        return pr.returncode

    # Non-JSON path: delegate directly
    from clematis.scripts.validate import main as _main

    return _main(rest)
