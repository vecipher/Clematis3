

#!/usr/bin/env python3
"""
Generate minimal man(1) pages for Clematis CLI from argparse --help output.

Design goals:
- Pure stdlib, offline, deterministic (honors SOURCE_DATE_EPOCH; fixed fallback)
- One page for the root command (clematis.1) and one per subcommand
- Minimal roff: .TH, .SH NAME, .SH SYNOPSIS, .SH DESCRIPTION, .SH OPTIONS
- Embed the exact --help text under OPTIONS in a no-fill block (.nf/.fi)
- Escape content safely for roff; avoid macros triggering inside help text

Usage (from repo root):
  SOURCE_DATE_EPOCH=1704067200 python scripts/gen_manpages.py \
      --outdir man --module clematis --section 1

Outputs:
  man/clematis.1
  man/clematis-<sub>.1 for every discovered subcommand
"""
from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import sys
from pathlib import Path
from typing import List

# Default to 2024-01-01 UTC if not provided (deterministic builds)
_DEFAULT_EPOCH = 1704067200


def _run(cmd: List[str]) -> str:
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
    )
    return p.stdout


def _date_str() -> str:
    try:
        epoch = int(os.environ.get("SOURCE_DATE_EPOCH", str(_DEFAULT_EPOCH)))
    except ValueError:
        epoch = _DEFAULT_EPOCH
    return datetime.datetime.fromtimestamp(epoch, datetime.timezone.utc).strftime("%Y-%m-%d")


def _version(module: str) -> str:
    code = (
        "import importlib,sys;"
        "m=importlib.import_module('%s');"
        "sys.stdout.write(getattr(m,'__version__','unknown'))"
        % module
    )
    try:
        out = _run([sys.executable, "-c", code])
        return out or "unknown"
    except Exception:
        return "unknown"


def _escape_roff_line(line: str) -> str:
    """Escape a single line for safe inclusion in roff.

    - Escape backslashes first
    - Escape hyphens to avoid option dashes being interpreted as hyphenation
    - If line begins with a roff control char ('.' or "'"), prefix with \\& to neutralize
    """
    s = line.replace("\\", "\\\\").replace("-", "\\-")
    if s.startswith(".") or s.startswith("'"):
        s = "\\&" + s
    return s


def _escape_roff_block(text: str) -> str:
    return "\n".join(_escape_roff_line(l) for l in text.splitlines())


def _emit_page(cmd_name: str, title: str, help_text: str, out_path: Path, section: str, module: str) -> None:
    ver = _version(module)
    date = _date_str()

    # SYNOPSIS: prefer the first line containing 'usage:'; fall back to a generic synopsis
    synopsis = next((ln.strip() for ln in help_text.splitlines() if ln.strip().startswith("usage:")), f"{cmd_name} [options]")

    name_line = f"{cmd_name} \\- {title}"  # hyphen escaped for roff NAME section

    help_block = "\n".join(l.rstrip() for l in help_text.splitlines())

    roff = []
    roff.append(f".TH {cmd_name} {section} \"{date}\" \"Clematis {ver}\" \"User Commands\"")
    roff.append(".SH NAME")
    roff.append(_escape_roff_line(name_line))
    roff.append(".SH SYNOPSIS")
    roff.append(_escape_roff_line(synopsis))
    roff.append(".SH DESCRIPTION")
    roff.append(_escape_roff_line(title))
    roff.append(".SH OPTIONS")
    roff.append(".nf")
    roff.append(_escape_roff_block(help_block))
    roff.append(".fi")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(roff) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


def _discover_subcommands(module: str) -> List[str]:
    """Ask the CLI parser for its subcommands in a robust, non-invasive way.

    We import clematis.cli.main.build_parser() in a subprocess and
    introspect the subparsers action (object with both 'choices' and 'add_parser').
    """
    snippet = (
        "import sys;"
        "import importlib;"
        "m = importlib.import_module('%s.cli.main' % sys.argv[1]);"
        "p = m.build_parser();"
        "sp = None\n"
        "for a in getattr(p, '_actions', []):\n"
        "    if hasattr(a, 'choices') and hasattr(a, 'add_parser'): sp=a; break\n"
        "subs = sorted(sp.choices.keys()) if sp else []\n"
        "sys.stdout.write(' '.join(subs))\n"
    )
    out = _run([sys.executable, "-c", snippet, module])
    return [s for s in out.strip().split() if s]


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate man pages for Clematis CLI deterministically (offline)")
    ap.add_argument("--outdir", default="man", help="Output directory for generated .1 pages (default: man)")
    ap.add_argument("--section", default="1", help="Man section (default: 1)")
    ap.add_argument("--module", default="clematis", help="Top-level package providing __version__ (default: clematis)")
    ap.add_argument("--root-title", default="umbrella CLI for Clematis", help="Title line for root page")
    ap.add_argument("--only", nargs="*", default=None, help="Optional list of subcommands to restrict generation")

    ns = ap.parse_args(argv)

    outdir = Path(ns.outdir)
    section = str(ns.section)
    module = str(ns.module)

    # Root page
    root_help = _run([sys.executable, "-m", module, "--help"])  # expects return code 0
    _emit_page(
        cmd_name=module,
        title=str(ns.root_title),
        help_text=root_help,
        out_path=outdir / f"{module}.1",
        section=section,
        module=module,
    )

    # Subcommands
    subs = ns.only or _discover_subcommands(module)
    for sub in subs:
        sub_help = _run([sys.executable, "-m", module, sub, "--help"])  # capture wrappers' stable phrase
        title = f"Delegates to scripts/ for '{sub}'"
        safe = sub.replace("/", "-")
        _emit_page(
            cmd_name=f"{module}-{safe}",
            title=title,
            help_text=sub_help,
            out_path=outdir / f"{module}-{safe}.1",
            section=section,
            module=module,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())