import os
import sys
import pathlib
import subprocess
import pytest

from tests.cli._normalize import normalize_help, write_or_assert


def _discover_subs():
    """Prefer clematis.cli.main.SUBS; fall back to introspecting the parser."""
    try:
        from clematis.cli.main import SUBS  # type: ignore[attr-defined]

        return list(SUBS)
    except Exception:
        # Fallback: walk argparse subparsers to list available subcommands
        import argparse  # local import to avoid test import side effects
        from clematis.cli.main import build_parser  # type: ignore

        parser = build_parser()
        subs = []
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                subs = sorted(action.choices.keys())
                break
        return subs


ROOT = pathlib.Path(__file__).resolve().parents[2]
GOLD = ROOT / "tests" / "cli" / "goldens" / "help"
_SUBS = _discover_subs()


def _run_help(args):
    env = os.environ.copy()
    # Stabilize layout/locale for argparse wrapping
    env.setdefault("COLUMNS", "80")
    env.setdefault("PYTHONHASHSEED", "0")
    env.setdefault("LC_ALL", "C.UTF-8")
    p = subprocess.run(
        [sys.executable, "-m", "clematis", *args, "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    assert p.returncode == 0, f"non-zero exit for {' '.join(args)}: {p.stderr}"
    return p.stdout


@pytest.mark.parametrize("target", ["__TOP__", *_SUBS])
def test_help_golden(target):
    bless = os.environ.get("BLESS") == "1" or os.environ.get("PYTEST_BLESS") == "1"
    args = [] if target == "__TOP__" else [target]
    raw = _run_help(args)
    norm = normalize_help(raw)

    name = "top" if target == "__TOP__" else target.replace("/", "_")
    out = GOLD / f"{name}.txt"
    write_or_assert(str(out), norm, bless)
