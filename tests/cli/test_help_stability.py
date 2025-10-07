

# tests/cli/test_help_stability.py
from __future__ import annotations

import os
import re
import subprocess
import sys
from typing import Dict, List, Optional

import pytest

from clematis.cli_utils import HELP_EPILOG


def _run_help(columns: Optional[int] = None, extra_env: Optional[Dict[str, str]] = None) -> str:
  """
  Invoke `python -m clematis --help` in a subprocess and return normalized text.
  Normalization: CRLF -> LF; strip trailing whitespace-only lines.
  """
  env = os.environ.copy()
  if columns is not None:
      env["COLUMNS"] = str(columns)
  # Deterministic env (mirrors __main__ defaults and CI); subprocess can still alter behavior
  env.setdefault("TZ", "UTC")
  env.setdefault("PYTHONUTF8", "1")
  env.setdefault("PYTHONHASHSEED", "0")
  env.setdefault("LC_ALL", "C.UTF-8")
  env.setdefault("SOURCE_DATE_EPOCH", "315532800")
  if extra_env:
      env.update(extra_env)

  proc = subprocess.run(
      [sys.executable, "-m", "clematis", "--help"],
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      env=env,
      check=False,
      text=True,
  )
  # Accept return code 0; on some argparse setups, --help exits 0. Anything else is a failure.
  assert proc.returncode == 0, f"--help failed: rc={proc.returncode}, stderr={proc.stderr}"
  out = proc.stdout.replace("\r\n", "\n")
  # Trim trailing spaces on each line (some shells inject spaces)
  out = "\n".join(line.rstrip() for line in out.splitlines()) + "\n"
  return out


@pytest.mark.parametrize("cols", [60, 80, 120])
def test_help_text_is_width_independent(cols: int) -> None:
  baseline = _run_help(80)
  probe = _run_help(cols)
  assert probe == baseline, f"Help output changed with terminal width={cols}. Ensure DeterministicHelpFormatter fixes width."


def _extract_subcommands(text: str) -> List[str]:
  """
  Parse the subcommands section produced by argparse.
  Looks for a 'subcommands:' heading added by the root parser.
  """
  m = re.search(r"(?ms)^subcommands:\n(?P<body>(?:^[ ]{2,}.+?\n)+)", text)
  if not m:
    return []
  body = m.group("body")
  cmds: List[str] = []
  for line in body.splitlines():
    line = line.strip()
    if not line:
      continue
    # argparse typically formats: "<name>  <help...>"
    parts = re.split(r"\s{2,}", line, maxsplit=1)
    if parts:
      cmds.append(parts[0])
  return cmds


def test_help_epilog_is_present_and_constant() -> None:
  text = _run_help().strip()
  assert HELP_EPILOG in text, "Expected constant epilog missing from --help output."
  assert text.endswith(HELP_EPILOG), "For determinism, the epilog should end the help text."


def test_subcommands_are_sorted_alphabetically() -> None:
  text = _run_help()
  cmds = _extract_subcommands(text)
  if not cmds:
    pytest.skip("No subcommands advertised in help; skip sort check.")
  assert cmds == sorted(cmds), f"Subcommands must be sorted for deterministic help: {cmds}"
