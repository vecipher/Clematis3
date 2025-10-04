

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

GOLDEN = Path("tests/reflection/goldens/enabled_rulebased/bench_rule.json")
CFG = "examples/reflection/enabled.yaml"


def _run_bench(cfg_path: str) -> dict:
  """
  Run the deterministic reflection microbench with CI-style normalization and
  return the parsed JSON result. If the bench accidentally emits extra lines,
  parse the last valid JSON object from stdout.
  """
  env = os.environ.copy()
  env.setdefault("CI", "true")
  env.setdefault("CLEMATIS_NETWORK_BAN", "1")

  cp = subprocess.run(
    [sys.executable, "-m", "clematis.scripts.bench_reflection", "-c", cfg_path],
    check=True,
    text=True,
    capture_output=True,
    env=env,
  )

  lines = [ln.strip() for ln in cp.stdout.strip().splitlines() if ln.strip()]
  last_json = None
  for ln in reversed(lines):
    try:
      last_json = json.loads(ln)
      break
    except Exception:
      continue

  assert isinstance(last_json, dict), (
    "Microbench did not emit a JSON object on stdout.\n"
    f"stdout was:\n{cp.stdout}"
  )
  return last_json


@pytest.mark.reflection
@pytest.mark.skipif(not GOLDEN.exists(), reason="Golden missing; run `just regen-reflection-goldens` (or see docs/m10/reflection.md).")
def test_rulebased_bench_matches_golden() -> None:
  got = _run_bench(CFG)
  want = json.loads(GOLDEN.read_text(encoding="utf-8"))

  assert got == want, (
    "Rule-based reflection bench output diverged from golden.\n"
    f"GOT:  {json.dumps(got, sort_keys=True)}\n"
    f"WANT: {json.dumps(want, sort_keys=True)}\n"
    "To refresh golden deterministically:\n"
    "  CI=true CLEMATIS_NETWORK_BAN=1 python -m clematis.scripts.bench_reflection "
    "-c examples/reflection/enabled.yaml | tee tests/reflection/goldens/enabled_rulebased/bench_rule.json\n"
  )
