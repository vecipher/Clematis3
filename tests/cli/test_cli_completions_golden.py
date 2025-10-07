import os
import pathlib
import pytest
import sys
import platform

try:
    import shtab  # type: ignore
except Exception:  # pragma: no cover
    shtab = None


from clematis.cli.main import build_parser  # type: ignore
from tests.cli._normalize import normalize_completion
from tests.cli._normalize import write_or_assert

pytestmark = pytest.mark.skipif(
    sys.version_info[:2] != (3, 13) or platform.system() != "Linux",
    reason="CLI goldens enforced on CPython 3.13 Linux only",
)

ROOT = pathlib.Path(__file__).resolve().parents[2]
GOLD = ROOT / "tests" / "cli" / "goldens" / "completions"


@pytest.mark.skipif(shtab is None, reason="shtab not installed")
@pytest.mark.parametrize("shell", ["bash", "zsh"])  # lock matrix; fish unsupported in shtab 1.7.1
def test_completions_golden(shell: str):
    bless = os.environ.get("BLESS") == "1" or os.environ.get("PYTEST_BLESS") == "1"
    parser = build_parser()

    # shtab generates deterministic completions given a fixed parser structure.
    script = shtab.complete(parser, shell=shell)  # returns a string
    norm = normalize_completion(script)

    out = GOLD / f"{shell}.txt"
    write_or_assert(str(out), norm, bless)
