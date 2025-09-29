import os
import pathlib
import pytest

try:
    import shtab  # type: ignore
except Exception:  # pragma: no cover
    shtab = None


from clematis.cli.main import build_parser  # type: ignore
from tests.cli._normalize import normalize_completion, write_or_assert

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
