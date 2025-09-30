import sys
import subprocess


def test_help_is_deterministic():
    out = subprocess.check_output([sys.executable, "-m", "clematis.cli.main", "--help"]).decode(
        "utf-8"
    )
    assert out.splitlines()[0].startswith("usage: clematis ")
    assert "--version" in out
