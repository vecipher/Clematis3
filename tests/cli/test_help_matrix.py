import sys
import subprocess

PY = sys.executable

SUBS = [
    "rotate-logs",
    "inspect-snapshot",
    "bench-t4",
    "seed-lance-demo",
    "validate",
    "demo",
]


def test_top_help():
    p = subprocess.run(
        [PY, "-m", "clematis", "--help"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    assert p.returncode == 0


def test_each_sub_help():
    for sub in SUBS:
        p = subprocess.run(
            [PY, "-m", "clematis", sub, "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert p.returncode == 0
        # Required deterministic affordance
        assert "Delegates to scripts/" in p.stdout
