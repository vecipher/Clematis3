import sys, subprocess
def test_unknown_subcommand_yields_code_2():
    p = subprocess.run([sys.executable, "-m", "clematis", "does-not-exist"],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert p.returncode == 2
    assert "usage:" in p.stderr.lower()