import os, sys, subprocess
PY = sys.executable

def _run(args, **kw):
    return subprocess.run([PY, "-m", "clematis", *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, **kw)

def test_wrapper_help_includes_phrase():
    p = _run(["rotate-logs", "--help"])
    assert p.returncode == 0
    assert "Delegates to scripts/" in p.stdout

def test_umbrella_help_mentions_version():
    p = _run(["--help"])
    assert p.returncode == 0
    assert "--version" in p.stdout