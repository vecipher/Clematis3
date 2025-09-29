import sys, subprocess


def _run(args, debug=False):
    argv = [sys.executable, "-m", "clematis"]
    if debug:
        argv.append("--debug")
    argv += args
    return subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def test_debug_breadcrumb_and_no_behavior_change():
    p1 = _run(["rotate-logs", "--dir", "./.logs", "--dry-run"], debug=False)
    p2 = _run(["rotate-logs", "--dir", "./.logs", "--dry-run"], debug=True)
    assert p1.returncode == p2.returncode == 0
    assert p1.stdout == p2.stdout
    assert p2.stderr.splitlines()[0].startswith("[clematis] delegate -> ")
