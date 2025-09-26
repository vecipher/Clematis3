import sys, subprocess, shlex, pathlib

PY = sys.executable

def _run(args):
    p = subprocess.run([PY, "-m", "clematis", "--debug", *args],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.returncode, p.stdout, p.stderr.splitlines()

def test_order_simple():
    rc, out, err = _run(["rotate-logs", "--dir", "./.logs", "--dry-run"])
    assert rc == 0
    assert err and "argv=['--dir', './.logs', '--dry-run']" in err[0]

def test_leading_double_dash():
    rc, out, err = _run(["rotate-logs", "--", "--dir", "./.logs", "--dry-run"])
    assert rc == 0
    assert "argv=['--dir', './.logs', '--dry-run']" in err[0]

def test_merge_extras_plus_remainder():
    # extras before subcommand; remainder after — must merge extras + ns.args (PR46 invariant)
    rc, out, err = _run(["--dir", "./.logs", "rotate-logs", "--dry-run"])
    assert rc == 0
    assert "argv=['--dir', './.logs', '--dry-run']" in err[0]

def test_pre_subcommand_extras_and_leading_double_dash():
    rc, out, err = _run(["--dir", "./.logs", "rotate-logs", "--", "--dry-run"])
    assert rc == 0
    assert "argv=['--dir', './.logs', '--dry-run']" in err[0]