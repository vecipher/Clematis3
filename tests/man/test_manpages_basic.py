import os, subprocess, sys, pathlib
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
GOLD_DIR = ROOT / "tests" / "golden" / "man"

def _read(p: pathlib.Path) -> str:
    return p.read_text(encoding="utf-8").replace("\r\n", "\n")

def _assert_minimal_stanzas(txt: str):
    for tok in (".TH ", ".SH NAME", ".SH SYNOPSIS", ".SH OPTIONS"):
        assert tok in txt, f"missing stanza {tok!r}"

def test_gen_manpages_minimal():
    env = dict(os.environ)
    env["SOURCE_DATE_EPOCH"] = "1704067200"  # 2024-01-01 UTC
    # generate
    subprocess.run([sys.executable, "scripts/gen_manpages.py", "--outdir", "man"],
                   cwd=ROOT, env=env, check=True)
    # root page
    root = ROOT / "man" / "clematis.1"
    assert root.exists()
    root_txt = _read(root)
    _assert_minimal_stanzas(root_txt)
    assert "2024-01-01" in root_txt  # deterministic date in .TH line

    # a couple of stable subcommands
    for sub in ("validate", "demo"):
        mp = ROOT / "man" / f"clematis-{sub}.1"
        assert mp.exists(), f"expected man page for {sub}"
        sub_txt = _read(mp)
        _assert_minimal_stanzas(sub_txt)
        assert "Delegates to scripts/" in sub_txt  # wrapper phrase enforced

    # Optional golden snapshot comparison (if goldens exist)
    required = ["clematis.1", "clematis-validate.1", "clematis-demo.1"]
    if all((GOLD_DIR / f).exists() for f in required):
        for f in required:
            got = _read(ROOT / "man" / f).rstrip()
            exp = _read(GOLD_DIR / f).rstrip()
            assert got == exp, f"snapshot mismatch for {f}"
    else:
        pytest.skip(f"goldens not present under {GOLD_DIR}, skipping snapshot equality")