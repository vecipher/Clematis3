import sys
from clematis.cli.main import build_parser

def _merge(argv0):
    p = build_parser()
    ns, extras = p.parse_known_args(argv0)
    args = list(getattr(ns, "args", []) or [])
    return (list(extras or []) + args)

def test_merge_preserves_flag_value_pair():
    merged = _merge(["rotate-logs", "--dir", "./.logs", "--dry-run"])
    assert merged == ["--dir", "./.logs", "--dry-run"]

def test_double_dash_passthrough_strips_leading_dashdash():
    # Subparser REMAINDER sees the leading '--'; wrapper strips it.
    p = build_parser()
    ns, extras = p.parse_known_args(["rotate-logs", "--", "--dir", "./.logs", "--dry-run"])
    # What main() passes to wrapper:
    merged = (list(extras or []) + (list(getattr(ns, "args", []) or [])))
    # Simulate wrapper’s leading '--' strip:
    if merged and merged[0] == "--":
        merged = merged[1:]
    assert merged == ["--dir", "./.logs", "--dry-run"]

def test_equals_form_and_numbers():
    merged = _merge(["bench-t4", "--num", "1", "--runs=1", "--json"])
    # Regardless of how argparse partitions, merged must start with any flags.
    assert merged[0].startswith("--")