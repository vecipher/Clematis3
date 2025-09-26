# clematis/cli/main.py
import argparse
from clematis import __version__

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="clematis",
        description="Clematis CLI",
        allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--version", action="version", version=f"clematis {__version__}")
    subs = p.add_subparsers(dest="command", metavar="<command>")

    # Register subcommands (wrappers) — keep imports local to avoid heavy imports at import time.
    from . import rotate_logs as _rotate_logs
    from . import inspect_snapshot as _inspect_snapshot
    from . import bench_t4 as _bench_t4
    from . import seed_lance_demo as _seed_lance_demo
    # (Optional extras; wire them now or in a follow-up)
    # from . import mem_inspect as _mem_inspect
    # from . import mem_compact as _mem_compact
    # from . import llm_smoke as _llm_smoke
    # from . import rq_trace_dump as _rq_trace_dump

    _rotate_logs.register(subs)
    _inspect_snapshot.register(subs)
    _bench_t4.register(subs)
    _seed_lance_demo.register(subs)
    # _mem_inspect.register(subs)
    # _mem_compact.register(subs)
    # _llm_smoke.register(subs)
    # _rq_trace_dump.register(subs)

    return p

def main(argv=None) -> int:
    parser = build_parser()
    ns, extras = parser.parse_known_args(argv)

    func = getattr(ns, "func", None)
    if func is None:
        parser.print_help()
        return 2

    # PR46: preserve user order. Top-level unknown flags (extras) may contain a flag whose value
    # was captured by the subparser's REMAINDER. Prepend extras before ns.args.
    if extras:
        if hasattr(ns, "args") and isinstance(getattr(ns, "args", None), list):
            ns.args = extras + ns.args
        else:
            ns.args = extras

    result = func(ns)
    return int(result) if isinstance(result, int) else 0

if __name__ == "__main__":
    raise SystemExit(main())