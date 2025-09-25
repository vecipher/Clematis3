import argparse
from clematis import __version__

def build_parser():
    p = argparse.ArgumentParser(prog="clematis", description="Clematis CLI")
    p.add_argument("--version", action="version", version=f"clematis {__version__}")
    p.add_subparsers(dest="command", metavar="<command>")
    return p

def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    return 0

if __name__ == "__main__":
    raise SystemExit(main())