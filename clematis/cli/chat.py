from __future__ import annotations

from typing import Optional

# Reuse the implementation that lives in scripts/chat.py
try:  # prefer packaged location
    from clematis.scripts.chat import main as _chat_main
except Exception:  # dev fallback when running from repo root
    try:
        from scripts.chat import main as _chat_main
    except Exception as e:  # helpful error if neither path is available
        raise ModuleNotFoundError(
            "Unable to import chat implementation. Expected 'clematis.scripts.chat' "
            "when installed, or 'scripts.chat' in a source checkout. If you're in 'dist/', "
            "run from the repo root or install the package."
        ) from e


def register(subparsers) -> None:
    """
    Register 'chat' as a pass-through subcommand so `python -m clematis chat` works.
    """
    p = subparsers.add_parser(
        "chat",
        help="Interactive chat loop with optional LLM backend",
    )

    def _run(ns) -> int:
        argv = list(getattr(ns, "args", []))
        if argv and argv[0] == "--":
            argv = argv[1:]
        return _chat_main(argv)

    p.set_defaults(func=_run)


def main(argv: Optional[list[str]] = None) -> int:
    """Direct entrypoint allowing `python -m clematis.chat` style execution."""
    return _chat_main(argv or [])
