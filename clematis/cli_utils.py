from __future__ import annotations

import argparse

HELP_EPILOG = "See Operator Guide: docs/operator-guide.md"


class DeterministicHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """
    Help formatter with fixed width/positions so `--help` output is deterministic
    across OS/terminals and independent of runtime terminal width.
    """

    def __init__(self, *args, **kwargs):
        # Fix wrap width and help column so CI/OS differences don't change text.
        kwargs.setdefault("max_help_position", 28)
        kwargs.setdefault("width", 80)
        super().__init__(*args, **kwargs)


def reorder_subparsers_alphabetically(parser: argparse.ArgumentParser) -> None:
    """
    Sort subcommands alphabetically in help output for determinism.

    Call this AFTER all subparsers (subcommands) are registered on the parser.
    """
    for action in parser._actions:  # noqa: SLF001 (private attribute access is intentional)
        if isinstance(action, argparse._SubParsersAction):
            # Order the public choices mapping
            ordered = dict(sorted(action.choices.items(), key=lambda kv: kv[0]))
            action.choices.clear()
            action.choices.update(ordered)

            # Rebuild the private list that argparse uses when rendering help.
            choices_actions = getattr(action, "_choices_actions", [])
            name_to_action = {getattr(a, "dest", None): a for a in choices_actions}
            # mypy/ruff: _choices_actions is private; attribute may not be declared
            action._choices_actions = [  # type: ignore[attr-defined]
                name_to_action[name] for name in ordered.keys() if name in name_to_action
            ]


__all__ = [
    "HELP_EPILOG",
    "DeterministicHelpFormatter",
    "reorder_subparsers_alphabetically",
]
