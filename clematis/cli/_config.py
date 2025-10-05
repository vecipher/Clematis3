from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, TextIO, Tuple

# Relative default searched under the current working directory
DEFAULT_REL = Path("configs") / "config.yaml"
# XDG subpath under $XDG_CONFIG_HOME (or ~/.config if unset)
XDG_SUBPATH = Path("clematis") / "config.yaml"


def _coerce_candidate(p: Path) -> Optional[Path]:
    """Return a concrete config file path if the candidate exists.

    Accepts a file path *or* a directory; directories are resolved to
    "config.yaml" inside that directory. Returns the resolved file path
    if it exists, else None.
    """
    if p.is_dir():
        p = p / "config.yaml"
    if p.is_file():
        return p.resolve()
    return None


def discover_config_path(
    explicit: Optional[str],
    cwd: Optional[Path] = None,
    env: Optional[Mapping[str, str]] = None,
) -> Tuple[Optional[Path], str]:
    """Deterministic config discovery.

    Order (only when `explicit`/`--config` is not provided):
      1) $CLEMATIS_CONFIG (file or dir→config.yaml)
      2) CWD: ./configs/config.yaml
      3) XDG: ${XDG_CONFIG_HOME:-$HOME/.config}/clematis/config.yaml

    Returns a tuple: (selected_path or None, source_tag).
    Source tags: 'explicit', 'explicit-missing', 'env:CLEMATIS_CONFIG',
    'cwd:configs/config.yaml', 'xdg', 'none'.
    """
    cwd = cwd or Path.cwd()
    env = dict(env or {})

    # 0) explicit --config always wins; allow reporting even if missing
    if explicit:
        expanded = Path(os.path.expandvars(explicit)).expanduser()
        sel = _coerce_candidate(expanded)
        if sel is not None:
            return sel, "explicit"
        return expanded, "explicit-missing"

    # 1) $CLEMATIS_CONFIG
    cenv = env.get("CLEMATIS_CONFIG")
    if cenv:
        epath = Path(os.path.expandvars(cenv)).expanduser()
        sel = _coerce_candidate(epath)
        if sel is not None:
            return sel, "env:CLEMATIS_CONFIG"

    # 2) ./configs/config.yaml
    sel = _coerce_candidate((cwd / DEFAULT_REL).expanduser())
    if sel is not None:
        return sel, "cwd:configs/config.yaml"

    # 3) XDG
    xdg_base = env.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    sel = _coerce_candidate((Path(xdg_base).expanduser() / XDG_SUBPATH).expanduser())
    if sel is not None:
        return sel, "xdg"

    return None, "none"


def maybe_log_selected(
    path: Optional[Path], source: str, *, verbose: bool = False, stream: Optional[TextIO] = None
) -> None:
    """Emit a one-line message about the selected config when verbose.

    Logged to stderr by default to avoid mixing with command outputs.
    """
    if not verbose:
        return
    if stream is None:
        stream = sys.stderr
    try:
        stream.write(f"[clematis] config: selected={path if path else 'none'} (source={source})\n")
        stream.flush()
    except Exception:
        # Don't allow logging failures to affect CLI behavior
        pass


def validate_or_die(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize a config dict against the frozen v1 schema.
    Exits with code 2 on invalid configs.
    """
    try:
        from configs.validate import validate_config
    except Exception as e:  # pragma: no cover
        sys.stderr.write(f"Config error: internal validator unavailable: {e}\n")
        sys.stderr.flush()
        raise SystemExit(2)
    try:
        return validate_config(dict(cfg))
    except ValueError as e:
        sys.stderr.write(f"Config error: {e}\n")
        sys.stderr.flush()
        raise SystemExit(2)


__all__ = [
    "DEFAULT_REL",
    "XDG_SUBPATH",
    "discover_config_path",
    "maybe_log_selected",
    "validate_or_die",
]
