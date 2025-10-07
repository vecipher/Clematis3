# clematis/__main__.py
from __future__ import annotations

import os

# Deterministic env defaults (no-ops if already set). Matches CI in PR115.
os.environ.setdefault("TZ", "UTC")
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("LC_ALL", "C.UTF-8")
os.environ.setdefault("SOURCE_DATE_EPOCH", "315532800")

# Delegate to the real CLI entrypoint.
from clematis.cli.main import main as _cli_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(_cli_main())
