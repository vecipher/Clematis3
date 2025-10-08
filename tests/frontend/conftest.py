import sys
import subprocess
import pytest
from pathlib import Path

# Build the static frontend exactly once per test session if dist/ is missing.
@pytest.fixture(scope="session", autouse=True)
def _ensure_frontend_built():
    # repo_root = .../Clematis3
    repo_root = Path(__file__).resolve().parents[2]
    dist_index = repo_root / "frontend" / "dist" / "index.html"
    if dist_index.exists():
        return
    # Build using the pure-Python builder (no make dependency; no network)
    builder = repo_root / "scripts" / "build_frontend.py"
    assert builder.exists(), f"missing builder: {builder}"
    subprocess.check_call([sys.executable, str(builder)], cwd=str(repo_root))
    assert dist_index.exists(), "frontend build did not produce dist/index.html"
