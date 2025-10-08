import os
import re


def test_index_exists_and_has_tabs():
    dist = os.path.join("frontend", "dist")
    idx = os.path.join(dist, "index.html")
    assert os.path.exists(idx), "frontend/dist/index.html missing; run: make frontend-build"

    # Compiled asset must exist (from tsdist if present; otherwise JS fallback)
    app = os.path.join(dist, "assets", "app.js")
    assert os.path.exists(app), "frontend/dist/assets/app.js missing; ensure TS build was committed or JS fallback exists"

    text = open(idx, "r", encoding="utf-8").read()

    # Basic markers
    assert "Clematis v3 — Viewer (M14)" in text
    assert re.search(r'data-tab="runs"', text)
    assert re.search(r'data-panel="logs"', text)

    # Perf toggle presence (default unchecked is enforced by JS at runtime)
    assert 'id="perf-toggle"' in text, "perf toggle checkbox missing in index.html"

    # Offline guarantee (index-level sanity; full grep is covered by make frontend-offline-check)
    assert not re.search(r'https?://', text), "index.html should not reference external URLs"
