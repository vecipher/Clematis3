import os
import re

def test_index_exists_and_has_tabs():
    dist = os.path.join("frontend", "dist")
    idx = os.path.join(dist, "index.html")
    assert os.path.exists(idx), "frontend/dist/index.html missing; run: make frontend-build"

    text = open(idx, "r", encoding="utf-8").read()
    # Basic markers
    assert "Clematis v3 — Viewer (M14)" in text
    assert re.search(r'data-tab="runs"', text)
    assert re.search(r'data-panel="logs"', text)
