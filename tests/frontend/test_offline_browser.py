import os
import sys
import json
from pathlib import Path
import pytest

playwright = pytest.importorskip("playwright.sync_api", reason="playwright not installed")
from playwright.sync_api import sync_playwright  # type: ignore

REPO = Path(__file__).resolve().parents[2]
INDEX = REPO / "frontend" / "dist" / "index.html"

@pytest.mark.skipif(not INDEX.exists(), reason="frontend/dist/index.html missing; run: make frontend-build")
def test_no_http_https_requests_on_load():
    # Open the viewer via file:// and assert absolutely no http(s) requests fire.
    uri = INDEX.resolve().as_uri()
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context()
        external = []

        def on_request(req):
            u = (req.url or "").lower()
            if u.startswith("http://") or u.startswith("https://"):
                external.append(u)

        context.on("request", on_request)
        page = context.new_page()
        page.goto(uri, wait_until="load")
        # basic sanity: tabs are present
        page.wait_for_selector('[data-tab="runs"]', timeout=5000)
        # Perf toggle exists but is hidden by default; presence is sufficient.
        page.wait_for_selector('#perf-toggle', state='attached', timeout=5000)
        # assert no external requests
        assert external == [], f"External network requests observed: {external}"
        context.close()
        browser.close()
