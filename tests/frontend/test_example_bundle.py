# tests/frontend/test_example_bundle.py
"""
Smoke test: the packaged demo bundle loads in the offline viewer (file://)

- Uses importlib.resources to resolve both the viewer entry (index.html)
  and the committed demo bundle under clematis/examples/run_bundles/.
- Does NOT perform any network requests; we keep this minimal and rely on
  existing offline tests for strict network banning.
- Skips if frontend/dist/index.html is not present in the installed package.
"""

from pathlib import Path
import pytest

try:
  from importlib.resources import files as r_files  # Python 3.9+
except Exception:  # pragma: no cover
  pytest.skip("importlib.resources.files unavailable", allow_module_level=True)

PLAYWRIGHT = pytest.importorskip("playwright.sync_api")

# Resolve packaged resources
DIST_INDEX = r_files("clematis").joinpath("frontend/dist/index.html")
DEMO_BUNDLE = r_files("clematis").joinpath("examples/run_bundles/run_demo_bundle.json")

pytestmark = pytest.mark.skipif(
    not Path(str(DIST_INDEX)).exists(),
    reason="frontend/dist/index.html missing; run: make frontend-build",
)

def _file_url(p: Path) -> str:
  return Path(p).resolve().as_uri()

def _msg_text(msg):
    # Robust across Playwright versions: method vs property vs str
    try:
        return msg.text()  # method (older Python API)
    except Exception:
        try:
            return msg.text  # property (some builds)
        except Exception:
            return str(msg)

def test_example_bundle_loads_without_network():
  from playwright.sync_api import sync_playwright

  idx = Path(str(DIST_INDEX))
  bundle = Path(str(DEMO_BUNDLE))

  assert bundle.exists(), "Demo bundle not present in package data"

  with sync_playwright() as pw:
    browser = pw.chromium.launch()
    page = browser.new_page()

    # Capture console for sanity (no http(s) errors expected)
    logs = []
    page.on("console", lambda msg: logs.append(_msg_text(msg)))

    # Open viewer over file://
    page.goto(_file_url(idx))

    # Viewer UI should be present
    page.wait_for_selector("#fileInput", state="attached")

    # Attach the committed demo bundle via the file input
    page.set_input_files("#fileInput", str(bundle))

    # Wait until snapshot panel gets content (more stable than runsList items)
    page.wait_for_selector("#snapshot", state="attached", timeout=10000)
    page.wait_for_function(
        "document.querySelector('#snapshot') && document.querySelector('#snapshot').textContent && document.querySelector('#snapshot').textContent.trim().length > 0",
        timeout=10000,
    )

    # Snapshot panel should render some text
    snap_txt = page.inner_text("#snapshot")
    assert snap_txt.strip(), "Snapshot panel is empty after loading demo bundle"

    # Quick sanity: no obvious http(s) errors in console
    page.wait_for_timeout(100)
    assert not any(
        ("http://" in s.lower()) or ("https://" in s.lower()) or ("network" in s.lower())
        for s in logs
    ), f"Unexpected network-looking console logs: {logs}"

    browser.close()
