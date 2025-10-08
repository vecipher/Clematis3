import { initTabs } from "./ui/tabs.js";
import { qs } from "./util/dom.js";
import { renderRuns } from "./components/runs.js";
import { renderSnapshots } from "./components/snapshots.js";
import { renderLogs } from "./components/logs.js";
const state = { bundle: null, showPerf: false, selectedPerfKey: null };
function setPerfToggleDefault() {
    const cb = qs("#perf-toggle");
    if (cb)
        cb.checked = false;
    state.showPerf = false;
}
async function loadBundleFromFile(file) {
    const text = await file.text();
    return JSON.parse(text);
}
async function loadBundleFromURL(u) {
    // Works when served via http(s); file:// will be blocked by fetch in most browsers.
    const res = await fetch(u, { cache: "no-store" });
    if (!res.ok)
        throw new Error(`fetch failed: ${res.status}`);
    return (await res.json());
}
function renderAll() {
    if (!state.bundle)
        return;
    const runsPane = qs("#pane-runs");
    const snapsPane = qs("#pane-snapshots");
    const logsPane = qs("#pane-logs");
    renderRuns(runsPane, state.bundle);
    renderSnapshots(snapsPane, state.bundle);
    renderLogs(logsPane, state.bundle, state.showPerf, state.selectedPerfKey);
}
function wireUI() {
    // Tabs
    const tabsRoot = qs("#tabs-root");
    initTabs(tabsRoot);
    // File input
    const input = qs("#bundle-input");
    input.addEventListener("change", async () => {
        const f = input.files?.[0];
        if (!f)
            return;
        state.bundle = await loadBundleFromFile(f);
        renderAll();
    });
    // Perf toggle
    const perfToggle = qs("#perf-toggle");
    perfToggle.addEventListener("change", () => {
        state.showPerf = !!perfToggle.checked;
        renderAll();
    });
    // Optional ?bundle=... for local serving
    const url = new URL(window.location.href);
    const q = url.searchParams.get("bundle");
    if (q) {
        loadBundleFromURL(q).then(b => { state.bundle = b; renderAll(); }).catch(console.error);
    }
}
function main() {
    setPerfToggleDefault();
    wireUI();
}
document.addEventListener("DOMContentLoaded", main);
