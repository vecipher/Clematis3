import { initTabs } from "./ui/tabs.js";
import { qs } from "./util/dom.js";
import { renderRuns } from "./components/runs.js";
import { renderSnapshots } from "./components/snapshots.js";
import { renderLogs } from "./components/logs.js";
const state = { bundle: null, showPerf: false, selectedPerfKey: null };
function setPerfToggleDefault() {
    const cb = qs("#perf-toggle");
    if (cb) {
        cb.checked = false;
        state.showPerf = false;
    }
}
async function loadBundleFromFile(file) {
    const text = await file.text();
    return JSON.parse(text);
}
async function loadBundleFromURL(u) {
    const res = await fetch(u, { cache: "no-store" });
    if (!res.ok)
        throw new Error(`fetch failed: ${res.status}`);
    return (await res.json());
}
function renderAll() {
    if (!state.bundle)
        return;
    const runsPane = qs("#runsList");
    const snapsPane = qs("#snapshot");
    const logsPane = qs("#logList");
    if (runsPane)
        renderRuns(runsPane, state.bundle);
    if (snapsPane)
        renderSnapshots(snapsPane, state.bundle);
    if (logsPane)
        renderLogs(logsPane, state.bundle, state.showPerf, state.selectedPerfKey);
}
function wireUI() {
    // Tabs nav lives at #tabs; panels are elsewhere (handled in initTabs)
    const tabsRoot = qs("#tabs");
    if (tabsRoot)
        initTabs(tabsRoot);
    // File input + clear
    const input = qs("#fileInput");
    if (input) {
        input.addEventListener("change", async () => {
            const f = input.files?.[0];
            if (!f)
                return;
            state.bundle = await loadBundleFromFile(f);
            renderAll();
        });
    }
    const clearBtn = qs("#btnClear");
    if (clearBtn) {
        clearBtn.addEventListener("click", () => {
            state.bundle = null;
            const targets = ["#runsList", "#snapshot", "#logList"];
            for (const sel of targets) {
                const el = qs(sel);
                if (el)
                    el.textContent = "—";
            }
            if (input)
                input.value = "";
        });
    }
    // Perf toggle
    const perfToggle = qs("#perf-toggle");
    if (perfToggle) {
        perfToggle.addEventListener("change", () => {
            state.showPerf = !!perfToggle.checked;
            renderAll();
        });
    }
    // Optional ?bundle=... (when served via http)
    const url = new URL(window.location.href);
    const q = url.searchParams.get("bundle");
    if (q) {
        loadBundleFromURL(q).then(b => { state.bundle = b; renderAll(); }).catch(console.error);
    }
}
document.addEventListener("DOMContentLoaded", () => {
    setPerfToggleDefault();
    wireUI();
});
