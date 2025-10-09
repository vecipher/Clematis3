import { initTabs } from "./ui/tabs.js";
import { qs } from "./util/dom.js";
import { renderRuns } from "./components/runs.js";
import { renderSnapshots } from "./components/snapshots.js";
import { renderLogs } from "./components/logs.js";
import { isExperimental, wireExperimentalRepaint } from "./exp/flags.js";
import { renderGraph } from "./exp/graph.js";
import { renderMemory } from "./exp/memory.js";
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
function toggleExperimentalTabs() {
    const on = isExperimental();
    const graphBtn = qs('[data-tab="graph"]');
    const memBtn = qs('[data-tab="memory"]');
    if (graphBtn)
        graphBtn.style.display = on ? "" : "none";
    if (memBtn)
        memBtn.style.display = on ? "" : "none";
    const graphPane = qs('[data-panel="graph"]');
    const memPane = qs('[data-panel="memory"]');
    if (!on) {
        if (graphPane)
            graphPane.style.display = "none";
        if (memPane)
            memPane.style.display = "none";
    }
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
    if (isExperimental()) {
        const gPane = qs('[data-panel="graph"]');
        const mPane = qs('[data-panel="memory"]');
        if (gPane)
            renderGraph(gPane, state.bundle);
        if (mPane)
            renderMemory(mPane, state.bundle);
    }
}
function wireUI() {
    const tabsRoot = qs("#tabs");
    if (tabsRoot)
        initTabs(tabsRoot);
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
    const perfToggle = qs("#perf-toggle");
    if (perfToggle) {
        perfToggle.addEventListener("change", () => {
            state.showPerf = !!perfToggle.checked;
            renderAll();
        });
    }
    // Experimental gating (checkbox + hash flag)
    toggleExperimentalTabs();
    wireExperimentalRepaint(() => {
        toggleExperimentalTabs();
        renderAll();
    });
    // Optional ?bundle=...
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
