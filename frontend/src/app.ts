import { initTabs } from "./ui/tabs.js";
import { qs } from "./util/dom.js";
import { renderRuns } from "./components/runs.js";
import { renderSnapshots } from "./components/snapshots.js";
import { renderLogs } from "./components/logs.js";
import { isExperimental, wireExperimentalRepaint } from "./exp/flags.js";
import { renderGraph } from "./exp/graph.js";
import { renderMemory } from "./exp/memory.js";
import type { AppState, RunBundle } from "./state.js";

const state: AppState = { bundle: null, showPerf: false, selectedPerfKey: null };

function setPerfToggleDefault() {
  const cb = qs<HTMLInputElement>("#perf-toggle");
  if (cb) {
    cb.checked = false;
    state.showPerf = false;
  }
}

async function loadBundleFromFile(file: File): Promise<RunBundle> {
  const text = await file.text();
  return JSON.parse(text) as RunBundle;
}

async function loadBundleFromURL(u: string): Promise<RunBundle> {
  const res = await fetch(u, { cache: "no-store" });
  if (!res.ok) throw new Error(`fetch failed: ${res.status}`);
  return (await res.json()) as RunBundle;
}

function toggleExperimentalTabs() {
  const on = isExperimental();
  const graphBtn = qs<HTMLElement>('[data-tab="graph"]');
  const memBtn   = qs<HTMLElement>('[data-tab="memory"]');
  if (graphBtn) graphBtn.style.display = on ? "" : "none";
  if (memBtn)   memBtn.style.display   = on ? "" : "none";

  const graphPane = qs<HTMLElement>('[data-panel="graph"]');
  const memPane   = qs<HTMLElement>('[data-panel="memory"]');
  if (!on) {
    if (graphPane) graphPane.style.display = "none";
    if (memPane)   memPane.style.display   = "none";
  }
}

function renderAll() {
  if (!state.bundle) return;

  const runsPane = qs<HTMLElement>("#runsList");
  const snapsPane = qs<HTMLElement>("#snapshot");
  const logsPane = qs<HTMLElement>("#logList");

  if (runsPane) renderRuns(runsPane, state.bundle);
  if (snapsPane) renderSnapshots(snapsPane, state.bundle);
  if (logsPane) renderLogs(logsPane, state.bundle, state.showPerf, state.selectedPerfKey);

  if (isExperimental()) {
    const gPane = qs<HTMLElement>('[data-panel="graph"]');
    const mPane = qs<HTMLElement>('[data-panel="memory"]');
    if (gPane) renderGraph(gPane, state.bundle);
    if (mPane) renderMemory(mPane, state.bundle);
  }
}

function wireUI() {
  const tabsRoot = qs<HTMLElement>("#tabs");
  if (tabsRoot) initTabs(tabsRoot);

  const input = qs<HTMLInputElement>("#fileInput");
  if (input) {
    input.addEventListener("change", async () => {
      const f = input.files?.[0];
      if (!f) return;
      state.bundle = await loadBundleFromFile(f);
      renderAll();
    });
  }

  const clearBtn = qs<HTMLButtonElement>("#btnClear");
  if (clearBtn) {
    clearBtn.addEventListener("click", () => {
      state.bundle = null;
      const targets = ["#runsList", "#snapshot", "#logList"];
      for (const sel of targets) {
        const el = qs<HTMLElement>(sel);
        if (el) el.textContent = "—";
      }
      if (input) input.value = "";
    });
  }

  const perfToggle = qs<HTMLInputElement>("#perf-toggle");
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
