import { initTabs } from "./ui/tabs.js";
import { qs } from "./util/dom.js";
import { renderRuns } from "./components/runs.js";
import { renderSnapshots } from "./components/snapshots.js";
import { renderLogs } from "./components/logs.js";
import type { AppState, RunBundle } from "./state.js";

const state: AppState = { bundle: null, showPerf: false, selectedPerfKey: null };

function setPerfToggleDefault() {
  const cb = qs<HTMLInputElement>("#perf-toggle");
  if (cb) cb.checked = false;
  state.showPerf = false;
}

async function loadBundleFromFile(file: File): Promise<RunBundle> {
  const text = await file.text();
  return JSON.parse(text) as RunBundle;
}

async function loadBundleFromURL(u: string): Promise<RunBundle> {
  // Works when served via http(s); file:// will be blocked by fetch in most browsers.
  const res = await fetch(u, { cache: "no-store" });
  if (!res.ok) throw new Error(`fetch failed: ${res.status}`);
  return (await res.json()) as RunBundle;
}

function renderAll() {
  if (!state.bundle) return;
  const runsPane = qs<HTMLElement>("#pane-runs")!;
  const snapsPane = qs<HTMLElement>("#pane-snapshots")!;
  const logsPane = qs<HTMLElement>("#pane-logs")!;

  renderRuns(runsPane, state.bundle);
  renderSnapshots(snapsPane, state.bundle);
  renderLogs(logsPane, state.bundle, state.showPerf, state.selectedPerfKey);
}

function wireUI() {
  // Tabs
  const tabsRoot = qs<HTMLElement>("#tabs-root")!;
  initTabs(tabsRoot);

  // File input
  const input = qs<HTMLInputElement>("#bundle-input")!;
  input.addEventListener("change", async () => {
    const f = input.files?.[0];
    if (!f) return;
    state.bundle = await loadBundleFromFile(f);
    renderAll();
  });

  // Perf toggle
  const perfToggle = qs<HTMLInputElement>("#perf-toggle")!;
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
