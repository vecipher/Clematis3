import {initTabs} from "./ui/tabs.js";
import {initRunsPanel} from "./components/runs.js";
import {initSnapshotsPanel} from "./components/snapshots.js";
import {initLogsPanel} from "./components/logs.js";
import {$} from "./util/dom.js";
import {addRun} from "./state.js";

function setStatus(msg) { $("#status").textContent = msg; }

function setupQueryBundle() {
  const params = new URLSearchParams(window.location.search);
  const bundle = params.get("bundle");
  if (!bundle) return;
  // Only works under local HTTP server; file:// will fail fetch
  fetch(bundle, {cache:"no-store"})
    .then(r => r.ok ? r.json() : Promise.reject(r.status))
    .then(obj => {
      addRun(obj, bundle.split("/").pop());
      window.dispatchEvent(new CustomEvent("run-selected", {}));
      setStatus("offline • loaded");
    })
    .catch(() => setStatus("offline • static"));
}

document.addEventListener("DOMContentLoaded", () => {
  initTabs($("#tabs"));
  initRunsPanel({onRunsChanged: () => setStatus("offline • loaded")});
  initSnapshotsPanel();
  initLogsPanel();
  setupQueryBundle(); // optional path on http server
});
