import {el, $, clear, copyToClipboard} from "../util/dom.js";
import {state, currentRun} from "../state.js";
import {renderJSONCollapsible} from "../ui/jsonview.js";

export function initSnapshotsPanel() {
  const runSelect = $("#runSelect");
  const btnCopy = $("#btnCopySnapshot");
  const snapBox = $("#snapshot");

  function refreshRunSelect() {
    clear(runSelect);
    for (const r of state.runs) {
      const opt = el("option", {value: r.id, text: r.name});
      if (r.id === (state.selectedRunId || "")) opt.selected = true;
      runSelect.append(opt);
    }
  }

  function renderSnapshot() {
    const r = currentRun();
    const snap = (r && r.bundle && r.bundle.snapshot) || {};
    snapBox.textContent = "";
    const view = {
      schema_version: snap.schema_version,
      version_etag: snap.version_etag,
      nodes: snap.nodes,
      edges: snap.edges,
      gel_nodes: snap.gel_nodes,
      gel_edges: snap.gel_edges,
      last_update: snap.last_update,
    };
    snapBox.append(renderJSONCollapsible(view));
  }

  runSelect.addEventListener("change", (e) => {
    state.selectedRunId = runSelect.value;
    renderSnapshot();
  });

  btnCopy.addEventListener("click", () => {
    const r = currentRun();
    const text = JSON.stringify((r && r.bundle && r.bundle.snapshot) || {}, null, 2).replace(/\r\n/g, "\n");
    copyToClipboard(text);
  });

  window.addEventListener("run-selected", () => { refreshRunSelect(); renderSnapshot(); });

  refreshRunSelect();
  renderSnapshot();
}
