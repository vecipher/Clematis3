import {el, clear, $} from "../util/dom.js";
import {addRun, state, selectRun} from "../state.js";

export function initRunsPanel({onRunsChanged}) {
  const fileInput = $("#fileInput");
  const btnClear = $("#btnClear");
  const runsList = $("#runsList");

  fileInput.addEventListener("change", async (ev) => {
    const files = Array.from(ev.target.files || []);
    for (const f of files) {
      const text = await f.text();
      const obj = JSON.parse(text);
      addRun(obj, f.name);
    }
    onRunsChanged();
    renderRunCards(runsList);
  });

  btnClear.addEventListener("click", () => {
    state.runs.splice(0, state.runs.length);
    state.selectedRunId = null;
    onRunsChanged();
    renderRunCards(runsList);
  });

  renderRunCards(runsList);
}

function renderRunCards(container) {
  clear(container);
  for (const r of state.runs) {
    const card = el("div", {class: "run-card"});
    const title = el("div", {class: "run-title", text: r.name});
    const meta = r.bundle && r.bundle.snapshot ? r.bundle.snapshot : {};
    const kv1 = el("div", {class:"kv", text: `schema=${meta.schema_version || "?"}`});
    const kv2 = el("div", {class:"kv", text: `nodes=${meta.nodes ?? "?"} edges=${meta.edges ?? "?"}`});
    const btn = el("button", {text: "Select"});
    btn.addEventListener("click", () => {
      selectRun(r.id);
      // fire a synthetic event others panels can listen to
      window.dispatchEvent(new CustomEvent("run-selected", {detail:{id:r.id}}));
    });
    card.append(title, kv1, kv2, btn);
    container.append(card);
  }
}
