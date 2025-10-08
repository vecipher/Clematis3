/* Minimal, deterministic UI logic. No network unless user passes ?bundle=... on a local server cs theyre retarded. */
const $ = (sel) => document.querySelector(sel);

function setStatus(msg) { $("#status").textContent = msg; }
function renderSnapshot(info) {
  const keys = ["schema_version","version_etag","nodes","edges","gel_nodes","gel_edges","last_update"];
  const view = {};
  for (const k of keys) if (info && info[k] != null) view[k] = info[k];
  $("#snapshot").textContent = Object.keys(view).length ? JSON.stringify(view, null, 2) : "—";
}
function renderCounts(bundle) {
  const ul = $("#logCounts");
  ul.innerHTML = "";
  const logs = (bundle && bundle.logs) || {};
  const stages = ["t1","t2","t4","apply","turn"];
  for (const k of stages) {
    const n = Array.isArray(logs[k]) ? logs[k].length : 0;
    const li = document.createElement("li");
    li.textContent = `${k}: ${n}`;
    ul.appendChild(li);
  }
}

function loadBundleObject(obj) {
  try {
    if (!obj || typeof obj !== "object") throw new Error("Invalid bundle");
    renderSnapshot(obj.snapshot || obj.meta || {});
    renderCounts(obj);
    setStatus("offline • loaded");
  } catch (e) {
    console.error(e);
    setStatus("offline • load error");
  }
}

function setupFileInput() {
  const input = $("#fileInput");
  const btnClear = $("#btnClear");
  input.addEventListener("change", async (ev) => {
    const file = ev.target.files && ev.target.files[0];
    if (!file) return;
    const text = await file.text();
    const obj = JSON.parse(text);
    loadBundleObject(obj);
  });
  btnClear.addEventListener("click", () => {
    input.value = "";
    renderSnapshot(null);
    renderCounts(null);
    setStatus("offline • static");
  });
}

async function tryLoadQueryBundle() {
  const params = new URLSearchParams(window.location.search);
  const bundle = params.get("bundle");
  if (!bundle) return;
  try {
    // Note: will only work under a local HTTP server due to browser security model.
    const resp = await fetch(bundle, { cache: "no-store" });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const obj = await resp.json();
    loadBundleObject(obj);
  } catch (e) {
    console.warn("Query bundle load failed (expected on file://):", e);
    setStatus("offline • static");
  }
}

setupFileInput();
tryLoadQueryBundle();
