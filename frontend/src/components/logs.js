import {el, $, $$, clear} from "../util/dom.js";
import {state, currentRun} from "../state.js";
import {renderEntries} from "../ui/jsonview.js";

const STAGES = ["t1","t2","t4","apply","turn"];

export function initLogsPanel() {
  const runSel = $("#logRunSelect");
  const togglePerf = $("#togglePerf");
  const search = $("#logSearch");
  const counts = $("#logCounts");
  const list = $("#logList");
  const stageTabs = $$(".stagetab");

  let activeStage = "t1";

  function refreshRunSelect() {
    clear(runSel);
    for (const r of state.runs) {
      const opt = el("option", {value: r.id, text: r.name});
      if (r.id === (state.selectedRunId || "")) opt.selected = true;
      runSel.append(opt);
    }
  }

  function filterFnFactory(q) {
    if (!q) return null;
    const s = q.trim().toLowerCase();
    const kv = s.split(":");
    if (kv.length === 2) {
      const k = kv[0].trim(); const v = kv[1].trim();
      return (obj) => {
        if (obj && typeof obj === "object" && obj[k] != null) {
          return String(obj[k]).toLowerCase().includes(v);
        }
        return false;
      };
    }
    return (obj) => JSON.stringify(obj || "").toLowerCase().includes(s);
  }

  function renderCountsAndList() {
    const r = currentRun();
    clear(counts); clear(list);
    if (!r) return;
    const logs = (r.bundle && r.bundle.logs) || {};
    const ul = el("ul", {class:"list"});
    for (const st of STAGES) {
      const n = Array.isArray(logs[st]) ? logs[st].length : 0;
      ul.append(el("li", {text: `${st}: ${n}`}));
    }
    counts.append(ul);

    const arr = Array.isArray(logs[activeStage]) ? logs[activeStage] : [];
    const filter = filterFnFactory(search.value);
    list.append(renderEntries(arr, {max: 500, filter}));
    // perf logs remain hidden unless toggle is on (we only show core stages in MVP)
  }

  runSel.addEventListener("change", () => { state.selectedRunId = runSel.value; renderCountsAndList(); });
  search.addEventListener("input", () => renderCountsAndList());
  togglePerf.addEventListener("change", () => {
    // In MVP, this is a no-op toggle that reserves the flag; perf data can be shown in PR130
    renderCountsAndList();
  });

  for (const t of stageTabs) {
    t.addEventListener("click", () => {
      stageTabs.forEach(x => x.classList.remove("active"));
      t.classList.add("active");
      activeStage = t.getAttribute("data-stage");
      renderCountsAndList();
    });
  }

  window.addEventListener("run-selected", () => { refreshRunSelect(); renderCountsAndList(); });

  refreshRunSelect();
  renderCountsAndList();
}
