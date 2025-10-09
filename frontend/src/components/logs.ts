import { el, clear } from "../util/dom.js";
import { stableSort } from "../util/sort.js";
import { renderJSONPre } from "../ui/jsonview.js";
import type { RunBundle, Stage } from "../state.js";

/**
 * Renders Logs with basic UX:
 * - Run selector (#logRunSelect) populated from bundle.runs or single-run fallback
 * - Stage chips (.stagetab[data-stage]) filter entries
 * - Optional text search (#logSearch) over JSON stringified entries
 * - Perf panel remains as before (gated by showPerf)
 */
export function renderLogs(container: HTMLElement, bundle: RunBundle, showPerf: boolean, selectedPerfKey: string | null) {
  clear(container);

  // Normalize to runs[]
  type RunLike = { id: string; logs: Record<string, unknown[]> };
  const runs: RunLike[] = Array.isArray(bundle.runs) && (bundle.runs as any[]).length
    ? (bundle.runs as any[]).map((r, i) => ({ id: String((r as any).id ?? `run-${i+1}`), logs: (r as any).logs ?? {} }))
    : [{ id: "run-1", logs: bundle.logs ?? {} }];

  // Wire run select
  const runSel = document.querySelector<HTMLSelectElement>("#logRunSelect");
  if (runSel) {
    const prev = runSel.value;
    runSel.innerHTML = "";
    runs.forEach((r, i) => {
      const opt = document.createElement("option");
      opt.value = String(i);
      opt.textContent = r.id;
      runSel.appendChild(opt);
    });
    const idx = /^\d+$/.test(prev) && Number(prev) < runs.length ? Number(prev) : 0;
    runSel.value = String(idx);
  }

  // Stage selection via chips
  const stages: Stage[] = ["t1", "t2", "t3", "t3_reflection", "t4", "apply", "turn"];
  const chips = Array.from(document.querySelectorAll<HTMLButtonElement>(".log-tabs .stagetab"));
  const activeChip = chips.find(b => b.classList.contains("active"));
  let selectedStage: Stage = (activeChip?.dataset.stage as Stage) || "t1";

  // Search box
  const searchBox = document.querySelector<HTMLInputElement>("#logSearch");

  // Counts header
  const countsPane = document.querySelector<HTMLElement>("#logCounts");

  function filterEntries(arr: unknown[]): unknown[] {
    const q = searchBox?.value?.trim() ?? "";
    if (!q) return arr;
    const needle = q.toLowerCase();
    return arr.filter(e => {
      try { return JSON.stringify(e).toLowerCase().includes(needle); } catch { return false; }
    });
  }

  function renderBody() {
    clear(container);
    const runIdx = runSel ? Math.min(Math.max(parseInt(runSel.value || "0", 10) || 0, 0), runs.length - 1) : 0;
    const run = runs[runIdx];

    // counts
    const stageCounts: Record<string, number> = {};
    stages.forEach(s => { stageCounts[s] = Array.isArray((run.logs as any)[s]) ? ((run.logs as any)[s] as any[]).length : 0; });
    if (countsPane) {
      countsPane.textContent = stages.map(s => `${s.toUpperCase()}: ${stageCounts[s]}`).join(" • ");
    }

    // render selected stage entries
    const entries: unknown[] = Array.isArray((run.logs as any)[selectedStage]) ? ((run.logs as any)[selectedStage] as unknown[]) : [];
    const filtered = filterEntries(entries);

    const base = el("div", { class: "log-block" }, [
      el("h4", {}, [`Logs — ${selectedStage.toUpperCase()} (${filtered.length})`]),
    ]);
    container.append(base);
    renderJSONPre(base.appendChild(el("pre")), filtered);

    // Perf logs (gated)
    if (showPerf) {
      const perf = (bundle as any).perf ?? {};
      const keys = stableSort(Object.keys(perf), k => k);
      const header = el("div", { class: "perf-header" }, [el("h4", {}, ["Perf logs (experimental)"])]);
      const select = el("select") as HTMLSelectElement;
      select.append(el("option", { value: "" }, ["— pick a file —"]));
      for (const k of keys) select.append(el("option", { value: k, selected: selectedPerfKey === k }, [k]));
      header.append(select);
      container.append(header);

      const pane = el("div", { class: "perf-pane" });
      container.append(pane);
      const renderPerf = (k: string) => {
        clear(pane);
        if (!k) return;
        pane.append(el("h5", {}, [k]));
        const entries = (perf as any)[k] ?? [];
        pane.append(el("div", { class: "perf-count" }, [`${entries.length} entries`]));
        renderJSONPre(pane.appendChild(el("pre")), entries);
      };
      renderPerf(select.value || "");
      select.addEventListener("change", () => renderPerf(select.value));
    }
  }

  // Wire events once per render (cheap)
  chips.forEach(b => {
    b.addEventListener("click", () => {
      chips.forEach(x => x.classList.toggle("active", x === b));
      selectedStage = (b.dataset.stage as Stage) || "t1";
      renderBody();
    });
  });
  if (runSel) runSel.addEventListener("change", renderBody);
  if (searchBox) searchBox.addEventListener("input", () => { renderBody(); });

  renderBody();
}
