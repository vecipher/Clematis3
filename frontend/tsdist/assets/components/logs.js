import { el, clear } from "../util/dom.js";
import { stableSort } from "../util/sort.js";
import { renderJSONPre } from "../ui/jsonview.js";
export function renderLogs(container, bundle, showPerf, selected) {
    clear(container);
    // Base logs
    const base = el("div", { class: "log-block" }, [
        el("h4", {}, ["Logs"]),
    ]);
    container.append(base);
    renderJSONPre(base.appendChild(el("pre")), bundle.logs ?? {});
    // Perf logs (gated)
    if (!showPerf)
        return;
    const perf = bundle.perf ?? {};
    const keys = stableSort(Object.keys(perf), k => k);
    const header = el("div", { class: "perf-header" }, [el("h4", {}, ["Perf logs (experimental)"])]);
    const select = el("select");
    select.append(el("option", { value: "" }, ["— pick a file —"]));
    for (const k of keys)
        select.append(el("option", { value: k, selected: selected === k }, [k]));
    header.append(select);
    container.append(header);
    const pane = el("div", { class: "perf-pane" });
    container.append(pane);
    const renderOne = (k) => {
        clear(pane);
        if (!k)
            return;
        pane.append(el("h5", {}, [k]));
        const entries = perf[k] ?? [];
        pane.append(el("div", { class: "perf-count" }, [`${entries.length} entries`]));
        renderJSONPre(pane.appendChild(el("pre")), entries);
    };
    renderOne(selected ?? "");
    select.addEventListener("change", () => renderOne(select.value));
}
