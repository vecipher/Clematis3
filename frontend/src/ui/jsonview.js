import {el} from "../util/dom.js";

function isObject(x) { return x && typeof x === "object" && !Array.isArray(x); }

export function renderJSONCollapsible(obj) {
  const root = el("div");
  const pre = el("pre", {class: "prebox"});
  // deterministic stringify: keys sorted
  const text = JSON.stringify(obj, Object.keys(obj).sort(), 2);
  pre.textContent = text;
  root.append(pre);
  return root;
}

// Minimal collapsible list rendering for arrays of entries (JSONL)
export function renderEntries(entries, {max = 100, filter = null} = {}) {
  const container = el("div", {class: "loglist"});
  let idx = 0;
  for (let i = 0; i < entries.length && idx < max; i++) {
    const e = entries[i];
    if (filter && !filter(e)) continue;
    const item = el("div", {class: "log-item"});
    const hdr = el("div", {class: "hdr"},
      [el("span", {text: `#${i}`}), el("span", {text: isObject(e) ? "object" : typeof e})]);
    const body = el("div", {class: "body"});
    const pre = el("pre", {class: "prebox"});
    pre.textContent = JSON.stringify(e, Object.keys(e || {}).sort(), 2);
    body.append(pre);
    item.append(hdr, body);
    hdr.addEventListener("click", () => item.classList.toggle("open"));
    container.append(item);
    idx++;
  }
  return container;
}
