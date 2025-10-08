import {$$} from "../util/dom.js";

export function initTabs(container) {
  const tabs = $$(".tab", container);
  const panels = $$(".tabpanel", document);
  for (const t of tabs) {
    t.addEventListener("click", () => {
      tabs.forEach(x => x.classList.remove("active"));
      t.classList.add("active");
      const target = t.getAttribute("data-tab");
      panels.forEach(p => {
        p.classList.toggle("active", p.getAttribute("data-panel") === target);
      });
    });
  }
}
