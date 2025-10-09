import type { RunBundle } from "../state.js";

function pickMemory(bundle: RunBundle): Array<Record<string, unknown>> {
  // Try a few obvious homes; fall back to empty.
  const m1 = (bundle as any).memory;
  if (Array.isArray(m1)) return m1 as Array<Record<string, unknown>>;
  const s0 = Array.isArray(bundle.snapshots) ? (bundle.snapshots as any[])[0] : undefined;
  const m2 = s0 && (s0.memory || s0.memories);
  if (Array.isArray(m2)) return m2 as Array<Record<string, unknown>>;
  return [];
}

export function renderMemory(container: HTMLElement, bundle: RunBundle) {
  container.textContent = "";
  const list = pickMemory(bundle);

  const header = document.createElement("div");
  header.className = "exp-mem-header";
  header.textContent = `${list.length} memory items`;
  container.appendChild(header);

  const ul = document.createElement("ul");
  ul.className = "exp-mem-list";
  for (const item of list) {
    const li = document.createElement("li");
    const keys = Object.keys(item).sort();
    const tsKey = keys.find(k => /time|ts|stamp|date/i.test(k));
    const label = keys.find(k => /id|label|name|key|title/i.test(k));
    const ts = tsKey ? String((item as any)[tsKey]) : "—";
    const id = label ? String((item as any)[label]) : "(item)";
    li.textContent = `${id} — ${ts}`;
    ul.appendChild(li);
  }
  container.appendChild(ul);
}
