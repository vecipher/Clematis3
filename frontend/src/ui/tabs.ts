export function initTabs(container: HTMLElement) {
  const buttons = Array.from(container.querySelectorAll<HTMLElement>("[data-tab]"));

  // Panels are siblings elsewhere in the DOM, not inside the nav
  const panelsList = Array.from(document.querySelectorAll<HTMLElement>("[data-panel]"));
  const panels = new Map<string, HTMLElement>();
  for (const p of panelsList) {
    const name = p.dataset.panel!;
    panels.set(name, p);
  }

  const activate = (name: string) => {
    for (const b of buttons) b.classList.toggle("active", b.dataset.tab === name);
    for (const [k, p] of panels) p.style.display = k === name ? "" : "none";
    for (const p of panelsList) p.classList.toggle("active", p.dataset.panel === name);
  };

  for (const b of buttons) b.addEventListener("click", () => activate(b.dataset.tab!));
  if (buttons.length) activate(buttons[0].dataset.tab!);
}
