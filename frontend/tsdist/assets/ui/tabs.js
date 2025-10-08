export function initTabs(container) {
    const buttons = Array.from(container.querySelectorAll("[data-tab]"));
    const panels = new Map();
    for (const p of Array.from(container.querySelectorAll("[data-panel]"))) {
        panels.set(p.dataset.panel, p);
    }
    const activate = (name) => {
        for (const b of buttons)
            b.classList.toggle("active", b.dataset.tab === name);
        for (const [k, p] of panels)
            p.style.display = k === name ? "" : "none";
    };
    for (const b of buttons)
        b.addEventListener("click", () => activate(b.dataset.tab));
    if (buttons.length)
        activate(buttons[0].dataset.tab);
}
