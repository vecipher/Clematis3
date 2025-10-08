import { renderJSONPre } from "../ui/jsonview.js";
import type { RunBundle } from "../state.js";

export function renderSnapshots(container: HTMLElement, bundle: RunBundle) {
  renderJSONPre(container, bundle.snapshots ?? []);
}
