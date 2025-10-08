import { renderJSONPre } from "../ui/jsonview.js";
export function renderSnapshots(container, bundle) {
    renderJSONPre(container, bundle.snapshots ?? []);
}
