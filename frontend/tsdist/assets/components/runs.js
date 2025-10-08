import { renderJSONPre } from "../ui/jsonview.js";
export function renderRuns(container, bundle) {
    renderJSONPre(container, bundle.runs ?? []);
}
