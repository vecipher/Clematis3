# Vendored JS (frontend/vendor)

This directory is reserved for third-party browser libraries (e.g., d3, dagre, JSONEditor),
copied as minified files for **offline** use (no CDNs). Add license snippets to top-level NOTICE.

PR127 intentionally ships without third-party JS; visuals arrive in later PRs.

We do not ignore frontend/dist/ in PR127 so we can hash it deterministically if you choose to commit it later. For now the build script creates it each run.
