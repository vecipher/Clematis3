

"""
Native (optional) acceleration backends for Clematis.

PR97 scaffold:
- Package marker only; no runtime imports.
- Actual kernels (e.g., T1) are gated behind `perf.native.*` config and will be added in later PRs.
- Keeping this init empty (besides this docstring) prevents accidental import side‑effects.
"""
