# tests/reflection/test_reflection_disabled_identity.py
import os
from pathlib import Path
import importlib
import json
import pytest
import shutil

from types import SimpleNamespace as _SNS
def _to_ns(x):
    if isinstance(x, dict):
        return _SNS(**{k: _to_ns(v) for k, v in x.items()})
    if isinstance(x, list):
        return [_to_ns(v) for v in x]
    return x

# Helper to convert namespaces/dicts/lists to plain dicts/lists
def _to_plain(x):
    from types import SimpleNamespace
    if isinstance(x, dict):
        return {k: _to_plain(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_plain(v) for v in x]
    if isinstance(x, SimpleNamespace):
        return {k: _to_plain(v) for k, v in x.__dict__.items()}
    return x

class _CfgAdaptor:
    def __init__(self, data):
        self._d = data if isinstance(data, dict) else {}
    def get(self, key, default=None):
        return self._d.get(key, default)
    def __getitem__(self, key):
        return self._d[key]
    def __getattr__(self, name):
        v = self._d.get(name)
        if isinstance(v, dict):
            return _CfgAdaptor(v)
        return v

NEEDED = ("t1.jsonl","t2.jsonl","t4.jsonl","apply.jsonl","turn.jsonl")

def _run_once(log_dir: str, cfg_overlay: dict):
    # Import lazily to avoid circulars if test discovery scans modules
    Core = importlib.import_module("clematis.engine.orchestrator.core")
    cfgmod = importlib.import_module("configs.validate")
    iolog = importlib.import_module("clematis.engine.util.io_logging")

    # Build ctx/state minimally; adjust to your real types as needed.
    from types import SimpleNamespace as SNS
    ctx = SNS(
        turn_id=1, agent_id="AgentA",
        now_ms=12_345, now="1970-01-01T00:00:12Z",
        _dry_run_until_t4=False
    )
    # Ensure log dir and point orchestrator writers at it via env override
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    prev_log_dir = os.environ.get("CLEMATIS_LOG_DIR")
    prev_logs_dir = os.environ.get("CLEMATIS_LOGS_DIR")
    os.environ["CLEMATIS_LOG_DIR"] = str(log_dir)
    os.environ["CLEMATIS_LOGS_DIR"] = str(log_dir)
    snap_dir = log_path.resolve().parent / "snapshots"
    if snap_dir.exists():
        shutil.rmtree(snap_dir)
    snap_dir.mkdir(parents=True, exist_ok=True)
    setattr(ctx, "log_dir", str(log_dir))

    # Build/validate config with overlay
    default_cfg = getattr(cfgmod, "DEFAULTS", {})
    # If you have a helper to deep-merge + validate, prefer it:
    merged = json.loads(json.dumps(default_cfg))  # deep copy via JSON
    # naive overlay:
    def _merge_dict(dst, src):
        for k, v in (src or {}).items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                _merge_dict(dst[k], v)
            else:
                dst[k] = v
        return dst

    _merge_dict(merged, cfg_overlay or {})
    merged.setdefault("t3", {}).update({"allow_reflection": False})
    merged.setdefault("t4", {})["snapshot_dir"] = str(snap_dir)
    merged.setdefault("scheduler", {}).setdefault("budgets", {}).update({
        "time_ms_reflection": 6000, "ops_reflection": 5
    })
    cfg = cfgmod.validate(merged) if hasattr(cfgmod, "validate") else merged
    cfg_plain = _to_plain(cfg)
    # Provide both attribute/dict hybrid views expected by orchestrator helpers
    setattr(ctx, "cfg", _CfgAdaptor(cfg_plain))
    setattr(ctx, "config", cfg_plain)

    # Prepare a minimal state; at least carry the cfg and an in-memory index placeholder
    state = {"cfg": cfg_plain, "memory_index": None}

    # Create orchestrator and run a single turn with a trivial input
    Orch = getattr(Core, "Orchestrator")
    try:
        orch = Orch()
        orch.run_turn(ctx, state, input_text="hello")  # side-effect: logs emitted to ctx.log_dir

        # Determine actual log directory: prefer requested, fallback to default package ./.logs
        actual = Path(log_dir)
        t1 = actual / "t1.jsonl"
        if not t1.exists():
            try:
                import clematis as _pkg
                pkg_logs = Path(_pkg.__file__).resolve().parent / ".logs"
                if (pkg_logs / "t1.jsonl").exists():
                    actual = pkg_logs
            except Exception:
                pass
        return str(actual)
    finally:
        if prev_log_dir is None:
            os.environ.pop("CLEMATIS_LOG_DIR", None)
        else:
            os.environ["CLEMATIS_LOG_DIR"] = prev_log_dir
        if prev_logs_dir is None:
            os.environ.pop("CLEMATIS_LOGS_DIR", None)
        else:
            os.environ["CLEMATIS_LOGS_DIR"] = prev_logs_dir


# Helper to snapshot identity logs
def _snapshot_identity(src_dir: str, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in NEEDED:
        sp = Path(src_dir) / name
        dp = dst_dir / name
        if not sp.exists():
            raise AssertionError(f"missing {name} in {src_dir}")
        if sp == dp:
            continue
        shutil.copyfile(sp, dp)

def test_reflection_disabled_identity_minimal(tmp_path, monkeypatch):
    monkeypatch.setenv("CI", "true")
    monkeypatch.setenv("CLEMATIS_NETWORK_BAN", "1")

    logs_a = tmp_path / "A"
    logs_b = tmp_path / "B"

    overlay = {"t3": {"allow_reflection": False}}
    src1 = _run_once(str(logs_a), overlay)
    _snapshot_identity(src1, logs_a)
    src2 = _run_once(str(logs_b), overlay)
    _snapshot_identity(src2, logs_b)

    # Assert reflection log absent
    assert not (logs_a / "t3_reflection.jsonl").exists()
    assert not (logs_b / "t3_reflection.jsonl").exists()

    # Byte-level compare
    for name in NEEDED:
        pa = logs_a / name
        pb = logs_b / name
        assert pa.read_bytes() == pb.read_bytes(), f"bytes differ in {name}"
