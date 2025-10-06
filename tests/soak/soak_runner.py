#!/usr/bin/env python3
import argparse, json, os, sys, time, subprocess, hashlib
from pathlib import Path

# psutil is test-only dep for this job, IGNORE THE PROBLEM IF ITS THERE
try:
    import psutil
except Exception:
    psutil = None

# Try to call an internal demo runner if present; otherwise fall back to CLI.
def _run_demo_steps(steps: int, log_dir: Path, snap_dir: Path, snapshot_every: int) -> None:
    # Ensure dirs
    log_dir.mkdir(parents=True, exist_ok=True)
    snap_dir.mkdir(parents=True, exist_ok=True)

    # Best-effort internal entrypoints (avoid hard dependency)
    candidates = [
        ("clematis.scripts.demo", "main"),
        ("clematis.world.demo", "run_demo"),
        ("clematis.demo", "main"),
    ]
    for mod_name, fn_name in candidates:
        try:
            mod = __import__(mod_name, fromlist=[fn_name])
            fn = getattr(mod, fn_name)
            fn(steps=steps, log_dir=str(log_dir), snapshot_dir=str(snap_dir), snapshot_every=snapshot_every)  # type: ignore
            return
        except Exception:
            pass

    # Fallback to CLI umbrella; pass via env to keep args stable
    env = os.environ.copy()
    env["CLEMATIS_DEMO_STEPS"] = str(steps)
    env["CLEMATIS_SNAPSHOT_EVERY"] = str(snapshot_every)
    env["CLEMATIS_LOG_DIR"] = str(log_dir)
    env["CLEMATIS_SNAPSHOT_DIR"] = str(snap_dir)
    # Delegate to scripts via the umbrella CLI; tolerate unknown flags by env usage
    cmd = [sys.executable, "-m", "clematis", "demo", "--"]
    subprocess.run(cmd, check=True, env=env)

def _hash_dir(path: Path) -> str:
    h = hashlib.sha256()
    if not path.exists():
        return h.hexdigest()
    for p in sorted(path.rglob("*")):
        if p.is_file():
            h.update(p.name.encode())
            with p.open("rb") as f:
                while True:
                    b = f.read(1 << 20)
                    if not b: break
                    h.update(b)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--snapshot-every", type=int, required=True)
    ap.add_argument("--log-dir", type=Path, required=True)
    ap.add_argument("--snap-dir", type=Path, required=True)
    ap.add_argument("--metrics", type=Path, required=True)
    args = ap.parse_args()

    proc = psutil.Process(os.getpid()) if psutil else None
    rss_series = []
    cpu_series = []
    checkpoints = []

    t0 = time.time()
    # Lightweight sampler
    def sample(checkpoint=None):
        if psutil:
            rss_mb = proc.memory_info().rss / (1024 * 1024)
            cpu_pct = psutil.cpu_percent(interval=None)
            rss_series.append(rss_mb)
            cpu_series.append(cpu_pct)
            if checkpoint is not None:
                checkpoints.append({"step": checkpoint, "rss_mb": rss_mb})
    sample()

    # Run
    _run_demo_steps(args.steps, args.log_dir, args.snap_dir, args.snapshot_every)
    sample(args.steps)

    dur = time.time() - t0
    logs_hash = _hash_dir(args.log_dir)
    snaps_hash = _hash_dir(args.snap_dir)

    metrics = {
        "steps": args.steps,
        "duration_s": round(dur, 3),
        "rss_peak_mb": round(max(rss_series) if rss_series else 0.0, 1),
        "cpu_mean_pct": round(sum(cpu_series)/len(cpu_series), 1) if cpu_series else 0.0,
        "rss_series_mb": [round(x, 1) for x in rss_series],
        "checkpoints": checkpoints,
        "logs_hash": logs_hash,
        "snapshots_hash": snaps_hash,
        "log_dir": str(args.log_dir),
        "snap_dir": str(args.snap_dir),
        "ts_utc": int(time.time()),
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2))
    print(json.dumps({"ok": True, "metrics_path": str(args.metrics)}, separators=(",", ":")))

if __name__ == "__main__":
    main()
