

#!/usr/bin/env bash
# Local reproducibility check: build sdist + wheel with pinned toolchain and stable env,
# print SHA256s, and (optionally) build twice to assert byte-for-byte equality.
#
# Usage:
#   scripts/repro_check_local.sh [--twice] [--sde <unix_ts>] [--no-install]
#
# Flags:
#   --twice       Build twice and compare SHA256 checksums (fails on mismatch).
#   --sde <ts>    Override SOURCE_DATE_EPOCH (defaults to last git commit timestamp, or 0).
#   --no-install  Skip installing pinned build tools (use current environment as-is).
#
# Environment overrides (optional):
#   SDE             = unix timestamp for SOURCE_DATE_EPOCH
#   BUILD_VER       = version for 'build'     (default 1.2.1)
#   WHEEL_VER       = version for 'wheel'     (default 0.45.1)
#   SETUPTOOLS_VER  = version for 'setuptools' (default 80.9.0)
#   TWINE_VER       = version for 'twine'     (default 5.1.1)
set -euo pipefail

die() { echo "error: $*" >&2; exit 2; }

# --- Parse args ---
TWICE=0
NO_INSTALL=0
SDE="${SDE:-}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --twice) TWICE=1; shift ;;
    --no-install) NO_INSTALL=1; shift ;;
    --sde) shift; [[ $# -gt 0 ]] || die "--sde requires a value"; SDE="$1"; shift ;;
    -h|--help) sed -n '1,50p' "$0"; exit 0 ;;
    *) die "unknown arg: $1" ;;
  esac
done

# --- Resolve repo root ---
if ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"; then
  cd "$ROOT"
else
  echo "[warn] Not in a git repo; using current directory: $(pwd)"
fi

# --- Defaults ---
BUILD_VER="${BUILD_VER:-1.2.1}"
WHEEL_VER="${WHEEL_VER:-0.45.1}"
SETUPTOOLS_VER="${SETUPTOOLS_VER:-80.9.0}"
TWINE_VER="${TWINE_VER:-5.1.1}"

if [[ -z "${SDE}" ]]; then
  if TS="$(git log -1 --pretty=%ct 2>/dev/null)"; then
    SDE="${TS}"
  else
    SDE="0"
  fi
fi

echo "[info] SOURCE_DATE_EPOCH=${SDE}"

export TZ=UTC
export PYTHONUTF8=1
export PYTHONIOENCODING=UTF-8
export PYTHONHASHSEED=0
export SOURCE_DATE_EPOCH="${SDE}"
export PIP_DISABLE_PIP_VERSION_CHECK=1

# --- Install pinned toolchain (unless skipped) ---
if [[ "${NO_INSTALL}" -eq 0 ]]; then
  python -m pip install -U pip >/dev/null
  python -m pip install --no-deps \
    "build==${BUILD_VER}" \
    "wheel==${WHEEL_VER}" \
    "setuptools==${SETUPTOOLS_VER}" \
    "twine==${TWINE_VER}" >/dev/null
else
  echo "[info] Skipping toolchain install (--no-install)"
fi

py_hash() {
  python - "$@" <<'PY'
import hashlib, pathlib, sys
p = pathlib.Path("dist")
if not p.exists():
    print("dist/ does not exist", file=sys.stderr); sys.exit(2)
for f in sorted(p.glob("*")):
    h = hashlib.sha256(f.read_bytes()).hexdigest()
    print(f"{h}  {f.name}")
PY
}

build_once() {
  rm -rf dist
  python -m build --sdist --wheel
  echo "--- SHA256 (dist) ---"
  py_hash
}

compare_dirs() {
  local A="$1" B="$2"
  python - "$A" "$B" <<'PY'
import hashlib, pathlib, sys
a, b = map(pathlib.Path, sys.argv[1:3])
def digest_dir(d: pathlib.Path):
    files = sorted(p for p in d.glob("*") if p.is_file())
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
da, db = digest_dir(a), digest_dir(b)
names = sorted(set(da) | set(db))
bad = False
for n in names:
    ha, hb = da.get(n), db.get(n)
    if ha != hb:
        print(f"[DIFF] {n}\n  A: {ha}\n  B: {hb}")
        bad = True
if bad:
    sys.exit(1)
print("Repro check passed: all artifact filenames and SHA256 digests match.")
PY
}

# --- Run ---
echo "[info] Building (pass #1) ..."
build_once

if [[ "${TWICE}" -eq 1 ]]; then
  echo "[info] Rebuilding (pass #2) and comparing ..."
  rm -rf dist.1 dist.2
  mv dist dist.1
  python -m build --sdist --wheel
  mv dist dist.2
  echo "--- SHA256 (dist.1) ---"; (cd dist.1 && py_hash || true)
  echo "--- SHA256 (dist.2) ---"; (cd dist.2 && py_hash || true)
  compare_dirs dist.1 dist.2
fi

echo "[done] Repro check completed."
