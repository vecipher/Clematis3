#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
source packaging/repro_env.sh

rm -rf dist build *.egg-info
python -m pip install --upgrade build

# build #1
python -m build
python - <<'PY'
import hashlib, glob, os, json
art = sorted(glob.glob("dist/*"))
shas = {os.path.basename(p): hashlib.sha256(open(p,"rb").read()).hexdigest() for p in art}
print(json.dumps(shas, sort_keys=True))
PY > .repro_sha_1.json

# clean and build #2
rm -rf dist build *.egg-info
python -m build
python - <<'PY'
import hashlib, glob, os, json
art = sorted(glob.glob("dist/*"))
shas = {os.path.basename(p): hashlib.sha256(open(p,"rb").read()).hexdigest() for p in art}
print(json.dumps(shas, sort_keys=True))
PY > .repro_sha_2.json

diff -u .repro_sha_1.json .repro_sha_2.json
echo "Reproducible: checksums identical."