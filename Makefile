# Deterministic frontend scaffolding (M14 / PR127)
.PHONY: frontend-build frontend-clean frontend-checksum frontend-offline-check frontend-stage frontend-repro demo-bundle
.ONESHELL:

FRONTEND_DIST := frontend/dist
PKG_FRONTEND_DIST := clematis/frontend/dist

frontend-build:
	@python scripts/build_frontend.py
	@$(MAKE) frontend-stage

frontend-clean:
	@rm -rf $(FRONTEND_DIST) $(PKG_FRONTEND_DIST)

frontend-checksum:
	@python scripts/hashdir.py $(FRONTEND_DIST)

# Fail if any external URLs appear in built assets
frontend-offline-check:
	@sh -c 'set -e; if rg -n "(https?://|//cdn)" $(FRONTEND_DIST) | grep -v "www\\.w3\\.org/2000/svg" | grep -q .; then echo "External URL found"; exit 2; else echo "OK: no external URLs"; fi'

# Mirror built assets into the Python package path for packaging/tests
frontend-stage:
	@rm -rf $(PKG_FRONTEND_DIST)
	@mkdir -p $(PKG_FRONTEND_DIST)
	@cp -R $(FRONTEND_DIST)/* $(PKG_FRONTEND_DIST)/

# Wrap local reproducibility check for the viewer
frontend-repro:
	@bash scripts/repro_check_local.sh --frontend

# Regenerate the deterministic demo bundle used by the viewer example (PR135)
demo-bundle:
	@mkdir -p clematis/examples/run_bundles
	@TZ=UTC PYTHONUTF8=1 PYTHONHASHSEED=0 LC_ALL=C.UTF-8 SOURCE_DATE_EPOCH=315532800 CLEMATIS_NETWORK_BAN=1 \
	python -m clematis console -- step --now-ms 315532800000 --out clematis/examples/run_bundles/run_demo_bundle.json
