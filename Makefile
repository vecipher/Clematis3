# Deterministic frontend scaffolding (M14 / PR127)
.PHONY: frontend-build frontend-clean frontend-checksum frontend-offline-check

FRONTEND_DIST := frontend/dist

frontend-build:
	@python scripts/build_frontend.py

frontend-clean:
	@rm -rf $(FRONTEND_DIST)

frontend-checksum:
	@python scripts/hashdir.py $(FRONTEND_DIST)

# Fail if any external URLs appear in built assets
frontend-offline-check:
	@rg -n "(https?://|//cdn)" $(FRONTEND_DIST) && (echo "External URL found"; exit 2) || echo "OK: no external URLs"
