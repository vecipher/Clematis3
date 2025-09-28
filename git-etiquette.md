

# Git etiquette & guardrails

Practical defaults to avoid detached HEAD, lost files, and broken CI. Use this as the house style for day‑to‑day work.

---

## Daily flow (feature → PR → merge)

1. **Sync main**
   ```bash
   git switch main
   git pull --ff-only
   ```
2. **Always branch before work**
   ```bash
   # preferred (uses our alias below)
   git start <feature-name> [base-ref]   # e.g., git start pr57 v0.8.0a2
   
   # regular git if you prefer
   git switch -c <feature-name> [base-ref]
   ```
3. **Edit → stage → commit**
   ```bash
   git add -A
   git commit -m "PRNN: concise summary"
   ```
4. **Push + open PR**
   ```bash
   git push -u origin <feature-name>
   gh pr create --fill --base main --head <feature-name>
   # optional: auto-merge when checks pass
   gh pr merge --squash --auto
   ```
5. **After merge**
   ```bash
   git switch main
   git pull --ff-only
   ```

### Tag & publish (when the PR ships artifacts)
- **SBOM/attest (pkg build):** push a tag `v*` to trigger SBOM + SLSA.
- **OCI image (GHCR):** push a tag `v*` to build/push multi‑arch images.
```bash
TAG=v0.8.0a2
git tag -a "$TAG" -m "Ship PRNN"
git push origin "$TAG"
```

---

## Local guardrails (install once)

### A) Hooks: block detached‑HEAD commits and pushes to `main`
> Install globally, then copy into this repo once.

```bash
# Global hooks directory
mkdir -p ~/.git-templates/hooks

# Block commits on detached HEAD or directly on main
cat > ~/.git-templates/hooks/pre-commit <<'SH'
#!/usr/bin/env bash
branch=$(git symbolic-ref --quiet --short HEAD) || { echo "❌ Detached HEAD. Create a branch: git start <name> [base]"; exit 1; }
if [[ "$branch" == "main" || "$branch" == "master" ]]; then
  echo "❌ You're on $branch. Create a feature branch: git start <name>"
  exit 1
fi
SH
chmod +x ~/.git-templates/hooks/pre-commit

# Block pushes from detached HEAD or to main
cat > ~/.git-templates/hooks/pre-push <<'SH'
#!/usr/bin/env bash
branch=$(git symbolic-ref --quiet --short HEAD) || { echo "❌ Detached HEAD. Create a branch: git start <name>"; exit 1; }
if [[ "$branch" == "main" || "$branch" == "master" ]]; then
  echo "❌ Refusing to push $branch directly. Open a PR from a feature branch."
  exit 1
fi
SH
chmod +x ~/.git-templates/hooks/pre-push

# Make these the default for new repos
git config --global init.templateDir ~/.git-templates

# Add to current repo
cp ~/.git-templates/hooks/* .git/hooks/
chmod +x .git/hooks/pre-commit .git/hooks/pre-push
```

### B) Alias: one command to “always branch”
```bash
git config --global alias.start '!f(){ 
  if [ -z "$1" ]; then echo "usage: git start <branch> [base-ref]"; exit 2; fi
  git switch -c "$1" "${2:-@}";
}; f'
```
Usage:
```bash
git start pr57            # from current HEAD
git start pr57 v0.8.0a2   # from a tag/sha without detached HEAD
```

### C) Quick sanity before switching
```bash
git symbolic-ref -q --short HEAD || echo "DETACHED"
git status --porcelain=v1
```

---

## CI expectations (what runs when)

- **pkg_build.yml**
  - Runs on **push to any branch** and **pull_request**.
  - Tag‑only job inside it handles **SBOM + attestations**.

- **oci_publish.yml**
  - **Dry build (no push)** runs on branch/PR pushes to catch Dockerfile breakage.
  - **Publish** runs on **tag `v*`** and **release: published** (multi‑arch, offline smokes).

**Tip:** use new tags (`v0.8.0a3`, `v0.8.0a4`, …) instead of force‑moving tags.

---

## Branch protection (server‑side)

Protect `main`:
- Require PRs (enforce for admins), at least 1 review.
- Require status checks: Tests, Wheel Smoke, CLI Smokes, Reproducible Build, OCI dry build, CodeQL (as applicable).
- Require conversation resolution; optionally require linear history.
- Disallow force pushes and deletion.

---

## Recovery checklist (if you slip)
```bash
# Find lost work
git reflog --date=iso | head -n 15

# Put it back on a branch
git switch -c rescue/<topic> <sha>

# If files were untracked but still on disk, re-add and commit
git add -A && git commit -m "restore"

# Merge back into main cleanly
git switch main && git pull --ff-only
git merge --no-ff rescue/<topic>   # or: git cherry-pick <sha>...
git push origin main
```

---

## Quick PR wrap‑up (copy/paste)
```bash
# Edit → commit → PR
git switch main && git pull --ff-only
git start prNN
git add -A && git commit -m "PRNN: concise summary"
git push -u origin prNN
gh pr create --fill --base main --head prNN
gh pr merge --squash --auto

# Tag if the PR publishes artifacts
TAG=v0.8.0a2
git tag -a "$TAG" -m "Ship PRNN"
git push origin "$TAG"
```