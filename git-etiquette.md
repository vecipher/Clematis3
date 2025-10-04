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
**Branch naming conventions**
- Use short, descriptive names: `feat/<topic>`, `fix/<bug>`, `chore/<task>`, or `prNN`.
- Avoid slashes that imply ownership or PII. Stick to ASCII and `-`/`_`.
- If the branch is stacked, suffix with an ordinal: `feat/<topic>-2-of-3`.
3. **Edit → stage → commit**
   ```bash
   git add -A
   git commit -m "PRNN: concise summary"
   ```
#### Commit message style
- Use **imperative mood**: “Add X”, not “Adds/Added X”.
- Keep the **subject ≤ 72 chars**; wrap the body at ~72 cols.
- Subject: `PRNN: concise summary` (include `PRNN` when known).
- Body (optional but encouraged): explain **why**, **scope**, and **trade‑offs**. Link issues/PRs.
- Useful trailers:
  ```
  Co-authored-by: Full Name <name@example.com>
  Reverts: <short-sha>
  ```
- For small follow‑ups, use **fixup commits** and autosquash (see below).
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

### Rebase & fixup workflow (before pushing)
Keep history clean without losing review context.
```bash
# Create a fixup commit against an earlier commit
git commit --fixup=<target-sha>
# Autosquash and rebase onto your base (origin/main or a tag)
git fetch origin
git rebase -i --autosquash origin/main
# If your branch was based on a tag:
git rebase -i --autosquash v0.8.0a2
```
**Notes**
- It’s fine to force‑push your **feature** branch: `git push -f` (never to `main`).
- Prefer **squash‑merge** for PRs (matches the flow above).
- If rebasing after review, leave a brief comment so reviewers aren’t surprised.

### Tag & publish (when the PR ships artifacts)
- **SBOM/attest (pkg build):** push a tag `v*` to trigger SBOM + SLSA.
- **OCI image (GHCR):** push a tag `v*` to build/push multi‑arch images.
```bash
TAG=v0.8.0a2
git tag -a "$TAG" -m "Ship PRNN"
git push origin "$TAG"
```
### Large files & generated assets
- Don’t commit build artifacts or huge binaries; put them in releases or an artifact store.
- If you must version binaries, use **Git LFS** and add patterns to `.gitattributes`.
  ```bash
  git lfs install
  git lfs track "*.bin" "*.onnx" "*.mp4"
  git add .gitattributes
  git commit -m "track large assets via LFS"
  ```
- Keep `.gitignore` updated for tool caches (e.g., `.pytest_cache`, `dist/`, `build/`, `.DS_Store`).

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

### (Optional) Add secret‑scanning to pre‑push
If you have `gitleaks` installed, add this to `.git/hooks/pre-push` *after* the branch checks:
```bash
if command -v gitleaks >/dev/null 2>&1; then
  echo "🛡  gitleaks protect..."
  gitleaks protect --staged --redact --no-git --verbose || {
    echo "❌ Possible secret detected. Aborting push."
    exit 1
  }
fi
```
Alternatives: `git-secrets` or a centralized pre‑commit framework can be used instead.

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

### C) Helpful aliases (fixups & undo)
```bash
# Create a fixup commit against the last commit (or provide a sha)
git config --global alias.fixup '!f(){ git commit --fixup="${1:-HEAD}"; }; f'
# Interactive autosquash rebase onto origin/main
git config --global alias.rsquash '!git fetch origin && git rebase -i --autosquash origin/main'
# Show current branch or DETACHED
git config --global alias.where '!git symbolic-ref -q --short HEAD || echo DETACHED'
# Soft undo (keep changes staged)
git config --global alias.undo '!git reset --soft HEAD~1'
```

### D) Quick sanity before switching
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

### Reverting vs resetting (public history)
- On **public branches** (e.g., `main`), prefer `git revert <sha>` to create a new commit that undoes a bad change.
- Avoid `git reset --hard` on public branches; it rewrites history and will be blocked by protection rules.
- When you revert, reference the original commit in the message (use the `Reverts:` trailer).
