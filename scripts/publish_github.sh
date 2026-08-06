#!/usr/bin/env bash
# Create GitHub repo and push the local CueSpace main branch.
#
# Prerequisites (one-time):
#   1. Install GitHub CLI: https://cli.github.com/
#   2. gh auth login
#
# Usage:
#   bash scripts/publish_github.sh                  # default: chelili/CueSpace
#   bash scripts/publish_github.sh my-org/CueSpace  # org repo
#   GITHUB_REPO=MyCueSpace bash scripts/publish_github.sh alice

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

GH_BIN="${GH_BIN:-gh}"
if ! command -v "$GH_BIN" >/dev/null 2>&1; then
  if [[ -x /tmp/gh_2.63.2_linux_amd64/bin/gh ]]; then
    GH_BIN=/tmp/gh_2.63.2_linux_amd64/bin/gh
  else
    echo "GitHub CLI (gh) not found. Install from https://cli.github.com/ or set GH_BIN."
    exit 1
  fi
fi

TARGET="${1:-${GITHUB_TARGET:-chelilia/Cuespace}}"
if [[ "$TARGET" == */* ]]; then
  REPO_ID="$TARGET"
else
  REPO_NAME="${GITHUB_REPO:-CueSpace}"
  REPO_ID="${TARGET}/${REPO_NAME}"
fi

if ! "$GH_BIN" auth status >/dev/null 2>&1; then
  echo "Not logged in to GitHub. Run:"
  echo "  $GH_BIN auth login"
  exit 1
fi

if [[ ! -d .git ]]; then
  echo "Missing .git — run local init first (see README or ask maintainer)."
  exit 1
fi

echo ">>> Creating public repo: $REPO_ID"
"$GH_BIN" repo create "$REPO_ID" \
  --public \
  --source=. \
  --remote=origin \
  --description "CueSpace: test-only AVQA inference and evaluation" \
  --push 2>/dev/null || {
  echo ">>> Repo may already exist; wiring remote and pushing..."
  if git remote get-url origin >/dev/null 2>&1; then
    git remote set-url origin "https://github.com/${REPO_ID}.git"
  else
    git remote add origin "https://github.com/${REPO_ID}.git"
  fi
  git push -u origin main
}

echo ""
echo "Done: https://github.com/${REPO_ID}"
