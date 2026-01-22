#!/bin/bash
# Git Commit Helper for Kernow Homelab Submodules
# Usage: git-commit-submodule.sh <submodule> <message>
# Example: git-commit-submodule.sh agentic_lab "feat: add new MCP server"

set -e

PARENT_DIR="/home"
SUBMODULE="$1"
MESSAGE="$2"

if [[ -z "$SUBMODULE" || -z "$MESSAGE" ]]; then
    echo "Usage: $0 <submodule> <message>"
    echo "Submodules: agentic_lab, prod_homelab, monit_homelab, mcp-servers"
    exit 1
fi

SUBMODULE_PATH="$PARENT_DIR/$SUBMODULE"

if [[ ! -e "$SUBMODULE_PATH/.git" ]]; then
    echo "Error: $SUBMODULE_PATH is not a git repository"
    exit 1
fi

# Step 1: Commit in submodule
echo "==> Committing in $SUBMODULE..."
cd "$SUBMODULE_PATH"

if [[ -z "$(git status --porcelain)" ]]; then
    echo "No changes to commit in $SUBMODULE"
    exit 0
fi

git add -A
git commit -m "$MESSAGE

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"

# Step 2: Push submodule
echo "==> Pushing $SUBMODULE to origin..."
git push origin main

# Step 3: Update parent reference
echo "==> Updating parent repo submodule reference..."
cd "$PARENT_DIR"
git add "$SUBMODULE"
git commit -m "chore: update $SUBMODULE submodule"
git push origin main

echo "==> Done! Submodule $SUBMODULE committed and parent updated."
