#!/usr/bin/env bash
# Sync agent-deployments docs into the bundled package data directory.
# Run this before building a release wheel.
set -euo pipefail

REPO_URL="${AGENT_DEPLOYMENTS_REPO:-https://github.com/jagguvarma15/agent-deployments.git}"
DEST="src/agent_scaffold/_bundled_deployments/docs"
TMP_DIR=$(mktemp -d)

echo "Cloning agent-deployments (shallow)..."
git clone --depth 1 "$REPO_URL" "$TMP_DIR"

echo "Copying docs into $DEST..."
rm -rf "$DEST"
mkdir -p "$DEST"

for subdir in recipes patterns frameworks stack cross-cutting capabilities; do
    if [ -d "$TMP_DIR/docs/$subdir" ]; then
        cp -r "$TMP_DIR/docs/$subdir" "$DEST/"
        echo "  copied docs/$subdir"
    fi
done

rm -rf "$TMP_DIR"
echo "Done. Bundled docs ready at $DEST"
