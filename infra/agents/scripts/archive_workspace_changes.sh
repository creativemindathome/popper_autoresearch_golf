#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="${1:-$PWD}"
ARCHIVE_ROOT="${2:-$PWD/infra/agents/var/symphony-archives}"
WORKSPACE_NAME="$(basename "$WORKSPACE_ROOT")"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
DEST="$ARCHIVE_ROOT/$WORKSPACE_NAME-$STAMP"

if [ ! -d "$WORKSPACE_ROOT" ]; then
  exit 0
fi

mkdir -p "$DEST"

CHANGED_LIST="$DEST/changed-files.txt"
if command -v git >/dev/null 2>&1 && git -C "$WORKSPACE_ROOT" rev-parse --verify HEAD >/dev/null 2>&1; then
  git -C "$WORKSPACE_ROOT" status --porcelain > "$CHANGED_LIST" 2>/dev/null || true
else
  find "$WORKSPACE_ROOT" -type f > "$CHANGED_LIST"
fi

if [ ! -s "$CHANGED_LIST" ]; then
  rmdir "$DEST" 2>/dev/null || true
  exit 0
fi

while IFS= read -r line; do
  path="${line#?? }"
  [ -n "$path" ] || continue
  [ -e "$WORKSPACE_ROOT/$path" ] || continue
  mkdir -p "$DEST/$(dirname "$path")"
  cp -R "$WORKSPACE_ROOT/$path" "$DEST/$path"
done < "$CHANGED_LIST"

