#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="${1:-$PWD}"
SOURCE_ROOT="${2:-${SYMPHONY_SOURCE_TREE_ROOT:-}}"

if [ -z "$SOURCE_ROOT" ] || [ ! -d "$SOURCE_ROOT" ]; then
  echo "sync_workspace_back: source tree missing" >&2
  exit 1
fi

if [ ! -d "$WORKSPACE_ROOT" ]; then
  echo "sync_workspace_back: workspace missing: $WORKSPACE_ROOT" >&2
  exit 1
fi

has_changes=0
git_available=0
if command -v git >/dev/null 2>&1 && git -C "$WORKSPACE_ROOT" rev-parse --verify HEAD >/dev/null 2>&1; then
  git_available=1
  if [ -n "$(git -C "$WORKSPACE_ROOT" status --porcelain 2>/dev/null)" ]; then
    has_changes=1
  fi
else
  has_changes=1
fi

if [ "$has_changes" -ne 1 ]; then
  exit 0
fi

default_excludes() {
  cat <<'EOF'
.git
.DS_Store
.symphony
infra/agents/var
.venv
venv
node_modules
dist
build
_build
deps
.pytest_cache
__pycache__
.mypy_cache
.ruff_cache
.next
.turbo
coverage
EOF
}

sync_excludes() {
  if [ -n "${SYMPHONY_SYNC_EXCLUDES:-}" ]; then
    printf '%s\n' "$SYMPHONY_SYNC_EXCLUDES" | tr ':,' '\n'
  else
    default_excludes
  fi
}

copy_git_changed_paths() {
  local paths_file
  paths_file="$(mktemp)"

  {
    git -C "$WORKSPACE_ROOT" diff --name-only --relative HEAD
    git -C "$WORKSPACE_ROOT" ls-files --others --exclude-standard
  } | awk 'NF && !seen[$0]++' > "$paths_file"

  while IFS= read -r relpath; do
    [ -n "$relpath" ] || continue
    [ -e "$WORKSPACE_ROOT/$relpath" ] || continue
    rsync -a --relative "$WORKSPACE_ROOT/./$relpath" "$SOURCE_ROOT/"
  done < "$paths_file"

  rm -f "$paths_file"
}

if command -v rsync >/dev/null 2>&1; then
  if [ "$git_available" -eq 1 ]; then
    copy_git_changed_paths
  else
    rsync_args=(-a)
    while IFS= read -r exclude; do
      [ -n "$exclude" ] || continue
      rsync_args+=("--exclude" "$exclude")
    done < <(sync_excludes)
    rsync "${rsync_args[@]}" "$WORKSPACE_ROOT"/ "$SOURCE_ROOT"/
  fi
else
  echo "sync_workspace_back: rsync is required" >&2
  exit 1
fi
