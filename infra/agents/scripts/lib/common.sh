#!/usr/bin/env bash

if [[ -n "${INFRA_AGENTS_COMMON_SH_LOADED:-}" ]]; then
  return 0
fi
INFRA_AGENTS_COMMON_SH_LOADED=1

COMMON_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFRA_AGENTS_ROOT="$(cd "$COMMON_SCRIPT_DIR/../.." && pwd)"
PROJECT_ROOT="$(cd "$INFRA_AGENTS_ROOT/../.." && pwd)"

default_stage_excludes() {
  cat <<'EOF'
.git
.DS_Store
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

repo_stage_excludes() {
  if [[ -n "${SYMPHONY_SYNC_EXCLUDES:-}" ]]; then
    printf '%s\n' "$SYMPHONY_SYNC_EXCLUDES" | tr ':,' '\n'
  else
    default_stage_excludes
  fi
}

load_env_file() {
  local env_file="$1"
  if [[ ! -f "$env_file" ]]; then
    echo "Missing env file: $env_file" >&2
    return 1
  fi

  set -a
  # shellcheck disable=SC1090
  . "$env_file"
  set +a
}

require_env_vars() {
  local missing=0
  local name

  for name in "$@"; do
    if [[ -z "${!name:-}" ]]; then
      echo "$name is required." >&2
      missing=1
    fi
  done

  return "$missing"
}

ensure_command() {
  local command_name="$1"
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Required command not found: $command_name" >&2
    return 1
  fi
}

ensure_docker_compose() {
  ensure_command docker
  if ! docker compose version >/dev/null 2>&1; then
    echo "docker compose is required." >&2
    return 1
  fi
}

ensure_runtime_dirs() {
  mkdir -p \
    "$INFRA_AGENTS_ROOT/var/hermes/workspace" \
    "$INFRA_AGENTS_ROOT/var/hermes/data" \
    "$INFRA_AGENTS_ROOT/var/hermes/config" \
    "$INFRA_AGENTS_ROOT/var/hermes/state" \
    "$INFRA_AGENTS_ROOT/var/hermes/tasks" \
    "$INFRA_AGENTS_ROOT/var/symphony-runtime" \
    "$INFRA_AGENTS_ROOT/var/symphony-workspaces"
}

ensure_codex_runtime_links() {
  local codex_home="$1"
  local global_codex_home="${HOME}/.codex"
  local entry=""

  mkdir -p "$codex_home"

  for entry in auth.json config.toml plugins; do
    if [[ -e "$global_codex_home/$entry" && ! -e "$codex_home/$entry" ]]; then
      ln -s "$global_codex_home/$entry" "$codex_home/$entry"
    fi
  done
}

stage_repo_copy() {
  local source_repo="$1"
  local destination_repo="$2"
  local -a exclude_args=()
  local exclude

  mkdir -p "$destination_repo"

  while IFS= read -r exclude; do
    [[ -n "$exclude" ]] || continue
    exclude_args+=("--exclude" "$exclude")
  done < <(repo_stage_excludes)

  if command -v rsync >/dev/null 2>&1; then
    rsync -a --delete "${exclude_args[@]}" "$source_repo/" "$destination_repo/"
  else
    rm -rf "$destination_repo"
    mkdir -p "$destination_repo"
    cp -R "$source_repo"/. "$destination_repo"/

    while IFS= read -r exclude; do
      [[ -n "$exclude" ]] || continue
      rm -rf "$destination_repo/$exclude" 2>/dev/null || true
    done < <(repo_stage_excludes)
  fi
}

initialize_staged_repo_git() {
  local repo_dir="$1"

  if ! command -v git >/dev/null 2>&1; then
    return 0
  fi

  (
    cd "$repo_dir"
    git init -q
    git config user.name "Hermes Workspace"
    git config user.email "hermes@local"
    git add .
    git commit -qm "Workspace snapshot" || true
  )
}

write_repo_change_artifacts() {
  local repo_dir="$1"
  local artifact_dir="$2"

  mkdir -p "$artifact_dir"

  (
    cd "$repo_dir"
    git status --short > "$artifact_dir/changed_files.txt" 2>/dev/null || true
    git diff --binary HEAD -- > "$artifact_dir/repo.diff" 2>/dev/null || true
  )
}

sync_repo_back() {
  local staged_repo="$1"
  local target_repo="$2"
  stage_repo_copy "$staged_repo" "$target_repo"
}
