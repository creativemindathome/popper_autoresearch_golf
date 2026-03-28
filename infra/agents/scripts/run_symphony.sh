#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_symphony.sh [symphony-args...]
  run_symphony.sh --doctor

Description:
  Start the repo-local Symphony orchestrator with env and runtime paths derived
  from this repository.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
. "$SCRIPT_DIR/lib/common.sh"

ROOT="$INFRA_AGENTS_ROOT"
SYMPHONY_DIR="$ROOT/symphony/elixir"
WORKFLOW_FILE="$ROOT/symphony/WORKFLOW.hermes.md"
ENV_FILE="$ROOT/env/.env.symphony"
RUNTIME_ROOT="$ROOT/var/symphony-runtime"

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

if [ "${1:-}" = "--doctor" ]; then
  exec python3 "$ROOT/scripts/check_symphony_readiness.py"
fi

load_env_file "$ENV_FILE"
require_env_vars LINEAR_API_KEY SYMPHONY_LINEAR_PROJECT_SLUG SYMPHONY_EXECUTION_PROJECT_SLUG
ensure_runtime_dirs

export SYMPHONY_WORKSPACE_ROOT="${SYMPHONY_WORKSPACE_ROOT:-$ROOT/var/symphony-workspaces}"
export SYMPHONY_SOURCE_TREE_ROOT="${SYMPHONY_SOURCE_TREE_ROOT:-$PROJECT_ROOT}"
export SYMPHONY_CODEX_HOME="${SYMPHONY_CODEX_HOME:-$ROOT/var/codex-symphony}"

mkdir -p "${SYMPHONY_WORKSPACE_ROOT:-$ROOT/var/symphony-workspaces}"
mkdir -p "$RUNTIME_ROOT/mise"
mkdir -p "$RUNTIME_ROOT/mix"
mkdir -p "$RUNTIME_ROOT/hex"
mkdir -p "$SYMPHONY_CODEX_HOME"
ensure_codex_runtime_links "$SYMPHONY_CODEX_HOME"

export MISE_DATA_DIR="$RUNTIME_ROOT/mise"
export MIX_HOME="$RUNTIME_ROOT/mix"
export HEX_HOME="$RUNTIME_ROOT/hex"
export PATH="/opt/homebrew/bin:$PATH"

if [ -z "${SYMPHONY_CODEX_COMMAND:-}" ]; then
  if [ -x "/Applications/Codex.app/Contents/Resources/codex" ]; then
    CODEX_BIN="/Applications/Codex.app/Contents/Resources/codex"
  else
    CODEX_BIN="codex"
  fi

  export SYMPHONY_CODEX_COMMAND="env RUST_LOG=error CODEX_HOME=$SYMPHONY_CODEX_HOME $CODEX_BIN --config shell_environment_policy.inherit=all app-server"
fi

if [ -z "${SOURCE_REPO_URL:-}" ] && [ ! -d "${SYMPHONY_SOURCE_TREE_ROOT:-}" ]; then
  echo "Set SOURCE_REPO_URL or SYMPHONY_SOURCE_TREE_ROOT before starting Symphony." >&2
  exit 1
fi

if [ ! -d "$SYMPHONY_DIR" ] || [ ! -f "$SYMPHONY_DIR/mix.exs" ]; then
  echo "Missing Symphony runtime at $SYMPHONY_DIR. Install or sync infra/agents/symphony/elixir first." >&2
  exit 1
fi

if [ -f "/opt/homebrew/etc/openssl@3/cert.pem" ]; then
  export SSL_CERT_FILE="/opt/homebrew/etc/openssl@3/cert.pem"
  export CURL_CA_BUNDLE="$SSL_CERT_FILE"
  export HEX_CACERTS_PATH="$SSL_CERT_FILE"
fi

cd "$SYMPHONY_DIR"

PREFER_HOST_MIX="${SYMPHONY_PREFER_HOST_MIX:-1}"

if [ "$PREFER_HOST_MIX" = "1" ] && command -v mix >/dev/null 2>&1 && command -v elixir >/dev/null 2>&1 && command -v erl >/dev/null 2>&1; then
  mix local.hex --force
  mix local.rebar --force
  mix setup
  mix build
  exec ./bin/symphony "$WORKFLOW_FILE" "$@"
fi

if command -v mise >/dev/null 2>&1 && [ -f "$SYMPHONY_DIR/mise.toml" ]; then
  mise trust
  mise install
  mise exec -- mix local.hex --force
  mise exec -- mix local.rebar --force
  mise exec -- mix setup
  mise exec -- mix build
  exec mise exec -- ./bin/symphony "$WORKFLOW_FILE" "$@"
fi

echo "Neither mix nor mise is available. Install Elixir tooling first." >&2
exit 1
