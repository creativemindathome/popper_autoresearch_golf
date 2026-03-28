#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
ENV_FILE="${REPO_ROOT}/infra/agents/env/.env.symphony"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "missing ${ENV_FILE}" >&2
  exit 1
fi

set -a
source "${ENV_FILE}"
set +a

echo "repo_root=${REPO_ROOT}"
echo "workflow=${REPO_ROOT}/infra/agents/symphony/WORKFLOW.hermes.md"
echo "linear_project_slug=${SYMPHONY_LINEAR_PROJECT_SLUG:-}"
echo "codex_command=${SYMPHONY_CODEX_COMMAND:-}"
echo "This script is the repo-local launch surface for Symphony."
echo "Replace this stub with the real Symphony invocation once the runtime is installed."
