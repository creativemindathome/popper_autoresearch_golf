#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || true
test -f "${REPO_ROOT}/infra/agents/env/.env.symphony"
test -f "${REPO_ROOT}/infra/agents/env/.env.hermes"

echo "python: ${PYTHON_BIN}"
echo "repo_root: ${REPO_ROOT}"
echo "env files present"
