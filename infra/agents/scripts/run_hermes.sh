#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
ENV_FILE="${REPO_ROOT}/infra/agents/env/.env.hermes"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "missing ${ENV_FILE}" >&2
  exit 1
fi

set -a
source "${ENV_FILE}"
set +a

echo "repo_root=${REPO_ROOT}"
echo "python_bin=${PYTHON_BIN:-python3}"
echo "venv_path=${VENV_PATH:-.venv}"
echo "This script is the repo-local launch surface for Hermes."
echo "Replace this stub with the real executor command once Hermes is wired."
