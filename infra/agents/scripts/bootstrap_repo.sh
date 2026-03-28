#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

required_dirs=(
  "${REPO_ROOT}/infra/agents/env"
  "${REPO_ROOT}/infra/agents/scripts"
  "${REPO_ROOT}/infra/agents/symphony"
  "${REPO_ROOT}/infra/agents/handoffs"
  "${REPO_ROOT}/research/profiles"
  "${REPO_ROOT}/research/theories"
  "${REPO_ROOT}/research/falsification"
  "${REPO_ROOT}/research/knowledge_graph"
  "${REPO_ROOT}/records/track_10min_16mb"
  "${REPO_ROOT}/records/track_non_record_16mb"
)

for dir in "${required_dirs[@]}"; do
  mkdir -p "${dir}"
done

required_files=(
  "${REPO_ROOT}/infra/agents/env/.env.symphony"
  "${REPO_ROOT}/infra/agents/env/.env.hermes"
  "${REPO_ROOT}/infra/agents/symphony/WORKFLOW.hermes.md"
  "${REPO_ROOT}/infra/agents/handoffs/README.md"
  "${REPO_ROOT}/ISSUE_TEMPLATE.md"
)

missing=0
for file in "${required_files[@]}"; do
  if [[ ! -f "${file}" ]]; then
    echo "missing required file: ${file}" >&2
    missing=1
  fi
done

if [[ ${missing} -ne 0 ]]; then
  exit 1
fi

echo "bootstrap check passed"
