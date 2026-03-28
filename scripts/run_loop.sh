#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_loop.sh [--iterations N] [--forever] [--sleep SECONDS] [--stage-only 1|2]

Runs the local loop:
  ideator (Gemini + OpenAI reviewer) → handoff to inbox → falsifier (Stage 1/2) → outbox

Requires environment variables:
  GEMINI_API_KEY   (required)
  OPENAI_API_KEY   (required)
  ANTHROPIC_API_KEY (optional; improves Stage 2 hypotheses)

Notes:
  - Uses `.venv/bin/python`. If missing, run: `uv sync --extra train`
  - Default parent is `parameter-golf/train_gpt.py` if it exists, else `train_gpt.py`.
EOF
}

ITERATIONS=1
FOREVER=0
SLEEP_SECONDS=0
STAGE_ONLY=""
KNOWLEDGE_DIR="${KNOWLEDGE_DIR:-knowledge_graph}"

while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --iterations|-n) ITERATIONS="${2:-}"; shift 2 ;;
    --forever) FOREVER=1; shift ;;
    --sleep) SLEEP_SECONDS="${2:-}"; shift 2 ;;
    --stage-only) STAGE_ONLY="${2:-}"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

PYTHON=".venv/bin/python"
if [ ! -x "$PYTHON" ]; then
  echo "Missing $PYTHON. Create the venv with: uv sync --extra train" >&2
  exit 1
fi

if [ -z "${GEMINI_API_KEY:-}" ]; then
  echo "Missing GEMINI_API_KEY in this terminal session." >&2
  exit 1
fi

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "Missing OPENAI_API_KEY in this terminal session." >&2
  exit 1
fi

PARENT_TRAIN_GPT="${PARENT_TRAIN_GPT:-}"
if [ -z "$PARENT_TRAIN_GPT" ]; then
  if [ -f "parameter-golf/train_gpt.py" ]; then
    PARENT_TRAIN_GPT="parameter-golf/train_gpt.py"
  elif [ -f "train_gpt.py" ]; then
    PARENT_TRAIN_GPT="train_gpt.py"
  else
    echo "Could not find a parent train_gpt.py. Set PARENT_TRAIN_GPT=/path/to/train_gpt.py" >&2
    exit 1
  fi
fi

mkdir -p \
  "${KNOWLEDGE_DIR}/inbox/approved" \
  "${KNOWLEDGE_DIR}/outbox/ideator" \
  "${KNOWLEDGE_DIR}/outbox/falsifier" \
  "${KNOWLEDGE_DIR}/work/in_falsification"

run_once() {
  local out_latest="${KNOWLEDGE_DIR}/outbox/ideator/latest.json"

  echo "==> ideator"
  "$PYTHON" -m ideator idea \
    --parent-train-gpt "$PARENT_TRAIN_GPT" \
    --knowledge-dir "$KNOWLEDGE_DIR" \
    >/dev/null

  if [ ! -f "$out_latest" ]; then
    echo "ideator did not write ${out_latest}" >&2
    exit 1
  fi

  local idea_id
  idea_id=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["idea_id"])' "$out_latest")

  echo "==> handoff (approve → inbox)"
  "$PYTHON" infra/agents/scripts/handoff_ideator_to_falsifier.py --knowledge-dir "$KNOWLEDGE_DIR" >/dev/null || true

  if [ ! -f "${KNOWLEDGE_DIR}/inbox/approved/${idea_id}.json" ]; then
    echo "Expected approved idea at ${KNOWLEDGE_DIR}/inbox/approved/${idea_id}.json" >&2
    echo "If reviewer rejected the idea, rerun ideator or lower IDEATOR_REVIEWER_MIN_SCORE." >&2
    exit 1
  fi

  echo "==> falsifier (${idea_id})"
  local out_path="${KNOWLEDGE_DIR}/outbox/falsifier/${idea_id}_result.json"
  local args=(
    -m falsifier.main
    --idea-id "$idea_id"
    --knowledge-dir "$KNOWLEDGE_DIR"
    --output-json "$out_path"
  )
  if [ -n "$STAGE_ONLY" ]; then
    args+=(--stage-only "$STAGE_ONLY")
  fi
  "$PYTHON" "${args[@]}"

  "$PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); print("verdict=", d.get("verdict"), "killed_by=", d.get("killed_by"))' "$out_path"
}

if [ "$FOREVER" -eq 1 ]; then
  i=0
  while true; do
    i=$((i + 1))
    echo ""
    echo "================================================================================"
    echo "LOOP ITERATION $i"
    echo "================================================================================"
    run_once
    if [ "$SLEEP_SECONDS" -gt 0 ]; then
      sleep "$SLEEP_SECONDS"
    fi
  done
fi

for ((i=1; i<=ITERATIONS; i++)); do
  echo ""
  echo "================================================================================"
  echo "LOOP ITERATION $i/$ITERATIONS"
  echo "================================================================================"
  run_once
  if [ "$i" -lt "$ITERATIONS" ] && [ "$SLEEP_SECONDS" -gt 0 ]; then
    sleep "$SLEEP_SECONDS"
  fi
done

