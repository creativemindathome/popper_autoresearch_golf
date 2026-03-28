---
tracker:
  kind: linear
  api_key: $LINEAR_API_KEY
  project_slug: parameter-golf-buildout-97b212957a6c
  active_states:
    - Todo
    - In Progress
  terminal_states:
    - Done
    - Closed
    - Cancelled
    - Canceled
    - Duplicate
polling:
  interval_ms: 5000
workspace:
  root: /Users/curiousmind/Desktop/null_fellow_hackathon/infra/agents/var/symphony-workspaces
hooks:
  after_create: |
    default_stage_excludes() {
      printf '%s\n' \
        '.git' \
        '.DS_Store' \
        'infra/agents/var' \
        '.venv' \
        'venv' \
        'node_modules' \
        'dist' \
        'build' \
        '_build' \
        'deps' \
        '.pytest_cache' \
        '__pycache__' \
        '.mypy_cache' \
        '.ruff_cache' \
        '.next' \
        '.turbo' \
        'coverage'
    }

    stage_excludes() {
      if [ -n "${SYMPHONY_SYNC_EXCLUDES:-}" ]; then
        printf '%s\n' "$SYMPHONY_SYNC_EXCLUDES" | tr ':,' '\n'
      else
        default_stage_excludes
      fi
    }

    hydrate_from_source_tree() {
      local source_root="$1"
      local -a rsync_args=()
      local exclude=""

      while IFS= read -r exclude; do
        [ -n "$exclude" ] || continue
        rsync_args+=("--exclude" "$exclude")
      done < <(stage_excludes)

      if command -v rsync >/dev/null 2>&1; then
        rsync -a --delete "${rsync_args[@]}" "$source_root"/ ./
      else
        cp -R "$source_root"/. ./
        while IFS= read -r exclude; do
          [ -n "$exclude" ] || continue
          rm -rf "./$exclude"
        done < <(stage_excludes)
      fi
    }

    if [ -n "${SYMPHONY_SOURCE_TREE_ROOT:-}" ] && [ -d "$SYMPHONY_SOURCE_TREE_ROOT" ]; then
      hydrate_from_source_tree "$SYMPHONY_SOURCE_TREE_ROOT"
      git init -q
      git config user.name "Symphony Workspace"
      git config user.email "symphony@local"
      git add .
      git commit -qm "Workspace snapshot" || true
    elif [ -n "${SOURCE_REPO_URL:-}" ]; then
      git clone --depth 1 "$SOURCE_REPO_URL" .
    else
      echo "Either SYMPHONY_SOURCE_TREE_ROOT or SOURCE_REPO_URL is required" >&2
      exit 1
    fi
  before_run: |
    if ! git rev-parse --verify HEAD >/dev/null 2>&1; then
      if [ -n "${SYMPHONY_SOURCE_TREE_ROOT:-}" ] && [ -d "$SYMPHONY_SOURCE_TREE_ROOT" ]; then
        find . -mindepth 1 -maxdepth 1 -exec rm -rf {} +
        rsync -a --delete \
          --exclude '.git' \
          --exclude '.DS_Store' \
          --exclude 'infra/agents/var' \
          --exclude '.venv' \
          --exclude 'venv' \
          --exclude 'node_modules' \
          --exclude 'dist' \
          --exclude 'build' \
          --exclude '_build' \
          --exclude 'deps' \
          --exclude '.pytest_cache' \
          --exclude '__pycache__' \
          --exclude '.mypy_cache' \
          --exclude '.ruff_cache' \
          --exclude '.next' \
          --exclude '.turbo' \
          --exclude 'coverage' \
          "$SYMPHONY_SOURCE_TREE_ROOT"/ ./
        git init -q
        git config user.name "Symphony Workspace"
        git config user.email "symphony@local"
        git add .
        git commit -qm "Workspace snapshot" || true
      fi
    fi
    if [ -d "./infra/agents/scripts" ] && [ -n "${SYMPHONY_SOURCE_TREE_ROOT:-}" ]; then
      python3 ./infra/agents/scripts/reconcile_delivery_contracts.py \
        --repo-root "$SYMPHONY_SOURCE_TREE_ROOT" \
        --issue "$(basename "$PWD")" \
        --reopen-missing || true
    fi
  after_run: |
    if [ -d "./infra/agents/scripts" ] && [ -n "${SYMPHONY_SOURCE_TREE_ROOT:-}" ]; then
      ./infra/agents/scripts/sync_workspace_back.sh "$PWD" "$SYMPHONY_SOURCE_TREE_ROOT"
      python3 ./infra/agents/scripts/reconcile_delivery_contracts.py \
        --repo-root "$SYMPHONY_SOURCE_TREE_ROOT" \
        --issue "$(basename "$PWD")" \
        --reopen-missing || true
    fi
  before_remove: |
    if [ -d "./infra/agents/scripts" ] && [ -n "${SYMPHONY_SOURCE_TREE_ROOT:-}" ]; then
      ./infra/agents/scripts/archive_workspace_changes.sh \
        "$PWD" \
        "$SYMPHONY_SOURCE_TREE_ROOT/infra/agents/var/symphony-archives"
    fi
agent:
  max_concurrent_agents: 4
  max_turns: 16
codex:
  command: env RUST_LOG=error CODEX_HOME=/Users/curiousmind/Desktop/null_fellow_hackathon/infra/agents/var/codex-symphony /Applications/Codex.app/Contents/Resources/codex --config shell_environment_policy.inherit=all app-server
  approval_policy: never
  thread_sandbox: workspace-write
  turn_sandbox_policy:
    type: workspaceWrite
---

You are working on Linear issue `{{ issue.identifier }}`.

Title: {{ issue.title }}
State: {{ issue.state }}
URL: {{ issue.url }}

Description:
{% if issue.description %}
{{ issue.description }}
{% else %}
No description provided.
{% endif %}

Execution policy:

- Treat checked-in handoffs under `infra/agents/handoffs/` as the repo-local source of truth.
- Only work on issues that are actually in `Todo` or `In Progress`.
- Do not retry a failed issue without a material issue mutation.
- Do not promote candidate-theory execution into the buildout project.
- Run the issue's declared validation commands before considering the issue complete.
- Do not mark work `Done` if listed delivery files are missing from the main repo tree.
- Keep scope narrow and respect `Ownership / Non-Overlap`.

Required startup checks:

1. Read `infra/agents/handoffs/README.md`.
2. Read the handoff file that matches the current issue.
3. Confirm the issue still matches the local handoff before editing code.
4. Treat missing env/tools as a blocker, not as a reason to improvise around the contract.

Loop policy:

- If the issue fails for environment reasons, leave a concise blocker note and stop.
- If the issue is under-specified, rewrite or narrow the issue instead of pushing low-confidence code.
- If the issue is over-broad, split it.
- Prefer deterministic local validation over broad speculative changes.
