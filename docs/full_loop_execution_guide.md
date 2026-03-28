# Full Loop Execution Guide

## Current Status: INFRASTRUCTURE READY ✓

The complete ideator→falsifier→knowledge graph loop is ready for execution.

### Readiness Check Results

```
✓ Knowledge Graph Infrastructure (all directories present)
✓ Python Modules (ideator/ and falsifier/ load correctly)
✓ Integration Components (lifecycle + adapter imports work)
○ API Keys (required for ideator + reviewer, falsifier works without)
```

## Quick Start for Cursor Agents

### Option 1: Full Loop (with API keys)

```bash
# 1. Set API keys
export GEMINI_API_KEY="your-gemini-key"
export OPENAI_API_KEY="your-openai-key"

# 2. Generate and review idea
python3 -m ideator idea \
    --parent-train-gpt tests/candidates/good_student.py \
    --reviewer-api-key $OPENAI_API_KEY

# 3. Falsify the approved idea
python3 -m falsifier.main \
    --idea-id $(ls knowledge_graph/inbox/approved/*.json | head -1 | xargs basename -s .json) \
    --knowledge-dir knowledge_graph
```

### Option 2: Falsifier Only (no APIs needed)

```bash
# Test any train_gpt.py directly
python3 -m falsifier.main \
    --candidate-json <path-to-candidate>.json \
    --output-json result.json
```

### Option 3: Cursor Agent Workflow

```python
# cursor_execute_loop.py
import subprocess
import json
from pathlib import Path

def execute_full_loop(parent_code_path: str) -> dict:
    """Execute ideator→falsifier→knowledge graph loop."""
    
    # Step 1: Generate idea with reviewer
    ideator_result = subprocess.run(
        [
            "python3", "-m", "ideator", "idea",
            "--parent-train-gpt", parent_code_path,
            "--knowledge-dir", "./knowledge_graph",
            "--reviewer-api-key", os.environ["OPENAI_API_KEY"]
        ],
        capture_output=True,
        text=True,
        timeout=300  # 5 minutes for generation + review
    )
    
    if ideator_result.returncode != 0:
        return {
            "status": "ideator_failed",
            "stderr": ideator_result.stderr,
            "stdout": ideator_result.stdout
        }
    
    # Step 2: Find the approved idea
    approved_dir = Path("knowledge_graph/inbox/approved")
    approved_files = sorted(
        approved_dir.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    
    if not approved_files:
        return {
            "status": "no_approved_ideas",
            "message": "Ideator ran but no ideas were approved by reviewer"
        }
    
    latest_idea = approved_files[0]
    idea_id = latest_idea.stem
    
    # Step 3: Run falsifier
    falsifier_result = subprocess.run(
        [
            "python3", "-m", "falsifier.main",
            "--idea-id", idea_id,
            "--knowledge-dir", "./knowledge_graph",
            "--graph-path", "./knowledge_graph/graph.json",
            "--output-json", f"./knowledge_graph/outbox/falsifier/{idea_id}_result.json"
        ],
        capture_output=True,
        text=True,
        timeout=120  # 2 minutes for falsification
    )
    
    # Step 4: Load results
    output_path = Path(f"knowledge_graph/outbox/falsifier/{idea_id}_result.json")
    if output_path.exists():
        result = json.loads(output_path.read_text())
        return {
            "status": "completed",
            "idea_id": idea_id,
            "verdict": result.get("verdict"),
            "killed_by": result.get("killed_by"),
            "kill_reason": result.get("kill_reason"),
            "tags_count": len(result.get("tags", [])),
            "falsifier_exit_code": falsifier_result.returncode
        }
    else:
        return {
            "status": "falsifier_no_output",
            "stderr": falsifier_result.stderr,
            "stdout": falsifier_result.stdout
        }
```

## Execution Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FULL LOOP EXECUTION                                │
└─────────────────────────────────────────────────────────────────────────────┘

Step 1: IDEATOR
├── Load knowledge context (knowledge_graph/graph.json)
├── Generate idea (Gemini API)
├── Review idea (OpenAI API)
├── Revise if rejected (up to max_review_rounds)
└── Output: knowledge_graph/outbox/ideator/<idea_id>.json

Step 2: AUTO-HANDOFF
├── Reviewer approves idea (score >= reviewer_min_score)
├── ideator.cli.py creates symlink
└── Output: knowledge_graph/inbox/approved/<idea_id>.json

Step 3: FALSIFIER
├── Load approved idea from inbox
├── Stage 1: T2→T3→T4→T5→T7 gates
│   ├── T2: Budget check (~16ms)
│   ├── T3: Compilation check (~16ms)
│   ├── T4: Signal propagation (~7ms)
│   ├── T5: Init diagnostics (~28ms)
│   └── T7: Microtrain (if T5 passes) (~3-5s)
├── Stage 2: Adversarial prosecution (if Stage 1 passes)
│   ├── Generate kill hypotheses (fallback or LLM)
│   ├── Execute experiments (training runs)
│   └── Verify trends
└── Output: knowledge_graph/outbox/falsifier/<idea_id>_result.json

Step 4: KNOWLEDGE GRAPH UPDATE
├── Update node status
├── Add tags and measurements
├── Link to parent nodes
└── Store failure analysis for learning
```

## Configuration for Different Environments

### Local Development

```bash
# No APIs needed for falsifier testing
python3 -m falsifier.main \
    --candidate-json my_candidate.json \
    --output-json result.json
```

### Cursor Agent (Single Run)

```bash
# Set APIs for full loop
export GEMINI_API_KEY="..."
export OPENAI_API_KEY="..."

# Run full loop
python3 -m ideator idea --parent-train-gpt <path> --reviewer-api-key $OPENAI_API_KEY
python3 -m falsifier.main --idea-id <id>
```

### Cursor Agent (With Symphony + Linear)

```bash
# Use Symphony for orchestration
export LINEAR_API_KEY="..."
export SYMPHONY_LINEAR_PROJECT_SLUG="parameter-golf-buildout"

# Run via Symphony
bash infra/agents/scripts/run_symphony.sh
```

### Production Server

```bash
# Environment file
# /etc/research-loop/env
GEMINI_API_KEY=...
OPENAI_API_KEY=...
KNOWLEDGE_DIR=/var/research/knowledge_graph
PARAMETER_GOLF_REPO=/var/research/parameter-golf

# Systemd service or Docker container
# Continuously polls inbox/approved/ for new ideas
```

## Verifying Execution

### Test 1: Check Infrastructure

```bash
python3 << 'EOF'
from pathlib import Path

dirs = [
    "knowledge_graph/inbox/approved",
    "knowledge_graph/outbox/ideator",
    "knowledge_graph/outbox/falsifier",
    "knowledge_graph/work/in_falsification"
]

for d in dirs:
    exists = Path(d).exists()
    print(f"{'✓' if exists else '✗'} {d}")
EOF
```

### Test 2: Quick Falsifier Run

```bash
# Create minimal test candidate
cat > /tmp/test_candidate.json << 'JSON'
{
  "theory_id": "test_run",
  "what_and_why": "Test that falsifier can execute",
  "train_gpt_path": "tests/candidates/good_student.py"
}
JSON

# Run falsifier
python3 -m falsifier.main \
    --candidate-json /tmp/test_candidate.json \
    --output-json /tmp/test_result.json

# Check result
cat /tmp/test_result.json | jq '.verdict, .killed_by'
```

Expected output:
```
"REFUTED"
"T5"
```

### Test 3: Knowledge Graph Update

```bash
# Check graph exists and can be updated
python3 << 'EOF'
from falsifier.graph.lifecycle import update_node_status
from pathlib import Path
import json

graph_path = Path("knowledge_graph/graph.json")

if not graph_path.exists():
    # Initialize empty graph
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    graph_path.write_text(json.dumps({"nodes": {}, "edges": []}))

# Test update
try:
    update_node_status(
        node_id="test_node",
        new_status="TEST_STATUS",
        graph_path=graph_path,
        actor="test_script",
        metadata={"test": True}
    )
    print("✓ Knowledge graph update works")
except Exception as e:
    print(f"✗ Error: {e}")
EOF
```

## Performance Expectations

| Phase | Typical Duration | API Calls |
|-------|------------------|-----------|
| Ideator Generation | 5-15s | 1 (Gemini) |
| Reviewer Evaluation | 3-10s | 1 (OpenAI) |
| Revision (if needed) | 5-10s × rounds | 2 × rounds |
| **Total Ideator** | **15-60s** | **1-4** |
| T2 Budget | 0.016s | 0 |
| T3 Compilation | 0.016s | 0 |
| T4 Signal | 0.007s | 0 |
| T5 Init | 0.028s | 0 |
| T7 Microtrain | 3-5s | 0 |
| **Stage 1 Total** | **~5s** | **0** |
| Stage 2 Hypotheses | 0.1s (fallback) or 1-3s (LLM) | 0 or 1 |
| Stage 2 Experiments | 10-30s (500 steps training) | 0 |
| **Stage 2 Total** | **10-35s** | **0-1** |
| **Full Loop** | **30-100s** | **1-5** |

## Error Handling

### Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| `GEMINI_API_KEY not set` | Export the API key or pass `--api-key` |
| `No approved ideas in inbox` | Check reviewer score threshold or manually approve |
| `Falsifier killed at T3` | Check train_gpt.py compiles (syntax errors) |
| `Falsifier killed at T5` | Check weight initialization (rank deficiency) |
| `Lock file exists` | Wait for other process or remove stale lock |
| `Graph update fails` | Check file permissions on knowledge_graph/ |

### Recovery Patterns

```python
# If ideator fails, retry with different seed
python3 -m ideator idea \
    --parent-train-gpt <path> \
    --seed $RANDOM

# If falsifier killed at T5, ideator should learn from feedback
# The feedback is stored in knowledge graph and used as context
# for next idea generation
```

## Summary

✓ **Infrastructure**: All knowledge graph directories present  
✓ **Modules**: ideator/ and falsifier/ import correctly  
✓ **Integration**: Lifecycle and adapter components working  
○ **API Keys**: Required for ideator + reviewer, set before execution  

**Full loop can execute** once API keys are configured. Falsifier alone can run without any APIs (deterministic testing).
