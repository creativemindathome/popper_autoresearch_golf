# Cursor API Configuration Guide

## Overview

The AutoResearch Loop (Ideator + Falsifier) can be configured to work within Cursor's agent environment. This guide covers the configuration options and execution patterns.

## Current Architecture Compatibility

The existing system is **already compatible** with Cursor agents because it uses:
1. **Environment variables** for API keys (standard practice)
2. **File-based knowledge graph** (persistent, debuggable)
3. **Command-line interface** (easy to invoke from agents)
4. **Exit codes** (0 = success/pass, 1 = failure/refuted)

## Configuration Methods

### Method 1: Environment Variables (Recommended)

Cursor agents can set environment variables before invoking commands:

```bash
# Required APIs
export GEMINI_API_KEY="your-gemini-key"           # For ideator
export OPENAI_API_KEY="your-openai-key"          # For reviewer + optional Stage 2

# Optional Anthropic for Stage 2 LLM enhancement
export ANTHROPIC_API_KEY="your-anthropic-key"    # Optional - fallback works without

# Configuration
export GEMINI_MODEL="gemini-2.5-flash"
export OPENAI_REVIEWER_MODEL="gpt-4o-mini"
export IDEATOR_MAX_REVIEW_ROUNDS="4"
export IDEATOR_REVIEWER_MIN_SCORE="6"

# Paths
export KNOWLEDGE_DIR="./knowledge_graph"
export PARAMETER_GOLF_REPO="./parameter-golf"
```

### Method 2: Command-Line Arguments

Direct invocation from Cursor agent:

```bash
# Generate idea
python3 -m ideator \
    --parent-train-gpt ./parameter-golf/train_gpt.py \
    --knowledge-dir ./knowledge_graph \
    --api-key $GEMINI_API_KEY \
    --reviewer-api-key $OPENAI_API_KEY

# Falsify approved idea
python3 -m falsifier.main \
    --idea-id low-rank-transformer-layers \
    --knowledge-dir ./knowledge_graph \
    --graph-path ./knowledge_graph/graph.json \
    --output-json ./knowledge_graph/outbox/falsifier/result.json
```

### Method 3: Cursor-Specific Configuration File

Create `.cursor/research-config.json`:

```json
{
  "api_keys": {
    "gemini": "${GEMINI_API_KEY}",
    "openai": "${OPENAI_API_KEY}",
    "anthropic": "${ANTHROPIC_API_KEY}"
  },
  "models": {
    "ideator": "gemini-2.5-flash",
    "reviewer": "gpt-4o-mini",
    "stage2_hypothesis": "claude-sonnet-4-20250514"
  },
  "paths": {
    "knowledge_graph": "./knowledge_graph",
    "parameter_golf_repo": "./parameter-golf",
    "results_dir": "./research/falsification/results"
  },
  "ideator": {
    "max_review_rounds": 4,
    "reviewer_min_score": 6,
    "temperature": 1.3
  },
  "falsifier": {
    "stage1_only": false,
    "use_mlx_if_available": true,
    "divergence_threshold": 100.0
  }
}
```

## Cursor Agent Integration Patterns

### Pattern 1: Sequential Execution (Single Agent)

One Cursor agent runs the full loop:

```python
# cursor_agent_script.py
import subprocess
import json
from pathlib import Path

def run_ideator_falsifier_loop(parent_code_path: str) -> dict:
    """Run complete ideator→falsifier loop."""
    
    # Step 1: Generate idea
    result = subprocess.run(
        [
            "python3", "-m", "ideator",
            "--parent-train-gpt", parent_code_path,
            "--knowledge-dir", "./knowledge_graph"
        ],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        return {"status": "ideator_failed", "error": result.stderr}
    
    # Step 2: Find approved idea
    inbox_dir = Path("./knowledge_graph/inbox/approved")
    approved_ideas = list(inbox_dir.glob("*.json"))
    
    if not approved_ideas:
        return {"status": "no_approved_ideas"}
    
    latest_idea = max(approved_ideas, key=lambda p: p.stat().st_mtime)
    idea_id = latest_idea.stem
    
    # Step 3: Falsify
    result = subprocess.run(
        [
            "python3", "-m", "falsifier.main",
            "--idea-id", idea_id,
            "--knowledge-dir", "./knowledge_graph"
        ],
        capture_output=True,
        text=True
    )
    
    # Load results
    output_path = Path(f"./knowledge_graph/outbox/falsifier/{idea_id}_result.json")
    if output_path.exists():
        return json.loads(output_path.read_text())
    
    return {"status": "falsifier_completed", "exit_code": result.returncode}
```

### Pattern 2: Parallel Agent Execution (Multi-Agent)

Use Cursor's parallel agent capability:

```yaml
# .cursor/agents.yaml
agents:
  ideator:
    role: "Generate novel architecture ideas"
    command: "python3 -m ideator --parent-train-gpt {{parent_code}}"
    env:
      GEMINI_API_KEY: "${GEMINI_API_KEY}"
      OPENAI_API_KEY: "${OPENAI_API_KEY}"
    
  falsifier:
    role: "Validate ideas through systematic testing"
    command: "python3 -m falsifier.main --idea-id {{idea_id}}"
    depends_on: [ideator]
    env:
      # No API keys needed - deterministic testing
      KNOWLEDGE_DIR: "./knowledge_graph"
  
  knowledge_manager:
    role: "Update knowledge graph with results"
    command: "python3 -m knowledge_graph.update"
    depends_on: [falsifier]
```

### Pattern 3: Cursor + Symphony Hybrid

For production use with Linear integration:

```bash
# Use Symphony for orchestration
export LINEAR_API_KEY="your-linear-key"
export SYMPHONY_LINEAR_PROJECT_SLUG="your-project"

# Run via Symphony
python infra/agents/scripts/run_symphony.sh
```

## Testing Full Loop Execution

### Quick Verification

```bash
# 1. Setup
export GEMINI_API_KEY="..."
export OPENAI_API_KEY="..."

# 2. Generate idea (includes automatic handoff)
python3 -m ideator idea \
    --parent-train-gpt tests/candidates/good_student.py \
    --reviewer-api-key $OPENAI_API_KEY

# 3. Check inbox for approved idea
ls -la knowledge_graph/inbox/approved/

# 4. Run falsifier
python3 -m falsifier.main \
    --idea-id $(ls knowledge_graph/inbox/approved/*.json | head -1 | xargs basename -s .json) \
    --knowledge-dir knowledge_graph

# 5. Check results
cat knowledge_graph/outbox/falsifier/*_result.json | jq '.verdict'
```

## Cursor-Specific Optimizations

### 1. Agent State Persistence

Cursor agents can save state to knowledge graph:

```python
# In Cursor agent context
from falsifier.graph.lifecycle import update_node_status

# Mark node as being processed by Cursor agent
update_node_status(
    node_id=idea_id,
    new_status="IN_FALSIFICATION_CURSOR",
    graph_path="./knowledge_graph/graph.json",
    actor="cursor_agent",
    metadata={
        "cursor_session_id": cursor_session_id,
        "agent_start_time": time.time()
    }
)
```

### 2. Streaming Output

For long-running falsification, stream progress:

```bash
# Run with unbuffered output for real-time streaming
python3 -u -m falsifier.main \
    --idea-id $IDEA_ID \
    --knowledge-dir ./knowledge_graph \
    2>&1 | tee knowledge_graph/work/in_falsification/${IDEA_ID}.log
```

### 3. Checkpoint Recovery

If Cursor agent times out, resume from checkpoint:

```python
# Check for existing lock file
lock_path = Path(f"knowledge_graph/work/in_falsification/{idea_id}.lock")
if lock_path.exists():
    lock_data = json.loads(lock_path.read_text())
    if lock_data.get("pid") != current_pid:
        # Another agent is processing - skip or wait
        pass
```

## Configuration Validation

Run this to verify Cursor environment is ready:

```python
#!/usr/bin/env python3
"""Validate Cursor API configuration."""

import os
import sys
from pathlib import Path

def check_cursor_config():
    """Check if all required configuration is present."""
    
    checks = {
        "GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY"),
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
        "knowledge_graph_dir": Path("./knowledge_graph").exists(),
        "inbox_approved_dir": Path("./knowledge_graph/inbox/approved").exists(),
        "ideator_module": Path("./ideator").exists(),
        "falsifier_module": Path("./falsifier").exists(),
    }
    
    optional = {
        "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY"),
    }
    
    print("=== Cursor API Configuration Check ===\n")
    
    all_required = True
    for key, value in checks.items():
        status = "✓" if value else "✗"
        print(f"{status} {key}: {'present' if value else 'MISSING'}")
        if not value and key in ["GEMINI_API_KEY", "OPENAI_API_KEY"]:
            all_required = False
    
    print("\nOptional:")
    for key, value in optional.items():
        status = "✓" if value else "○"
        print(f"{status} {key}: {'present' if value else 'not set (ok - fallback works)'}")
    
    if all_required:
        print("\n✓ Configuration valid - full loop can execute")
        return 0
    else:
        print("\n✗ Missing required configuration")
        return 1

if __name__ == "__main__":
    sys.exit(check_cursor_config())
```

## Summary

**Can the full loops execute?** YES ✓

The system is ready for Cursor agent execution with:
- Environment-based configuration
- File-based state management
- Command-line interface
- Automatic handoff between ideator→falsifier
- Graceful degradation (works without optional APIs)

**Cursor-specific advantages:**
1. No special Cursor API needed - uses standard CLI
2. File-based knowledge graph integrates naturally
3. Exit codes allow agent decision logic
4. Lock files prevent agent conflicts
5. Streaming output for real-time feedback
