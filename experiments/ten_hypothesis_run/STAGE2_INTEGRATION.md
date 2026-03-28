# Stage 2 Anthropic Integration

## Overview

The full live experiment now includes **Stage 2 adversarial falsification** using Anthropic Claude Sonnet as the executing model.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    FULL PIPELINE WITH STAGE 2                     │
└─────────────────────────────────────────────────────────────────┘

Stage 0: Idea Generation (Anthropic Sonnet)
├── Generates novel architecture ideas
├── 2-3 paragraphs of "what and why"
├── Complete trainable code
└── Novelty claims and expected behavior
         │
         ▼
Stage 1: Falsifier Gates (Local MLX/PyTorch)
├── T2 Budget Check: Resource constraints
├── T3 Compilation: Syntax/semantic validation
├── T4 Signal Check: Forward pass produces valid output
├── T5 Init Check: Numerical stability at initialization
└── T7 Microtrain: 100-step training dynamics
         │
         ├── Killed → Knowledge Graph (FALSIFIED)
         │
         ▼
Stage 2: Adversarial Falsifier (Anthropic Sonnet)
├── Generate 3-5 kill hypotheses
├── Design specific experiments
├── Run extended training/ablation tests
├── Evaluate each hypothesis
└── Final verdict: PASSED or KILLED with specific reason
         │
         ▼
Knowledge Graph Update
├── Track node status (APPROVED/FALSIFIED)
├── Record which stage killed it
├── Store kill hypothesis (if Stage 2)
└── Link to parent/child theories
```

## Configuration

### Default Stage 2 Settings

```python
Stage2Config(
    enabled=True,
    model="claude-sonnet-4-20250514",  # Sonnet for hypothesis generation
    max_hypotheses=5,
    max_tokens=4000,
    temperature=0.7,
)
```

### Running with Stage 2

```bash
# Full experiment with Stage 2 enabled (default)
bash run_live_with_stage2.sh

# Or directly with Python
python3 run_full_live_experiment.py \
    --num-hypotheses 10 \
    --stage2-model "claude-sonnet-4-20250514"

# Disable Stage 2 (Stage 1 only)
python3 run_full_live_experiment.py --disable-stage2
```

## Stage 2 Process

### 1. Context Building

The falsifier context includes:
- **Theory Details**: ID, what_and_why, config changes
- **Stage 1 Results**: Which gates passed/failed, tags
- **Checkpoint Profile**: Gradient norms, entropy, ratios
- **Knowledge Graph**: Previous theories and failures

### 2. Hypothesis Generation

Anthropic Sonnet generates kill hypotheses in this format:

```json
{
  "hypothesis_id": "H1",
  "confidence": "high",
  "failure_mode": "Gradient explosion in attention heads",
  "mechanism": "Query-key dot products grow unbounded during training",
  "experiment_type": "temporal",
  "experiment_spec": {
    "metric": "grad_norm",
    "threshold": 100.0,
    "comparator": ">",
    "step": 500,
    "needs_ablation": true,
    "ablation_target": "attention_heads"
  },
  "evidence": "Stage 1 T5: init_logit_max=2.3, gradient_norm_ratio=1.8"
}
```

### 3. Experiment Execution

Each hypothesis triggers:
- Extended training run (default 500 steps)
- Metric monitoring
- Ablation test (if needed)
- Threshold evaluation

### 4. Verdict Determination

```python
if any(hypothesis.confirmed for hypothesis in hypotheses):
    verdict = "STAGE_2_KILLED"
    killed_by = confirmed_hypothesis.failure_mode
else:
    verdict = "SURVIVED"
    killed_by = None
```

## Output Structure

With Stage 2 enabled, the output includes:

```json
{
  "run_id": "h001_abc123",
  "hypothesis_number": 1,
  "stage1_result": {
    "verdict": "STAGE_1_PASSED",
    "killed_by": null,
    "summary": "All gates passed"
  },
  "stage2_result": {
    "verdict": "STAGE_2_KILLED",
    "hypotheses_tested": 3,
    "experiments_run": 3,
    "kill_hypothesis": {
      "hypothesis_id": "H2",
      "failure_mode": "Attention entropy collapse",
      "confidence": "high"
    }
  },
  "verdict": "STAGE_2_KILLED"
}
```

## Comparison: With vs Without Stage 2

### Without Stage 2
- Theories can pass Stage 1 but fail in extended training
- No adversarial testing of mechanisms
- Limited kill reasoning

### With Stage 2
- Extended training validation (500 steps)
- Specific failure mode identification
- Mechanism-grounded kills
- Richer knowledge graph entries

## Performance

### Timing per Hypothesis

| Stage | Duration | Notes |
|-------|----------|-------|
| Generation | 5-10s | API call to Anthropic |
| Stage 1 | 30-60s | Local computation (T2-T7) |
| Stage 2 | 2-5 min | API + extended training |
| **Total** | **3-6 min** | Per hypothesis |

### 10 Hypothesis Run

- **Without Stage 2**: ~15-20 minutes
- **With Stage 2**: ~40-60 minutes
- **With Stage 2 (survivors only)**: ~30-50 minutes

Stage 2 only runs on theories that pass Stage 1 (typically 20-40% of theories).

## Cost Estimation

### API Calls per Hypothesis

1. **Ideation**: 1 call to generate idea
2. **Stage 2**: 1 call to generate kill hypotheses (only if Stage 1 passed)

### Estimated Cost (10 hypotheses)

- ~10 ideation calls
- ~2-4 Stage 2 calls (only survivors)
- Total: ~$2-5 USD depending on output length

## Customization

### Use Different Anthropic Model

```bash
# Use Haiku for faster/cheaper Stage 2
python3 run_full_live_experiment.py \
    --stage2-model "claude-haiku-3-20240307"

# Use Opus for more thorough analysis
python3 run_full_live_experiment.py \
    --stage2-model "claude-opus-4-20250514"
```

### Adjust Hypothesis Count

```python
# In run_full_live_experiment.py
stage2_config = Stage2Config(
    max_hypotheses=3,  # Reduce for faster execution
    max_tokens=2000,    # Reduce for lower cost
)
```

### Custom Kill Prompts

Edit `falsifier/stage2/hypothesis_gen.py`:

```python
FALSIFIER_SYSTEM_PROMPT = """Your custom prompt here...
Focus on specific failure modes you care about.
"""
```

## Debugging Stage 2

### Enable Verbose Logging

```python
# In hypothesis_gen.py
def _generate_llm_hypotheses(...):
    print(f"[DEBUG] Context length: {len(context)} chars")
    print(f"[DEBUG] Sending to Anthropic...")
    response = client.messages.create(...)
    print(f"[DEBUG] Response: {response.content[0].text[:200]}...")
```

### Test Stage 2 in Isolation

```bash
# Run a single hypothesis with detailed logging
python3 -c "
from falsifier.stage2.hypothesis_gen import generate_kill_hypotheses
# ... test code ...
"
```

## Future Enhancements

### Planned Improvements

1. **Multi-model Stage 2**: Compare Sonnet vs Haiku vs Opus kills
2. **Iterative Refinement**: Generate hypotheses → test → refine based on results
3. **Knowledge Graph Context**: Richer context from graph history
4. **Parallel Execution**: Run experiments for multiple hypotheses concurrently

### Research Questions

- Does Sonnet generate more accurate kills than heuristics?
- Do Stage 2 kills correlate with real training failures?
- Can we learn a meta-classifier from Stage 2 results?

## Integration with Visualization

Stage 2 results appear in the visualization:

```json
{
  "evolution": [
    {
      "frame": 5,
      "stage2_triggered": 2,
      "stage2_passed": 1,
      "stage2_killed": 1
    }
  ]
}
```

The HTML viewer shows:
- Which theories triggered Stage 2
- Which survived vs were killed
- Kill hypothesis details on hover

## Troubleshooting

### Stage 2 Not Triggering

```bash
# Check if ANTHROPIC_API_KEY is set
echo $ANTHROPIC_API_KEY | head -c 20

# Check if anthropic package is installed
python3 -c "import anthropic; print('OK')"
```

### High API Costs

```python
# Reduce max_tokens
stage2_config = Stage2Config(
    max_tokens=2000,  # Down from 4000
)

# Reduce hypotheses per theory
max_hypotheses=3,  # Down from 5
```

### Slow Execution

```bash
# Run Stage 2 only on most promising theories
# Modify run_full_live_experiment.py to filter by Stage 1 score
```

## References

- Anthropic Sonnet: https://www.anthropic.com/news/claude-3-5-sonnet
- Falsifier Pipeline: ../../docs/falsification_analysis.md
- Knowledge Graph: ../../knowledge_graph/
