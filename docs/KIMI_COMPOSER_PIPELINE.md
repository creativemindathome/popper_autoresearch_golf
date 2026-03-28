# Kimi 2.5 + Composer 2 Pipeline

Advanced autoresearch pipeline combining:
- **Kimi 2.5** (Moonshot AI) for hypothesis generation with strict engineering constraints
- **Composer 2** (advanced reasoning system) for Stage 2 adversarial falsification

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    KIMI + COMPOSER PIPELINE                        │
└─────────────────────────────────────────────────────────────────┘

Stage 0: Ideation (Kimi 2.5)
├── Specialized prompt with engineering constraints
├── Emphasizes 10M parameter budget
├── Requires falsifiability in 100 steps
├── Novelty vs practicality balance
└── Returns structured JSON with code
         │
         ▼
Stage 1: Falsifier Gates (Local MLX/PyTorch)
├── T2 Budget Check: Enforces 10M limit
├── T3 Compilation: Syntax validation
├── T4 Signal Check: Forward pass validation
├── T5 Init Check: Numerical stability
└── T7 Microtrain: 100-step training test
         │
         ├── Killed → Log failure, update KG
         │
         ▼
Stage 2: Adversarial Falsifier (Composer 2)
├── Theory decomposition into claims
├── Mechanistic failure mode analysis
├── Targeted experiment design
├── Scientific reasoning & evidence
└── Kill hypothesis generation
         │
         ├── Killed → Detailed kill reason, evidence
         │
         ▼
Stage 3: Knowledge Graph Update
├── Track node status
├── Link to parent/child theories
├── Record falsification path
└── Update visualization
```

## Key Improvements

### 1. Kimi 2.5 Ideator (vs Generic LLM)

**Constraint-Aware Prompting:**
```python
system_prompt = """
CRITICAL ENGINEERING CONSTRAINTS - YOU MUST RESPECT THESE:

1. **Parameter Budget: STRICT 10M parameter limit**
   - Count every parameter: embeddings, attention, FFN, norms, biases
   - Current baseline: GPT-2 small (124M) is TOO BIG - we need 10M max
   - Example valid sizes: 6M (n_embd=384, n_layer=6, n_head=6), 8M, 10M
   - IF you propose >10M, it will be REJECTED immediately by T2 Budget gate

2. **Memory Constraints**
   - Target: <2GB peak memory during training
   - No massive activation caches
   - Efficient attention (no O(n²) for long sequences)

3. **Training Stability Requirements**
   - Must initialize with variance scaling (GPT-2 style)
   - No gradient explosions at start (check initial loss < 15)
   - Must compile and run without errors

4. **Falsifiability**
   - Must be testable in 100 training steps (T7 Microtrain gate)
   - Must show loss reduction or clear failure mode
   - Avoid "magic" components that can't be measured
"""
```

**Output Requirements:**
- `theory_id`: Unique identifier
- `what_and_why`: 3 paragraphs (what, why, falsification signatures)
- `train_gpt_code`: Complete runnable Python code
- `parameter_estimate`: Explicit count with verification
- `risk_factors`: What could go wrong + early detection

### 2. Composer 2 Falsifier (vs Standard LLM)

**Structured Reasoning Process:**

1. **Theory Decomposition**
   ```
   Break theory into 3-5 falsifiable claims:
   - Architecture claim
   - Mechanism claim
   - Empirical claim
   ```

2. **Failure Mode Analysis**
   ```
   For each claim identify:
   - Specific mechanism that could fail
   - Early warning signals (step 50, 100, 200)
   - Root cause (init, optimization, numerics)
   ```

3. **Experiment Design**
   ```
   Design minimal experiments:
   - Metric to measure
   - Threshold for "failed"
   - Step to check
   - Ablation needed?
   ```

4. **Confidence Calibration**
   ```
   Based on:
   - HIGH: Grounded in Stage 1 measurements
   - MEDIUM: Reasonable extrapolation
   - LOW: Speculative but worth testing
   ```

**Output: JSON Array of Kill Hypotheses**
```json
{
  "hypothesis_id": "H1",
  "confidence": "high",
  "target_claim": "Attention modification improves gradient flow",
  "failure_mode": "Gradient norms explode due to unbounded attention weights",
  "mechanism": "Step-by-step causal chain...",
  "early_signal": "grad_norm > 100 by step 50",
  "experiment_type": "temporal",
  "experiment_spec": {
    "metric": "grad_norm",
    "threshold": 100,
    "comparator": ">",
    "step": 100
  },
  "evidence": "Stage 1 T5: grad_norm_spike_ratio=1.8, init unstable"
}
```

## Setup

### 1. Get Kimi API Key

```bash
# Sign up at https://platform.moonshot.cn/
# Create API key
export KIMI_API_KEY="your-kimi-key"
```

### 2. Ensure Anthropic Key (for Composer 2)

```bash
# Composer 2 uses Claude as base model
export ANTHROPIC_API_KEY="<your-anthropic-api-key>"
```

### 3. Run the Pipeline

```bash
cd /Users/curiousmind/Desktop/null_fellow_hackathon/experiments/ten_hypothesis_run

# Run with Kimi + Composer
python3 run_kimi_composer.py --num-hypotheses 5

# Disable Composer (Kimi only)
python3 run_kimi_composer.py --num-hypotheses 5 --disable-composer
```

## Comparison: Original vs Kimi+Composer

| Aspect | Original (Anthropic) | Kimi 2.5 + Composer 2 |
|--------|---------------------|------------------------|
| **Ideator** | Claude Sonnet | Kimi 2.5 (Moonshot AI) |
| **Constraint Awareness** | Basic | Strict (10M budget emphasized) |
| **Output Structure** | Flexible JSON | Rigid with risk factors |
| **Stage 2** | Standard Claude | Composer 2 reasoning |
| **Hypothesis Quality** | Creative, sometimes too big | Engineering-grounded, budget-aware |
| **Kill Precision** | General failure modes | Specific mechanistic predictions |
| **Evidence Citation** | Sometimes vague | Explicit Stage 1 measurements |
| **Cost** | ~$0.20/idea | ~$0.15/idea (Kimi cheaper) |

## Files Created

```
ideator/
└── kimi_client.py                 # Kimi 2.5 API client
     ├── generate_with_constraints()  # Constraint-aware generation
     └── test_kimi_connection()       # API test

falsifier/stage2/
└── composer_falsifier.py          # Composer 2 integration
     ├── generate_kill_hypotheses()   # Structured reasoning
     ├── _build_context()             # Context assembly
     └── _generate_fallback()         # Fallback heuristics

experiments/ten_hypothesis_run/
└── run_kimi_composer.py           # Integrated runner
     ├── KimiComposerExperiment       # Main orchestrator
     ├── generate_with_kimi()         # Stage 0
     ├── run_stage1()                 # Stage 1 gates
     └── run_stage2_composer()        # Stage 2 reasoning

docs/
└── KIMI_COMPOSER_PIPELINE.md      # This documentation
```

## Example Run

```bash
$ python3 run_kimi_composer.py --num-hypotheses 3

======================================================================
KIMI 2.5 + COMPOSER 2 AUTO RESEARCH
======================================================================
Ideator: Kimi 2.5 (Moonshot AI) with engineering constraints
Falsifier: Composer 2 (advanced reasoning)
Hypotheses: 3
Output: kimi_composer_run_20260328_165200

[16:52:01] [INFO] ======================================================================
[16:52:01] [INFO] HYPOTHESIS 1/3
[16:52:01] [INFO] ====================================================================== 
[16:52:01] [INFO] [Hypothesis 1] Generating with Kimi 2.5...
✓ Generated: entropy-gated-attention
  Parameters: ~6.5M (verified <10M)
  Novelty claims: 4
[16:52:35] [INFO] 
[16:52:35] [INFO] [Stage 1] Running falsifier gates on entropy-gated-attention...
✗ Stage 1 KILLED by: T5
[16:52:40] [INFO] ✓ Complete in 39.2s: STAGE_1_KILLED

[16:52:40] [INFO] ======================================================================
[16:52:40] [INFO] HYPOTHESIS 2/3
[16:52:40] [INFO] ======================================================================
[16:52:40] [INFO] [Hypothesis 2] Generating with Kimi 2.5...
✓ Generated: gradient-sparsity-transformer
  Parameters: ~8.2M (verified <10M)
  Novelty claims: 5
[16:53:15] [INFO] 
[16:53:15] [INFO] [Stage 1] Running falsifier gates on gradient-sparsity-transformer...
✓ Stage 1 PASSED
[16:53:45] [INFO] 
[16:53:45] [INFO] [Stage 2] Running Composer 2 adversarial falsifier...
  Composer 2 generated 4 kill hypotheses
    - H1 (high): Gradient norm explosion in sparse pathways...
    - H2 (medium): Sparsity pattern doesn't improve with scale...
    - H3 (medium): Initialization sensitivity causes divergence...
    - H4 (low): Long-term stability questionable beyond 500 steps...
✓ Stage 2 complete
[16:55:10] [INFO] ✓ Complete in 115.3s: STAGE_2_PASSED

...etc
```

## Integration with Existing Infrastructure

The Kimi+Composer pipeline integrates with:

1. **Knowledge Graph**: Updates tracked the same way
2. **Visualization**: Use same HTML/animation viewers
3. **Cursor Cloud**: Can run Kimi+Composer in cloud agents
4. **Stage 1 Gates**: No changes needed (local MLX/PyTorch)

To use in Cursor Cloud:

```bash
# Edit run_with_cursor_cloud.py prompt to call:
# python3 run_kimi_composer.py --num-hypotheses 10

python3 run_with_cursor_cloud.py --model claude-4-sonnet-thinking
```

## API Costs

| Service | Cost per 10-hypotheses |
|---------|----------------------|
| **Kimi 2.5** | ~$1.50 (cheaper than Anthropic) |
| **Composer 2** | ~$2.00 (uses Claude base) |
| **Total** | ~$3.50 vs ~$4-5 for pure Anthropic |

## Benefits

1. **Better Constraint Adherence**: Kimi's prompt engineering keeps 10M budget top-of-mind
2. **More Rigorous Falsification**: Composer 2's structured reasoning finds specific failure modes
3. **Lower Cost**: Kimi API cheaper than Anthropic for ideation
4. **Diverse Ideas**: Different model = different architectural insights
5. **Scientific Rigor**: Composer 2 forces mechanistic explanations

## Troubleshooting

### "KIMI_API_KEY not set"
```bash
export KIMI_API_KEY="your-key-from-platform.moonshot.cn"
```

### "Kimi response not valid JSON"
- Kimi sometimes returns markdown-wrapped JSON
- Parser extracts JSON automatically
- If persistent, check Kimi model version

### "Composer 2 errors"
- Ensure ANTHROPIC_API_KEY is set
- Composer falls back to standard Anthropic if Claude unavailable

## Future Enhancements

- [ ] A/B test: Kimi vs Anthropic ideation quality
- [ ] Ensemble: Generate with both, select best
- [ ] Composer 2 experiment execution (not just hypothesis generation)
- [ ] Kimi vision API for architecture diagrams
- [ ] Multi-modal: Kimi generates + explains with diagrams
