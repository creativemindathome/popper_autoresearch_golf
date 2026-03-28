# Hypothesis History and Dead Ends

## Run Summary (2026-03-28)

**Total Hypotheses Generated:** 10
**Verdicts:** 8 REFUTED | 2 UNKNOWN | 0 VERIFIED

## Critical Pattern Identified

**ALL attention-related modifications FAILED T2 Budget Check**

Every hypothesis involving attention mechanism modifications has been refuted due to exceeding the parameter/compute budget. This is a strong signal to pivot away from attention modifications.

## Refuted Hypotheses (Dead Ends)

### Attention Modifications (8 failures)
1. **entropy-guided-sparse-attention** - T2 Killed (35.2s)
2. **differential-attention-routing** - T2 Killed (40.8s)
3. **gradient-guided-sparse-attention** - T2 Killed (38.3s)
4. **fourier-attention-gates** - T2 Killed (32.2s)
5. **gradient-informed-attention-v7** - T2 Killed (41.7s)
6. **gradient-conditioned-attention-v8** - T2 Killed (37.9s)

### Architecture Modifications (1 failure)
7. **temporal-depth-modulation** - T2 Killed (36.8s)

### MLP Modifications (1 failure)
8. **adaptive-sparse-moe-6** - T2 Killed (39.9s)

## Patterns to AVOID

### 1. Sparse Attention Mechanisms
- Any form of "sparse attention" adds too many parameters
- Budget overrun is immediate (T2 fails within 30-45 seconds)
- Examples: entropy-guided, gradient-guided, token-modulated

### 2. Attention Routing/Gating
- Differential routing increases compute overhead
- Fourier gating requires additional transforms
- Gradient conditioning adds backward-pass complexity

### 3. Architecture Depth Modifications
- Temporal depth modulation increases memory
- Fractal/recursive structures explode parameter count

## Suggested PIVOTS (Untried Areas)

### High Potential (No attempts yet)
1. **Data Pipeline Optimizations**
   - Curriculum learning strategies
   - Tokenization improvements
   - Sequence packing optimizations

2. **Training Efficiency**
   - 8-bit optimizer states (AdamW 8-bit)
   - Adafactor with factorized moments
   - Mixed precision strategies (FP8)

3. **Memory Optimization**
   - Gradient checkpointing variations
   - KV-cache quantization (INT8)
   - Activation storage strategies

4. **Normalization & Regularization**
   - RMSNorm vs LayerNorm tradeoffs
   - Dropout scheduling
   - Label smoothing variations

### Medium Potential (Related to known successes)
- Embedding factorization (low-rank)
- Weight tying between embeddings and output
- Attention projection sharing (GQA/MQA)

## Context for Next Run

The system has hit a wall with attention modifications. All 8 attention-related ideas failed immediately at the budget check stage. This suggests:

1. The parameter budget is too tight for attention innovations
2. Need to find wins in data/training efficiency instead
3. Consider architectural simplifications rather than additions

**IDEATOR GUIDANCE:** Do not generate attention-related hypotheses. Focus on data pipeline, training optimization, or memory efficiency improvements.
