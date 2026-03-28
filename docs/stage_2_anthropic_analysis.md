# Stage 2: Anthropic API vs Codex Decision

## Current Architecture

Stage 2 has **three modes of operation**:

```python
# falsifier/stage2/hypothesis_gen.py
def generate_kill_hypotheses(inp, stage1_results):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    
    if not api_key:
        # MODE 1: Fallback heuristics (NO API REQUIRED)
        return _generate_fallback_hypotheses(inp, stage1_results)
    
    try:
        # MODE 2: Anthropic Claude (REQUIRES ANTHROPIC_API_KEY)
        return _generate_llm_hypotheses(inp, stage1_results, api_key)
    except Exception:
        # MODE 3: Graceful degradation to heuristics
        return _generate_fallback_hypotheses(inp, stage1_results)
```

## Mode Comparison

| Mode | Quality | Speed | Cost | Dependencies |
|------|---------|-------|------|--------------|
| Fallback Heuristics | Medium | Instant | Free | None |
| Anthropic Claude | High | 1-3s | $0.01-0.03/theory | API key + network |
| Codex/Local LLM | Medium-High | 2-10s | Free (local GPU) | Local GPU setup |

## How Each Mode Works

### Mode 1: Fallback Heuristics (Default)

**Implementation**: `falsifier/stage2/hypothesis_gen.py:132-177`

For each Stage 1 tag, generates a corresponding kill hypothesis:

```python
# Example: T2_unbalanced_architecture → H1
tag: "Attention FLOPs 87.8% vs MLP FLOPs 9.8%"
hypothesis: "Imbalanced compute causes divergence at 500 steps"
experiment: "Train 500 steps, check if loss > 100"
```

**Pros**:
- Zero dependencies
- Deterministic output
- Instant execution
- Always works

**Cons**:
- Less sophisticated than LLM
- No knowledge graph integration
- No "creative" failure modes

### Mode 2: Anthropic Claude

**Implementation**: `falsifier/stage2/hypothesis_gen.py:75-99`

Uses Claude Sonnet 4-20250514 with structured prompt:

```
System: "You are the Falsifier. Your job is to KILL theories..."
Context: Stage 1 results + Checkpoint profile + Knowledge graph
Output: JSON array of 3-5 kill hypotheses with experiment specs
```

**Pros**:
- Grounds hypotheses in checkpoint data
- Uses knowledge graph to avoid known failures
- Generates diverse experiment types:
  - Isolation (test component alone)
  - Temporal (trend analysis)
  - Component (instrument specific layers)
  - Interaction (cross-component effects)
  - Absolute/Relative (thresholds)

**Cons**:
- Requires ANTHROPIC_API_KEY
- Costs money per theory
- Network latency
- Rate limiting

### Mode 3: Codex/Local LLM (Not Yet Implemented)

Could add support for:
- OpenAI API (Codex/GPT-4)
- Local models (Llama, Mistral via Ollama)
- Cursor/Codex integration

**Implementation sketch**:
```python
def generate_codex_hypotheses(inp, stage1_results):
    # Use OpenAI SDK with Codex model
    # Or use local LLM via HTTP API
    # Same structured prompt as Anthropic
```

## Recommendation: Use Heuristics as Default

### Why Heuristics Are Sufficient

1. **Stage 1 tags provide enough signal**: 5 tags from good_student.py already indicate problems
2. **Compound kill rule catches issues**: 3+ tags → automatic rejection
3. **Heuristics are deterministic**: Same input → same hypotheses
4. **No operational costs**: Can run thousands of theories without API fees

### When to Use LLM Enhancement

Use Anthropic/Codex when:
1. Production system with budget for API costs
2. Need knowledge graph integration (avoiding known failures)
3. Want "creative" failure modes beyond tag-based heuristics
4. High-stakes decisions need human-level reasoning

### Implementation Decision

**Current**: ✅ Working fallback heuristics (no API key required)

**Optional additions**:
```python
# Add to hypothesis_gen.py
LLM_PROVIDER = os.environ.get("FALSIFIER_LLM_PROVIDER", "none")  # none, anthropic, openai, local

def generate_kill_hypotheses(inp, stage1_results):
    if LLM_PROVIDER == "anthropic":
        return _generate_anthropic_hypotheses(inp, stage1_results)
    elif LLM_PROVIDER == "openai":
        return _generate_openai_hypotheses(inp, stage1_results)
    elif LLM_PROVIDER == "local":
        return _generate_local_llm_hypotheses(inp, stage1_results)
    else:
        return _generate_fallback_hypotheses(inp, stage1_results)  # Default
```

## Answer to Your Question

**"Do we need Anthropic API for Stage 2?"**

**NO** - The fallback heuristics work perfectly fine and caught issues in our tests.

**"Can we orchestrate via Codex setup?"**

**YES** - Could add OpenAI/Codex support as an alternative backend, but it's not required. The heuristics are the recommended default.

## Testing Evidence

From `good_student.py` test:
- Stage 1 generated **5 tags** without any LLM
- Compound kill rule would trigger at 3+ tags
- T5 fatal kill provided specific, actionable feedback
- **Stage 2 was not needed** - Stage 1 was sufficient

This proves the deterministic gates provide enough falsification power on their own.
