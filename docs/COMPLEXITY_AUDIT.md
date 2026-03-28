# Experimental Loop Complexity Audit

## The Pipeline (Current State)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. IDEATOR                                                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  Output Schema A (CLI ideator)          Output Schema B (Experiment)      │
│  ──────────────────────────────────     ─────────────────────────────     │
│  {                                        {                                 │
│    "schema_version": "ideator.idea.v1",     "theory_id": "...",             │
│    "idea_id": "...",                       "what_and_why": "...",           │
│    "title": "...",                         "train_gpt_code": "...",         │
│    "novelty_summary": "...",               "novelty_claims": [...],           │
│    "implementation_steps": [...],          "expected_behavior": "...",      │
│    "parent_implementation": {...}          "parameter_estimate": "..."      │
│  }                                        }                                 │
│                                                                             │
│  🔴 DRIFT: Two schemas for the same concept!                                │
│  🔴 COMPLEXITY: _translate_idea_for_reviewer() invents "ideator.idea.v2"    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. REVIEWER GATE (Optional)                                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  Input: Fake ideator.idea.v2 structure                                     │
│  Problem: Translates experiment output to synthetic ideator format          │
│                                                                             │
│  🔴 COMPLEXITY: Translation layer for translation layer!                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. FALSIFIER INPUT TRANSFORMATION                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│  Path A: CLI (--idea-id)                                                   │
│  ─────────────────────────────────────────────────────────────────────    │
│  ideator.idea.v1 → ideator_adapter.py → FalsifierInput                      │
│  (parses implementation_steps with regex heuristics!)                       │
│                                                                             │
│  Path B: Experiment                                                         │
│  ─────────────────────────────────────────────────────────────────────    │
│  experiment idea → direct FalsifierInput construction                     │
│  (skips adapter, different path!)                                           │
│                                                                             │
│  🔴 DRIFT: Two entry paths into falsifier with different transforms        │
│  🔴 COMPLEXITY: Adapter does regex parsing on natural language              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. STAGE 1 EXECUTION                                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  Input: FalsifierInput (dataclass)                                         │
│  Output: FalsifierOutput (dataclass)                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 5. STAGE 1 SERIALIZATION (THE WASTEFUL PART)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Step A: run_stage1() returns FalsifierOutput (dataclass)                 │
│          │                                                                  │
│          ▼                                                                  │
│  Step B: Immediately serialized to dict via asdict()                      │
│          result = {                                                         │
│            "verdict": output.verdict,                                     │
│            "t2_budget": asdict(output.t2_budget),  // ← dataclass → dict  │
│            ...                                                              │
│          }                                                                  │
│          │                                                                  │
│          ▼                                                                  │
│  Step C: Stored in run.stage1_result (dict)                                │
│          │                                                                  │
│          ▼                                                                  │
│  Step D: _reconstruct_falsifier_output() rebuilds dataclasses              │
│          t2_budget = _rebuild(T2Result, stage1_result.get("t2_budget"))   │
│          // ← dict → dataclass (reverse of Step B!)                         │
│          │                                                                  │
│          ▼                                                                  │
│  Step E: FalsifierOutput passed to Stage 2                                  │
│          │                                                                  │
│          ▼                                                                  │
│  Step F: At sync time, build_falsifier_output_from_results()               │
│          rebuilds AGAIN for lifecycle module                               │
│          t2_budget = _rebuild_dataclass(T2Result, ...)  // ← 3rd rebuild! │
│                                                                             │
│  🔴 COMPLEXITY: 3 round-trip conversions dataclass↔dict                     │
│  🔴 WASTE: Field filtering at each boundary loses data                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 6. KNOWLEDGE GRAPH WRITE (THE DRIFT PART)                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Path A: legacy update.py (DEPRECATED but still exists)                   │
│  ─────────────────────────────────────────────────────────────────────    │
│  - Direct file I/O, no locking                                              │
│  - Flat structure: node.measured_metrics = {...}                           │
│  - Old field names: test_results.T2 (uppercase)                          │
│                                                                             │
│  Path B: lifecycle.py (CANONICAL)                                          │
│  ─────────────────────────────────────────────────────────────────────    │
│  - AtomicGraphUpdate with file locking                                     │
│  - Nested structure: falsification.test_results.t2_budget                │
│  - New field names: t2_budget (lowercase)                                  │
│  - Status history tracking                                                  │
│                                                                             │
│  Path C: experiment_sync.py (WRAPPER around B)                             │
│  ─────────────────────────────────────────────────────────────────────    │
│  - Calls lifecycle helpers                                                  │
│  - Reconstructs FalsifierOutput yet again                                   │
│                                                                             │
│  🔴 DRIFT: Nodes in graph.json have MIXED schemas!                         │
│  🔴 DRIFT: Some have T2 (old), some have t2_budget (new)                  │
│  🔴 COMPLEXITY: Reader code must handle both schemas (_get_t2_feedback)     │
└─────────────────────────────────────────────────────────────────────────────┘

```

## The Complexity Taxonomy

### Type 1: Schema Drift (Data Model Fragmentation)

| Location | Old Schema | New Schema | Problem |
|----------|------------|------------|---------|
| Ideator output | `ideator.idea.v1` with `implementation_steps` | Experiment's ad-hoc dict with `train_gpt_code` | Two parallel universes |
| Test results | `T2`, `T3`, `T4`, `T5`, `T7` (uppercase) | `t2_budget`, `t3_compilation`, etc. (lowercase) | Readers need dual parsers |
| Graph structure | `node.measured_metrics` (flat) | `node.falsification.test_results` (nested) | Queries need multiple paths |
| Outcome values | `FAILED`, `PASSED`, `ITERATE` | `REFUTED`, `STAGE_1_PASSED`, `STAGE_2_PASSED` | Logic branches diverge |

### Type 2: Redundant Serialization (Round-Trip Waste)

```python
# The Cycle (happens for EVERY hypothesis):

FalsifierOutput(dataclass)          # 1. Original
    ↓ asdict()
dict (stage1_result)              # 2. Serialized
    ↓ stored in run object
dict (run.stage1_result)            # 3. Retrieved
    ↓ _rebuild()
FalsifierOutput(dataclass)          # 4. Reconstructed for Stage 2
    ↓ stage2 processing
FalsifierOutput(modified)           # 5. Stage 2 results
    ↓ stored in run.stage2_result
dict                              # 6. Serialized again
    ↓ sync_experiment_results()
    ↓ build_falsifier_output_from_results()
FalsifierOutput(dataclass)          # 7. Reconstructed AGAIN
    ↓ update_node_with_falsification_results()
JSON in graph.json                  # 8. Final destination

# Total: 3 dataclass→dict→dataclass round trips!
# We could go: dataclass → JSON directly (1 trip)
```

### Type 3: Path Multiplication (Entry Point Explosion)

```
Ideator Output
    ├── CLI Path → ideator.idea.v1 → adapter.py → FalsifierInput
    └── Experiment Path → experiment dict → direct construction

Falsifier Result
    ├── Legacy Path → update.py → graph.json (flat, old schema)
    └── Canonical Path → lifecycle.py → graph.json (nested, new schema)
```

### Type 4: Translation Layers (Middleman Bloat)

| Layer | Purpose | Problem |
|-------|---------|---------|
| `_translate_idea_for_reviewer()` | Convert experiment idea to fake ideator.v2 | Invented schema doesn't exist |
| `ideator_adapter.py` | Parse natural language → config_delta | Regex heuristics on unstructured text |
| `experiment_sync.py` | Wrap lifecycle helpers | Another indirection layer |
| `_reconstruct_falsifier_output()` | Dict → dataclass | Reverse of asdict() we just did |
| `_get_t2_feedback_from_graph()` | Read graph → feedback | Must handle both T2 and t2_budget |

## The Fix Strategy

### Immediate (Low Risk)

1. **Keep FalsifierOutput as dataclass throughout**
   - Don't `asdict()` in run_stage1()
   - Store dataclass directly in HypothesisRun
   - Only serialize at final write to graph

2. **Delete redundant _reconstruct_falsifier_output**
   - If we keep dataclass, no need to rebuild

3. **Consolidate on lifecycle.py path**
   - Delete or deprecate update.py
   - Single schema for all graph writes

### Medium (Medium Risk)

4. **Unify ideator output schema**
   - Either: Add `train_gpt_code` to ideator.idea.v1
   - Or: Make experiment use real ideator output format

5. **Remove _translate_idea_for_reviewer**
   - If schemas unified, no translation needed

### Long-term (Higher Risk)

6. **Migrate graph.json to single schema**
   - One-time migration script
   - All nodes use t2_budget format
   - Remove backward compatibility code

## Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Schema versions | 4+ (v1, v2-fake, experiment, internal) | 1 |
| Serialization round-trips per hypothesis | 3 | 1 |
| Graph write paths | 3 (update.py, lifecycle.py, experiment_sync.py) | 1 |
| Translation layers | 5+ | 0-1 |
| Lines of transformation code | ~500 | ~100 |
