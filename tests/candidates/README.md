# Stage 1 Gate Test Candidates

This directory contains 5 test candidate theories designed to validate the Stage 1 gate signals (T2-T7) of the falsifier. Each candidate is a complete training script based on `train_gpt_mlx.py` with specific modifications to trigger expected pass/fail behavior.

## Candidate Overview

### 1. `good_student.py` - The Baseline Candidate
**Expected: PASS T2-T7, Reach Stage 2**

A minimal modification to the baseline training script with a slightly reduced learning rate (0.03 → 0.025). This represents a "well-behaved" theory that should sail through all Stage 1 gates.

**Modification:**
- Reduced `matrix_lr` from 0.04 to 0.025 (via env override in code)

**Why it passes:**
- T2: Fits within parameter budget (same architecture)
- T3: Imports and compiles correctly
- T4: Normal gradient flow, healthy entropy
- T5: Proper initialization statistics
- T7: Normal learning trajectory over 100 steps

---

### 2. `overconfident_logits.py` - T5 Init Gate Failure
**Expected: FAIL T5 (Initialization Statistics)**

Multiplies final logits by 2.0 in the forward pass, creating extreme output values that skew initialization statistics.

**Modification:**
- In `GPT.loss()`: `logits = self.softcap(logits_proj) * 2.0`

**Why it fails T5:**
- T5 compares minimal-model init statistics (kurtosis, effective rank) against baseline
- Extreme logits cause abnormal weight distributions
- Weight kurtosis mean will be outside the acceptable log-band
- Effective rank mean will deviate from baseline

---

### 3. `gradient_flow_issues.py` - T4/T7 Failure
**Expected: FAIL T4 (Gradient Ratio) or T7 (Slow Learning)**

Removes the attention residual connection (`x = x + attn_out`) in the transformer block, breaking gradient flow and causing vanishing gradients in early layers.

**Modification:**
- In `Block.__call__()`: Removed `x = x + self.attn_scale.astype(x.dtype)[None, None, :] * attn_out`

**Why it fails:**
- **T4**: Gradient norm ratio will exceed the fatal threshold (max/min across layers > 1000x baseline)
  - Early layers receive almost no gradient signal
  - Later layers have normal gradients
  - Creates extreme gradient norm ratio
- **T7**: Slow learning due to blocked gradient flow
  - Learning ratio will be below kill threshold
  - Loss won't decrease significantly over 100 steps

---

### 4. `over_parametrized.py` - T2 Budget Failure
**Expected: FAIL T2 (Parameter Budget)**

Dramatically increases model capacity to 32 layers and 1024 dimensions, far exceeding the 16MB artifact size limit.

**Modification:**
- `num_layers = 32` (default: 9)
- `model_dim = 1024` (default: 512)

**Why it fails T2:**
- T2 checks artifact size against 16MB limit (minus 200KB safety margin)
- 32 layers × 1024 dim model has ~100M+ parameters
- Estimated artifact size will be >> 16MB effective limit
- Returns `FAIL_FATAL` with reason about exceeding size limit

---

### 5. `broken_architecture.py` - T3 Compilation Failure
**Expected: FAIL T3 (Compilation/Construction)**

Sets `model_dim = 31` which is not divisible by `num_heads = 4`, causing a ValueError during model instantiation.

**Modification:**
- `model_dim = 31` (not divisible by num_heads=4)

**Why it fails T3:**
- T3 instantiates a minimal model to verify compilation
- `CausalSelfAttention.__init__()` validates: `if dim % num_heads != 0: raise ValueError`
- Model construction fails before forward/backward pass
- Returns `FAIL_FATAL` with `InstantiationError`

---

## Usage

Run candidates through the falsifier to verify gate behavior:

```python
from pathlib import Path
from falsifier.main import run_falsifier
from tests.candidates import get_candidate_path

# Test a specific candidate
candidate_path = get_candidate_path("overconfident_logits")
result = run_falsifier(
    proposed_train_gpt=candidate_path.read_text(),
    train_gpt_path=candidate_path,
    # ... other required inputs
)

# Check which gate failed
print(result.stage1.t5.status)  # Expected: "FAIL_FATAL"
print(result.stage1.t5.kill_reason)  # Expected: kurtosis/rank outside band
```

## Gate Mapping Reference

| Candidate | Expected Fail Gate | Failure Mode |
|-----------|-------------------|--------------|
| good_student | None | Passes all gates |
| overconfident_logits | T5 | Init stats outside band |
| gradient_flow_issues | T4/T7 | Gradient pathology / Slow learning |
| over_parametrized | T2 | Budget exceeded |
| broken_architecture | T3 | Construction error |

## Testing Strategy

1. **Validate gate detection**: Each candidate should trigger its expected gate
2. **Verify error messages**: Check that kill_reason contains expected text
3. **Test boundary conditions**: Candidates represent clear pass/fail cases
4. **Regression testing**: Run all candidates when modifying gate logic

## Files Generated

Each candidate is a standalone training script (~1100 lines) that can be:
- Imported as a module
- Executed directly (`python tests/candidates/good_student.py`)
- Passed to the falsifier as `proposed_train_gpt` content
