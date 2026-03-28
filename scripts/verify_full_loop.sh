#!/bin/bash
# Verify Full Loop Execution Readiness
# Usage: bash scripts/verify_full_loop.sh

set -e

echo "================================================================================"
echo "FULL LOOP EXECUTION VERIFICATION"
echo "================================================================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track results
PASSED=0
FAILED=0

check_pass() {
    echo -e "${GREEN}✓${NC} $1"
    PASSED=$((PASSED + 1))
}

check_fail() {
    echo -e "${RED}✗${NC} $1"
    FAILED=$((FAILED + 1))
}

check_warn() {
    echo -e "${YELLOW}○${NC} $1"
}

choose_python() {
    if [ -x ".venv/bin/python" ]; then
        echo ".venv/bin/python"
        return 0
    fi
    if command -v python3.12 &> /dev/null; then
        command -v python3.12
        return 0
    fi
    if command -v python3 &> /dev/null; then
        command -v python3
        return 0
    fi
    return 1
}

python_version_tuple() {
    "$1" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
}

python_is_3_12_plus() {
    "$1" -c "import sys; raise SystemExit(0 if (sys.version_info.major, sys.version_info.minor) >= (3, 12) else 1)"
}

run_with_timeout() {
    local seconds="$1"
    shift
    if command -v timeout &> /dev/null; then
        timeout "$seconds" "$@"
        return $?
    fi
    if command -v gtimeout &> /dev/null; then
        gtimeout "$seconds" "$@"
        return $?
    fi
    check_warn "timeout not found; running without a timeout"
    "$@"
}

echo "1. KNOWLEDGE GRAPH INFRASTRUCTURE"
echo "----------------------------------------"

DIRS=(
    "knowledge_graph"
    "knowledge_graph/inbox/approved"
    "knowledge_graph/outbox/ideator"
    "knowledge_graph/outbox/falsifier"
    "knowledge_graph/work/in_falsification"
)

for dir in "${DIRS[@]}"; do
    if [ -d "$dir" ]; then
        check_pass "$dir/"
    else
        check_warn "$dir/ (missing; creating)"
        mkdir -p "$dir"
        check_pass "$dir/ (created)"
    fi
done

echo ""
echo "2. PYTHON ENVIRONMENT"
echo "----------------------------------------"

# Check Python (requires Python >= 3.12)
PYTHON=""
PYTHON_OK=0
if PYTHON=$(choose_python); then
    PYTHON_VERSION=$(python_version_tuple "$PYTHON" 2>/dev/null || echo "unknown")
    if python_is_3_12_plus "$PYTHON" 2>/dev/null; then
        PYTHON_OK=1
        check_pass "Python $PYTHON_VERSION ($PYTHON)"
    else
        check_fail "Python >= 3.12 required (found $PYTHON_VERSION at $PYTHON)"
    fi
else
    check_fail "Python not found (.venv/bin/python, python3.12, or python3)"
fi

# Check imports
if [ "$PYTHON_OK" -eq 1 ]; then
    if "$PYTHON" -c "import ideator" 2>/dev/null; then
        check_pass "ideator module imports"
    else
        check_fail "ideator module import failed"
    fi

    if "$PYTHON" -c "import falsifier" 2>/dev/null; then
        check_pass "falsifier module imports"
    else
        check_fail "falsifier module import failed"
    fi

    if "$PYTHON" -c "from falsifier.graph.lifecycle import update_node_status" 2>/dev/null; then
        check_pass "graph.lifecycle imports"
    else
        check_fail "graph.lifecycle import failed"
    fi

    if "$PYTHON" -c "from falsifier.adapters.ideator_adapter import load_and_adapt_ideator_idea" 2>/dev/null; then
        check_pass "ideator_adapter imports"
    else
        check_fail "ideator_adapter import failed"
    fi
else
    check_warn "Skipping import checks (Python >= 3.12 not available)"
fi

echo ""
echo "3. API KEYS"
echo "----------------------------------------"

if [ -n "$GEMINI_API_KEY" ]; then
    check_pass "GEMINI_API_KEY set"
else
    check_warn "GEMINI_API_KEY not set (required for ideator)"
fi

if [ -n "$OPENAI_API_KEY" ]; then
    check_pass "OPENAI_API_KEY set"
else
    check_warn "OPENAI_API_KEY not set (required for reviewer)"
fi

if [ -n "$ANTHROPIC_API_KEY" ]; then
    check_pass "ANTHROPIC_API_KEY set (optional)"
else
    check_warn "ANTHROPIC_API_KEY not set (optional - fallback works)"
fi

echo ""
echo "4. TEST DATA"
echo "----------------------------------------"

if [ -f "tests/candidates/good_student.py" ]; then
    check_pass "test candidate: good_student.py"
else
    check_fail "test candidate: good_student.py not found"
fi

echo ""
echo "5. QUICK FALSIFIER TEST"
echo "----------------------------------------"

# Create test candidate
cat > /tmp/test_candidate.json << 'JSON'
{
  "theory_id": "verify_test",
  "what_and_why": "Smoke test to verify the falsifier can load a candidate train_gpt.py, run Stage 1 gates (T2/T3/T4/T5/T7), and write a structured verdict JSON without crashing.",
  "train_gpt_path": "tests/candidates/good_student.py"
}
JSON

# Run falsifier
if [ "$PYTHON_OK" -ne 1 ]; then
    check_fail "Falsifier test skipped (Python >= 3.12 required)"
elif run_with_timeout 30 "$PYTHON" -m falsifier.main \
    --candidate-json /tmp/test_candidate.json \
    --output-json /tmp/test_result.json; then
    
    # Check result
    if [ -f /tmp/test_result.json ]; then
        VERDICT=$("$PYTHON" -c "import json; print(json.load(open('/tmp/test_result.json'))['verdict'])" 2>/dev/null || echo "unknown")
        KILLED_BY=$("$PYTHON" -c "import json; print(json.load(open('/tmp/test_result.json'))['killed_by'])" 2>/dev/null || echo "unknown")
        
        if [ "$VERDICT" = "REFUTED" ]; then
            check_pass "Falsifier runs and produces REFUTED (killed by $KILLED_BY)"
        elif [ "$VERDICT" = "STAGE_1_PASSED" ]; then
            check_pass "Falsifier runs and produces STAGE_1_PASSED"
        else
            check_warn "Falsifier produced unexpected verdict: $VERDICT"
        fi
    else
        check_fail "Falsifier did not produce output"
    fi
else
    check_fail "Falsifier execution failed or timed out"
fi

echo ""
echo "================================================================================"
echo "SUMMARY"
echo "================================================================================"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓✓ FULL LOOP INFRASTRUCTURE READY${NC}"
    echo ""
    echo "Infrastructure: $PASSED checks passed"
    echo ""
    echo "To execute the full loop:"
    echo "  1. Set API keys:"
    echo "     export GEMINI_API_KEY='your-key'"
    echo "     export OPENAI_API_KEY='your-key'"
    echo ""
    echo "  2. Generate and falsify an idea:"
    echo "     .venv/bin/python -m ideator idea --parent-train-gpt <path>"
    echo "     .venv/bin/python -m falsifier.main --idea-id <id>"
    echo ""
    echo "For Cursor agents: See docs/cursor_api_configuration.md"
    exit 0
else
    echo -e "${RED}✗ INFRASTRUCTURE ISSUES DETECTED${NC}"
    echo ""
    echo "Passed: $PASSED"
    echo "Failed: $FAILED"
    echo ""
    echo "Please fix the issues above before running the full loop."
    exit 1
fi
