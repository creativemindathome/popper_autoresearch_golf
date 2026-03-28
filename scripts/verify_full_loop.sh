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
    ((PASSED++))
}

check_fail() {
    echo -e "${RED}✗${NC} $1"
    ((FAILED++))
}

check_warn() {
    echo -e "${YELLOW}○${NC} $1"
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
        check_fail "$dir/ (missing)"
        mkdir -p "$dir"
        echo "    Created: $dir/"
    fi
done

echo ""
echo "2. PYTHON ENVIRONMENT"
echo "----------------------------------------"

# Check Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
    check_pass "Python $PYTHON_VERSION"
else
    check_fail "Python 3 not found"
fi

# Check imports
if python3 -c "import ideator" 2>/dev/null; then
    check_pass "ideator module imports"
else
    check_fail "ideator module import failed"
fi

if python3 -c "import falsifier" 2>/dev/null; then
    check_pass "falsifier module imports"
else
    check_fail "falsifier module import failed"
fi

if python3 -c "from falsifier.graph.lifecycle import update_node_status" 2>/dev/null; then
    check_pass "graph.lifecycle imports"
else
    check_fail "graph.lifecycle import failed"
fi

if python3 -c "from falsifier.adapters.ideator_adapter import load_and_adapt_ideator_idea" 2>/dev/null; then
    check_pass "ideator_adapter imports"
else
    check_fail "ideator_adapter import failed"
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
  "what_and_why": "Quick test that falsifier executes",
  "train_gpt_path": "tests/candidates/good_student.py"
}
JSON

# Run falsifier
if timeout 30 python3 -m falsifier.main \
    --candidate-json /tmp/test_candidate.json \
    --output-json /tmp/test_result.json 2>/dev/null; then
    
    # Check result
    if [ -f /tmp/test_result.json ]; then
        VERDICT=$(python3 -c "import json; print(json.load(open('/tmp/test_result.json'))['verdict'])" 2>/dev/null || echo "unknown")
        KILLED_BY=$(python3 -c "import json; print(json.load(open('/tmp/test_result.json'))['killed_by'])" 2>/dev/null || echo "unknown")
        
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
    echo "     python3 -m ideator idea --parent-train-gpt <path>"
    echo "     python3 -m falsifier.main --idea-id <id>"
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
