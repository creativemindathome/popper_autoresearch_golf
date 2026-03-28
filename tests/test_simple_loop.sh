#!/bin/bash
# Simple end-to-end test of the continuous loop

set -e

echo "================================================================================"
echo "SIMPLE CONTINUOUS LOOP TEST"
echo "================================================================================"
echo ""

IDEA_ID="loop_test_$(date +%s)"
KNOWLEDGE_DIR="knowledge_graph"

echo "Step 1: Create ideator output"
echo "----------------------------------------"

# Create sample train_gpt.py
cat > /tmp/${IDEA_ID}_train_gpt.py << 'EOF'
import torch
import torch.nn as nn
import torch.nn.functional as F

class Hyperparameters:
    vocab_size = 50257
    d_model = 128
    n_heads = 4
    n_layers = 2
    d_mlp = 512

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_mlp):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(1024, d_model)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, d_mlp, batch_first=True)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
    def forward(self, input_ids, targets=None):
        B, T = input_ids.size()
        x = self.tok_emb(input_ids) + self.pos_emb(torch.arange(T, device=input_ids.device))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if targets is not None:
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits
EOF

# Create ideator JSON
cat > ${KNOWLEDGE_DIR}/outbox/ideator/${IDEA_ID}.json << EOF
{
  "idea_id": "${IDEA_ID}",
  "timestamp": "$(date -Iseconds)",
  "what_and_why": "Test continuous loop with standard transformer architecture using proper initialization",
  "config_changes": {
    "d_model": 128,
    "n_heads": 4
  },
  "review": {
    "approved": true,
    "score": 7
  }
}
EOF

cp /tmp/${IDEA_ID}_train_gpt.py ${KNOWLEDGE_DIR}/outbox/ideator/${IDEA_ID}_train_gpt.py

echo "✓ Created: ${KNOWLEDGE_DIR}/outbox/ideator/${IDEA_ID}.json"
echo "✓ Created: ${KNOWLEDGE_DIR}/outbox/ideator/${IDEA_ID}_train_gpt.py"
echo ""

echo "Step 2: Simulate reviewer approval (handoff to inbox)"
echo "----------------------------------------"

ln -sf "${PWD}/${KNOWLEDGE_DIR}/outbox/ideator/${IDEA_ID}.json" \
       "${KNOWLEDGE_DIR}/inbox/approved/${IDEA_ID}.json"

echo "✓ Created symlink: inbox/approved/${IDEA_ID}.json"
echo ""

echo "Step 3: Run falsifier"
echo "----------------------------------------"

# Create falsifier input JSON
cat > /tmp/${IDEA_ID}_falsifier.json << EOF
{
  "theory_id": "${IDEA_ID}",
  "what_and_why": "Test continuous loop with standard transformer architecture using proper initialization and residual connections for language modeling",
  "train_gpt_path": "${KNOWLEDGE_DIR}/outbox/ideator/${IDEA_ID}_train_gpt.py"
}
EOF

python3 -m falsifier.main \
    --candidate-json /tmp/${IDEA_ID}_falsifier.json \
    --output-json ${KNOWLEDGE_DIR}/outbox/falsifier/${IDEA_ID}_result.json \
    --knowledge-dir ./${KNOWLEDGE_DIR} \
    2>&1 | grep -E "(Running Stage|Stage [12]|Killed by|verdict|REFUTED|PASSED|✓|✗)" || true

echo ""

echo "Step 4: Check knowledge graph"
echo "----------------------------------------"

# Check if graph.json exists and was updated
if [ -f "${KNOWLEDGE_DIR}/graph.json" ]; then
    echo "✓ Knowledge graph exists: ${KNOWLEDGE_DIR}/graph.json"
    
    # Check if node was added
    if python3 -c "
import json
import sys
graph = json.load(open('${KNOWLEDGE_DIR}/graph.json'))
nodes = graph.get('nodes', {})
if '${IDEA_ID}' in nodes:
    node = nodes['${IDEA_ID}']
    print(f'✓ Node ${IDEA_ID} in graph')
    print(f'  Status: {node.get(\"status\", \"unknown\")}')
    sys.exit(0)
else:
    print('○ Node not in graph (may need manual update)')
    sys.exit(0)
" 2>/dev/null; then
        : # Success
    fi
else
    echo "○ Knowledge graph not initialized yet"
fi

echo ""

echo "Step 5: Verify information flow"
echo "----------------------------------------"

# Check file chain
echo "Checking file chain:"
[ -f "${KNOWLEDGE_DIR}/outbox/ideator/${IDEA_ID}.json" ] && echo "  ✓ ideator output" || echo "  ✗ ideator output"
[ -L "${KNOWLEDGE_DIR}/inbox/approved/${IDEA_ID}.json" ] && echo "  ✓ inbox handoff" || echo "  ✗ inbox handoff"
[ -f "${KNOWLEDGE_DIR}/outbox/falsifier/${IDEA_ID}_result.json" ] && echo "  ✓ falsifier output" || echo "  ✗ falsifier output"

echo ""

# Show falsifier result
if [ -f "${KNOWLEDGE_DIR}/outbox/falsifier/${IDEA_ID}_result.json" ]; then
    echo "Falsifier Result:"
    python3 -c "
import json
data = json.load(open('${KNOWLEDGE_DIR}/outbox/falsifier/${IDEA_ID}_result.json'))
print(f\"  Verdict: {data.get('verdict', 'UNKNOWN')}\")
print(f\"  Killed by: {data.get('killed_by', 'N/A')}\")
print(f\"  Tags: {len(data.get('tags', []))}\")
print(f\"  Total wall time: {data.get('total_wall_seconds', 0):.3f}s\")
" 2>/dev/null || echo "  (could not parse result)"
fi

echo ""
echo "================================================================================"
echo "CONTINUOUS LOOP TEST COMPLETE"
echo "================================================================================"
echo ""
echo "Summary:"
echo "  Idea ID: ${IDEA_ID}"
echo "  Flow: ideator → inbox → falsifier → outbox"
echo ""
echo "Files created:"
echo "  ${KNOWLEDGE_DIR}/outbox/ideator/${IDEA_ID}.json"
echo "  ${KNOWLEDGE_DIR}/outbox/ideator/${IDEA_ID}_train_gpt.py"
echo "  ${KNOWLEDGE_DIR}/inbox/approved/${IDEA_ID}.json → (symlink)"
echo "  ${KNOWLEDGE_DIR}/outbox/falsifier/${IDEA_ID}_result.json"
echo ""
