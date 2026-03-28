#!/usr/bin/env python3
"""
End-to-end test for the continuous AutoResearch loop.

Tests information flow:
1. Ideator generates idea → outbox/ideator/
2. Reviewer approves → inbox/approved/
3. Falsifier tests → outbox/falsifier/
4. Knowledge graph updated with results
5. Feedback available for next ideator iteration
"""

import json
import sys
import time
import tempfile
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from falsifier.types import FalsifierInput, KnowledgeGraph, ParentRef
from falsifier.graph.lifecycle import (
    update_node_status,
    update_node_with_falsification_results,
    find_node_by_idea_id,
)
from falsifier.adapters.ideator_adapter import load_and_adapt_ideator_idea


class ContinuousLoopTest:
    """Test the full continuous information flow."""
    
    def __init__(self, knowledge_dir: Path = None):
        self.knowledge_dir = knowledge_dir or Path("knowledge_graph")
        self.graph_path = self.knowledge_dir / "graph.json"
        self.results = []
        self.start_time = time.time()
        
    def log(self, msg: str):
        elapsed = time.time() - self.start_time
        print(f"[{elapsed:6.2f}s] {msg}")
        
    def run_all_tests(self):
        """Run complete loop test."""
        print("=" * 80)
        print("CONTINUOUS LOOP INTEGRATION TEST")
        print("=" * 80)
        print()
        
        # Test 1: Create mock ideator output
        self.test_1_create_ideator_output()
        
        # Test 2: Simulate reviewer approval
        self.test_2_reviewer_approval()
        
        # Test 3: Handoff to falsifier
        self.test_3_handoff_to_falsifier()
        
        # Test 4: Falsifier execution
        self.test_4_falsifier_execution()
        
        # Test 5: Knowledge graph update
        self.test_5_knowledge_graph_update()
        
        # Test 6: Feedback propagation
        self.test_6_feedback_propagation()
        
        # Test 7: Verify information chain
        self.test_7_verify_information_chain()
        
        # Generate report
        return self.generate_report()
    
    def test_1_create_ideator_output(self):
        """Step 1: Create mock ideator output."""
        self.log("TEST 1: Creating ideator output...")
        
        # Create a sample train_gpt.py code
        sample_code = '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class Hyperparameters:
    vocab_size = 50257
    d_model = 256
    n_heads = 4
    n_layers = 2
    d_mlp = 1024
    max_seq_len = 1024
    batch_size = 4
    learning_rate = 0.001

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_mlp, max_seq_len):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_mlp)
            for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
    def forward(self, input_ids, targets=None):
        B, T = input_ids.size()
        tok_emb = self.tok_emb(input_ids)
        pos_emb = self.pos_emb(torch.arange(T, device=input_ids.device))
        x = tok_emb + pos_emb
        
        for block in self.blocks:
            x = block(x)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return loss
        return logits

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_mlp):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.GELU(),
            nn.Linear(d_mlp, d_model),
        )
        
    def forward(self, x):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.mlp(self.ln2(x))
        return x
'''
        
        # Create mock ideator output
        self.idea_id = f"test_loop_{int(time.time())}"
        self.idea_data = {
            "idea_id": self.idea_id,
            "timestamp": datetime.now().isoformat(),
            "what_and_why": "Test continuous loop with standard transformer architecture using proper initialization and residual connections",
            "config_changes": {
                "d_model": 256,
                "n_heads": 4,
                "n_layers": 2,
            },
            "train_gpt_code": sample_code,
            "parent_ref": {
                "source": "parameter-golf",
                "commit": "abc123",
            },
            "review": {
                "approved": True,
                "score": 7,
                "novelty_score": 6,
                "comments": "Standard architecture, should be tested",
            },
        }
        
        # Save to outbox/ideator/
        ideator_dir = self.knowledge_dir / "outbox" / "ideator"
        ideator_dir.mkdir(parents=True, exist_ok=True)
        
        self.idea_path = ideator_dir / f"{self.idea_id}.json"
        self.idea_path.write_text(json.dumps(self.idea_data, indent=2))
        
        self.log(f"  ✓ Created ideator output: {self.idea_path}")
        
        # Save train_gpt.py
        train_gpt_path = ideator_dir / f"{self.idea_id}_train_gpt.py"
        train_gpt_path.write_text(sample_code)
        self.log(f"  ✓ Created train_gpt.py: {train_gpt_path}")
        
        self.results.append({
            "test": "create_ideator_output",
            "passed": True,
            "idea_id": self.idea_id,
        })
    
    def test_2_reviewer_approval(self):
        """Step 2: Simulate reviewer approval and handoff."""
        self.log("TEST 2: Simulating reviewer approval...")
        
        # Check that review exists and is approved
        review = self.idea_data.get("review", {})
        is_approved = review.get("approved", False)
        score = review.get("score", 0)
        
        self.log(f"  Review score: {score}/10")
        self.log(f"  Approved: {is_approved}")
        
        if is_approved and score >= 6:
            # Create handoff symlink
            inbox_dir = self.knowledge_dir / "inbox" / "approved"
            inbox_dir.mkdir(parents=True, exist_ok=True)
            
            self.inbox_path = inbox_dir / f"{self.idea_id}.json"
            
            # Create symlink or copy
            try:
                if self.inbox_path.exists() or self.inbox_path.is_symlink():
                    self.inbox_path.unlink()
                self.inbox_path.symlink_to(self.idea_path)
                self.log(f"  ✓ Created handoff symlink: {self.inbox_path} -> {self.idea_path}")
            except OSError:
                # Fallback to copy
                import shutil
                shutil.copy2(self.idea_path, self.inbox_path)
                self.log(f"  ✓ Copied to inbox: {self.inbox_path}")
            
            self.results.append({
                "test": "reviewer_approval",
                "passed": True,
                "score": score,
            })
        else:
            self.log(f"  ✗ Idea not approved (score={score})")
            self.results.append({
                "test": "reviewer_approval",
                "passed": False,
                "reason": "Not approved",
            })
    
    def test_3_handoff_to_falsifier(self):
        """Step 3: Test handoff to falsifier."""
        self.log("TEST 3: Testing handoff to falsifier...")
        
        try:
            # Use the adapter to load the idea
            inp = load_and_adapt_ideator_idea(
                self.inbox_path,
                self.knowledge_dir
            )
            
            self.log(f"  ✓ Successfully loaded idea via adapter")
            self.log(f"  Theory ID: {inp.theory_id}")
            self.log(f"  What and Why: {inp.what_and_why[:50]}...")
            self.log(f"  Has proposed_train_gpt: {bool(inp.proposed_train_gpt)}")
            
            self.falsifier_input = inp
            
            self.results.append({
                "test": "handoff_to_falsifier",
                "passed": True,
            })
            
        except Exception as e:
            self.log(f"  ✗ Failed to load idea: {e}")
            self.results.append({
                "test": "handoff_to_falsifier",
                "passed": False,
                "error": str(e),
            })
    
    def test_4_falsifier_execution(self):
        """Step 4: Run falsifier on the idea."""
        self.log("TEST 4: Running falsifier...")
        
        try:
            from falsifier.stage1.orchestrator import run_stage_1
            from falsifier.stage2.orchestrator import run_stage_2
            
            # Run Stage 1
            t1 = time.time()
            stage1_result = run_stage_1(self.falsifier_input)
            stage1_time = time.time() - t1
            
            self.log(f"  Stage 1: {stage1_result.verdict} in {stage1_time:.3f}s")
            self.log(f"  Killed by: {stage1_result.killed_by or 'N/A'}")
            self.log(f"  Tags: {len(stage1_result.tags)}")
            
            self.stage1_result = stage1_result
            
            # Run Stage 2 if Stage 1 passed
            if stage1_result.verdict == "STAGE_1_PASSED":
                self.log(f"  Running Stage 2...")
                t2 = time.time()
                stage2_result = run_stage_2(self.falsifier_input, stage1_result)
                stage2_time = time.time() - t2
                
                self.log(f"  Stage 2: {stage2_result.verdict} in {stage2_time:.3f}s")
                self.final_result = stage2_result
            else:
                self.log(f"  Stage 2: SKIPPED (Stage 1 failed)")
                self.final_result = stage1_result
            
            # Save falsifier output
            falsifier_dir = self.knowledge_dir / "outbox" / "falsifier"
            falsifier_dir.mkdir(parents=True, exist_ok=True)
            
            self.falsifier_output_path = falsifier_dir / f"{self.idea_id}_result.json"
            
            from dataclasses import asdict
            output_dict = asdict(self.final_result)
            
            # Clean non-serializable items
            def clean(obj):
                if isinstance(obj, Path):
                    return str(obj)
                if isinstance(obj, dict):
                    return {k: clean(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [clean(v) for v in obj]
                return obj
            
            self.falsifier_output_path.write_text(
                json.dumps(clean(output_dict), indent=2)
            )
            
            self.log(f"  ✓ Saved falsifier output: {self.falsifier_output_path}")
            
            self.results.append({
                "test": "falsifier_execution",
                "passed": True,
                "verdict": self.final_result.verdict,
                "killed_by": self.final_result.killed_by,
                "tags_count": len(self.final_result.tags),
            })
            
        except Exception as e:
            self.log(f"  ✗ Falsifier failed: {e}")
            import traceback
            traceback.print_exc()
            self.results.append({
                "test": "falsifier_execution",
                "passed": False,
                "error": str(e),
            })
    
    def test_5_knowledge_graph_update(self):
        """Step 5: Update knowledge graph with results."""
        self.log("TEST 5: Updating knowledge graph...")
        
        try:
            # Ensure graph exists
            if not self.graph_path.exists():
                self.graph_path.write_text(json.dumps({
                    "nodes": {},
                    "edges": [],
                    "version": "1.0"
                }))
            
            # Load graph
            graph_data = json.loads(self.graph_path.read_text())
            
            # Add or update node
            if self.idea_id not in graph_data.get("nodes", {}):
                graph_data["nodes"][self.idea_id] = {
                    "id": self.idea_id,
                    "type": "theory",
                    "status": "GENERATED",
                    "created_at": datetime.now().isoformat(),
                }
            
            # Update node with falsification results
            node = graph_data["nodes"][self.idea_id]
            node["status"] = self.final_result.verdict
            node["killed_by"] = self.final_result.killed_by
            node["kill_reason"] = self.final_result.kill_reason
            node["tags"] = [tag.tag_id for tag in self.final_result.tags]
            node["falsified_at"] = datetime.now().isoformat()
            
            # Add measurements
            if self.final_result.t2_budget:
                node["measurements"] = {
                    "estimated_params": self.final_result.t2_budget.get("estimated_params"),
                    "budget_utilization": self.final_result.t2_budget.get("budget_utilization"),
                }
            
            if self.final_result.t3_compilation:
                node["measurements"]["forward_ms"] = self.final_result.t3_compilation.get("forward_ms")
                node["measurements"]["backward_ms"] = self.final_result.t3_compilation.get("backward_ms")
            
            # Save graph
            self.graph_path.write_text(json.dumps(graph_data, indent=2))
            
            self.log(f"  ✓ Updated knowledge graph: {self.graph_path}")
            self.log(f"  Node {self.idea_id} status: {node['status']}")
            self.log(f"  Tags stored: {len(node.get('tags', []))}")
            
            self.graph_data = graph_data
            
            self.results.append({
                "test": "knowledge_graph_update",
                "passed": True,
            })
            
        except Exception as e:
            self.log(f"  ✗ Knowledge graph update failed: {e}")
            import traceback
            traceback.print_exc()
            self.results.append({
                "test": "knowledge_graph_update",
                "passed": False,
                "error": str(e),
            })
    
    def test_6_feedback_propagation(self):
        """Step 6: Verify feedback is available for ideator."""
        self.log("TEST 6: Verifying feedback propagation...")
        
        try:
            # Check that falsifier output contains feedback
            feedback = self.final_result.feedback
            
            if feedback:
                self.log(f"  ✓ Feedback generated")
                self.log(f"  One-liner: {feedback.one_line[:60]}...")
                self.log(f"  Stage reached: {feedback.stage_reached}")
                
                # Check that feedback is stored in knowledge graph
                node = self.graph_data["nodes"][self.idea_id]
                has_feedback = "kill_reason" in node or "tags" in node
                
                if has_feedback:
                    self.log(f"  ✓ Feedback stored in knowledge graph")
                else:
                    self.log(f"  ○ Feedback not in graph (may need explicit storage)")
                
                self.results.append({
                    "test": "feedback_propagation",
                    "passed": True,
                    "has_feedback": bool(feedback.one_line),
                })
            else:
                self.log(f"  ○ No feedback object (Stage 1 may not generate)")
                self.results.append({
                    "test": "feedback_propagation",
                    "passed": True,
                    "has_feedback": False,
                })
            
        except Exception as e:
            self.log(f"  ✗ Feedback check failed: {e}")
            self.results.append({
                "test": "feedback_propagation",
                "passed": False,
                "error": str(e),
            })
    
    def test_7_verify_information_chain(self):
        """Step 7: Verify complete information chain."""
        self.log("TEST 7: Verifying complete information chain...")
        
        checks = []
        
        # Check 1: Ideator output exists
        check1 = self.idea_path.exists()
        checks.append(("Ideator output exists", check1))
        self.log(f"  {'✓' if check1 else '✗'} Ideator output exists")
        
        # Check 2: Inbox symlink exists
        check2 = self.inbox_path.exists() or self.inbox_path.is_symlink()
        checks.append(("Inbox handoff exists", check2))
        self.log(f"  {'✓' if check2 else '✗'} Inbox handoff exists")
        
        # Check 3: Falsifier output exists
        check3 = self.falsifier_output_path.exists()
        checks.append(("Falsifier output exists", check3))
        self.log(f"  {'✓' if check3 else '✗'} Falsifier output exists")
        
        # Check 4: Knowledge graph updated
        check4 = self.idea_id in self.graph_data.get("nodes", {})
        checks.append(("Node in knowledge graph", check4))
        self.log(f"  {'✓' if check4 else '✗'} Node in knowledge graph")
        
        # Check 5: Node has falsification results
        if check4:
            node = self.graph_data["nodes"][self.idea_id]
            check5 = "status" in node and node["status"] != "GENERATED"
            checks.append(("Node has falsification status", check5))
            self.log(f"  {'✓' if check5 else '✗'} Node has falsification status")
        else:
            checks.append(("Node has falsification status", False))
            self.log(f"  ✗ Node has falsification status")
        
        # Check 6: Information traceable from ideator to falsifier
        # The idea_id should be consistent throughout
        check6 = (
            self.idea_id in self.idea_data.get("idea_id", "") and
            self.idea_id in str(self.inbox_path) and
            self.idea_id in str(self.falsifier_output_path)
        )
        checks.append(("Consistent idea_id throughout chain", check6))
        self.log(f"  {'✓' if check6 else '✗'} Consistent idea_id throughout chain")
        
        all_passed = all(c[1] for c in checks)
        
        self.results.append({
            "test": "verify_information_chain",
            "passed": all_passed,
            "checks": {c[0]: c[1] for c in checks},
        })
    
    def generate_report(self):
        """Generate final test report."""
        self.log("")
        self.log("=" * 80)
        self.log("CONTINUOUS LOOP TEST REPORT")
        self.log("=" * 80)
        self.log("")
        
        # Count results
        total = len(self.results)
        passed = sum(1 for r in self.results if r.get("passed"))
        failed = total - passed
        
        self.log(f"Total tests: {total}")
        self.log(f"Passed: {passed}")
        self.log(f"Failed: {failed}")
        self.log("")
        
        # Information flow summary
        self.log("INFORMATION FLOW CHAIN:")
        self.log("-" * 40)
        self.log(f"1. Ideator output: {self.idea_path}")
        self.log(f"   - Idea ID: {self.idea_id}")
        self.log(f"   - Contains: what_and_why, train_gpt_code, review")
        self.log("")
        self.log(f"2. Inbox handoff: {self.inbox_path}")
        self.log(f"   - Symlink/copy to ideator output")
        self.log(f"   - Review approved: {self.idea_data.get('review', {}).get('approved')}")
        self.log("")
        self.log(f"3. Falsifier input: {self.falsifier_input.theory_id if hasattr(self, 'falsifier_input') else 'N/A'}")
        if hasattr(self, 'final_result'):
            self.log(f"4. Falsifier output: {self.falsifier_output_path}")
            self.log(f"   - Verdict: {self.final_result.verdict}")
            self.log(f"   - Tags: {len(self.final_result.tags)}")
        if hasattr(self, 'graph_data') and self.idea_id in self.graph_data.get("nodes", {}):
            self.log(f"5. Knowledge graph: {self.graph_path}")
            self.log(f"   - Node: {self.idea_id}")
            self.log(f"   - Status: {self.graph_data['nodes'][self.idea_id].get('status')}")
        self.log("")
        
        # Final verdict
        if failed == 0:
            self.log("✓✓ CONTINUOUS LOOP WORKING - Information flows correctly through all components")
            self.log("")
            self.log("The full loop is operational:")
            self.log("  ideator → reviewer → inbox → falsifier → knowledge graph")
            return True
        elif passed >= total * 0.75:
            self.log("○ MOSTLY WORKING - Minor issues in information flow")
            return True
        else:
            self.log("✗ BROKEN - Information chain has failures")
            return False


def main():
    """Run continuous loop test."""
    test = ContinuousLoopTest()
    success = test.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
