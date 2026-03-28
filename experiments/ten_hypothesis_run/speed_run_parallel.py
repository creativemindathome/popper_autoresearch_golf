#!/usr/bin/env python3
"""SPEED RUN: Maximum throughput parallel hypothesis generation.

Runs 100+ hypotheses in parallel using thread pools.
Optimizations:
- Disables reviewer gate (saves 1-3 min per hypothesis)
- Disables stage 2 (saves 1-3 min per hypothesis)
- Runs 10-20 concurrent workers
- Only runs Stage 1 falsifier (T2-T7 gates)
- Minimal logging overhead

Target: 100 hypotheses in 60 minutes = 36s per hypothesis
"""

import concurrent.futures
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import threading

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ideator.anthropic_client import AnthropicClient, get_anthropic_api_key
from ideator.knowledge import load_knowledge_context, choose_knowledge_dir
from falsifier.stage1.orchestrator import run_stage_1
from falsifier.types import FalsifierInput
from falsifier.graph.experiment_sync import create_node_from_experiment_idea, sync_experiment_results


class ParallelSpeedRun:
    """Parallel speed-optimized experiment runner."""
    
    def __init__(self, num_hypotheses: int = 100, max_workers: int = 10):
        self.num_hypotheses = num_hypotheses
        self.max_workers = max_workers
        self.output_dir = Path(__file__).parent / f"speed_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir.mkdir(exist_ok=True)
        
        # Stats
        self.stats_lock = threading.Lock()
        self.completed = 0
        self.passed = 0
        self.killed = 0
        self.failed = 0
        self.stage2_triggered = 0  # Count how many passed Stage 1
        self.stage2_killed = 0     # Count how many Stage 2 killed
        self.start_time = None
        
        # Anthropic client (thread-safe)
        self.client = AnthropicClient(api_key=get_anthropic_api_key())
        
        # Knowledge context (shared)
        try:
            kdir = choose_knowledge_dir(explicit=None, cwd=project_root)
            self.knowledge_context = load_knowledge_context(kdir) if kdir else ""
        except:
            self.knowledge_context = ""
        
        print(f"🚀 SPEED RUN CONFIG")
        print(f"   Hypotheses: {num_hypotheses}")
        print(f"   Workers: {max_workers}")
        print(f"   Output: {self.output_dir}")
        print(f"   Target: {num_hypotheses} in 60 min = {3600/num_hypotheses:.1f}s each")
        print()
    
    def log(self, msg: str, level: str = "INFO"):
        """Minimal logging."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        with self.stats_lock:
            progress = f"[{self.completed}/{self.num_hypotheses}]"
        print(f"[{timestamp}] {progress} {msg}")
    
    def generate_idea_fast(self, hypothesis_num: int, max_retries: int = 2) -> Optional[Dict]:
        """Generate idea with retry logic."""
        system_prompt = f"""You are an expert ML researcher. Generate a novel transformer architecture modification.

CRITICAL CONSTRAINTS:
1. Parameter budget: MAX 10M parameters
   - Safe config: n_embd=384, n_layer=6, n_head=6 (~6.5M params)
   - Calculate: vocab_size * n_embd + n_layer * (4 * n_embd * n_embd)
2. Must compile and run without errors
3. Must print parameter count at startup
4. Must achieve loss < 15 on first forward pass
5. Must show decreasing loss over 100 training steps

NOVELTY IDEAS (pick one):
- Attention: sliding window, dilated, cross-layer sharing
- Architecture: mixture of experts (small), reversible layers
- Optimization: adaptive learning rate per layer, gradient compression
- Efficiency: sparse activations, parameter sharing patterns

OUTPUT FORMAT - EXACTLY TWO SECTIONS:

SECTION 1 - JSON METADATA:
```json
{{
  "idea_id": "descriptive-name-with-hyphens",
  "title": "Short descriptive title",
  "novelty_summary": "2-3 sentences describing the novel architectural change",
  "implementation_steps": [
    {{
      "step_id": "1",
      "file": "train_gpt.py", 
      "locate": "Find the X class",
      "change": "Modify Y to implement Z",
      "done_when": "Specific verification criteria"
    }}
  ],
  "falsifier_smoke_tests": [
    "Test 1 description",
    "Test 2 description"
  ]
}}
```

SECTION 2 - PYTHON CODE:
```python
###TRAIN_GPT_CODE_START###
# Complete train_gpt.py implementation
# Include: imports, Model class with modifications, training loop
# Config: n_embd=384, n_layer=6, n_head=6
# MUST print parameter count
###TRAIN_GPT_CODE_END###
```

IMPORTANT:
- JSON and code MUST be in separate markdown blocks
- Code MUST be complete and runnable
- Include ###TRAIN_GPT_CODE_START### and ###TRAIN_GPT_CODE_END### markers exactly"""

        for attempt in range(max_retries + 1):
            user_prompt = f"Generate hypothesis #{hypothesis_num} (attempt {attempt + 1}/{max_retries + 1}). Create a working implementation."
            
            try:
                response = self.client.generate_idea(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=min(0.8 + attempt * 0.1, 1.0),
                    max_tokens=4096,
                    model="claude-sonnet-4-20250514",
                )
                
                # Parse response
                idea = self._parse_response(response, hypothesis_num)
                if idea and idea.get("train_gpt_code") and len(idea["train_gpt_code"]) > 500:
                    self.log(f"H{hypothesis_num}: Generated {len(idea['train_gpt_code'])} chars of code", "DEBUG")
                    return idea
                
                if attempt < max_retries:
                    self.log(f"H{hypothesis_num}: Attempt {attempt + 1} failed (no code), retrying...")
                
            except Exception as e:
                self.log(f"H{hypothesis_num}: Generation error on attempt {attempt + 1}: {e}", "WARN")
        
        return None
    
    def _parse_response(self, response: str, hypothesis_num: int) -> Optional[Dict]:
        """Fast parsing of split-section response."""
        import re
        
        # Extract JSON
        json_match = re.search(r'```json\s*([\s\S]*?)```', response)
        if not json_match:
            json_match = re.search(r'\{[\s\S]*?"idea_id"[\s\S]*?\}', response)
        
        if not json_match:
            return None
        
        try:
            idea = json.loads(json_match.group(1) if json_match.group(1) else json_match.group(0))
        except:
            return None
        
        # Extract code
        code_match = re.search(r'###TRAIN_GPT_CODE_START###\s*([\s\S]*?)###TRAIN_GPT_CODE_END###', response)
        if not code_match:
            code_match = re.search(r'```python\s*([\s\S]*?)```', response)
        
        if code_match:
            idea["train_gpt_code"] = code_match.group(1).strip()
        
        # Ensure ID
        idea["idea_id"] = idea.get("idea_id", f"h{hypothesis_num:03d}")
        return idea
    
    def run_stage1_fast(self, idea: Dict) -> Dict:
        """Run stage 1 falsifier (T2-T7 gates)."""
        from falsifier.adapters.ideator_adapter import adapt_ideator_to_falsifier
        
        try:
            inp = adapt_ideator_to_falsifier(idea)
            output = run_stage_1(inp, verbosity=0)  # Minimal verbosity
            
            return {
                "verdict": output.verdict,
                "killed_at": output.killed_at,
                "measurements": {k: v for k, v in output.measurements.items()} if output.measurements else {},
                "s2_output": output,  # Keep full output for Stage 2
            }
        except Exception as e:
            return {"verdict": "ERROR", "error": str(e)}
    
    def run_stage2_fast(self, idea: Dict, stage1_output) -> Dict:
        """Run Stage 2 adversarial falsifier for ideas that passed Stage 1."""
        from falsifier.stage2.orchestrator import run_stage_2
        from falsifier.adapters.ideator_adapter import adapt_ideator_to_falsifier
        
        try:
            inp = adapt_ideator_to_falsifier(idea)
            s2_result = run_stage_2(
                inp, 
                stage1_output, 
                model=self.client.model or "claude-sonnet-4-20250514",
                max_hypotheses=3,  # Reduced for speed
                verbosity=0
            )
            
            return {
                "s2_verdict": s2_result.verdict if s2_result else "NO_STAGE2",
                "s2_killed": s2_result.verdict == "KILLED" if s2_result else False,
                "s2_hypotheses_tested": len(s2_result.test_results) if s2_result and s2_result.test_results else 0,
            }
        except Exception as e:
            return {"s2_verdict": "ERROR", "error": str(e)}
    
    def process_single_hypothesis(self, hypothesis_num: int) -> Dict:
        """Process one hypothesis - generation + Stage 1 + Stage 2 (if passed)."""
        start = time.time()
        h_id = f"H{hypothesis_num:03d}"
        
        # Generate
        idea = self.generate_idea_fast(hypothesis_num)
        if not idea:
            with self.stats_lock:
                self.completed += 1
                self.failed += 1
            return {"id": h_id, "status": "FAILED", "time": time.time() - start}
        
        # Stage 1
        stage1_result = self.run_stage1_fast(idea)
        stage1_verdict = stage1_result.get("verdict", "UNKNOWN")
        
        # Stage 2 (only if Stage 1 passed)
        stage2_result = None
        if stage1_verdict == "PASSED":
            stage2_result = self.run_stage2_fast(idea, stage1_result.get("s2_output"))
        
        # Sync to KG
        try:
            kg_dir = project_root / "knowledge_graph"
            graph_path = kg_dir / "graph.json"
            
            # Create node
            create_node_from_experiment_idea(idea, graph_path)
            
            # Build final verdict
            final_verdict = stage1_verdict
            if stage1_verdict == "PASSED" and stage2_result:
                if stage2_result.get("s2_killed"):
                    final_verdict = "KILLED_STAGE2"
                else:
                    final_verdict = "PASSED_STAGE2"
            
            # Sync Stage 1 results
            from falsifier.types import FalsifierOutput
            output = FalsifierOutput(
                theory_id=idea["idea_id"],
                verdict=final_verdict,
                killed_at=stage1_result.get("killed_at"),
                measurements=stage1_result.get("measurements", {}),
            )
            from falsifier.graph.lifecycle import update_node_with_falsification_results
            update_node_with_falsification_results(idea["idea_id"], output, graph_path)
        except Exception as e:
            pass  # Don't fail on KG sync
        
        duration = time.time() - start
        
        with self.stats_lock:
            self.completed += 1
            if final_verdict in ("PASSED", "PASSED_STAGE2"):
                self.passed += 1
            elif final_verdict in ("KILLED", "KILLED_STAGE2"):
                self.killed += 1
            else:
                self.failed += 1
        
        # Log with Stage 2 info
        if stage2_result:
            s2_status = "S2-KILLED" if stage2_result.get("s2_killed") else "S2-PASSED"
            self.log(f"{h_id}: S1-PASSED → {s2_status} in {duration:.1f}s | Progress: {self.passed}P/{self.killed}K/{self.failed}F")
        else:
            status_emoji = "✅" if stage1_verdict == "PASSED" else "❌" if stage1_verdict == "KILLED" else "⚠️"
            self.log(f"{h_id}: {status_emoji} {stage1_verdict} in {duration:.1f}s | Progress: {self.passed}P/{self.killed}K/{self.failed}F")
        
        return {
            "id": h_id,
            "status": final_verdict,
            "stage1": stage1_verdict,
            "stage2": stage2_result.get("s2_verdict") if stage2_result else None,
            "time": duration,
            "idea_id": idea.get("idea_id"),
        }
    
    def run(self) -> Dict:
        """Execute parallel speed run."""
        self.start_time = time.time()
        print(f"⏱️  Starting at {datetime.now().strftime('%H:%M:%S')}")
        print(f"🎯 ETA: {(datetime.fromtimestamp(self.start_time + 3600)).strftime('%H:%M:%S')}")
        print()
        
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(self.process_single_hypothesis, i): i 
                      for i in range(1, self.num_hypotheses + 1)}
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(futures):
                hypothesis_num = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.log(f"H{hypothesis_num:03d}: Exception: {e}", "ERROR")
                    with self.stats_lock:
                        self.completed += 1
                        self.failed += 1
        
        # Summary
        total_time = time.time() - self.start_time
        print()
        print("=" * 60)
        print("🏁 SPEED RUN COMPLETE")
        print("=" * 60)
        print(f"⏱️  Total time: {total_time/60:.1f} minutes")
        print(f"📊 Results: {self.passed} PASSED / {self.killed} KILLED / {self.failed} FAILED")
        print(f"⚡ Throughput: {self.num_hypotheses/total_time:.2f} hypotheses/second")
        print(f"🎯 Target was 100 in 60 min = 1.67/min")
        print(f"✅ Actual: {self.num_hypotheses/(total_time/60):.2f}/min")
        print()
        
        # Save results
        summary = {
            "total_time_seconds": total_time,
            "hypotheses": self.num_hypotheses,
            "passed": self.passed,
            "killed": self.killed,
            "failed": self.failed,
            "throughput_per_minute": self.num_hypotheses / (total_time / 60),
            "results": results,
        }
        
        summary_path = self.output_dir / "speed_run_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"💾 Summary saved to: {summary_path}")
        
        return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Parallel speed run for maximum throughput")
    parser.add_argument("--num-hypotheses", type=int, default=100, help="Number of hypotheses (default: 100)")
    parser.add_argument("--max-workers", type=int, default=10, help="Concurrent workers (default: 10)")
    args = parser.parse_args()
    
    runner = ParallelSpeedRun(
        num_hypotheses=args.num_hypotheses,
        max_workers=args.max_workers
    )
    runner.run()
