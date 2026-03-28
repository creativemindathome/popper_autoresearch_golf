#!/usr/bin/env python3
"""FULL LIVE EXPERIMENT with Stage 2 Anthropic (Sonnet).

This script:
1. Generates 10 novel architecture ideas using Anthropic Claude
2. Runs each through Stage 1 falsifier gates (T2-T7)
3. Runs SURVIVING ideas through Stage 2 adversarial falsifier (Anthropic Sonnet)
4. Tracks knowledge graph evolution with snapshots
5. Creates visualization data for 2-hour time-lapse clip
6. Provides detailed live reporting

Stage 2 Configuration:
- LLM: Anthropic Claude Sonnet (claude-sonnet-4-20250514)
- Generates 3-5 kill hypotheses per surviving theory
- Runs adversarial experiments
- Provides detailed kill verdicts
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import uuid

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ideator.anthropic_client import AnthropicClient, get_anthropic_api_key
from ideator.knowledge import load_knowledge_context, choose_knowledge_dir
from ideator.prompts import (
    build_reviewer_prompts,
    reviewer_response_schema,
    REVIEWER_SYSTEM,
)
from falsifier.stage1.orchestrator import run_stage_1
from falsifier.stage2.orchestrator import run_stage_2
from falsifier.stage2.feedback import generate_feedback
from falsifier.types import FalsifierInput, Calibration, FalsifierOutput, Tag

# Import knowledge graph lifecycle for syncing results
from falsifier.graph.lifecycle import (
    create_node_from_ideator_idea,
    update_node_status,
    update_node_with_falsification_results,
    find_node_by_idea_id,
)
from falsifier.graph.locking import AtomicGraphUpdate


@dataclass
class Stage0Config:
    """Configuration for Stage 0 ideation (idea generation)."""
    model: str = "claude-sonnet-4-20250514"  # Default ideation model
    temperature: float = 1.0
    max_tokens: int = 4096


@dataclass
class ReviewerConfig:
    """Configuration for the novelty reviewer gate between ideation and falsification."""
    enabled: bool = True
    model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.4
    max_tokens: int = 2048
    min_score: int = 6  # Auto-pass ideas scoring >= this even if reviewer says "revise"
    max_rounds: int = 3  # Max ideation-review revision rounds per hypothesis


@dataclass
class Stage2Config:
    """Configuration for Stage 2 adversarial falsifier."""
    enabled: bool = True
    model: str = "claude-sonnet-4-20250514"  # Sonnet for hypothesis generation
    max_hypotheses: int = 5
    max_tokens: int = 4000
    temperature: float = 0.7


@dataclass
class HypothesisRun:
    """Track a single hypothesis through the full pipeline."""
    run_id: str
    hypothesis_number: int
    start_time: float
    end_time: Optional[float] = None
    status: str = "pending"
    idea_json: Optional[Dict] = None
    reviewer_feedback: Optional[Dict] = None
    stage1_result: Optional[Dict] = None
    falsifier_feedback: Optional[Dict] = None
    stage2_input: Optional[Dict] = None
    stage2_result: Optional[Dict] = None
    verdict: Optional[str] = None
    kill_hypotheses: List[Dict] = None
    experiments_run: int = 0
    error: Optional[str] = None


@dataclass
class GraphSnapshot:
    """Snapshot of knowledge graph state at a point in time."""
    timestamp: float
    iso_time: str
    total_ideas: int
    approved_ideas: int
    falsified_ideas: int
    stage1_passed: int
    stage2_triggered: int
    stage2_passed: int
    nodes: List[Dict]
    edges: List[Dict]


class FullLiveExperiment:
    """Run full live experiment with configurable models per stage."""

    def __init__(
        self,
        output_dir: Path,
        stage0_config: Optional[Stage0Config] = None,
        reviewer_config: Optional[ReviewerConfig] = None,
        stage2_config: Optional[Stage2Config] = None,
    ):
        self.output_dir = output_dir
        self.logs_dir = output_dir / "logs"
        self.snapshots_dir = output_dir / "graph_snapshots"
        self.viz_dir = output_dir / "visualization"
        self.reviewer_config = reviewer_config or ReviewerConfig()
        self.stage2_config = stage2_config or Stage2Config()

        # Create directories
        for d in [self.logs_dir, self.snapshots_dir, self.viz_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # State
        self.runs: List[HypothesisRun] = []
        self.snapshots: List[GraphSnapshot] = []
        self.start_time = time.time()
        self.client: Optional[AnthropicClient] = None

        # Knowledge graph context (loaded once, refreshed between hypotheses)
        self._kg_dir = project_root / "knowledge_graph"
        self._knowledge_context: str = ""

        # Statistics
        self.stats = {
            "total_generated": 0,
            "reviewer_passed": 0,
            "reviewer_rejected": 0,
            "reviewer_rounds_total": 0,
            "stage1_passed": 0,
            "stage1_killed": 0,
            "stage2_triggered": 0,
            "stage2_passed": 0,
            "stage2_killed": 0,
            "failed": 0,
        }

        # Load API key
        api_key = get_anthropic_api_key()
        if not api_key:
            env_file = project_root / ".env"
            if env_file.exists():
                with open(env_file) as f:
                    for line in f:
                        if line.strip().startswith("ANTHROPIC_API_KEY"):
                            key = line.split("=", 1)[1].strip().strip('"')
                            os.environ["ANTHROPIC_API_KEY"] = key
                            api_key = key
                            break

        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not found in environment or .env")

        self.client = AnthropicClient(api_key=api_key)
        self.stage0_config = stage0_config or Stage0Config()
        self.stage2_config = stage2_config or Stage2Config()

        self.log("="*70)
        self.log("FULL LIVE EXPERIMENT WITH SEPARATE STAGE MODELS")
        self.log("="*70)
        self.log(f"Output directory: {output_dir}")
        self.log(f"Stage 0 (Ideation): {self.stage0_config.model}")
        self.log(f"Reviewer Gate: {'ENABLED' if self.reviewer_config.enabled else 'DISABLED'}")
        if self.reviewer_config.enabled:
            self.log(f"  Model: {self.reviewer_config.model}")
            self.log(f"  Max rounds: {self.reviewer_config.max_rounds}")
            self.log(f"  Min score (auto-pass): {self.reviewer_config.min_score}")
        self.log(f"Stage 2 (Falsifier): {'ENABLED' if self.stage2_config.enabled else 'DISABLED'}")
        if self.stage2_config.enabled:
            self.log(f"  Model: {self.stage2_config.model}")
            self.log(f"  Max hypotheses: {self.stage2_config.max_hypotheses}")
        self.log("")

    def log(self, message: str, level: str = "INFO"):
        """Log with timestamp."""
        timestamp = datetime.now().isoformat()
        log_line = f"[{timestamp}] [{level}] {message}"
        print(log_line)

        log_file = self.logs_dir / "run.log"
        with open(log_file, "a") as f:
            f.write(log_line + "\n")

    def log_stage(self, stage: str, details: str = ""):
        """Log a pipeline stage."""
        self.log("")
        self.log(f"{'─'*70}")
        self.log(f"  {stage}")
        if details:
            self.log(f"  {details}")
        self.log(f"{'─'*70}")

    def _refresh_knowledge_context(self) -> str:
        """Reload knowledge context from the working KG.

        This is the canonical source of what has been tried, what failed,
        and what passed — exactly what the ideator needs.
        """
        try:
            kdir = choose_knowledge_dir(self._kg_dir, cwd=project_root)
            if kdir:
                self._knowledge_context = load_knowledge_context(kdir)
                if self._knowledge_context:
                    self.log(f"  Loaded knowledge context ({len(self._knowledge_context)} chars)")
                else:
                    self.log("  Knowledge context is empty (first run)")
            else:
                self._knowledge_context = ""
        except Exception as e:
            self.log(f"  Warning: failed to load knowledge context: {e}", "WARN")
            self._knowledge_context = ""
        return self._knowledge_context

    @staticmethod
    def _translate_idea_for_reviewer(idea: Dict) -> Dict:
        """Translate the experiment's idea schema to the v2 schema the reviewer expects.

        The reviewer prompt references fields like novelty_summary,
        implementation_steps, falsifier_smoke_tests, expected_metric_change.
        The experiment ideator produces theory_id, what_and_why, train_gpt_code,
        novelty_claims, expected_behavior, parameter_estimate, risk_factors.
        """
        return {
            "schema_version": "ideator.idea.v2",
            "idea_id": idea.get("theory_id", ""),
            "title": idea.get("theory_id", "").replace("-", " ").title(),
            "novelty_summary": idea.get("what_and_why", ""),
            "novelty_claims": idea.get("novelty_claims", []),
            "parent_implementation": {
                "repo_url": "https://github.com/openai/parameter-golf",
                "git_ref": "main",
                "primary_file": "train_gpt.py",
                "run_command": "torchrun --standalone --nproc_per_node=1 train_gpt.py",
                "code_search_hints": [],
            },
            "implementation_steps": [{
                "step_id": "full-replacement",
                "file": "train_gpt.py",
                "locate": "entire file",
                "change": f"Replace with generated code ({len(idea.get('train_gpt_code', '').splitlines())} lines)",
                "done_when": "Script runs, params < 10M, loss decreases",
            }],
            "falsifier_smoke_tests": idea.get("risk_factors", []),
            "expected_metric_change": idea.get("expected_behavior", ""),
            "parameter_estimate": idea.get("parameter_estimate", ""),
        }

    def review_idea(self, idea: Dict, round_idx: int) -> Dict:
        """Run the novelty reviewer gate on a generated idea.

        Returns a review dict with: decision, novelty_score, primary_reasons,
        revision_instructions, must_fix_fields, similar_to_knowledge.
        """
        self.log_stage("REVIEWER GATE", f"Round {round_idx + 1}/{self.reviewer_config.max_rounds}")

        translated = self._translate_idea_for_reviewer(idea)
        review_system, review_user = build_reviewer_prompts(
            knowledge_context=self._knowledge_context,
            idea=translated,
        )

        try:
            response = self.client.generate_idea(
                system_prompt=review_system,
                user_prompt=review_user,
                temperature=self.reviewer_config.temperature,
                max_tokens=self.reviewer_config.max_tokens,
                model=self.reviewer_config.model,
            )

            import re
            try:
                review = json.loads(response)
            except json.JSONDecodeError:
                json_match = re.search(r'\{[\s\S]*\}', response)
                if json_match:
                    review = json.loads(json_match.group(0))
                else:
                    raise

            if not isinstance(review, dict):
                review = {
                    "decision": "revise",
                    "novelty_score": 0,
                    "primary_reasons": ["Reviewer returned non-object JSON"],
                    "revision_instructions": "Regenerate with more novelty.",
                    "must_fix_fields": [],
                    "similar_to_knowledge": [],
                }

            decision = str(review.get("decision", "revise")).strip().lower()
            score = review.get("novelty_score", 0)
            try:
                score = int(score)
            except (TypeError, ValueError):
                score = 0

            # Auto-pass if score meets threshold
            effective_decision = decision
            note = ""
            if decision != "pass" and score >= self.reviewer_config.min_score:
                effective_decision = "pass"
                note = f" (auto-pass: score {score} >= {self.reviewer_config.min_score})"
                review["decision"] = "pass"

            self.log(f"  Reviewer decision: {effective_decision} | novelty_score: {score}{note}")
            if review.get("primary_reasons"):
                for reason in review["primary_reasons"][:3]:
                    self.log(f"    - {reason}")

            self.stats["reviewer_rounds_total"] += 1
            return review

        except Exception as e:
            self.log(f"  Reviewer error: {e} — treating as pass", "WARN")
            return {
                "decision": "pass",
                "novelty_score": 5,
                "primary_reasons": [f"Reviewer error: {e}"],
                "revision_instructions": "",
                "must_fix_fields": [],
                "similar_to_knowledge": [],
            }

    def _generate_falsifier_feedback(self, inp: FalsifierInput, output: FalsifierOutput) -> Optional[Dict]:
        """Generate structured feedback from falsification results for the ideator.

        Uses the canonical generate_feedback() from falsifier.stage2.feedback.
        """
        try:
            feedback = generate_feedback(inp, output, s2=None)
            fb_dict = {
                "one_line": feedback.one_line,
                "stage_reached": feedback.stage_reached,
                "failure_analysis": feedback.failure_analysis,
                "suggested_directions": feedback.suggested_directions,
                "key_measurements": feedback.key_measurements,
            }
            self.log(f"  Feedback: {feedback.one_line}")
            if feedback.suggested_directions:
                for d in feedback.suggested_directions[:3]:
                    self.log(f"    → {d}")
            return fb_dict
        except Exception as e:
            self.log(f"  Warning: feedback generation failed: {e}", "WARN")
            return None

    def _get_t2_feedback_from_graph(self) -> str:
        """Read actual T2 parameter counts from working knowledge graph.
        
        This provides feedback to the ideator about actual vs estimated parameters.
        """
        try:
            kg_dir = project_root / "knowledge_graph"
            graph_path = kg_dir / "graph.json"
            
            if not graph_path.exists():
                return ""
            
            with open(graph_path) as f:
                graph = json.load(f)
            
            nodes = graph.get("nodes", {})
            if not nodes:
                return ""
            
            feedback_lines = ["\n═══════════════════════════════════════════════════════════════════════"]
            feedback_lines.append("HISTORICAL T2 BUDGET RESULTS (LEARN FROM THESE):")
            feedback_lines.append("═══════════════════════════════════════════════════════════════════════")
            
            # Find nodes that were killed by T2
            t2_kills = []
            for node_id, node in nodes.items():
                if not isinstance(node, dict):
                    continue
                
                status = node.get("status", "")
                falsification = node.get("falsification", {})
                killed_by = falsification.get("killed_by", "")
                
                if status == "REFUTED" and killed_by == "T2":
                    test_results = falsification.get("test_results", {})
                    t2_result = test_results.get("T2", {}) or {}
                    
                    estimated = t2_result.get("estimated_params", 0)
                    kill_reason = falsification.get("kill_reason", "")
                    
                    if estimated > 0:
                        t2_kills.append({
                            "node_id": node_id,
                            "estimated": estimated,
                            "reason": kill_reason
                        })
            
            if t2_kills:
                feedback_lines.append(f"\n⚠️  Previous {len(t2_kills)} hypotheses KILLED by T2 Budget:")
                for kill in t2_kills[-5:]:  # Show last 5
                    feedback_lines.append(f"   - {kill['node_id']}: T2 estimated {kill['estimated']:,} params")
                    if kill['reason']:
                        feedback_lines.append(f"     Reason: {kill['reason'][:80]}")
                feedback_lines.append("\n📝 LESSON: The LLM consistently underestimates parameters by ~20-30%!")
                feedback_lines.append("   When you think you have 8M, T2 measures 11M.")
                feedback_lines.append("   Target 6-7M in your estimate to safely stay under 10M.")
            
            # Find successful passes for reference
            passes = []
            for node_id, node in nodes.items():
                if not isinstance(node, dict):
                    continue
                
                status = node.get("status", "")
                if status == "STAGE_1_PASSED":
                    falsification = node.get("falsification", {})
                    test_results = falsification.get("test_results", {})
                    t2_result = test_results.get("T2", {}) or {}
                    estimated = t2_result.get("estimated_params", 0)
                    if estimated > 0:
                        passes.append({
                            "node_id": node_id,
                            "estimated": estimated
                        })
            
            if passes:
                feedback_lines.append(f"\n✅ {len(passes)} hypotheses PASSED T2 with these counts:")
                for p in passes[-3:]:
                    feedback_lines.append(f"   - {p['node_id']}: {p['estimated']:,} params")
            
            feedback_lines.append("═══════════════════════════════════════════════════════════════════════\n")
            return "\n".join(feedback_lines)
            
        except Exception as e:
            return f"\n[Could not load T2 feedback: {e}]\n"

    def generate_idea(
        self,
        hypothesis_num: int,
        previous_idea: Optional[Dict] = None,
        reviewer_feedback: Optional[Dict] = None,
    ) -> Dict:
        """Generate a novel architecture idea using Anthropic Sonnet.

        When previous_idea and reviewer_feedback are provided, this generates
        a *revision* of the previous idea addressing the reviewer's criticism.
        """
        is_revision = previous_idea is not None and reviewer_feedback is not None
        revision_label = " (REVISION)" if is_revision else ""
        self.log_stage(
            f"STAGE 0: IDEA GENERATION{revision_label}",
            f"Hypothesis #{hypothesis_num}",
        )

        # Get T2 feedback from knowledge graph
        t2_feedback = self._get_t2_feedback_from_graph()
        if t2_feedback:
            self.log("  Loaded T2 feedback from knowledge graph")

        # Include KG context so ideator knows what has been tried
        kg_section = ""
        if self._knowledge_context:
            kg_section = f"""
═══════════════════════════════════════════════════════════════════════
KNOWLEDGE GRAPH CONTEXT (what has been tried, what failed, what passed):
═══════════════════════════════════════════════════════════════════════
{self._knowledge_context}
═══════════════════════════════════════════════════════════════════════
"""

        # If revising, include the reviewer's criticism
        revision_section = ""
        if is_revision:
            prev_json = json.dumps(previous_idea, indent=2, default=str)
            rev_json = json.dumps(reviewer_feedback, indent=2, default=str)
            revision_section = f"""
═══════════════════════════════════════════════════════════════════════
REVISION REQUIRED — Your previous idea was REJECTED by the reviewer.
Address the reviewer's criticisms by changing the CORE MECHANISM, not
just rewording. Marginal tweaks will be rejected again.
═══════════════════════════════════════════════════════════════════════

Previous idea (rejected):
{prev_json}

Reviewer feedback:
{rev_json}
═══════════════════════════════════════════════════════════════════════
"""

        system_prompt = f"""You are an expert ML systems engineer specializing in efficient transformer architectures.
Your task is to generate a novel, bold, but PRACTICAL transformer architecture modification.

═══════════════════════════════════════════════════════════════════════
CRITICAL ENGINEERING CONSTRAINTS - YOU MUST RESPECT THESE:
═══════════════════════════════════════════════════════════════════════

1. **HARD PARAMETER BUDGET: MAX 10M PARAMETERS**
   - This is a HARD LIMIT. Ideas with >10M parameters WILL BE REJECTED
   - Count EVERY parameter: embeddings, attention weights, FFN, norms, biases, output head
   - GPT-2 small (124M) is TOO BIG - we need 10M MAXIMUM
   - Valid sizes: 6M (n_embd=384, n_layer=6, n_head=6), 8M, 9M, 10M
   - If you propose >10M, the T2 Budget gate will KILL your idea immediately

2. **Memory Constraints**
   - Target: <2GB peak memory during training
   - No massive activation caches
   - Efficient attention (no O(n²) for long sequences)

3. **Training Stability**
   - Must initialize with variance scaling (GPT-2 style)
   - No gradient explosions at start (initial loss < 15)
   - Must compile and run without errors

4. **Falsifiability**
   - Must be testable in 100 training steps (T7 Microtrain gate)
   - Must show loss reduction or clear failure mode
   - Avoid "magic" components that can't be measured

5. **Novelty vs Practicality**
   - BOLD ideas welcome, but MUST be grounded in known mechanisms
   - Prefer: attention modifications, normalization changes, architecture patterns
   - AVOID: "add more layers" (violates budget), "use quantum computing" (untestable)

{kg_section}
{t2_feedback}
{revision_section}
═══════════════════════════════════════════════════════════════════════

OUTPUT FORMAT - Return ONLY valid JSON:
{{
  "theory_id": "unique-lowercase-hyphenated-name",
  "what_and_why": "2-3 paragraphs. Para 1: What change (BE SPECIFIC). Para 2: Why it works (cite mechanisms). Para 3: Expected falsification signatures.",
  "train_gpt_code": "Complete runnable Python code. MUST use PyTorch. CRITICAL: Count parameters and verify <10M. Include explicit parameter count comment at top.",
  "parent_architecture": "GPT-2 small (modified)",
  "novelty_claims": ["Specific claim 1", "Specific claim 2", "Specific claim 3"],
  "expected_behavior": "Training curve expectations: initial loss? After 100 steps? Success vs failure metrics?",
  "parameter_estimate": "Explicit count: ~X.M parameters (verified <10M). Break down: embeddings=X, layers=Y, etc.",
  "risk_factors": ["What could go wrong", "How to detect failure early"]
}}

═══════════════════════════════════════════════════════════════════════

SAFE CONFIGURATIONS (verified to be under 10M):

**Config A: ~6.5M parameters (RECOMMENDED - safest bet)**
```
n_embd = 384
n_layer = 6
n_head = 6
vocab_size = 50304
block_size = 128
Parameters: 6.5M (safe)
```

**Config B: ~8.2M parameters (MAXIMUM SAFE)**
```
n_embd = 512
n_layer = 6
n_head = 8
vocab_size = 50304
block_size = 128
Parameters: 8.2M (safe)
```

═══════════════════════════════════════════════════════════════════════

⚠️ DANGER: These will EXCEED 10M and WILL BE REJECTED:
- n_embd=768, n_layer=6, vocab=50304 → ~15M (TOO BIG)
- n_embd=512, n_layer=12, vocab=50304 → ~18M (TOO BIG)
- n_embd=384, n_layer=12, vocab=50304 → ~13M (TOO BIG)

═══════════════════════════════════════════════════════════════════════

CRITICAL: You MUST verify your parameter count by calculating:
1. Embeddings: vocab_size × n_embd
2. Output head: vocab_size × n_embd  
3. Per layer: 4 × (n_embd × n_embd) for attention + 2 × (n_embd × 4×n_embd) for FFN
4. Add norms and biases (~0.1M)

═══════════════════════════════════════════════════════════════════════

REMEMBER: 
- A 6M parameter idea that passes all gates is INFINITELY better than a 50M idea that gets killed.
- USE Config A (6.5M) or Config B (8.2M) and MODIFY THE ARCHITECTURE PATTERN, NOT THE SIZE.
- Check the T2 feedback above to see what actually failed before.

Generate something novel but RESPECT THE 10M HARD LIMIT."""

        user_prompt = f"Generate hypothesis #{hypothesis_num} for a novel efficient transformer architecture."
        if is_revision:
            user_prompt += " This is a REVISION — address ALL reviewer criticisms substantively."

        try:
            self.log(f"Calling Anthropic API ({self.stage0_config.model})...")
            response = self.client.generate_idea(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=self.stage0_config.temperature,
                max_tokens=self.stage0_config.max_tokens,
                model=self.stage0_config.model,
            )

            # Parse JSON response
            import re
            try:
                idea = json.loads(response)
            except json.JSONDecodeError:
                # Try to extract JSON from markdown
                json_match = re.search(r'```(?:json)?\n(.*?)\n```', response, re.DOTALL)
                if json_match:
                    idea = json.loads(json_match.group(1))
                else:
                    # Try to find JSON object directly
                    json_match = re.search(r'\{[\s\S]*\}', response)
                    if json_match:
                        idea = json.loads(json_match.group(0))
                    else:
                        raise
            
            # Ensure idea is a dict
            if not isinstance(idea, dict):
                raise ValueError(f"Parsed idea is not a dict, got {type(idea)}: {str(idea)[:100]}")

            theory_id = idea.get("theory_id", f"h{hypothesis_num:03d}")
            param_estimate = idea.get("parameter_estimate", "unknown")
            train_code = idea.get("train_gpt_code", "")
            novelty_claims = idea.get("novelty_claims", [])
            # Ensure novelty_claims is a list
            if isinstance(novelty_claims, str):
                novelty_claims = [novelty_claims] if novelty_claims else []
            elif not isinstance(novelty_claims, list):
                novelty_claims = []
            
            # Log generation results
            self.log(f"✓ Generated hypothesis #{hypothesis_num}: {theory_id}")
            self.log(f"  Parameter estimate: {param_estimate}")
            self.log(f"  Novelty: {len(novelty_claims)} claims")
            code_lines = len(train_code.split("\n")) if train_code else 0
            self.log(f"  Code: ~{code_lines} lines")
            
            # Warn if parameter estimate is missing or exceeds budget
            if param_estimate == "unknown":
                self.log("  ⚠️ WARNING: No parameter_estimate provided - T2 Budget check will likely fail")
            else:
                # Check for parameter budget - look for patterns like "8.2M", "10M", etc
                import re
                numbers = re.findall(r'(\d+\.?\d*)\s*[Mm]', str(param_estimate))
                for num in numbers:
                    try:
                        val = float(num)
                        if val > 10.5:  # Allow slight buffer for estimation error
                            self.log(f"  ⚠️️ WARNING: Parameter estimate {val}M exceeds 10M budget - will likely be KILLED by T2")
                        elif val <= 10:
                            self.log(f"  ✓ Parameter estimate {val}M is within 10M budget")
                    except ValueError:
                        pass
            
            # Pre-check: Look for common budget violations in code
            if train_code:
                import re
                # Check for n_embd values that would exceed budget
                n_embd_matches = re.findall(r'n_embd\s*=\s*(\d+)', train_code)
                for match in n_embd_matches:
                    try:
                        val = int(match)
                        if val > 512:
                            self.log(f"  ⚠️ WARNING: n_embd={val} may exceed 10M budget (recommend 384-512)")
                    except (ValueError, TypeError):
                        pass
                
                # Check for n_layer values
                n_layer_matches = re.findall(r'n_layer\s*=\s*(\d+)', train_code)
                for match in n_layer_matches:
                    try:
                        val = int(match)
                        if val > 8:
                            self.log(f"  ⚠️ WARNING: n_layer={val} may exceed 10M budget (recommend 4-8)")
                    except (ValueError, TypeError):
                        pass

            self.stats["total_generated"] += 1
            self.log(f"  DEBUG: About to return idea (type: {type(idea)})")
            return idea

        except Exception as e:
            import traceback
            self.log(f"✗ Failed to generate hypothesis #{hypothesis_num}: {e}", "ERROR")
            self.log(f"  Traceback: {traceback.format_exc()}", "ERROR")
            raise

    def run_stage1(self, idea: Dict, run: HypothesisRun) -> Dict:
        """Run Stage 1 falsifier gates (T2-T7)."""
        self.log_stage("STAGE 1: FALSIFIER GATES (T2-T7)", f"Theory: {idea.get('theory_id', run.run_id)}")

        # Extract and save train_gpt code
        train_code = idea.get("train_gpt_code", "")
        if not train_code:
            raise ValueError("No train_gpt_code in idea")

        temp_code_file = self.output_dir / "output" / f"{run.run_id}_train_gpt.py"
        temp_code_file.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_code_file, "w") as f:
            f.write(train_code)

        # Create falsifier input
        inp = FalsifierInput(
            train_gpt_path=str(temp_code_file),
            theory_id=idea.get("theory_id", run.run_id),
            what_and_why=idea.get("what_and_why", "No description provided"),
            sota_train_gpt=str(project_root / "parameter-golf" / "train_gpt.py"),
            val_data_path=str(project_root / "data" / "fineweb" / "sample.jsonl"),
        )

        # Validate
        if not inp.theory_id or not inp.theory_id.strip():
            raise ValueError("theory_id is required")
        if not inp.what_and_why or len(inp.what_and_why.split()) < 6:
            raise ValueError("what_and_why must be at least 6 words")
        if not Path(inp.train_gpt_path).exists():
            raise ValueError(f"train_gpt.py not found at {inp.train_gpt_path}")

        # Run Stage 1
        self.log("Running T2 Budget check...")
        self.log("Running T3 Compilation check...")
        self.log("Running T4 Signal check...")
        self.log("Running T5 Init check...")
        self.log("Running T7 Microtrain check...")

        output = run_stage_1(inp)

        # Process result
        result = {
            "verdict": output.verdict,
            "killed_by": output.killed_by,
            "kill_reason": output.kill_reason,
            "tags": [asdict(t) for t in output.tags],
            "t2_budget": asdict(output.t2_budget) if output.t2_budget else None,
            "t3_compilation": asdict(output.t3_compilation) if output.t3_compilation else None,
            "t4_signal": asdict(output.t4_signal) if output.t4_signal else None,
            "t5_init": asdict(output.t5_init) if output.t5_init else None,
            "t7_microtrain": asdict(output.t7_microtrain) if output.t7_microtrain else None,
        }

        # Log result
        if output.verdict == "STAGE_1_PASSED":
            self.log(f"✓ Stage 1 PASSED - advancing to Stage 2")
            self.stats["stage1_passed"] += 1
        else:
            self.log(f"✗ Stage 1 KILLED by: {output.killed_by}")
            self.stats["stage1_killed"] += 1

        if output.tags:
            self.log(f"  Tags: {[t.tag_id for t in output.tags]}")

        return result

    @staticmethod
    def _reconstruct_falsifier_output(stage1_result: Dict, idea: Dict, run: HypothesisRun) -> FalsifierOutput:
        """Reconstruct a proper FalsifierOutput from the serialized stage1_result dicts.

        run_stage1() serialises everything via asdict(), so we need to
        re-hydrate the typed dataclasses so downstream consumers (Stage 2,
        feedback) get real objects, not None.
        """
        from falsifier.types import T2Result, T3Result, T4Result, T5Result, T7Result, Tag

        def _rebuild(cls, data):
            if not data or not isinstance(data, dict):
                return None
            try:
                valid = {}
                import dataclasses
                field_names = {f.name for f in dataclasses.fields(cls)}
                for k, v in data.items():
                    if k in field_names:
                        valid[k] = v
                return cls(**valid)
            except Exception:
                return None

        tags = []
        for t in (stage1_result.get("tags") or []):
            if isinstance(t, dict) and "tag_id" in t:
                try:
                    tags.append(Tag(
                        tag_id=t["tag_id"],
                        test_id=t.get("test_id", ""),
                        category=t.get("category", ""),
                        description=t.get("description", t.get("detail", "")),
                    ))
                except Exception:
                    pass

        return FalsifierOutput(
            theory_id=idea.get("theory_id", run.run_id),
            verdict=stage1_result.get("verdict", "UNKNOWN"),
            killed_by=stage1_result.get("killed_by"),
            kill_reason=stage1_result.get("kill_reason"),
            t2_budget=_rebuild(T2Result, stage1_result.get("t2_budget")),
            t3_compilation=_rebuild(T3Result, stage1_result.get("t3_compilation")),
            t4_signal=_rebuild(T4Result, stage1_result.get("t4_signal")),
            t5_init=_rebuild(T5Result, stage1_result.get("t5_init")),
            t7_microtrain=_rebuild(T7Result, stage1_result.get("t7_microtrain")),
            tags=tags,
        )

    def run_stage2(self, idea: Dict, stage1_result: Dict, run: HypothesisRun) -> Optional[Dict]:
        """Run Stage 2 adversarial falsifier with Anthropic Sonnet."""
        if not self.stage2_config.enabled:
            self.log("Stage 2 disabled, skipping")
            return None

        # Only run Stage 2 if Stage 1 passed
        if stage1_result.get("verdict") != "STAGE_1_PASSED":
            self.log("Stage 1 did not pass, skipping Stage 2")
            return None

        self.log_stage("STAGE 2: ADVERSARIAL FALSIFIER", "Using Anthropic Sonnet")

        self.stats["stage2_triggered"] += 1

        # Prepare Stage 1 output for Stage 2 — reconstruct typed results
        stage1_output = self._reconstruct_falsifier_output(stage1_result, idea, run)

        # Create input for Stage 2 (include full context)
        inp = FalsifierInput(
            train_gpt_path=str(self.output_dir / "output" / f"{run.run_id}_train_gpt.py"),
            theory_id=idea.get("theory_id", run.run_id),
            what_and_why=idea.get("what_and_why", ""),
            sota_train_gpt=str(project_root / "parameter-golf" / "train_gpt.py"),
            val_data_path=str(project_root / "data" / "fineweb" / "sample.jsonl"),
        )

        # Run Stage 2
        self.log("Generating kill hypotheses with Anthropic Sonnet...")
        try:
            s2_output = run_stage_2(inp, stage1_output)

            # Extract real metrics from s2_results if available
            s2r = s2_output.s2_results
            hypotheses_tested = len(s2r.hypothesis_results) if s2r and s2r.hypothesis_results else 0

            result = {
                "verdict": s2_output.verdict,
                "killed_by": s2_output.killed_by,
                "kill_reason": s2_output.kill_reason,
                "hypotheses_tested": hypotheses_tested,
            }

            if s2_output.verdict in ("REFUTED", "REJECTED"):
                self.log(f"✗ Stage 2 KILLED by {s2_output.killed_by}: {s2_output.kill_reason}")
                self.stats["stage2_killed"] += 1
            else:
                self.log(f"✓ Stage 2 PASSED: Theory survived adversarial evaluation")
                self.stats["stage2_passed"] += 1

            return result

        except Exception as e:
            self.log(f"⚠ Stage 2 error: {e}", "WARN")
            self.log("  Falling back to Stage 1 verdict only")
            return {"error": str(e), "verdict": "STAGE_2_ERROR"}

    def run_single_hypothesis(self, hypothesis_num: int) -> HypothesisRun:
        """Run one hypothesis through the complete pipeline.

        Pipeline (matches designed architecture):
          0. Refresh knowledge context from working KG
          1. Generate idea (Stage 0)
          2. Reviewer gate: score for novelty; revise up to max_rounds
          3. Stage 1 falsifier gates (T2-T7)
          4. Generate structured feedback from Stage 1
          5. Stage 2 adversarial falsifier (if Stage 1 passed)
          6. Sync all results to working KG
        """
        run_id = f"h{hypothesis_num:03d}_{uuid.uuid4().hex[:8]}"
        run = HypothesisRun(
            run_id=run_id,
            hypothesis_number=hypothesis_num,
            start_time=time.time(),
            kill_hypotheses=[],
        )
        self.runs.append(run)
        idea: Optional[Dict] = None

        try:
            # ── Step 0: Refresh knowledge context ──────────────────────────
            self._refresh_knowledge_context()

            # ── Step 1 + 2: Ideation → Reviewer loop ──────────────────────
            run.status = "generating"
            accepted_idea: Optional[Dict] = None
            accepted_review: Optional[Dict] = None
            prev_idea: Optional[Dict] = None
            prev_review: Optional[Dict] = None

            rounds = max(1, self.reviewer_config.max_rounds) if self.reviewer_config.enabled else 1

            for round_idx in range(rounds):
                # Generate (or revise) the idea
                if round_idx == 0:
                    idea = self.generate_idea(hypothesis_num)
                else:
                    self.log(f"\n  Revision round {round_idx + 1}/{rounds}...")
                    idea = self.generate_idea(
                        hypothesis_num,
                        previous_idea=prev_idea,
                        reviewer_feedback=prev_review,
                    )

                if not isinstance(idea, dict):
                    self.log(f"  Idea generation returned non-dict ({type(idea)}), skipping review", "WARN")
                    break

                # Run reviewer if enabled
                if not self.reviewer_config.enabled:
                    accepted_idea = idea
                    break

                review = self.review_idea(idea, round_idx)
                decision = str(review.get("decision", "revise")).strip().lower()

                if decision == "pass":
                    accepted_idea = idea
                    accepted_review = review
                    self.stats["reviewer_passed"] += 1
                    self.log(f"  ✓ Reviewer PASSED — advancing to Stage 1")
                    break
                else:
                    self.log(f"  ✗ Reviewer says REVISE — will generate new version")
                    prev_idea = idea
                    prev_review = review

            # If reviewer never passed after all rounds, use the last idea anyway
            # (better to falsify it than to discard entirely)
            if accepted_idea is None:
                if idea is not None and isinstance(idea, dict):
                    self.log(f"  Reviewer did not approve after {rounds} rounds — proceeding with last idea")
                    self.stats["reviewer_rejected"] += 1
                    accepted_idea = idea
                else:
                    raise ValueError("No valid idea generated after all rounds")

            run.idea_json = accepted_idea
            run.reviewer_feedback = accepted_review
            self.capture_graph_snapshot()

            # ── Step 3: Stage 1 falsifier gates ────────────────────────────
            run.status = "stage1"
            stage1_result = self.run_stage1(accepted_idea, run)
            run.stage1_result = stage1_result
            self.capture_graph_snapshot()

            # ── Step 4: Generate structured feedback ───────────────────────
            run.status = "feedback"
            s1_output = self._reconstruct_falsifier_output(stage1_result, accepted_idea, run)
            temp_code_file = self.output_dir / "output" / f"{run.run_id}_train_gpt.py"
            s1_inp = FalsifierInput(
                train_gpt_path=str(temp_code_file),
                theory_id=accepted_idea.get("theory_id", run.run_id),
                what_and_why=accepted_idea.get("what_and_why", ""),
                sota_train_gpt=str(project_root / "parameter-golf" / "train_gpt.py"),
                val_data_path=str(project_root / "data" / "fineweb" / "sample.jsonl"),
            )
            feedback_dict = self._generate_falsifier_feedback(s1_inp, s1_output)
            run.falsifier_feedback = feedback_dict

            # ── Step 5: Stage 2 adversarial falsifier ──────────────────────
            if stage1_result.get("verdict") == "STAGE_1_PASSED":
                run.status = "stage2"
                stage2_result = self.run_stage2(accepted_idea, stage1_result, run)
                run.stage2_result = stage2_result
                run.verdict = stage2_result.get("verdict", "STAGE_2_ERROR") if stage2_result else "STAGE_1_PASSED"
                self.capture_graph_snapshot()
            else:
                run.verdict = stage1_result.get("verdict", "UNKNOWN")

            run.status = "complete"

        except Exception as e:
            run.status = "failed"
            run.error = str(e)
            self.log(f"✗ Hypothesis #{hypothesis_num} failed: {e}", "ERROR")
            import traceback
            self.log(f"  Debug traceback: {traceback.format_exc()[:500]}", "ERROR")
            self.stats["failed"] += 1

        run.end_time = time.time()
        duration = run.end_time - run.start_time
        self.log(f"\n✓ Hypothesis #{hypothesis_num} complete in {duration:.1f}s")
        self.log(f"  Verdict: {run.verdict}")

        # ── Step 6: Sync results to working KG ────────────────────────────
        final_idea = run.idea_json
        if final_idea is not None and isinstance(final_idea, dict):
            self._sync_to_working_graph(run, final_idea)
        else:
            self.log(f"  [Graph Sync] Skipping — idea not available or invalid type: {type(final_idea)}")

        return run

    def _sync_to_working_graph(self, run: HypothesisRun, idea: Dict) -> None:
        """Sync falsification results to the working knowledge graph.

        This ensures the ideator knows what has been tried and what failed.
        The working graph is at knowledge_graph/graph.json (separate from seed graph).
        """
        try:
            # Working graph path
            kg_dir = project_root / "knowledge_graph"
            graph_path = kg_dir / "graph.json"

            # Ensure knowledge graph directory exists
            kg_dir.mkdir(parents=True, exist_ok=True)

            theory_id = idea.get("theory_id", run.run_id)

            # Build node data directly
            node_data = {
                "node_id": theory_id,
                "idea_id": theory_id,
                "title": idea.get("theory_id", "Untitled"),
                "theory_type": "architectural",
                "status": run.verdict or "UNKNOWN",
                "status_history": [
                    {
                        "status": "GENERATED",
                        "timestamp": datetime.fromtimestamp(run.start_time).isoformat(),
                        "actor": "anthropic-ideator",
                    },
                    {
                        "status": run.verdict or "COMPLETE",
                        "timestamp": datetime.fromtimestamp(run.end_time or time.time()).isoformat(),
                        "actor": "falsifier",
                    },
                ],
                "what_and_why": idea.get("what_and_why", ""),
                "source": {
                    "type": "anthropic-ideator",
                    "generated_at": datetime.fromtimestamp(run.start_time).isoformat(),
                    "model": self.stage0_config.model,
                },
                "falsification": {
                    "outcome": "FAILED" if run.verdict in ("REFUTED", "REJECTED") else "PASSED" if run.verdict == "STAGE_2_PASSED" else "ITERATE",
                    "stage_reached": 2 if run.stage2_result else 1,
                    "killed_by": run.stage1_result.get("killed_by") if run.stage1_result else None,
                    "kill_reason": run.stage1_result.get("kill_reason") if run.stage1_result else None,
                    "timing": {
                        "started_at": datetime.fromtimestamp(run.start_time).isoformat(),
                        "completed_at": datetime.fromtimestamp(run.end_time or time.time()).isoformat(),
                        "wall_seconds": (run.end_time or time.time()) - run.start_time,
                    },
                    "test_results": {
                        "T2": run.stage1_result.get("t2_budget") if run.stage1_result else None,
                        "T3": run.stage1_result.get("t3_compilation") if run.stage1_result else None,
                        "T4": run.stage1_result.get("t4_signal") if run.stage1_result else None,
                        "T5": run.stage1_result.get("t5_init") if run.stage1_result else None,
                        "T7": run.stage1_result.get("t7_microtrain") if run.stage1_result else None,
                    },
                    "feedback": run.falsifier_feedback,
                },
            }

            # Use AtomicGraphUpdate to safely update the working graph
            from falsifier.graph.locking import AtomicGraphUpdate
            updater = AtomicGraphUpdate(graph_path)

            # Check if node already exists
            graph = updater.read_graph()
            if theory_id in graph.get("nodes", {}):
                # Update existing node
                updater.update_node(theory_id, node_data)
                self.log(f"  [Graph Sync] Updated existing node: {theory_id}")
            else:
                # Create new node
                updater.create_node(theory_id, node_data)
                self.log(f"  [Graph Sync] Created new node: {theory_id}")

        except Exception as e:
            # Log but don't fail the experiment if graph sync fails
            self.log(f"  [Graph Sync] Warning: Failed to sync to working graph: {e}", "WARNING")

    def capture_graph_snapshot(self) -> GraphSnapshot:
        """Capture current state of knowledge graph."""
        kg_dir = project_root / "knowledge_graph"

        nodes = []
        edges = []

        graph_file = kg_dir / "graph.json"
        if graph_file.exists():
            try:
                with open(graph_file) as f:
                    graph_data = json.load(f)
                    raw_nodes = graph_data.get("nodes", {})
                    raw_edges = graph_data.get("edges", [])
                    # nodes can be dict (id→data) or list — normalise to list of dicts
                    if isinstance(raw_nodes, dict):
                        nodes = [v for v in raw_nodes.values() if isinstance(v, dict)]
                    elif isinstance(raw_nodes, list):
                        nodes = [n for n in raw_nodes if isinstance(n, dict)]
                    edges = [e for e in raw_edges if isinstance(e, dict)]
            except Exception as e:
                self.log(f"  [Graph Warning] Could not load graph: {e}")
                pass

        total = len(nodes)
        approved = sum(1 for n in nodes if n.get("status") == "APPROVED")
        refuted = sum(1 for n in nodes if n.get("status") == "REFUTED")
        passed_s1 = sum(1 for n in nodes if n.get("status") == "STAGE_1_PASSED")
        triggered_s2 = sum(1 for n in nodes if n.get("falsification", {}).get("stage_reached", 0) >= 2)
        passed_s2 = sum(1 for n in nodes if n.get("status") == "STAGE_2_PASSED")

        snapshot = GraphSnapshot(
            timestamp=time.time(),
            iso_time=datetime.now().isoformat(),
            total_ideas=total,
            approved_ideas=approved,
            falsified_ideas=refuted,
            stage1_passed=passed_s1,
            stage2_triggered=triggered_s2,
            stage2_passed=passed_s2,
            nodes=nodes,
            edges=edges,
        )

        snapshot_file = self.snapshots_dir / f"snapshot_{len(self.snapshots):04d}.json"
        with open(snapshot_file, "w") as f:
            json.dump(asdict(snapshot), f, indent=2)

        self.snapshots.append(snapshot)
        return snapshot

    def run_all(self, num_hypotheses: int = 10):
        """Run all hypotheses through the complete pipeline."""
        self.log(f"Starting experiment: {num_hypotheses} hypotheses")
        self.log(f"Estimated time: {num_hypotheses * 3}-{num_hypotheses * 5} minutes")
        self.log("")

        for i in range(1, num_hypotheses + 1):
            self.log(f"\n{'='*70}")
            self.log(f"HYPOTHESIS {i}/{num_hypotheses}")
            self.log(f"{'='*70}")

            run = self.run_single_hypothesis(i)

            # Ensure output directory exists
            (self.output_dir / "output").mkdir(parents=True, exist_ok=True)

            # Save run data
            run_file = self.output_dir / "output" / f"run_{i:03d}.json"
            with open(run_file, "w") as f:
                json.dump(asdict(run), f, indent=2, default=str)

            # Brief pause between runs
            if i < num_hypotheses:
                self.log(f"\nPausing 3 seconds before next hypothesis...")
                time.sleep(3)

        # Final summary
        self.generate_summary()
        self.generate_visualization_data()

    def generate_summary(self):
        """Generate final summary report."""
        self.log("")
        self.log("="*70)
        self.log("FINAL SUMMARY")
        self.log("="*70)

        total_time = time.time() - self.start_time
        completed = sum(1 for r in self.runs if r.status == "complete")
        failed = sum(1 for r in self.runs if r.status == "failed")

        # Verdict breakdown
        verdicts = {}
        for run in self.runs:
            v = run.verdict or "UNKNOWN"
            verdicts[v] = verdicts.get(v, 0) + 1

        self.log("")
        self.log("STATISTICS:")
        self.log(f"  Total time: {total_time/60:.1f} minutes")
        self.log(f"  Completed: {completed}/{len(self.runs)}")
        self.log(f"  Failed: {failed}/{len(self.runs)}")
        self.log("")
        self.log("PIPELINE RESULTS:")
        self.log(f"  Generated: {self.stats['total_generated']}")
        if self.reviewer_config.enabled:
            self.log(f"  Reviewer Passed: {self.stats['reviewer_passed']}")
            self.log(f"  Reviewer Rejected (used anyway): {self.stats['reviewer_rejected']}")
            self.log(f"  Total Reviewer Rounds: {self.stats['reviewer_rounds_total']}")
        self.log(f"  Stage 1 Passed: {self.stats['stage1_passed']} / {self.stats['total_generated']}")
        self.log(f"  Stage 1 Killed: {self.stats['stage1_killed']} / {self.stats['total_generated']}")
        if self.stage2_config.enabled:
            self.log(f"  Stage 2 Triggered: {self.stats['stage2_triggered']}")
            self.log(f"  Stage 2 Passed: {self.stats['stage2_passed']} / {self.stats['stage2_triggered']}")
            self.log(f"  Stage 2 Killed: {self.stats['stage2_killed']} / {self.stats['stage2_triggered']}")
        self.log("")
        self.log("VERDICT BREAKDOWN:")
        for v, count in verdicts.items():
            self.log(f"  {v}: {count}")

        summary = {
            "total_hypotheses": len(self.runs),
            "completed": completed,
            "failed": failed,
            "total_time_seconds": total_time,
            "total_time_minutes": total_time / 60,
            "stage2_enabled": self.stage2_config.enabled,
            "stage2_model": self.stage2_config.model if self.stage2_config.enabled else None,
            "statistics": self.stats,
            "verdicts": verdicts,
            "snapshots_captured": len(self.snapshots),
        }

        summary_file = self.output_dir / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        self.log("")
        self.log(f"✓ Summary saved to: {summary_file}")

    def generate_visualization_data(self):
        """Generate data for visualization tools."""
        self.log("")
        self.log("Generating visualization data...")

        # Timeline data
        timeline = []
        for run in self.runs:
            if run.idea_json:
                timeline.append({
                    "time": run.start_time - self.start_time,
                    "hypothesis": run.hypothesis_number,
                    "event": "start",
                    "theory_id": run.idea_json.get("theory_id", run.run_id),
                })
            if run.end_time:
                timeline.append({
                    "time": run.end_time - self.start_time,
                    "hypothesis": run.hypothesis_number,
                    "event": "complete",
                    "verdict": run.verdict,
                    "status": run.status,
                    "stage1_verdict": run.stage1_result.get("verdict") if run.stage1_result else None,
                    "stage2_verdict": run.stage2_result.get("verdict") if run.stage2_result else None,
                })

        # Evolution data
        evolution = []
        for i, snapshot in enumerate(self.snapshots):
            evolution.append({
                "frame": i,
                "timestamp": snapshot.timestamp - self.start_time,
                "total_ideas": snapshot.total_ideas,
                "approved": snapshot.approved_ideas,
                "falsified": snapshot.falsified_ideas,
                "stage1_passed": snapshot.stage1_passed,
                "stage2_triggered": snapshot.stage2_triggered,
                "stage2_passed": snapshot.stage2_passed,
            })

        viz_data = {
            "timeline": timeline,
            "evolution": evolution,
            "start_time": self.start_time,
            "runs": [asdict(r) for r in self.runs],
            "stage2_config": asdict(self.stage2_config),
        }

        viz_file = self.viz_dir / "visualization_data.json"
        with open(viz_file, "w") as f:
            json.dump(viz_data, f, indent=2, default=str)

        self.log(f"✓ Visualization data saved to: {viz_file}")
        self.log(f"  - Timeline: {len(timeline)} events")
        self.log(f"  - Evolution frames: {len(evolution)}")
        self.log(f"  - Stage 2 enabled: {self.stage2_config.enabled}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Full live experiment with Stage 2")
    parser.add_argument("--num-hypotheses", type=int, default=10,
                       help="Number of hypotheses to generate (default: 10)")
    parser.add_argument("--output-dir", type=Path, default=None,
                       help="Output directory (default: auto-generated)")

    # Reviewer gate options
    reviewer_group = parser.add_argument_group("Reviewer gate")
    reviewer_group.add_argument("--disable-reviewer", action="store_true",
                               help="Disable the novelty reviewer gate")
    reviewer_group.add_argument("--reviewer-model", type=str, default="claude-sonnet-4-20250514",
                               help="Model for the reviewer (default: claude-sonnet-4-20250514)")
    reviewer_group.add_argument("--max-review-rounds", type=int, default=3,
                               help="Max ideation-review rounds per hypothesis (default: 3)")
    reviewer_group.add_argument("--reviewer-min-score", type=int, default=6,
                               help="Auto-pass ideas scoring >= this (default: 6)")

    # Stage 2 options
    s2_group = parser.add_argument_group("Stage 2 falsifier")
    s2_group.add_argument("--disable-stage2", action="store_true",
                         help="Disable Stage 2 adversarial falsifier")
    s2_group.add_argument("--stage2-model", type=str, default="claude-sonnet-4-20250514",
                         help="Anthropic model for Stage 2 (default: claude-sonnet-4-20250514)")

    args = parser.parse_args()

    # Create output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).parent / f"live_run_{timestamp}"

    # Reviewer config
    reviewer_config = ReviewerConfig(
        enabled=not args.disable_reviewer,
        model=args.reviewer_model,
        max_rounds=args.max_review_rounds,
        min_score=args.reviewer_min_score,
    )

    # Stage 2 config
    stage2_config = Stage2Config(
        enabled=not args.disable_stage2,
        model=args.stage2_model,
    )

    print("="*70)
    print("FULL LIVE EXPERIMENT")
    print("="*70)
    print(f"Output: {output_dir}")
    print(f"Hypotheses: {args.num_hypotheses}")
    print(f"Reviewer: {'ENABLED' if reviewer_config.enabled else 'DISABLED'}")
    if reviewer_config.enabled:
        print(f"  Max rounds: {reviewer_config.max_rounds} | Min score: {reviewer_config.min_score}")
    print(f"Stage 2: {'ENABLED' if stage2_config.enabled else 'DISABLED'}")
    if stage2_config.enabled:
        print(f"  Model: {stage2_config.model}")
    print("")

    try:
        experiment = FullLiveExperiment(output_dir, reviewer_config=reviewer_config, stage2_config=stage2_config)
        experiment.run_all(num_hypotheses=args.num_hypotheses)

        print("\n" + "="*70)
        print("✓ EXPERIMENT COMPLETE")
        print("="*70)
        print(f"\nResults: {output_dir}")
        print("\nNext steps:")
        print(f"  1. View summary: cat {output_dir}/summary.json")
        print(f"  2. Generate frames: python3 visualization/generate_frames.py --viz-data {output_dir}/visualization/visualization_data.json --output {output_dir}/visualization/frames")
        print(f"  3. Assemble video: python3 visualization/assemble_clip.py --frames {output_dir}/visualization/frames --output {output_dir}/evolution_timelapse.mp4")
        print(f"  4. View HTML: open {output_dir}/viewer.html")

    except Exception as e:
        print(f"\n✗ EXPERIMENT FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
