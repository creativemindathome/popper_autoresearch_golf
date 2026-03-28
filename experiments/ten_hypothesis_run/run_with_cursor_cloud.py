#!/usr/bin/env python3
"""Run autoresearch experiments using Cursor Cloud Agents API.

This launches cloud agents that execute the full pipeline:
1. Generate hypotheses using Anthropic (via cloud agent)
2. Run falsifier gates (in cloud agent environment)
3. Track knowledge graph evolution
4. Return results

Requires: CURSOR_API_KEY environment variable
"""

import json
import os
import sys
import time
import base64
from pathlib import Path
from typing import Optional, Dict, Any
import urllib.request
import urllib.error


class CursorCloudAPIError(Exception):
    """Error from Cursor Cloud API."""
    pass


class CursorCloudClient:
    """Client for Cursor Cloud Agents API."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.cursor.com"
    
    def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Make API request with basic auth."""
        url = f"{self.base_url}{endpoint}"
        
        # Create request
        req = urllib.request.Request(
            url,
            method=method,
            headers={
                "Content-Type": "application/json",
            }
        )
        
        # Add basic auth (API key as username, no password)
        import base64
        credentials = base64.b64encode(f"{self.api_key}:".encode()).decode()
        req.add_header("Authorization", f"Basic {credentials}")
        
        # Add data if provided
        if data:
            req.data = json.dumps(data).encode("utf-8")
        
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            raise CursorCloudAPIError(f"HTTP {e.code}: {error_body}")
        except Exception as e:
            raise CursorCloudAPIError(f"Request failed: {e}")
    
    def launch_agent(
        self,
        prompt: str,
        repository: str,
        ref: str = "main",
        model: str = "default",
        auto_create_pr: bool = False,
        branch_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Launch a cloud agent to run autoresearch."""
        data = {
            "prompt": {"text": prompt},
            "model": model,
            "source": {
                "repository": repository,
                "ref": ref
            },
            "target": {
                "autoCreatePr": auto_create_pr
            }
        }
        
        if branch_name:
            data["target"]["branchName"] = branch_name
        
        return self._request("POST", "/v0/agents", data)
    
    def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Get status of a running agent."""
        return self._request("GET", f"/v0/agents/{agent_id}")
    
    def get_agent_conversation(self, agent_id: str) -> Dict[str, Any]:
        """Get conversation history from agent."""
        return self._request("GET", f"/v0/agents/{agent_id}/conversation")
    
    def list_agents(self, limit: int = 20) -> Dict[str, Any]:
        """List all cloud agents."""
        return self._request("GET", f"/v0/agents?limit={limit}")
    
    def stop_agent(self, agent_id: str) -> Dict[str, Any]:
        """Stop a running agent."""
        return self._request("POST", f"/v0/agents/{agent_id}/stop", {})
    
    def delete_agent(self, agent_id: str) -> Dict[str, Any]:
        """Delete an agent permanently."""
        return self._request("DELETE", f"/v0/agents/{agent_id}")
    
    def list_models(self) -> Dict[str, Any]:
        """List available models for cloud agents."""
        return self._request("GET", "/v0/models")
    
    def get_api_info(self) -> Dict[str, Any]:
        """Get info about the API key."""
        return self._request("GET", "/v0/me")
    
    def add_followup(self, agent_id: str, prompt: str) -> Dict[str, Any]:
        """Add follow-up instruction to running agent."""
        data = {
            "prompt": {"text": prompt}
        }
        return self._request("POST", f"/v0/agents/{agent_id}/followup", data)


class CursorCloudAutoresearch:
    """Run autoresearch using Cursor Cloud Agents."""
    
    def __init__(self, api_key: str, repo_url: str):
        self.client = CursorCloudClient(api_key)
        self.repo_url = repo_url
        self.agent_id: Optional[str] = None
    
    def create_experiment_prompt(self, num_hypotheses: int = 10) -> str:
        """Create the experiment prompt for the cloud agent."""
        return f"""Run an autoresearch experiment with {num_hypotheses} hypotheses.

This is an automated research pipeline that:
1. Generates novel transformer architecture ideas using Anthropic Claude
2. Runs each through Stage 1 falsifier gates (T2-T7)
3. Runs Stage 2 adversarial falsifier on survivors
4. Tracks knowledge graph evolution

Steps to execute:
1. Load environment from .env file (ANTHROPIC_API_KEY should be set)
2. Run: cd experiments/ten_hypothesis_run && python3 run_full_live_experiment.py --num-hypotheses {num_hypotheses}
3. Wait for completion (estimated 30-60 minutes)
4. Save results to the experiment output directory
5. Generate visualization frames
6. Create summary report

The experiment uses:
- Anthropic Sonnet for idea generation (Stage 0)
- Local MLX/PyTorch for Stage 1 gates (T2 Budget, T3 Compilation, T4 Signal, T5 Init, T7 Microtrain)
- Anthropic Sonnet for Stage 2 adversarial falsifier
- Knowledge graph tracking with snapshots

Monitor progress in the console output. When complete, the results will be in experiments/ten_hypothesis_run/live_run_*/

Key files to check:
- summary.json (final statistics)
- logs/run.log (detailed log)
- visualization/frames/ (evolution frames)
- knowledge_graph_evolution.html (interactive viewer)

Do not stop until the experiment completes and all visualizations are generated.
"""
    
    def launch_experiment(
        self,
        num_hypotheses: int = 10,
        model: str = "claude-4-sonnet-thinking",
        branch: str = "main"
    ) -> str:
        """Launch the autoresearch experiment in Cursor Cloud."""
        print("🚀 Launching autoresearch experiment in Cursor Cloud...")
        print(f"   Repository: {self.repo_url}")
        print(f"   Branch: {branch}")
        print(f"   Hypotheses: {num_hypotheses}")
        print(f"   Model: {model}")
        
        prompt = self.create_experiment_prompt(num_hypotheses)
        
        try:
            result = self.client.launch_agent(
                prompt=prompt,
                repository=self.repo_url,
                ref=branch,
                model=model,
                auto_create_pr=False,
                branch_name=f"autoresearch-run-{int(time.time())}"
            )
            
            self.agent_id = result.get("id")
            status = result.get("status")
            url = result.get("target", {}).get("url", "N/A")
            
            print(f"\n✓ Agent launched!")
            print(f"   Agent ID: {self.agent_id}")
            print(f"   Status: {status}")
            print(f"   URL: {url}")
            
            return self.agent_id
            
        except CursorCloudAPIError as e:
            print(f"\n✗ Failed to launch: {e}")
            raise
    
    def monitor_progress(self, poll_interval: int = 30) -> Dict[str, Any]:
        """Monitor experiment progress until completion."""
        if not self.agent_id:
            raise ValueError("No agent running. Call launch_experiment first.")
        
        print(f"\n📊 Monitoring agent {self.agent_id}...")
        print(f"   Polling every {poll_interval} seconds")
        print("   (Press Ctrl+C to stop monitoring - agent will continue running)")
        print()
        
        last_status = None
        last_summary = None
        
        try:
            while True:
                status_info = self.client.get_agent_status(self.agent_id)
                status = status_info.get("status", "UNKNOWN")
                summary = status_info.get("summary", "")
                
                # Only print if status changed
                if status != last_status or summary != last_summary:
                    timestamp = time.strftime("%H:%M:%S")
                    status_emoji = {
                        "FINISHED": "✅",
                        "RUNNING": "🔄",
                        "CREATING": "🆕",
                        "FAILED": "❌",
                        "STOPPED": "⏹️"
                    }.get(status, "❓")
                    
                    print(f"[{timestamp}] {status_emoji} {status}")
                    
                    if summary and summary != last_summary:
                        print(f"         📝 {summary[:120]}...")
                    
                    last_status = status
                    last_summary = summary
                
                if status in ["FINISHED", "FAILED", "STOPPED"]:
                    print(f"\n{'='*60}")
                    print(f"✓ Agent finished with status: {status}")
                    print(f"{'='*60}")
                    return status_info
                
                time.sleep(poll_interval)
                
        except KeyboardInterrupt:
            print("\n\n⚠ Monitoring stopped (agent still running)")
            print(f"   To check: python3 run_with_cursor_cloud.py --status {self.agent_id}")
            print(f"   To stop:  python3 run_with_cursor_cloud.py --stop {self.agent_id}")
            return {"status": "RUNNING", "id": self.agent_id}
    
    def get_results(self) -> Dict[str, Any]:
        """Get experiment results from agent."""
        if not self.agent_id:
            raise ValueError("No agent ID specified")
        
        print(f"\n📥 Retrieving results for agent {self.agent_id}...")
        
        # Get status
        status = self.client.get_agent_status(self.agent_id)
        
        # Get conversation
        try:
            conversation = self.client.get_agent_conversation(self.agent_id)
            messages = conversation.get("messages", [])
        except Exception as e:
            messages = []
            print(f"   Warning: Could not retrieve conversation: {e}")
        
        return {
            "status": status,
            "conversation": messages,
            "agent_id": self.agent_id
        }
    
    def stop(self) -> bool:
        """Stop the running agent."""
        if not self.agent_id:
            print("No agent to stop")
            return False
        
        print(f"\n🛑 Stopping agent {self.agent_id}...")
        try:
            self.client.stop_agent(self.agent_id)
            print("✓ Agent stopped")
            return True
        except Exception as e:
            print(f"✗ Failed to stop: {e}")
            return False


def get_cursor_api_key() -> str:
    """Get Cursor API key from environment."""
    api_key = os.environ.get("CURSOR_API_KEY")
    if not api_key:
        print("✗ CURSOR_API_KEY not found!")
        print("\nTo get your API key:")
        print("  1. Go to https://cursor.com/settings")
        print("  2. Create an API key")
        print("  3. Set it: export CURSOR_API_KEY='your-key'")
        sys.exit(1)
    return api_key


def print_models(models_data: Dict):
    """Pretty print available models."""
    models = models_data.get("models", [])
    print("\n📋 Available Cursor Cloud Models:")
    print("-" * 50)
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model}")
    print()
    print("💡 Use --model MODEL_NAME to specify which agent to use")
    print("   Example: --model claude-4-sonnet-thinking")


def print_agents(agents_data: Dict, status_filter: Optional[str] = None):
    """Pretty print agents list."""
    agents = agents_data.get("agents", [])
    
    if status_filter:
        agents = [a for a in agents if a.get("status") == status_filter.upper()]
    
    if not agents:
        print("\n📭 No agents found" + (f" with status: {status_filter}" if status_filter else ""))
        return
    
    print(f"\n📋 Your Cursor Cloud Agents ({len(agents)} total):")
    print("-" * 80)
    print(f"{'ID':<20} {'Name':<30} {'Status':<12} {'Created'}")
    print("-" * 80)
    
    for agent in agents[:20]:  # Show first 20
        agent_id = agent.get("id", "N/A")[:18]
        name = agent.get("name", "Unnamed")[:28]
        status = agent.get("status", "UNKNOWN")
        created = agent.get("createdAt", "")[:16]
        
        status_icon = {
            "FINISHED": "✅",
            "RUNNING": "🔄",
            "CREATING": "🆕",
            "FAILED": "❌",
            "STOPPED": "⏹️"
        }.get(status, "❓")
        
        print(f"{agent_id:<20} {name:<30} {status_icon} {status:<10} {created}")
    
    if len(agents) > 20:
        print(f"\n... and {len(agents) - 20} more agents")
    
    print()
    print("💡 Commands:")
    print("   --status AGENT_ID    - Check agent details")
    print("   --stop AGENT_ID      - Stop running agent")
    print("   --delete AGENT_ID    - Delete agent permanently")
    print("   --conversation ID    - View full conversation")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run autoresearch with Cursor Cloud Agents API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run 10 hypotheses with specific model
  python3 run_with_cursor_cloud.py --model claude-4-sonnet-thinking

  # Run with specific branch (e.g., Leonard)
  python3 run_with_cursor_cloud.py --model composer-2 --branch Leonard

  # List available models
  python3 run_with_cursor_cloud.py --list-models

  # List all your agents
  python3 run_with_cursor_cloud.py --list-agents

  # List only running agents
  python3 run_with_cursor_cloud.py --list-agents --filter running

  # Check specific agent status
  python3 run_with_cursor_cloud.py --status bc_abc123

  # View agent conversation
  python3 run_with_cursor_cloud.py --conversation bc_abc123

  # Add follow-up to running agent
  python3 run_with_cursor_cloud.py --followup bc_abc123 "Generate 5 more hypotheses"
        """
    )
    
    # Experiment options
    parser.add_argument("--num-hypotheses", type=int, default=10,
                       help="Number of hypotheses to generate (default: 10)")
    parser.add_argument("--model", type=str, default="claude-4-sonnet-thinking",
                       help="Cursor Cloud model/agent to use (see --list-models)")
    parser.add_argument("--repo", type=str,
                       default="https://github.com/creativemindathome/popper_autoresearch_golf",
                       help="GitHub repository URL")
    parser.add_argument("--branch", type=str, default="main",
                       help="Git branch to use (default: main)")
    
    # Agent management
    parser.add_argument("--list-models", action="store_true",
                       help="List available Cursor Cloud models")
    parser.add_argument("--list-agents", action="store_true",
                       help="List all your cloud agents")
    parser.add_argument("--filter", type=str, metavar="STATUS",
                       choices=["finished", "running", "creating", "failed", "stopped"],
                       help="Filter agents by status (with --list-agents)")
    parser.add_argument("--status", type=str, metavar="AGENT_ID",
                       help="Check detailed status of specific agent")
    parser.add_argument("--conversation", type=str, metavar="AGENT_ID",
                       help="Get conversation history from agent")
    parser.add_argument("--stop", type=str, metavar="AGENT_ID",
                       help="Stop a running agent")
    parser.add_argument("--delete", type=str, metavar="AGENT_ID",
                       help="Delete an agent permanently")
    parser.add_argument("--followup", type=str, metavar="AGENT_ID",
                       help="Add follow-up to running agent (use with --prompt)")
    parser.add_argument("--prompt", type=str,
                       help="Follow-up prompt text (use with --followup)")
    
    # Execution options
    parser.add_argument("--no-monitor", action="store_true",
                       help="Launch without monitoring (runs in background)")
    parser.add_argument("--poll-interval", type=int, default=30,
                       help="Status polling interval in seconds (default: 30)")
    
    args = parser.parse_args()
    
    # Get API key
    api_key = get_cursor_api_key()
    client = CursorCloudClient(api_key)
    
    # Handle list models
    if args.list_models:
        try:
            models = client.list_models()
            print_models(models)
        except Exception as e:
            print(f"✗ Failed to list models: {e}")
        return
    
    # Handle list agents
    if args.list_agents:
        try:
            agents = client.list_agents(limit=100)
            print_agents(agents, args.filter)
        except Exception as e:
            print(f"✗ Failed to list agents: {e}")
        return
    
    # Handle API info
    if args.status == "me" or args.status == "api":
        try:
            info = client.get_api_info()
            print("\n🔑 API Key Info:")
            print(json.dumps(info, indent=2))
        except Exception as e:
            print(f"✗ Failed to get API info: {e}")
        return
    
    # Handle status check
    if args.status:
        try:
            status = client.get_agent_status(args.status)
            print(f"\n📊 Agent {args.status} Status:")
            print(json.dumps(status, indent=2))
        except Exception as e:
            print(f"✗ Failed to get status: {e}")
        return
    
    # Handle conversation retrieval
    if args.conversation:
        try:
            conv = client.get_agent_conversation(args.conversation)
            messages = conv.get("messages", [])
            print(f"\n💬 Agent {args.conversation} Conversation:")
            print("-" * 60)
            for msg in messages[-20:]:  # Show last 20 messages
                msg_type = msg.get("type", "unknown")
                text = msg.get("text", "")[:200]
                icon = "👤" if msg_type == "user_message" else "🤖"
                print(f"\n{icon} {msg_type}:")
                print(f"   {text}...")
        except Exception as e:
            print(f"✗ Failed to get conversation: {e}")
        return
    
    # Handle stop
    if args.stop:
        try:
            client.stop_agent(args.stop)
            print(f"✓ Stopped agent {args.stop}")
        except Exception as e:
            print(f"✗ Failed to stop agent: {e}")
        return
    
    # Handle delete
    if args.delete:
        try:
            confirm = input(f"⚠️  Delete agent {args.delete} permanently? [y/N]: ")
            if confirm.lower() == 'y':
                client.delete_agent(args.delete)
                print(f"✓ Deleted agent {args.delete}")
            else:
                print("Cancelled")
        except Exception as e:
            print(f"✗ Failed to delete agent: {e}")
        return
    
    # Handle follow-up
    if args.followup:
        if not args.prompt:
            print("✗ --prompt required with --followup")
            print("   Example: --followup bc_123 --prompt 'Generate 5 more'")
            return
        try:
            client.add_followup(args.followup, args.prompt)
            print(f"✓ Added follow-up to agent {args.followup}")
        except Exception as e:
            print(f"✗ Failed to add follow-up: {e}")
        return
    
    # Launch new experiment
    runner = CursorCloudAutoresearch(api_key, args.repo)
    
    # Show which model we're using
    print(f"\n🤖 Using Cursor Cloud model: {args.model}")
    
    try:
        agent_id = runner.launch_experiment(
            num_hypotheses=args.num_hypotheses,
            model=args.model,
            branch=args.branch
        )
        
        if not args.no_monitor:
            # Monitor until completion
            final_status = runner.monitor_progress(poll_interval=args.poll_interval)
            
            # Get results
            if final_status.get("status") == "FINISHED":
                results = runner.get_results()
                print("\n" + "="*60)
                print("EXPERIMENT COMPLETE")
                print("="*60)
                print(f"\nResults summary:")
                print(f"  Agent ID: {agent_id}")
                print(f"  Status: {final_status.get('status')}")
                print(f"  Summary: {final_status.get('summary', 'N/A')}")
                print(f"\nTo view conversation:")
                print(f"  python3 run_with_cursor_cloud.py --conversation {agent_id}")
                
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        if runner.agent_id:
            print(f"\nAgent {runner.agent_id} is still running!")
            print(f"To check status: python3 run_with_cursor_cloud.py --status {runner.agent_id}")
            print(f"To stop: python3 run_with_cursor_cloud.py --stop {runner.agent_id}")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
