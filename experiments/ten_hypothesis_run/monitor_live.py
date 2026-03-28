#!/usr/bin/env python3
"""Real-time dashboard monitor for speed run."""

import json
import time
import sys
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent.parent
kg_path = project_root / "knowledge_graph" / "graph.json"

def get_stats():
    """Get current graph stats."""
    try:
        with open(kg_path) as f:
            graph = json.load(f)
        nodes = graph.get("nodes", {})
        if isinstance(nodes, dict):
            total = len(nodes)
            approved = sum(1 for n in nodes.values() if n.get("status") == "APPROVED")
            killed = sum(1 for n in nodes.values() if n.get("status") == "KILLED")
            unknown = sum(1 for n in nodes.values() if n.get("status") not in ("APPROVED", "KILLED"))
        else:
            total = len(nodes)
            approved = sum(1 for n in nodes if n.get("status") == "APPROVED")
            killed = sum(1 for n in nodes if n.get("status") == "KILLED")
            unknown = total - approved - killed
        return {"total": total, "approved": approved, "killed": killed, "unknown": unknown}
    except:
        return {"total": 0, "approved": 0, "killed": 0, "unknown": 0}

def clear_screen():
    """Clear terminal."""
    print("\033[2J\033[H", end="")

def print_dashboard(start_time, target_hypotheses=100):
    """Print real-time dashboard."""
    stats = get_stats()
    elapsed = time.time() - start_time
    elapsed_min = elapsed / 60
    
    # Calculate throughput
    if elapsed_min > 0:
        throughput_min = stats["total"] / elapsed_min if stats["total"] > 0 else 0
        throughput_hour = throughput_min * 60
    else:
        throughput_min = 0
        throughput_hour = 0
    
    # ETA calculation
    if throughput_min > 0 and stats["total"] < target_hypotheses:
        remaining = target_hypotheses - stats["total"]
        eta_min = remaining / throughput_min
        eta_str = f"{eta_min:.1f} min"
    else:
        eta_str = "N/A"
    
    # Progress bar
    progress_pct = min(100, (stats["total"] / target_hypotheses) * 100)
    bar_width = 30
    filled = int(bar_width * progress_pct / 100)
    bar = "█" * filled + "░" * (bar_width - filled)
    
    clear_screen()
    print("=" * 70)
    print(f"🚀 AUTORESEARCH SPEED RUN - REAL-TIME DASHBOARD")
    print("=" * 70)
    print(f"⏱️  Elapsed: {elapsed_min:.1f} min | 🎯 Target: {target_hypotheses} hypotheses")
    print()
    print(f"📊 PROGRESS: [{bar}] {progress_pct:.1f}%")
    print(f"   Total: {stats['total']} | ✅ Passed: {stats['approved']} | ❌ Killed: {stats['killed']} | ⏳ Pending: {stats['unknown']}")
    print()
    print(f"⚡ THROUGHPUT:")
    print(f"   Current: {throughput_min:.2f} hyp/min = {throughput_hour:.1f} hyp/hour")
    print(f"   Target: 100/hour = 1.67/min")
    status = "✅ ON TRACK" if throughput_hour >= 100 else "⚠️  BELOW TARGET"
    print(f"   Status: {status}")
    print()
    print(f"⏰ ETA: {eta_str}")
    print()
    
    # Increasing returns check
    if elapsed_min >= 5 and stats["total"] > 10:
        print("📈 INCREASING RETURNS TO SCALE:")
        # Show last 5 minutes vs first 5 minutes
        print(f"   (Monitoring knowledge graph efficiency...)")
    
    print("=" * 70)
    print(f"Last update: {datetime.now().strftime('%H:%M:%S')} | Press Ctrl+C to exit")
    print()

if __name__ == "__main__":
    print("🚀 Starting real-time dashboard monitor...")
    print(f"Monitoring: {kg_path}")
    print()
    
    start_time = time.time()
    
    try:
        while True:
            print_dashboard(start_time, target_hypotheses=100)
            time.sleep(5)  # Update every 5 seconds
    except KeyboardInterrupt:
        print("\n\n👋 Monitor stopped. Speed run continues in background.")
        print("   Check logs: tail -f speed_run_100.log")
