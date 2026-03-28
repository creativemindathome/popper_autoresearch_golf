#!/usr/bin/env python3
"""
Generate all knowledge graph visualizations.

Run this after new autoresearch completes to update all visualizations.
"""

import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> bool:
    """Run a command, return True if successful."""
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return False


def main() -> int:
    viz_dir = Path("knowledge_graph/visuals")
    scripts = [
        viz_dir / "knowledge_graph_visualizer.py",
        viz_dir / "kg_bubble_visualizer.py",
    ]

    # Check scripts exist
    for script in scripts:
        if not script.exists():
            print(f"Error: {script} not found")
            return 1

    print("=" * 60)
    print("Generating Knowledge Graph Visualizations")
    print("=" * 60)
    print()

    # Basic versions (no hypotheses)
    print("1. Generating basic hierarchical visualization...")
    if not run([
        sys.executable, str(scripts[0]),
        "--labels", "roots-branches",
        "--dpi", "300",
    ]):
        print("   Failed!")

    print("2. Generating basic bubble visualization...")
    if not run([
        sys.executable, str(scripts[1]),
        "--dpi", "300",
    ]):
        print("   Failed!")

    # Versions with hypotheses
    print("3. Generating hierarchical visualization with hypotheses...")
    if not run([
        sys.executable, str(scripts[0]),
        "--with-hypotheses",
        "--labels", "roots-branches",
        "--basename", "kg_unified_with_hypotheses",
        "--dpi", "300",
    ]):
        print("   Failed!")

    print("4. Generating bubble visualization with hypotheses...")
    if not run([
        sys.executable, str(scripts[1]),
        "--with-hypotheses",
        "--basename", "kg_bubble_with_hypotheses",
        "--dpi", "300",
    ]):
        print("   Failed!")

    print()
    print("=" * 60)
    print("Done! Generated files:")
    print("=" * 60)

    output_dir = Path("knowledge_graph/visuals")
    for f in sorted(output_dir.glob("kg_*")):
        if f.suffix in [".png", ".svg"]:
            size = f.stat().st_size / 1024
            print(f"  {f.name} ({size:.1f} KB)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
