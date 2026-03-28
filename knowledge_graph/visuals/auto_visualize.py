#!/usr/bin/env python3
"""
Auto-Visualization Module for Pipeline Integration

Automatically generates all visualizations after a pipeline run:
- Evolution timeline (PNG)
- Evolution movie (MP4) 
- Knowledge graph with dead ends
- Executive summary

Usage (add to end of your pipeline script):
    from knowledge_graph.visuals.auto_visualize import generate_all_visualizations
    generate_all_visualizations(experiment_dir="/path/to/run")
"""

import json
import subprocess
from pathlib import Path
from typing import Optional
import tempfile
import os


def generate_all_visualizations(
    experiment_dir: str,
    seed_graph: str = "knowledge_graph/seed_parameter_golf_kg.json",
    output_dir: str = "knowledge_graph/visuals",
    fps: int = 1,
) -> dict:
    """
    Generate all visualizations for a completed pipeline run.

    Args:
        experiment_dir: Path to the experiment run directory
        seed_graph: Path to baseline knowledge graph
        output_dir: Where to save visualizations
        fps: Frames per second for evolution movie

    Returns:
        Dict with paths to all generated files
    """
    experiment_path = Path(experiment_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "experiment": str(experiment_path),
        "files": {},
        "success": True,
        "errors": []
    }

    print("=" * 60)
    print("Generating Knowledge Graph Visualizations")
    print("=" * 60)

    # 1. Generate evolution movie (v2 with full graph)
    print("\n1. Creating evolution movie v2 (full knowledge graph)...")
    try:
        movie_path = output_path / "evolution_movie_v2.mp4"
        cmd = [
            "python3",
            "knowledge_graph/visuals/generate_evolution_movie_v2.py",
            "--seed", seed_graph,
            "--experiment-dir", str(experiment_path),
            "--output", str(movie_path),
            "--fps", str(fps),
            "--dpi", "150"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())

        if result.returncode == 0 and movie_path.exists():
            results["files"]["evolution_movie_v2"] = str(movie_path)
            print(f"   ✓ {movie_path}")
        else:
            results["errors"].append(f"Movie generation failed: {result.stderr}")
            print(f"   ✗ Movie generation failed")
    except Exception as e:
        results["errors"].append(f"Movie error: {e}")
        print(f"   ✗ Error: {e}")

    # 2. Generate evolution timeline (static)
    print("\n2. Creating evolution timeline...")
    try:
        viz_data = experiment_path / "visualization" / "visualization_data.json"
        if viz_data.exists():
            cmd = [
                "python3",
                "knowledge_graph/visuals/render_evolution.py"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())

            timeline_path = output_path / "evolution_timeline.png"
            if timeline_path.exists():
                results["files"]["evolution_timeline"] = str(timeline_path)
                print(f"   ✓ {timeline_path}")
        else:
            print("   - No visualization data found")
    except Exception as e:
        results["errors"].append(f"Timeline error: {e}")
        print(f"   ✗ Error: {e}")

    # 3. Update hypothesis registry
    print("\n3. Updating hypothesis history...")
    try:
        from generate_evolution_movie import load_hypotheses_from_run, load_seed_graph

        nodes, edges = load_seed_graph(Path(seed_graph))
        hypotheses = load_hypotheses_from_run(experiment_path)

        if hypotheses:
            # Build registry
            registry = {
                "version": "1.0",
                "last_updated": subprocess.check_output(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"]).decode().strip(),
                "experiment_dir": str(experiment_path),
                "total_hypotheses": len(hypotheses),
                "verdict_summary": {
                    "REFUTED": sum(1 for h in hypotheses if h.get("verdict") == "REFUTED"),
                    "VERIFIED": sum(1 for h in hypotheses if h.get("verdict") == "VERIFIED"),
                    "UNKNOWN": sum(1 for h in hypotheses if h.get("verdict") not in ["REFUTED", "VERIFIED"]),
                },
                "hypotheses": hypotheses
            }

            registry_path = Path("knowledge_graph/history") / f"registry_{experiment_path.name}.json"
            registry_path.parent.mkdir(parents=True, exist_ok=True)

            with open(registry_path, 'w') as f:
                json.dump(registry, f, indent=2)

            results["files"]["hypothesis_registry"] = str(registry_path)
            print(f"   ✓ {registry_path}")
        else:
            print("   - No hypotheses found")

    except Exception as e:
        results["errors"].append(f"Registry error: {e}")
        print(f"   ✗ Error: {e}")

    # 4. Generate knowledge graph with dead ends
    print("\n4. Creating knowledge graph with dead ends...")
    try:
        cmd = [
            "python3",
            "knowledge_graph/visuals/kg_with_dead_ends.py"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())

        dead_ends_path = output_path / "kg_with_dead_ends.png"
        if dead_ends_path.exists():
            results["files"]["kg_with_dead_ends"] = str(dead_ends_path)
            print(f"   ✓ {dead_ends_path}")
    except Exception as e:
        results["errors"].append(f"Dead ends error: {e}")
        print(f"   ✗ Error: {e}")

    # 5. Generate executive visualizations
    print("\n5. Creating executive visualizations...")
    try:
        cmd = [
            "python3",
            "knowledge_graph/visuals/kg_executive.py",
            "--with-hypotheses"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())

        exec_path = output_path / "kg_executive_with_research.png"
        if exec_path.exists():
            results["files"]["executive"] = str(exec_path)
            print(f"   ✓ {exec_path}")
    except Exception as e:
        results["errors"].append(f"Executive error: {e}")
        print(f"   ✗ Error: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("Visualization Generation Complete")
    print("=" * 60)
    print(f"Generated {len(results['files'])} files:")
    for name, path in results["files"].items():
        print(f"  - {name}: {path}")

    if results["errors"]:
        print(f"\nErrors ({len(results['errors'])}):")
        for err in results["errors"]:
            print(f"  ! {err}")
        results["success"] = False
    else:
        print("\n✓ All visualizations generated successfully")

    return results


def main():
    """CLI entry point for standalone execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Auto-generate all visualizations")
    parser.add_argument("--experiment-dir", required=True,
                        help="Path to experiment run directory")
    parser.add_argument("--seed-graph", default="knowledge_graph/seed_parameter_golf_kg.json",
                        help="Path to seed knowledge graph")
    parser.add_argument("--output-dir", default="knowledge_graph/visuals",
                        help="Output directory")
    parser.add_argument("--fps", type=int, default=1,
                        help="Movie frame rate")

    args = parser.parse_args()

    results = generate_all_visualizations(
        experiment_dir=args.experiment_dir,
        seed_graph=args.seed_graph,
        output_dir=args.output_dir,
        fps=args.fps
    )

    # Save results manifest
    manifest_path = Path(args.output_dir) / "visualization_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nManifest saved: {manifest_path}")

    return 0 if results["success"] else 1


if __name__ == "__main__":
    exit(main())
