#!/usr/bin/env python3
"""Generate visualization frames for 2-hour evolution time-lapse.

This creates frames that can be assembled into a video showing:
- Knowledge graph growth over time
- Hypothesis generation and falsification
- Pipeline stage progression
- Statistics evolution
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Try to import matplotlib for frame generation
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, generating text-based frames only")


@dataclass
class FrameData:
    """Data for a single frame."""
    frame_number: int
    timestamp: float  # seconds from start
    total_ideas: int
    approved: int
    falsified: int
    stage1_passed: int
    stage2_passed: int
    active_runs: List[Dict]
    completed_runs: List[Dict]
    failed_runs: List[Dict]


class FrameGenerator:
    """Generate visualization frames."""

    def __init__(self, viz_data_path: Path, output_dir: Path):
        self.viz_data_path = viz_data_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        with open(viz_data_path) as f:
            self.data = json.load(f)

        self.timeline = self.data.get("timeline", [])
        self.evolution = self.data.get("evolution", [])
        self.runs = self.data.get("runs", [])
        self.start_time = self.data.get("start_time", 0)

        # Frame configuration
        self.target_duration = 2 * 60 * 60  # 2 hours in seconds for time-lapse
        self.fps = 30
        self.total_frames = int(self.target_duration * self.fps)

    def interpolate_state(self, target_time: float) -> FrameData:
        """Interpolate graph state at a given time."""
        # Find surrounding evolution points
        before = None
        after = None

        for i, ev in enumerate(self.evolution):
            ev_time = ev.get("timestamp", 0)
            if ev_time <= target_time:
                before = ev
            if ev_time > target_time and after is None:
                after = ev
                break

        # Use closest or interpolate
        if before is None:
            state = after if after else {"frame": 0, "timestamp": 0, "total_ideas": 0}
        elif after is None:
            state = before
        else:
            # Linear interpolation
            t_before = before.get("timestamp", 0)
            t_after = after.get("timestamp", 0)
            if t_after == t_before:
                state = before
            else:
                ratio = (target_time - t_before) / (t_after - t_before)
                state = {
                    "frame": before["frame"],
                    "timestamp": target_time,
                    "total_ideas": self._lerp(before.get("total_ideas", 0), after.get("total_ideas", 0), ratio),
                    "approved": self._lerp(before.get("approved", 0), after.get("approved", 0), ratio),
                    "falsified": self._lerp(before.get("falsified", 0), after.get("falsified", 0), ratio),
                    "stage1_passed": self._lerp(before.get("stage1_passed", 0), after.get("stage1_passed", 0), ratio),
                    "stage2_passed": self._lerp(before.get("stage2_passed", 0), after.get("stage2_passed", 0), ratio),
                }

        # Get active/completed runs at this time
        active = [r for r in self.runs if r.get("start_time", 0) <= target_time + self.start_time]
        active = [r for r in active if r.get("end_time", float('inf')) > target_time + self.start_time]

        completed = [r for r in self.runs if r.get("end_time", 0) <= target_time + self.start_time and r.get("status") == "complete"]
        failed = [r for r in self.runs if r.get("end_time", 0) <= target_time + self.start_time and r.get("status") == "failed"]

        return FrameData(
            frame_number=0,  # Set later
            timestamp=target_time,
            total_ideas=int(state.get("total_ideas", 0)),
            approved=int(state.get("approved", 0)),
            falsified=int(state.get("falsified", 0)),
            stage1_passed=int(state.get("stage1_passed", 0)),
            stage2_passed=int(state.get("stage2_passed", 0)),
            active_runs=active,
            completed_runs=completed,
            failed_runs=failed,
        )

    def _lerp(self, a: float, b: float, t: float) -> float:
        """Linear interpolation."""
        return a + (b - a) * t

    def generate_matplotlib_frame(self, frame_data: FrameData, frame_number: int) -> Path:
        """Generate a single frame using matplotlib."""
        if not HAS_MATPLOTLIB:
            return None

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Knowledge Graph Evolution - Frame {frame_number:05d}\nTime: {frame_data.timestamp:.1f}s', fontsize=14)

        # 1. Graph topology visualization (top-left)
        ax = axes[0, 0]
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Knowledge Graph Structure', fontsize=12)

        # Draw nodes
        total_nodes = frame_data.total_ideas
        if total_nodes > 0:
            # Position nodes in a circle
            angles = [2 * math.pi * i / max(total_nodes, 1) for i in range(total_nodes)]
            radius = 3.5

            for i, angle in enumerate(angles):
                x = 5 + radius * math.cos(angle)
                y = 5 + radius * math.sin(angle)

                # Color by status
                if i < frame_data.falsified:
                    color = '#ff6b6b'  # Red for falsified
                    label = 'Falsified'
                elif i < frame_data.falsified + frame_data.approved:
                    color = '#51cf66'  # Green for approved
                    label = 'Approved'
                else:
                    color = '#339af0'  # Blue for pending
                    label = 'Pending'

                circle = Circle((x, y), 0.3, color=color, alpha=0.8)
                ax.add_patch(circle)
                ax.text(x, y, str(i+1), ha='center', va='center', fontsize=8, color='white', weight='bold')

        # Legend
        legend_elements = [
            mpatches.Patch(color='#ff6b6b', label=f'Falsified ({frame_data.falsified})'),
            mpatches.Patch(color='#51cf66', label=f'Approved ({frame_data.approved})'),
            mpatches.Patch(color='#339af0', label=f'Pending ({total_nodes - frame_data.approved - frame_data.falsified})'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

        # 2. Statistics bars (top-right)
        ax = axes[0, 1]
        categories = ['Total', 'Approved', 'Falsified', 'Stage 1\nPassed', 'Stage 2\nPassed']
        values = [
            frame_data.total_ideas,
            frame_data.approved,
            frame_data.falsified,
            frame_data.stage1_passed,
            frame_data.stage2_passed,
        ]
        colors = ['#339af0', '#51cf66', '#ff6b6b', '#fcc419', '#cc5de8']

        bars = ax.bar(categories, values, color=colors, alpha=0.8)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title('Pipeline Statistics', fontsize=12)
        ax.set_ylim(0, max(10, max(values) + 2))

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(val)}', ha='center', va='bottom', fontsize=9)

        # 3. Timeline progress (bottom-left)
        ax = axes[1, 0]

        # Timeline events
        timeline_y = 5
        ax.set_xlim(0, max(100, frame_data.timestamp + 10))
        ax.set_ylim(0, 10)
        ax.set_xlabel('Time (seconds)', fontsize=10)
        ax.set_title('Hypothesis Timeline', fontsize=12)

        # Draw timeline line
        ax.axhline(y=timeline_y, color='#495057', linewidth=2)

        # Draw events
        for run in frame_data.completed_runs:
            start_t = run.get("start_time", 0) - self.start_time
            end_t = run.get("end_time", 0) - self.start_time

            # Start marker
            ax.plot(start_t, timeline_y, 'o', color='#339af0', markersize=8)

            # Duration bar
            if end_t > start_t:
                verdict = run.get("verdict", "")
                color = '#51cf66' if "PASSED" in verdict else '#ff6b6b'
                ax.plot([start_t, end_t], [timeline_y, timeline_y], color=color, linewidth=3, alpha=0.7)

            # End marker
            ax.plot(end_t, timeline_y, 's', color=color, markersize=8)

        # Current time marker
        ax.axvline(x=frame_data.timestamp, color='#fa5252', linewidth=2, linestyle='--', label='Current')
        ax.legend(loc='upper right')

        # 4. Run status table (bottom-right)
        ax = axes[1, 1]
        ax.axis('off')
        ax.set_title('Current Run Status', fontsize=12)

        # Status text
        status_text = f"""
Active Runs: {len(frame_data.active_runs)}
Completed: {len(frame_data.completed_runs)}
Failed: {len(frame_data.failed_runs)}

Pipeline Progress:
- Stage 1 Gates: {frame_data.stage1_passed}/{frame_data.total_ideas}
- Stage 2 Falsifier: {frame_data.stage2_passed}/{frame_data.total_ideas}

Current Time: {frame_data.timestamp:.1f}s
Frame: {frame_number}
        """
        ax.text(0.1, 0.9, status_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.8))

        plt.tight_layout()

        # Save frame
        frame_path = self.output_dir / f"frame_{frame_number:05d}.png"
        plt.savefig(frame_path, dpi=100, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        return frame_path

    def generate_text_frame(self, frame_data: FrameData, frame_number: int) -> Path:
        """Generate a simple text-based frame."""
        frame_path = self.output_dir / f"frame_{frame_number:05d}.txt"

        content = f"""
================================================================================
KNOWLEDGE GRAPH EVOLUTION - FRAME {frame_number:05d}
Time: {frame_data.timestamp:.1f}s
================================================================================

STATISTICS:
  Total Ideas:      {frame_data.total_ideas}
  Approved:         {frame_data.approved}
  Falsified:        {frame_data.falsified}
  Stage 1 Passed:   {frame_data.stage1_passed}
  Stage 2 Passed:   {frame_data.stage2_passed}

RUN STATUS:
  Active:    {len(frame_data.active_runs)}
  Completed: {len(frame_data.completed_runs)}
  Failed:    {len(frame_data.failed_runs)}

ACTIVE RUNS:
"""
        for run in frame_data.active_runs:
            content += f"  - #{run.get('hypothesis_number', '?')}: {run.get('status', 'unknown')}\n"

        content += "\nCOMPLETED RUNS:\n"
        for run in frame_data.completed_runs:
            verdict = run.get('verdict', 'unknown')
            content += f"  - #{run.get('hypothesis_number', '?')}: {verdict}\n"

        content += "\n================================================================================\n"

        with open(frame_path, 'w') as f:
            f.write(content)

        return frame_path

    def generate_all_frames(self, max_frames: Optional[int] = None):
        """Generate all frames for the time-lapse."""
        print(f"Generating frames...")
        print(f"  Target duration: {self.target_duration/3600:.1f} hours")
        print(f"  FPS: {self.fps}")
        print(f"  Total frames: {self.total_frames}")

        # Determine actual time range
        if self.evolution:
            actual_duration = max(e.get("timestamp", 0) for e in self.evolution)
        else:
            actual_duration = 100  # Default

        print(f"  Actual experiment duration: {actual_duration:.1f}s")

        # Generate frames
        num_frames = min(max_frames, self.total_frames) if max_frames else min(100, self.total_frames)

        for i in range(num_frames):
            # Map frame to time
            progress = i / num_frames
            target_time = actual_duration * progress

            # Get state at this time
            frame_data = self.interpolate_state(target_time)
            frame_data.frame_number = i

            # Generate frame
            if HAS_MATPLOTLIB:
                frame_path = self.generate_matplotlib_frame(frame_data, i)
                if frame_path:
                    print(f"  Frame {i+1}/{num_frames}: {frame_path.name}")
            else:
                frame_path = self.generate_text_frame(frame_data, i)
                print(f"  Frame {i+1}/{num_frames}: {frame_path.name} (text)")

        print(f"\n✓ Generated {num_frames} frames in: {self.output_dir}")

        # Generate metadata
        metadata = {
            "total_frames": num_frames,
            "fps": self.fps,
            "target_duration_seconds": self.target_duration,
            "actual_duration_seconds": actual_duration,
            "has_matplotlib": HAS_MATPLOTLIB,
            "frame_pattern": "frame_%05d.png" if HAS_MATPLOTLIB else "frame_%05d.txt",
        }

        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ Metadata saved to: {metadata_path}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate visualization frames")
    parser.add_argument("--viz-data", type=Path, required=True,
                       help="Path to visualization_data.json")
    parser.add_argument("--output", type=Path, default=Path("frames"),
                       help="Output directory for frames")
    parser.add_argument("--max-frames", type=int, default=None,
                       help="Maximum number of frames to generate")

    args = parser.parse_args()

    if not args.viz_data.exists():
        print(f"Error: Visualization data not found: {args.viz_data}")
        sys.exit(1)

    generator = FrameGenerator(args.viz_data, args.output)
    generator.generate_all_frames(max_frames=args.max_frames)


if __name__ == "__main__":
    main()
