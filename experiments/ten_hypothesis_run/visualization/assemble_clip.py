#!/usr/bin/env python3
"""Assemble frames into video clip using ffmpeg or alternative tools.

Creates a time-lapse video showing the 2-hour evolution of the knowledge graph.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, List


class ClipAssembler:
    """Assemble frames into video clip."""

    def __init__(self, frames_dir: Path, output_path: Path):
        self.frames_dir = frames_dir
        self.output_path = output_path
        self.metadata: Optional[dict] = None

        # Load metadata
        metadata_path = frames_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                self.metadata = json.load(f)

    def check_ffmpeg(self) -> bool:
        """Check if ffmpeg is available."""
        return shutil.which("ffmpeg") is not None

    def assemble_with_ffmpeg(self) -> bool:
        """Assemble frames using ffmpeg."""
        if not self.check_ffmpeg():
            print("✗ ffmpeg not found")
            return False

        fps = self.metadata.get("fps", 30) if self.metadata else 30
        frame_pattern = str(self.frames_dir / "frame_%05d.png")

        print(f"Assembling video with ffmpeg...")
        print(f"  Input: {frame_pattern}")
        print(f"  Output: {self.output_path}")
        print(f"  FPS: {fps}")

        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-framerate", str(fps),
            "-i", frame_pattern,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "23",  # Quality (lower is better)
            "-preset", "medium",
            str(self.output_path),
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✓ Video created: {self.output_path}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ ffmpeg failed: {e}")
            print(f"  stderr: {e.stderr}")
            return False

    def assemble_with_moviepy(self) -> bool:
        """Assemble frames using moviepy (Python alternative)."""
        try:
            from moviepy import ImageSequenceClip
            print("Using moviepy for assembly...")
        except ImportError:
            try:
                from moviepy.editor import ImageSequenceClip
                print("Using moviepy.editor for assembly...")
            except ImportError:
                print("✗ moviepy not available")
                return False

        # Find all frame files
        frame_files = sorted(self.frames_dir.glob("frame_*.png"))
        if not frame_files:
            print("✗ No frame files found")
            return False

        fps = self.metadata.get("fps", 30) if self.metadata else 30

        print(f"Loading {len(frame_files)} frames...")

        try:
            clip = ImageSequenceClip([str(f) for f in frame_files], fps=fps)
            clip.write_videofile(str(self.output_path), fps=fps, codec='libx264')
            print(f"✓ Video created: {self.output_path}")
            return True
        except Exception as e:
            print(f"✗ moviepy failed: {e}")
            return False

    def create_ffmpeg_script(self) -> Path:
        """Create a shell script for manual ffmpeg execution."""
        script_path = self.frames_dir.parent / "assemble_video.sh"

        fps = self.metadata.get("fps", 30) if self.metadata else 30
        frame_pattern = "visualization/frames/frame_%05d.png"
        output = "evolution_timelapse.mp4"

        script_content = f"""#!/bin/bash
# Assemble frames into video using ffmpeg
# Run this script from the experiment directory

FPS={fps}
INPUT_PATTERN="{frame_pattern}"
OUTPUT="{output}"

echo "Assembling time-lapse video..."
echo "  FPS: $FPS"
echo "  Input: $INPUT_PATTERN"
echo "  Output: $OUTPUT"

ffmpeg -y \\
    -framerate $FPS \\
    -i "$INPUT_PATTERN" \\
    -c:v libx264 \\
    -pix_fmt yuv420p \\
    -crf 23 \\
    -preset medium \\
    "$OUTPUT"

echo "✓ Video created: $OUTPUT"
"""

        with open(script_path, 'w') as f:
            f.write(script_content)

        script_path.chmod(0o755)
        print(f"✓ Created assembly script: {script_path}")
        return script_path

    def generate_html_viewer(self) -> Path:
        """Generate an HTML viewer for the frames."""
        html_path = self.frames_dir.parent / "viewer.html"

        # Find all frames
        frame_files = sorted(self.frames_dir.glob("frame_*.png"))
        if not frame_files:
            print("✗ No frames found for HTML viewer")
            return None

        # Generate image tags
        images_html = ""
        for i, frame in enumerate(frame_files):
            rel_path = f"visualization/frames/{frame.name}"
            images_html += f'    <img src="{rel_path}" alt="Frame {i}" data-frame="{i}">\n'

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Knowledge Graph Evolution Viewer</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }}
        header {{
            background: #16213e;
            padding: 1rem 2rem;
            border-bottom: 2px solid #0f3460;
        }}
        h1 {{ font-size: 1.5rem; color: #e94560; }}
        .subtitle {{ color: #888; font-size: 0.9rem; margin-top: 0.25rem; }}
        .container {{
            flex: 1;
            display: flex;
            padding: 1rem;
            gap: 1rem;
        }}
        .viewer {{
            flex: 2;
            background: #0f3460;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
            overflow: hidden;
        }}
        .viewer img {{
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
            display: none;
        }}
        .viewer img.active {{ display: block; }}
        .controls {{
            flex: 1;
            background: #16213e;
            border-radius: 8px;
            padding: 1rem;
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }}
        .control-group {{
            background: #1a1a2e;
            padding: 1rem;
            border-radius: 6px;
        }}
        .control-group h3 {{
            color: #e94560;
            font-size: 0.9rem;
            margin-bottom: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        button {{
            background: #e94560;
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.9rem;
            transition: background 0.2s;
            width: 100%;
            margin-bottom: 0.5rem;
        }}
        button:hover {{ background: #ff6b6b; }}
        button.secondary {{
            background: #0f3460;
        }}
        button.secondary:hover {{ background: #1a4a7a; }}
        .slider-container {{
            margin-top: 0.5rem;
        }}
        input[type="range"] {{
            width: 100%;
            margin-top: 0.5rem;
        }}
        .frame-info {{
            font-family: 'Monaco', 'Consolas', monospace;
            font-size: 0.85rem;
            color: #aaa;
        }}
        .stats {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.5rem;
        }}
        .stat {{
            background: #0f3460;
            padding: 0.75rem;
            border-radius: 4px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 1.5rem;
            font-weight: bold;
            color: #e94560;
        }}
        .stat-label {{
            font-size: 0.75rem;
            color: #888;
            margin-top: 0.25rem;
        }}
        .instructions {{
            background: #1a1a2e;
            padding: 1rem;
            border-radius: 6px;
            font-size: 0.85rem;
            color: #aaa;
        }}
        .instructions code {{
            background: #0f3460;
            padding: 0.2rem 0.4rem;
            border-radius: 3px;
            color: #e94560;
        }}
    </style>
</head>
<body>
    <header>
        <h1>🔬 Knowledge Graph Evolution</h1>
        <div class="subtitle">10 Hypothesis Experiment - 2 Hour Time-lapse</div>
    </header>

    <div class="container">
        <div class="viewer">
{images_html}        </div>

        <div class="controls">
            <div class="control-group">
                <h3>Playback</h3>
                <button id="playBtn">▶ Play</button>
                <button id="pauseBtn" class="secondary" style="display:none">⏸ Pause</button>
                <div class="slider-container">
                    <label>Frame: <span id="frameNum">0</span> / {len(frame_files)}</label>
                    <input type="range" id="frameSlider" min="0" max="{len(frame_files)-1}" value="0">
                </div>
            </div>

            <div class="control-group">
                <h3>Statistics</h3>
                <div class="stats">
                    <div class="stat">
                        <div class="stat-value" id="totalIdeas">-</div>
                        <div class="stat-label">Total Ideas</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value" id="approved">-</div>
                        <div class="stat-label">Approved</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value" id="falsified">-</div>
                        <div class="stat-label">Falsified</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value" id="passed">-</div>
                        <div class="stat-label">Passed</div>
                    </div>
                </div>
            </div>

            <div class="control-group">
                <h3>Export</h3>
                <button id="exportScriptBtn" class="secondary">📹 Generate Video Script</button>
            </div>

            <div class="instructions">
                <strong>Instructions:</strong><br><br>
                Use the slider or Play button to navigate through the evolution.<br><br>
                To create a video, click "Generate Video Script" and run the command in your terminal.
            </div>
        </div>
    </div>

    <script>
        const images = document.querySelectorAll('.viewer img');
        const slider = document.getElementById('frameSlider');
        const frameNum = document.getElementById('frameNum');
        const playBtn = document.getElementById('playBtn');
        const pauseBtn = document.getElementById('pauseBtn');

        let currentFrame = 0;
        let isPlaying = false;
        let playInterval;

        function showFrame(n) {{
            images.forEach((img, i) => {{
                img.classList.toggle('active', i === n);
            }});
            currentFrame = n;
            slider.value = n;
            frameNum.textContent = n;

            // Update stats from metadata (would need to be populated)
            updateStats(n);
        }}

        function updateStats(frame) {{
            // This would be populated with actual data
            document.getElementById('totalIdeas').textContent = Math.floor(frame / 10);
        }}

        slider.addEventListener('input', (e) => {{
            showFrame(parseInt(e.target.value));
        }});

        playBtn.addEventListener('click', () => {{
            isPlaying = true;
            playBtn.style.display = 'none';
            pauseBtn.style.display = 'block';

            playInterval = setInterval(() => {{
                let next = currentFrame + 1;
                if (next >= images.length) next = 0;
                showFrame(next);
            }}, 100);  // 10 fps playback
        }});

        pauseBtn.addEventListener('click', () => {{
            isPlaying = false;
            clearInterval(playInterval);
            playBtn.style.display = 'block';
            pauseBtn.style.display = 'none';
        }});

        document.getElementById('exportScriptBtn').addEventListener('click', () => {{
            alert('Run this command in the terminal to create the video:\\n\\n' +
                  'bash assemble_video.sh');
        }});

        // Show first frame
        showFrame(0);
    </script>
</body>
</html>
"""

        with open(html_path, 'w') as f:
            f.write(html_content)

        print(f"✓ Created HTML viewer: {html_path}")
        print(f"  Open this file in a browser to view the evolution")
        return html_path

    def assemble(self, method: str = "auto") -> bool:
        """Assemble clip using best available method."""
        print(f"\nAssembling video clip...")
        print(f"  Frames: {self.frames_dir}")
        print(f"  Output: {self.output_path}")

        if method == "auto":
            if self.check_ffmpeg():
                method = "ffmpeg"
            else:
                method = "moviepy"

        success = False

        if method == "ffmpeg":
            success = self.assemble_with_ffmpeg()
        elif method == "moviepy":
            success = self.assemble_with_moviepy()

        if not success:
            # Create fallback script
            print("\n⚠ Video assembly failed, creating fallback options...")
            self.create_ffmpeg_script()
            self.generate_html_viewer()
            return False

        return success


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Assemble frames into video")
    parser.add_argument("--frames", type=Path, required=True,
                       help="Directory containing frame images")
    parser.add_argument("--output", type=Path, default=Path("evolution_timelapse.mp4"),
                       help="Output video path")
    parser.add_argument("--method", choices=["auto", "ffmpeg", "moviepy"], default="auto",
                       help="Assembly method")

    args = parser.parse_args()

    if not args.frames.exists():
        print(f"Error: Frames directory not found: {args.frames}")
        sys.exit(1)

    assembler = ClipAssembler(args.frames, args.output)
    success = assembler.assemble(method=args.method)

    if success:
        print("\n✓ Video assembly complete!")
    else:
        print("\n⚠ Video assembly incomplete - check fallback options above")
        sys.exit(1)


if __name__ == "__main__":
    main()
