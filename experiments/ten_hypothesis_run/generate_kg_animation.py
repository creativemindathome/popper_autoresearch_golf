#!/usr/bin/env python3
"""Generate animated knowledge graph evolution visualization.

Matches the original knowledge graph visual style with:
- RootBoxes at the top
- Branches as colored clusters  
- Leaves as nodes
- Animated hypotheses appearing and being falsified
"""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any
import colorsys


def hsl_to_hex(h: int, s: float, l: float) -> str:
    """Convert HSL to hex color."""
    r, g, b = colorsys.hls_to_rgb(h / 360.0, l, s)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def stable_hue(key: str) -> int:
    """Generate stable hue from string."""
    hue = 0
    for ch in key:
        hue = (hue * 31 + ord(ch)) % 360
    return hue


@dataclass
class EvolutionEvent:
    time: float
    event_type: str  # 'generate', 'stage1_start', 'stage1_kill', 'stage2_start', 'stage2_pass', 'stage2_kill'
    hypothesis_num: int
    theory_id: str
    details: Dict[str, Any]


def load_experiment_data(run_dir: Path) -> Dict:
    """Load experiment data from run directory."""
    summary_path = run_dir / "summary.json"
    viz_path = run_dir / "visualization" / "visualization_data.json"
    
    with open(viz_path) as f:
        viz_data = json.load(f)
    
    return viz_data


def build_events(viz_data: Dict) -> List[EvolutionEvent]:
    """Build chronological event list from timeline."""
    events = []
    timeline = viz_data.get("timeline", [])
    runs = viz_data.get("runs", [])
    
    for item in timeline:
        time = item.get("time", 0)
        hypothesis = item.get("hypothesis", 0)
        event_type = item.get("event", "")
        theory_id = item.get("theory_id", "")
        
        # Find corresponding run data
        run = None
        for r in runs:
            if r.get("hypothesis_number") == hypothesis:
                run = r
                break
        
        if event_type == "start":
            events.append(EvolutionEvent(
                time=time,
                event_type="generate",
                hypothesis_num=hypothesis,
                theory_id=theory_id,
                details={"run": run}
            ))
            # Immediately start stage 1
            events.append(EvolutionEvent(
                time=time + 0.1,
                event_type="stage1_start",
                hypothesis_num=hypothesis,
                theory_id=theory_id,
                details={"run": run}
            ))
        elif event_type == "complete":
            verdict = item.get("verdict", "")
            stage1_verdict = item.get("stage1_verdict")
            stage2_verdict = item.get("stage2_verdict")
            
            if verdict == "REFUTED" or stage1_verdict == "REFUTED":
                kill_reason = ""
                if run and run.get("stage1_result"):
                    kill_reason = run["stage1_result"].get("kill_reason", "")
                events.append(EvolutionEvent(
                    time=time,
                    event_type="stage1_kill",
                    hypothesis_num=hypothesis,
                    theory_id=theory_id,
                    details={"kill_reason": kill_reason, "run": run}
                ))
            elif stage1_verdict == "STAGE_1_PASSED":
                events.append(EvolutionEvent(
                    time=time,
                    event_type="stage2_start",
                    hypothesis_num=hypothesis,
                    theory_id=theory_id,
                    details={"run": run}
                ))
    
    return sorted(events, key=lambda e: e.time)


def generate_html(viz_data: Dict, events: List[EvolutionEvent], output_path: Path):
    """Generate animated HTML visualization."""
    
    # Build event data for JavaScript
    events_js = []
    for e in events:
        events_js.append({
            "time": e.time,
            "type": e.event_type,
            "hypothesis": e.hypothesis_num,
            "theory_id": e.theory_id,
            "details": e.details
        })
    
    # Create base structure matching original KG
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Knowledge Graph Evolution - Animated</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f8fafc;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }}
        header {{
            background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
            color: white;
            padding: 1.5rem 2rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }}
        h1 {{ font-size: 1.5rem; margin-bottom: 0.5rem; }}
        .subtitle {{ opacity: 0.8; font-size: 0.9rem; }}
        .main-container {{
            flex: 1;
            display: flex;
            padding: 1rem;
            gap: 1rem;
            overflow: hidden;
        }}
        #graph-area {{
            flex: 1;
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            position: relative;
            overflow: hidden;
        }}
        .controls-panel {{
            width: 320px;
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            padding: 1.5rem;
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }}
        .control-section {{
            border-bottom: 1px solid #e2e8f0;
            padding-bottom: 1rem;
        }}
        .control-section:last-child {{ border-bottom: none; }}
        .control-section h3 {{
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #64748b;
            margin-bottom: 1rem;
        }}
        .play-btn {{
            width: 100%;
            padding: 0.875rem;
            background: #3b82f6;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .play-btn:hover {{ background: #2563eb; }}
        .play-btn.playing {{ background: #22c55e; }}
        .timeline-container {{
            margin-top: 0.75rem;
        }}
        #timeline-slider {{
            width: 100%;
            height: 6px;
            -webkit-appearance: none;
            background: #e2e8f0;
            border-radius: 3px;
            outline: none;
        }}
        #timeline-slider::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 18px;
            height: 18px;
            border-radius: 50%;
            background: #3b82f6;
            cursor: pointer;
        }}
        .time-display {{
            display: flex;
            justify-content: space-between;
            margin-top: 0.5rem;
            font-size: 0.85rem;
            color: #64748b;
            font-family: 'Monaco', monospace;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.75rem;
        }}
        .stat-item {{
            background: #f1f5f9;
            padding: 0.75rem;
            border-radius: 6px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 1.5rem;
            font-weight: bold;
            color: #3b82f6;
        }}
        .stat-label {{
            font-size: 0.75rem;
            color: #64748b;
            margin-top: 0.25rem;
        }}
        .legend {{
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.85rem;
        }}
        .legend-dot {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }}
        .event-log {{
            max-height: 150px;
            overflow-y: auto;
            font-family: 'Monaco', monospace;
            font-size: 0.75rem;
            background: #f8fafc;
            padding: 0.75rem;
            border-radius: 6px;
            border: 1px solid #e2e8f0;
        }}
        .event-item {{
            padding: 0.25rem 0;
            border-bottom: 1px solid #e2e8f0;
        }}
        .event-item:last-child {{ border-bottom: none; }}
        .event-generate {{ color: #3b82f6; }}
        .event-kill {{ color: #ef4444; }}
        .event-pass {{ color: #22c55e; }}
        
        /* Graph styles */
        .node-root {{
            fill: #1e293b;
            stroke: white;
            stroke-width: 3px;
        }}
        .node-branch {{
            stroke: white;
            stroke-width: 2px;
        }}
        .node-leaf {{
            stroke: white;
            stroke-width: 1px;
        }}
        .node-hypothesis {{
            stroke: #3b82f6;
            stroke-width: 3px;
            stroke-dasharray: 5,5;
        }}
        .node-hypothesis.running {{
            stroke: #f59e0b;
            animation: pulse 1s infinite;
        }}
        .node-hypothesis.killed {{
            stroke: #ef4444;
            fill: #fee2e2;
        }}
        .node-hypothesis.passed {{
            stroke: #22c55e;
            fill: #dcfce7;
        }}
        @keyframes pulse {{
            0% {{ stroke-opacity: 1; }}
            50% {{ stroke-opacity: 0.5; }}
            100% {{ stroke-opacity: 1; }}
        }}
        .link {{
            stroke: #cbd5e1;
            stroke-width: 1.5px;
        }}
        .link-hypothesis {{
            stroke: #3b82f6;
            stroke-width: 2px;
            stroke-dasharray: 5,5;
        }}
        .node-label {{
            font-size: 11px;
            fill: #334155;
            pointer-events: none;
            text-anchor: middle;
            font-weight: 500;
        }}
        .node-label.root {{
            font-size: 13px;
            font-weight: bold;
            fill: white;
        }}
    </style>
</head>
<body>
    <header>
        <h1>🌳 Knowledge Graph Evolution</h1>
        <div class="subtitle">Animated falsification pipeline over time</div>
    </header>

    <div class="main-container">
        <div id="graph-area">
            <svg id="graph-svg"></svg>
        </div>

        <div class="controls-panel">
            <div class="control-section">
                <h3>Playback</h3>
                <button id="play-btn" class="play-btn">▶ Play Evolution</button>
                <div class="timeline-container">
                    <input type="range" id="timeline-slider" min="0" max="{len(events_js)-1}" value="0">
                    <div class="time-display">
                        <span id="current-time">00:00</span>
                        <span id="total-time">{format_time(max(e['time'] for e in events_js) if events_js else 0)}</span>
                    </div>
                </div>
            </div>

            <div class="control-section">
                <h3>Statistics</h3>
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-value" id="stat-generated">0</div>
                        <div class="stat-label">Generated</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="stat-running">0</div>
                        <div class="stat-label">Running</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="stat-passed">0</div>
                        <div class="stat-label">Passed</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="stat-killed">0</div>
                        <div class="stat-label">Killed</div>
                    </div>
                </div>
            </div>

            <div class="control-section">
                <h3>Legend</h3>
                <div class="legend">
                    <div class="legend-item">
                        <div class="legend-dot" style="background: #1e293b;"></div>
                        <span>Root</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-dot" style="background: #3b82f6; border: 2px dashed #3b82f6;"></div>
                        <span>Generated (pending)</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-dot" style="background: #f59e0b;"></div>
                        <span>Stage 1 Running</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-dot" style="background: #ef4444;"></div>
                        <span>Killed</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-dot" style="background: #22c55e;"></div>
                        <span>Passed</span>
                    </div>
                </div>
            </div>

            <div class="control-section">
                <h3>Event Log</h3>
                <div class="event-log" id="event-log">
                    Ready to play...
                </div>
            </div>
        </div>
    </div>

    <script>
        // Events data
        const events = {json.dumps(events_js)};
        
        // Base knowledge graph structure (matching seed KG style)
        const baseNodes = [
            // Roots
            {{id: "root_pipeline", label: "Data Pipeline", type: "RootBox", x: 200, y: 80, color: "#3b82f6"}},
            {{id: "root_network", label: "Neural Network", type: "RootBox", x: 500, y: 80, color: "#ef4444"}},
            {{id: "root_training", label: "Training", type: "RootBox", x: 800, y: 80, color: "#22c55e"}},
            
            // Branches
            {{id: "branch_data", label: "Data", type: "Branch", x: 150, y: 200, color: "{hsl_to_hex(stable_hue('data'), 0.75, 0.60)}"}},
            {{id: "branch_arch", label: "Architecture", type: "Branch", x: 450, y: 200, color: "{hsl_to_hex(stable_hue('arch'), 0.75, 0.60)}"}},
            {{id: "branch_train", label: "Optimization", type: "Branch", x: 750, y: 200, color: "{hsl_to_hex(stable_hue('train'), 0.75, 0.60)}"}},
            
            // Leaves (representing specific concepts)
            {{id: "leaf_web", label: "Web Text", type: "Leaf", x: 100, y: 320, color: "{hsl_to_hex(stable_hue('web'), 0.6, 0.75)}"}},
            {{id: "leaf_code", label: "Code", type: "Leaf", x: 200, y: 320, color: "{hsl_to_hex(stable_hue('code'), 0.6, 0.75)}"}},
            {{id: "leaf_attn", label: "Attention", type: "Leaf", x: 400, y: 320, color: "{hsl_to_hex(stable_hue('attn'), 0.6, 0.75)}"}},
            {{id: "leaf_ffn", label: "FFN", type: "Leaf", x: 500, y: 320, color: "{hsl_to_hex(stable_hue('ffn'), 0.6, 0.75)}"}},
            {{id: "leaf_opt", label: "Optimizer", type: "Leaf", x: 700, y: 320, color: "{hsl_to_hex(stable_hue('opt'), 0.6, 0.75)}"}},
            {{id: "leaf_loss", label: "Loss", type: "Leaf", x: 800, y: 320, color: "{hsl_to_hex(stable_hue('loss'), 0.6, 0.75)}"}},
        ];
        
        const baseLinks = [
            {{source: "root_pipeline", target: "branch_data"}},
            {{source: "root_network", target: "branch_arch"}},
            {{source: "root_training", target: "branch_train"}},
            {{source: "branch_data", target: "leaf_web"}},
            {{source: "branch_data", target: "leaf_code"}},
            {{source: "branch_arch", target: "leaf_attn"}},
            {{source: "branch_arch", target: "leaf_ffn"}},
            {{source: "branch_train", target: "leaf_opt"}},
            {{source: "branch_train", target: "leaf_loss"}},
        ];

        // Hypothesis nodes (will be added dynamically)
        let hypothesisNodes = [];
        let hypothesisLinks = [];
        
        // Set up SVG
        const svg = d3.select("#graph-svg")
            .attr("width", "100%")
            .attr("height", "100%");
        
        const width = document.getElementById('graph-area').clientWidth;
        const height = document.getElementById('graph-area').clientHeight;
        svg.attr("viewBox", `0 0 ${{width}} ${{height}}`);
        
        const g = svg.append("g");
        
        // Scale base structure to fit
        const scaleX = width / 1000;
        const scaleY = height / 450;
        const scale = Math.min(scaleX, scaleY) * 0.9;
        const offsetX = (width - 1000 * scale) / 2;
        const offsetY = (height - 450 * scale) / 2;
        
        baseNodes.forEach(n => {{
            n.sx = n.x * scale + offsetX;
            n.sy = n.y * scale + offsetY;
        }});
        
        // Draw base links
        const linkElements = g.selectAll(".link")
            .data(baseLinks)
            .enter()
            .append("line")
            .attr("class", "link")
            .attr("x1", d => baseNodes.find(n => n.id === d.source).sx)
            .attr("y1", d => baseNodes.find(n => n.id === d.source).sy)
            .attr("x2", d => baseNodes.find(n => n.id === d.target).sx)
            .attr("y2", d => baseNodes.find(n => n.id === d.target).sy);
        
        // Draw base nodes
        const baseNodeElements = g.selectAll(".base-node")
            .data(baseNodes)
            .enter()
            .append("g")
            .attr("class", "base-node")
            .attr("transform", d => `translate(${{d.sx}}, ${{d.sy}})`);
        
        baseNodeElements.append("circle")
            .attr("r", d => d.type === "RootBox" ? 30 : d.type === "Branch" ? 20 : 12)
            .attr("class", d => `node-${{d.type.toLowerCase()}}`)
            .attr("fill", d => d.color);
        
        baseNodeElements.append("text")
            .attr("class", d => `node-label ${{d.type === "RootBox" ? "root" : ""}}`)
            .attr("dy", d => d.type === "RootBox" ? 45 : d.type === "Branch" ? 35 : 25)
            .text(d => d.label);
        
        // Hypothesis layer
        const hypothesisG = g.append("g").attr("class", "hypothesis-layer");
        
        // Animation state
        let currentEvent = 0;
        let isPlaying = false;
        let animationInterval;
        
        function formatTime(seconds) {{
            const mins = Math.floor(seconds / 60);
            const secs = Math.floor(seconds % 60);
            return `${{mins.toString().padStart(2, '0')}}:${{secs.toString().padStart(2, '0')}}`;
        }}
        
        function logEvent(msg, type) {{
            const log = document.getElementById('event-log');
            const div = document.createElement('div');
            div.className = `event-item event-${{type}}`;
            div.textContent = msg;
            log.insertBefore(div, log.firstChild);
            if (log.children.length > 10) log.removeChild(log.lastChild);
        }}
        
        function updateStats() {{
            const generated = hypothesisNodes.length;
            const running = hypothesisNodes.filter(n => n.status === 'running').length;
            const killed = hypothesisNodes.filter(n => n.status === 'killed').length;
            const passed = hypothesisNodes.filter(n => n.status === 'passed').length;
            
            document.getElementById('stat-generated').textContent = generated;
            document.getElementById('stat-running').textContent = running;
            document.getElementById('stat-killed').textContent = killed;
            document.getElementById('stat-passed').textContent = passed;
        }}
        
        function renderEvent(index) {{
            if (index >= events.length) return;
            
            const event = events[index];
            document.getElementById('current-time').textContent = formatTime(event.time);
            document.getElementById('timeline-slider').value = index;
            
            if (event.type === 'generate') {{
                // Add new hypothesis node
                const branch = baseNodes.find(n => n.type === "Branch" && n.id.includes("arch"));
                const angle = (hypothesisNodes.length * 0.5) - 1;
                const radius = 120;
                const hx = branch.sx + Math.cos(angle) * radius;
                const hy = branch.sy + Math.sin(angle) * radius + 80;
                
                const node = {{
                    id: `hyp_${{event.hypothesis}}`,
                    label: `H${{event.hypothesis}}`,
                    fullId: event.theory_id,
                    x: hx,
                    y: hy,
                    status: 'pending',
                    hypothesis: event.hypothesis
                }};
                hypothesisNodes.push(node);
                
                const link = {{source: branch.id, target: node.id, type: 'hypothesis'}};
                hypothesisLinks.push(link);
                
                // Draw link
                hypothesisG.append("line")
                    .attr("class", "link-hypothesis")
                    .attr("x1", branch.sx)
                    .attr("y1", branch.sy)
                    .attr("x2", hx)
                    .attr("y2", hy)
                    .attr("id", `link_${{node.id}}`);
                
                // Draw node
                const nodeG = hypothesisG.append("g")
                    .attr("class", "hypothesis-node")
                    .attr("transform", `translate(${{hx}}, ${{hy}})`)
                    .attr("id", node.id);
                
                nodeG.append("circle")
                    .attr("r", 18)
                    .attr("class", "node-hypothesis")
                    .attr("fill", "#dbeafe");
                
                nodeG.append("text")
                    .attr("class", "node-label")
                    .attr("dy", 30)
                    .text(node.label);
                
                logEvent(`Generated H${{event.hypothesis}}: ${{event.theory_id.substring(0, 30)}}...`, 'generate');
                
            }} else if (event.type === 'stage1_start') {{
                const node = hypothesisNodes.find(n => n.hypothesis === event.hypothesis);
                if (node) {{
                    node.status = 'running';
                    d3.select(`#${{node.id}} circle`)
                        .attr("class", "node-hypothesis running");
                }}
                
            }} else if (event.type === 'stage1_kill') {{
                const node = hypothesisNodes.find(n => n.hypothesis === event.hypothesis);
                if (node) {{
                    node.status = 'killed';
                    d3.select(`#${{node.id}} circle`)
                        .transition()
                        .duration(500)
                        .attr("class", "node-hypothesis killed");
                    
                    logEvent(`Killed H${{event.hypothesis}}: ${{event.details.kill_reason?.substring(0, 40) || 'Stage 1 failed'}}...`, 'kill');
                }}
                
            }} else if (event.type === 'stage2_start') {{
                const node = hypothesisNodes.find(n => n.hypothesis === event.hypothesis);
                if (node) {{
                    node.status = 'running';
                    d3.select(`#${{node.id}} circle`)
                        .attr("class", "node-hypothesis running");
                }}
                
            }} else if (event.type === 'stage2_pass') {{
                const node = hypothesisNodes.find(n => n.hypothesis === event.hypothesis);
                if (node) {{
                    node.status = 'passed';
                    d3.select(`#${{node.id}} circle`)
                        .transition()
                        .duration(500)
                        .attr("class", "node-hypothesis passed");
                    
                    logEvent(`H${{event.hypothesis}} passed Stage 2!`, 'pass');
                }}
            }}
            
            updateStats();
        }}
        
        function play() {{
            if (isPlaying) return;
            isPlaying = true;
            document.getElementById('play-btn').textContent = '⏸ Pause';
            document.getElementById('play-btn').classList.add('playing');
            
            animationInterval = setInterval(() => {{
                if (currentEvent < events.length) {{
                    renderEvent(currentEvent);
                    currentEvent++;
                }} else {{
                    stop();
                }}
            }}, 1500);
        }}
        
        function stop() {{
            isPlaying = false;
            clearInterval(animationInterval);
            document.getElementById('play-btn').textContent = '▶ Play Evolution';
            document.getElementById('play-btn').classList.remove('playing');
        }}
        
        function reset() {{
            stop();
            currentEvent = 0;
            hypothesisNodes = [];
            hypothesisLinks = [];
            hypothesisG.selectAll("*").remove();
            document.getElementById('event-log').innerHTML = 'Ready to play...';
            document.getElementById('current-time').textContent = '00:00';
            document.getElementById('timeline-slider').value = 0;
            updateStats();
        }}
        
        function goToEvent(index) {{
            stop();
            reset();
            for (let i = 0; i <= index && i < events.length; i++) {{
                renderEvent(i);
            }}
            currentEvent = index + 1;
        }}
        
        // Controls
        document.getElementById('play-btn').addEventListener('click', () => {{
            if (isPlaying) stop(); else play();
        }});
        
        document.getElementById('timeline-slider').addEventListener('input', (e) => {{
            goToEvent(parseInt(e.target.value));
        }});
        
        // Initialize
        updateStats();
    </script>
</body>
</html>
'''
    
    with open(output_path, 'w') as f:
        f.write(html)
    
    print(f"✓ Generated: {output_path}")


def format_time(seconds: float) -> str:
    """Format seconds as MM:SS."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def main():
    """Generate animation for the experiment."""
    run_dir = Path("/Users/curiousmind/Desktop/null_fellow_hackathon/experiments/ten_hypothesis_run/live_experiment_20260328_163530")
    output_path = run_dir / "kg_evolution_animated.html"
    
    print("Loading experiment data...")
    viz_data = load_experiment_data(run_dir)
    
    print("Building event timeline...")
    events = build_events(viz_data)
    
    print(f"Found {len(events)} events")
    
    print("Generating animated visualization...")
    generate_html(viz_data, events, output_path)
    
    print(f"\n✓ Animation complete!")
    print(f"Open: {output_path}")


if __name__ == "__main__":
    main()
