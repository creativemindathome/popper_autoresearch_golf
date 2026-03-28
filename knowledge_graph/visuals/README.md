# Knowledge Graph Visualizations

Professional visualization suite for the Parameter Golf knowledge graph.

## Quick Start

```bash
# Generate all visualizations
cd knowledge_graph/visuals && python3 generate_all.py

# Or generate specific styles
python3 kg_executive.py --with-hypotheses
python3 kg_premium.py --theme dark
```

## Visualization Styles

### 1. **Executive** (`kg_executive.py`) - RECOMMENDED FOR PITCHES

**Best for:** Investor presentations, executive briefings, slide decks

- Three clean columns (Data Pipeline, Neural Network, Training & Evaluation)
- Clear visual hierarchy with color-coded pillars
- Professional typography (SF Pro/Inter)
- Box-style nodes with proper labels
- Optimized for 16:9 presentation slides

```bash
python3 knowledge_graph/visuals/kg_executive.py
python3 knowledge_graph/visuals/kg_executive.py --with-hypotheses
python3 knowledge_graph/visuals/kg_executive.py --dpi 300 --title "Your Title"
```

**Outputs:**
- `kg_executive.png` - Clean hierarchy view
- `kg_executive_with_research.png` - With active hypotheses

### 2. **Premium** (`kg_premium.py`)

**Best for:** Publications, research papers, technical documentation

- Hierarchical tree layout
- Light and dark themes
- Publication-ready color palette (colorblind-safe)
- Orthogonal edge routing
- High-resolution output (300 DPI print-ready)

```bash
python3 knowledge_graph/visuals/kg_premium.py --theme light
python3 knowledge_graph/visuals/kg_premium.py --theme dark --with-hypotheses
python3 knowledge_graph/visuals/kg_premium.py --dpi 600  # For print
```

**Outputs:**
- `kg_premium.png` - Light theme
- `kg_premium_dark.png` - Dark theme for screen presentations
- `kg_premium_dark_with_research.png` - With hypotheses

### 3. **Pitch Quality** (`kg_pitch_quality.py`)

**Best for:** Research team presentations, academic conferences

- Force-directed with collision avoidance
- Paul Tol's accessible color palette
- Dark mode option
- Hypothesis overlay with status indicators
- Legend included

```bash
python3 knowledge_graph/visuals/kg_pitch_quality.py
python3 knowledge_graph/visuals/kg_pitch_quality.py --dark-mode
```

### 4. **Unified** (`knowledge_graph_visualizer.py`)

**Best for:** Full system overview with clusters

- Hierarchical cluster layout
- Subgraph clusters for each pillar
- Configurable label visibility
- Legend showing node types

```bash
python3 knowledge_graph/visuals/knowledge_graph_visualizer.py --labels all
python3 knowledge_graph/visuals/knowledge_graph_visualizer.py --with-hypotheses
```

### 5. **Bubble** (`kg_bubble_visualizer.py`)

**Best for:** Network-style exploration

- Circular bubble clusters
- Radial layout
- H1-style hypothesis nodes
- Gephi-like aesthetic

## Color Coding

### Pillars (Three Main Categories)

| Pillar | Primary | Light | Use Case |
|--------|---------|-------|----------|
| **Data Pipeline** | Blue (#2563eb) | Light blue | Data sources, tokenization, cleaning |
| **Neural Network** | Red (#dc2626) | Light red | Architecture, attention, MLP, embeddings |
| **Training & Eval** | Green (#16a34a) | Light green | Loss, optimizer, precision, evaluation |

### Node Types

- **Root** (Dark fill): Main pillars at the top
- **Branch** (Medium fill): Component categories
- **Leaf** (Light fill): Specific techniques/options

### Hypothesis Status

- **Pending** (Amber): New research ideas not yet tested
- **Verified** (Emerald): Successfully validated hypotheses
- **Refuted** (Red): Failed validation

## Auto-Extension with Research

All visualizers automatically:

1. **Read ideas** from `knowledge_graph/outbox/ideator/*.json`
2. **Fetch verdicts** from `knowledge_graph/outbox/falsifier/*_result.json`
3. **Connect hypotheses** to relevant knowledge nodes by keyword matching
4. **Style by status** (pending/verified/refuted)

Simply run any visualizer with `--with-hypotheses` flag after generating new research.

## Requirements

```bash
# macOS
brew install graphviz

# Ubuntu/Debian
sudo apt-get install graphviz

# Verify installation
dot -V
```

## Output Reference

```
knowledge_graph/visuals/
├── kg_executive.png                    ← Recommended for pitches
├── kg_executive_with_research.png
├── kg_premium.png
├── kg_premium_dark.png                 ← Best for dark presentations
├── kg_premium_dark_with_research.png
├── kg_pitch.png
├── kg_pitch_dark.png
├── kg_unified.png
├── kg_bubble.png
└── README.md
```

## DPI Guidelines

| Use Case | DPI | Command |
|----------|-----|---------|
| Web/Screens | 150 | `--dpi 150` |
| Presentations | 300 | `--dpi 300` (default) |
| Print/Posters | 600 | `--dpi 600` |

## Customization

### Change Title
```bash
python3 kg_executive.py --title "OpenAI Parameter Golf: System Architecture"
```

### Change Label Visibility
```bash
python3 kg_premium.py --labels roots           # Only pillar labels
python3 kg_premium.py --labels roots-branches  # Pillars + components (default)
python3 kg_premium.py --labels all             # All labels
```

### Select Theme
```bash
python3 kg_premium.py --theme light  # Clean white background
python3 kg_premium.py --theme dark   # Dark mode for presentations
```

## Legacy Visualizers

Original visualizers still available:
- `render_seed_kg_graphviz.py` - Tree-style hierarchical
- `render_seed_kg_force_graphviz.py` - Force-directed "hairball"

## Tips for Pitches

1. **For OpenAI Research Team:**
   - Use `kg_executive.png` for overview slides
   - Use `kg_premium_dark.png` for technical deep-dives
   - Include `kg_executive_with_research.png` to show active work

2. **File Formats:**
   - PNG: For presentations (PowerPoint, Keynote)
   - SVG: For web, editing in Illustrator/Figma
   - DOT: For manual tweaking

3. **Dimensions:**
   - Executive: Optimized for 1920x1080 slides
   - Premium: Square-ish, good for papers
   - All: Scalable via DPI setting
