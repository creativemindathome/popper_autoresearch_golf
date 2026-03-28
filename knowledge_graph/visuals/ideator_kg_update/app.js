(function () {
  "use strict";

  const COLORS = {
    bg: "#0b1220",
    seedEdge: "rgba(255,255,255,0.10)",
    buildsOnEdge: "rgba(96,165,250,0.35)",
    similarEdge: "rgba(167,139,250,0.35)",
    revisionEdge: "rgba(245,158,11,0.40)",
    mentionsEdge: "rgba(148,163,184,0.28)",
    root: "#fbbf24",
    branch: "#60a5fa",
    leaf: "#9ca3af",
    ideaPending: "#a78bfa",
    ideaApproved: "#22c55e",
    ideaRevise: "#f59e0b",
    ideaKnown: "#94a3b8",
    label: "rgba(255,255,255,0.85)",
    labelMuted: "rgba(255,255,255,0.62)",
  };

  const UI = {
    canvas: document.getElementById("graph"),
    tooltip: document.getElementById("tooltip"),
    stepTitle: document.getElementById("stepTitle"),
    stepSubtitle: document.getElementById("stepSubtitle"),
    stepIndex: document.getElementById("stepIndex"),
    stepDuration: document.getElementById("stepDuration"),
    stepNotes: document.getElementById("stepNotes"),
    btnPlay: document.getElementById("btnPlay"),
    btnPause: document.getElementById("btnPause"),
    btnPrev: document.getElementById("btnPrev"),
    btnNext: document.getElementById("btnNext"),
    btnDemo: document.getElementById("btnDemo"),
    btnLoad: document.getElementById("btnLoad"),
    fileInput: document.getElementById("fileInput"),
    speed: document.getElementById("speed"),
    speedVal: document.getElementById("speedVal"),
  };

  const BUILTIN_DEMO_TIMELINE = {
    schema_version: "knowledge_graph.ideator_visual_timeline.v1",
    generated_at: new Date().toISOString(),
    seed: {
      nodes: [
        { id: "node_root_data_pipeline", label: "Data Pipeline", type: "RootBox", status: "BASE_KNOWLEDGE" },
        { id: "node_root_neural_network", label: "Neural Network", type: "RootBox", status: "BASE_KNOWLEDGE" },
        { id: "node_root_training_eval", label: "Training & Evaluation", type: "RootBox", status: "BASE_KNOWLEDGE" },

        { id: "node_data_sources", label: "Data Sources", type: "Branch", status: "BASE_KNOWLEDGE" },
        { id: "node_data_source_web_text", label: "Web Text", type: "Leaf", status: "BASE_KNOWLEDGE" },
        { id: "node_data_source_code", label: "Code", type: "Leaf", status: "BASE_KNOWLEDGE" },

        { id: "node_mlp_variant", label: "MLP Variant", type: "Branch", status: "BASE_KNOWLEDGE" },
        { id: "node_mlp_low_rank", label: "Low-rank / Factorized MLP", type: "Leaf", status: "BASE_KNOWLEDGE" },
        { id: "node_mlp_moe", label: "Mixture-of-Experts (Sparse)", type: "Leaf", status: "BASE_KNOWLEDGE" },
        { id: "node_embed_factorized", label: "Factorized Embeddings", type: "Leaf", status: "BASE_KNOWLEDGE" },
        { id: "node_head_adaptive_softmax", label: "Adaptive / Factorized Softmax", type: "Leaf", status: "BASE_KNOWLEDGE" },

        { id: "node_optimizer_state_strategy", label: "Optimizer State Strategy", type: "Branch", status: "BASE_KNOWLEDGE" },
        { id: "node_opt_adamw_8bit_state", label: "AdamW (8-bit moments)", type: "Leaf", status: "BASE_KNOWLEDGE" },
        { id: "node_opt_adafactor", label: "Adafactor (Factorized)", type: "Leaf", status: "BASE_KNOWLEDGE" },
        { id: "node_kv_cache_int8", label: "INT8 KV Cache", type: "Leaf", status: "BASE_KNOWLEDGE" },
      ],
      edges: [
        { source: "node_root_data_pipeline", target: "node_data_sources", kind: "seed" },
        { source: "node_data_sources", target: "node_data_source_web_text", kind: "seed" },
        { source: "node_data_sources", target: "node_data_source_code", kind: "seed" },

        { source: "node_root_neural_network", target: "node_mlp_variant", kind: "seed" },
        { source: "node_mlp_variant", target: "node_mlp_low_rank", kind: "seed" },
        { source: "node_mlp_variant", target: "node_mlp_moe", kind: "seed" },
        { source: "node_root_neural_network", target: "node_embed_factorized", kind: "seed" },
        { source: "node_root_neural_network", target: "node_head_adaptive_softmax", kind: "seed" },

        { source: "node_root_training_eval", target: "node_optimizer_state_strategy", kind: "seed" },
        { source: "node_optimizer_state_strategy", target: "node_opt_adamw_8bit_state", kind: "seed" },
        { source: "node_optimizer_state_strategy", target: "node_opt_adafactor", kind: "seed" },
        { source: "node_root_training_eval", target: "node_kv_cache_int8", kind: "seed" },
      ],
    },
    steps: [
      {
        title: "Load knowledge graph",
        subtitle: "Seed: 15 nodes, 12 edges",
        duration_ms: 1500,
        actions: [
          {
            type: "highlight",
            ids: ["node_root_data_pipeline", "node_root_neural_network", "node_root_training_eval"],
            style: "pulse",
          },
        ],
      },
      {
        title: "Existing ideas in the knowledge graph",
        subtitle: "Low-Rank Factorized Transformer Layers",
        duration_ms: 2100,
        actions: [
          {
            type: "add_node",
            node: {
              id: "idea_low-rank-transformer-layers",
              label: "Low-Rank Factorized Transformer Layers",
              type: "Idea",
              status: "KNOWN",
              meta: {
                idea_id: "low-rank-transformer-layers",
                source_path: "DEMO",
                novelty_summary: "Factorize dense layers into low-rank factors to save params and optimizer state.",
              },
            },
          },
          { type: "add_edge", edge: { source: "idea_low-rank-transformer-layers", target: "node_mlp_low_rank", kind: "mentions" } },
          { type: "add_edge", edge: { source: "idea_low-rank-transformer-layers", target: "node_embed_factorized", kind: "mentions" } },
          { type: "highlight", ids: ["node_mlp_low_rank", "node_embed_factorized"], style: "glow" },
        ],
      },
      {
        title: "Scan: retrieve relevant concepts",
        subtitle: "Token‑Modulated Prototypes (Discrete + Low‑Rank)",
        duration_ms: 1400,
        actions: [
          { type: "highlight", ids: ["node_mlp_moe", "node_mlp_low_rank", "idea_low-rank-transformer-layers"], style: "glow" },
        ],
      },
      {
        title: "Ideate: propose new node",
        subtitle: "token-modulated-prototypes",
        duration_ms: 2200,
        actions: [
          {
            type: "add_node",
            node: {
              id: "idea_token-modulated-prototypes",
              label: "Token‑Modulated Prototypes (Discrete + Low‑Rank)",
              type: "Idea",
              status: "PENDING_REVIEW",
              meta: {
                idea_id: "token-modulated-prototypes",
                source_path: "DEMO",
                novelty_summary: "Each token selects a prototype weight (discrete routing) and applies a tiny low‑rank modulation.",
              },
            },
          },
          { type: "add_edge", edge: { source: "idea_token-modulated-prototypes", target: "node_mlp_moe", kind: "builds_on" } },
          { type: "add_edge", edge: { source: "idea_token-modulated-prototypes", target: "node_mlp_low_rank", kind: "builds_on" } },
          { type: "add_edge", edge: { source: "idea_token-modulated-prototypes", target: "idea_low-rank-transformer-layers", kind: "similar_to" } },
        ],
      },
      {
        title: "Review: revise",
        subtitle: "Novelty score: 5",
        duration_ms: 2400,
        actions: [
          { type: "set_status", id: "idea_token-modulated-prototypes", status: "REVISE" },
          { type: "highlight", ids: ["idea_token-modulated-prototypes"], style: "pulse" },
        ],
        notes: {
          primary_reasons: [
            "Resembles MoE routing plus low-rank adapters; novelty unclear.",
            "Selection mechanism needs a clearer advantage over standard gating.",
            "Make the falsifiable win concrete (bbp target, memory delta).",
          ],
          revision_instructions:
            "Clarify what is genuinely new vs existing MoE/LoRA patterns, and define one measurable win (compressed size vs val_bpb).",
        },
      },
      {
        title: "Scan: retrieve relevant concepts",
        subtitle: "Adaptive Representation Strategy (QLT → LRF → FP)",
        duration_ms: 1400,
        actions: [
          { type: "highlight", ids: ["node_kv_cache_int8", "node_mlp_low_rank", "idea_token-modulated-prototypes"], style: "glow" },
        ],
      },
      {
        title: "Ideate: propose new node",
        subtitle: "adaptive-representation-strategy",
        duration_ms: 2200,
        actions: [
          {
            type: "add_node",
            node: {
              id: "idea_adaptive-representation-strategy",
              label: "Adaptive Representation Strategy (QLT → LRF → FP)",
              type: "Idea",
              status: "PENDING_REVIEW",
              meta: {
                idea_id: "adaptive-representation-strategy",
                source_path: "DEMO",
                novelty_summary:
                  "Start heavily compressed, then promote only bottleneck layers to low‑rank and full precision based on gradient signals.",
              },
            },
          },
          { type: "add_edge", edge: { source: "idea_adaptive-representation-strategy", target: "node_kv_cache_int8", kind: "builds_on" } },
          { type: "add_edge", edge: { source: "idea_adaptive-representation-strategy", target: "node_mlp_low_rank", kind: "builds_on" } },
          { type: "add_edge", edge: { source: "idea_adaptive-representation-strategy", target: "idea_token-modulated-prototypes", kind: "similar_to" } },
          { type: "add_edge", edge: { source: "idea_token-modulated-prototypes", target: "idea_adaptive-representation-strategy", kind: "revision" } },
        ],
      },
      {
        title: "Review: pass",
        subtitle: "Novelty score: 7",
        duration_ms: 2400,
        actions: [
          { type: "set_status", id: "idea_adaptive-representation-strategy", status: "APPROVED" },
          { type: "highlight", ids: ["idea_adaptive-representation-strategy"], style: "pulse" },
        ],
        notes: {
          primary_reasons: [
            "Clearer falsifiability: explicit promotion triggers and expected memory/val_bpb movement.",
            "Adds an adaptive mechanism vs a fixed compression choice.",
          ],
        },
      },
    ],
  };

  function easeOutCubic(t) {
    const x = Math.max(0, Math.min(1, t));
    return 1 - Math.pow(1 - x, 3);
  }

  function clamp(v, lo, hi) {
    return Math.max(lo, Math.min(hi, v));
  }

  function nowMs() {
    return performance.now();
  }

  function formatMs(ms) {
    if (!Number.isFinite(ms)) return "—";
    return `${Math.round(ms)}ms`;
  }

  function byLabel(labelById) {
    return (a, b) => (labelById.get(a) || a).localeCompare(labelById.get(b) || b);
  }

  function setTooltipVisible(visible) {
    if (visible) UI.tooltip.classList.remove("hidden");
    else UI.tooltip.classList.add("hidden");
  }

  function setTooltipContent(title, subtitle) {
    UI.tooltip.innerHTML = `<div class="t-title"></div><div class="t-sub"></div>`;
    UI.tooltip.querySelector(".t-title").textContent = title || "";
    UI.tooltip.querySelector(".t-sub").textContent = subtitle || "";
  }

  function setTooltipPosition(x, y) {
    UI.tooltip.style.left = `${x}px`;
    UI.tooltip.style.top = `${y}px`;
  }

  function safeArray(v) {
    return Array.isArray(v) ? v : [];
  }

  function isTimelineObject(obj) {
    if (!obj || typeof obj !== "object") return false;
    const sv = obj.schema_version;
    if (typeof sv !== "string") return false;
    if (!sv.startsWith("knowledge_graph.ideator_visual_timeline.")) return false;
    const seed = obj.seed;
    if (!seed || typeof seed !== "object") return false;
    if (!Array.isArray(seed.nodes) || !Array.isArray(seed.edges)) return false;
    if (!Array.isArray(obj.steps)) return false;
    return true;
  }

  function readFileAsText(file) {
    if (!file) return Promise.reject(new Error("No file selected"));
    if (typeof file.text === "function") return file.text();
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onerror = () => reject(reader.error || new Error("FileReader error"));
      reader.onload = () => resolve(String(reader.result || ""));
      reader.readAsText(file);
    });
  }

  function isSeedType(t) {
    return t === "RootBox" || t === "Branch" || t === "Leaf";
  }

  function isIdeaType(t) {
    return t === "Idea";
  }

  function nodeRadius(node) {
    if (node.type === "RootBox") return 10;
    if (node.type === "Branch") return 7;
    if (node.type === "Leaf") return 4.5;
    if (node.type === "Idea") return 9;
    return 6;
  }

  function nodeFill(node) {
    if (node.type === "RootBox") return COLORS.root;
    if (node.type === "Branch") return COLORS.branch;
    if (node.type === "Leaf") return COLORS.leaf;
    if (node.type === "Idea") {
      if (node.status === "APPROVED") return COLORS.ideaApproved;
      if (node.status === "REVISE") return COLORS.ideaRevise;
      if (node.status === "KNOWN") return COLORS.ideaKnown;
      return COLORS.ideaPending;
    }
    return "#ffffff";
  }

  function edgeStroke(edge) {
    if (edge.kind === "seed") return COLORS.seedEdge;
    if (edge.kind === "builds_on") return COLORS.buildsOnEdge;
    if (edge.kind === "similar_to") return COLORS.similarEdge;
    if (edge.kind === "revision") return COLORS.revisionEdge;
    if (edge.kind === "mentions") return COLORS.mentionsEdge;
    return COLORS.seedEdge;
  }

  function edgeWidth(edge) {
    if (edge.kind === "seed") return 1;
    if (edge.kind === "revision") return 2.0;
    return 1.4;
  }

  function resetCanvasSize() {
    const dpr = window.devicePixelRatio || 1;
    const rect = UI.canvas.getBoundingClientRect();
    UI.canvas.width = Math.max(2, Math.floor(rect.width * dpr));
    UI.canvas.height = Math.max(2, Math.floor(rect.height * dpr));
    return { dpr, width: rect.width, height: rect.height };
  }

  function buildChildren(edges) {
    const children = new Map();
    for (const e of edges) {
      const s = String(e.source);
      const t = String(e.target);
      if (!children.has(s)) children.set(s, []);
      children.get(s).push(t);
    }
    return children;
  }

  function reachableFrom(rootId, children) {
    const seen = new Set();
    const queue = [rootId];
    seen.add(rootId);
    while (queue.length) {
      const cur = queue.shift();
      for (const kid of children.get(cur) || []) {
        if (seen.has(kid)) continue;
        seen.add(kid);
        queue.push(kid);
      }
    }
    return seen;
  }

  function computeDepths(rootId, children, withinSet) {
    const depth = new Map();
    depth.set(rootId, 0);
    const queue = [rootId];
    while (queue.length) {
      const cur = queue.shift();
      const d = depth.get(cur) || 0;
      for (const kid of children.get(cur) || []) {
        if (withinSet && !withinSet.has(kid)) continue;
        if (!depth.has(kid)) {
          depth.set(kid, d + 1);
          queue.push(kid);
        }
      }
    }
    return depth;
  }

  function computeTreeY(rootId, children, labelById, withinSet) {
    const yIndex = new Map();
    const visiting = new Set();
    let leafCursor = 0;

    function assign(nodeId) {
      if (yIndex.has(nodeId)) return yIndex.get(nodeId);
      if (visiting.has(nodeId)) {
        const fallback = leafCursor;
        leafCursor += 1;
        yIndex.set(nodeId, fallback);
        return fallback;
      }
      visiting.add(nodeId);
      const kids = (children.get(nodeId) || [])
        .filter((k) => !withinSet || withinSet.has(k))
        .slice()
        .sort(byLabel(labelById));
      let y;
      if (kids.length === 0) {
        y = leafCursor;
        leafCursor += 1;
      } else {
        const ys = kids.map(assign);
        y = ys.reduce((a, b) => a + b, 0) / ys.length;
      }
      visiting.delete(nodeId);
      yIndex.set(nodeId, y);
      return y;
    }

    assign(rootId);
    return { yIndex, leafCount: Math.max(leafCursor, 1) };
  }

  function layoutSeed(nodesById, seedEdges, viewport) {
    const seedNodes = Array.from(nodesById.values()).filter((n) => isSeedType(n.type));
    const labelById = new Map(seedNodes.map((n) => [n.id, n.label || n.id]));

    const children = buildChildren(seedEdges);
    const roots = seedNodes.filter((n) => n.type === "RootBox").slice().sort(byLabel(labelById));
    const rootIds = roots.map((r) => r.id);

    const width = viewport.width;
    const height = viewport.height;
    const dpr = viewport.dpr;

    const leftW = width * 0.68;
    const padX = 28;
    const padY = 28;
    const bandW = Math.max(220, (leftW - 2 * padX) / Math.max(1, rootIds.length));
    const depthDx = clamp(bandW / 3.2, 70, 110);

    for (let i = 0; i < rootIds.length; i++) {
      const rootId = rootIds[i];
      const within = reachableFrom(rootId, children);
      const depths = computeDepths(rootId, children, within);
      const { yIndex, leafCount } = computeTreeY(rootId, children, labelById, within);

      const top = padY + 44;
      const bottom = height - padY - 18;
      const yScale = (bottom - top) / Math.max(1, leafCount - 1);

      const bandLeft = padX + i * bandW;
      for (const nodeId of within) {
        const n = nodesById.get(nodeId);
        if (!n) continue;
        const depth = depths.get(nodeId) || 0;
        const yi = yIndex.get(nodeId) || 0;
        n.x = (bandLeft + 14 + depth * depthDx) * dpr;
        n.y = (top + yi * yScale) * dpr;
      }
    }

    const ideaLaneX = (leftW + (width - leftW) * 0.52) * dpr;
    return { ideaLaneX, leftW, roots: rootIds };
  }

  function computeIdeaSlotY(slotIndex, viewport) {
    const height = viewport.height;
    const dpr = viewport.dpr;
    const top = 90;
    const bottom = height - 50;
    const slotH = 64;
    const maxSlots = Math.max(1, Math.floor((bottom - top) / slotH));
    const wrapped = slotIndex % maxSlots;
    const column = Math.floor(slotIndex / maxSlots);
    const xOffset = column * 170;
    return { y: (top + wrapped * slotH) * dpr, xOffset: xOffset * dpr };
  }

  function rebuildGraphFromTimeline(timeline, uptoStepIndex, viewport) {
    const nodesById = new Map();
    const edges = [];
    const ideaOrder = [];

    for (const n of safeArray(timeline.seed?.nodes)) {
      nodesById.set(String(n.id), {
        id: String(n.id),
        label: String(n.label || n.id),
        type: String(n.type || "Node"),
        status: String(n.status || "BASE_KNOWLEDGE"),
        meta: n.meta || null,
        x: 0,
        y: 0,
        highlight: null,
        move: null,
      });
    }
    for (const e of safeArray(timeline.seed?.edges)) {
      edges.push({
        source: String(e.source),
        target: String(e.target),
        kind: String(e.kind || "seed"),
        addedAt: 0,
      });
    }

    const seedLayout = layoutSeed(nodesById, edges.filter((e) => e.kind === "seed"), viewport);

    function applyAction(action, stepDurationMs, tNow) {
      if (!action || typeof action !== "object") return;
      if (action.type === "add_node") {
        const node = action.node || {};
        const nodeId = String(node.id || "");
        if (!nodeId) return;
        const n = {
          id: nodeId,
          label: String(node.label || nodeId),
          type: String(node.type || "Idea"),
          status: String(node.status || "PENDING_REVIEW"),
          meta: node.meta || null,
          x: 0,
          y: 0,
          highlight: null,
          move: null,
          slotIndex: null,
        };
        const spawnX = 90 * viewport.dpr;
        const spawnY = 70 * viewport.dpr;
        n.x = spawnX;
        n.y = spawnY;

        if (isIdeaType(n.type)) {
          n.slotIndex = ideaOrder.length;
          ideaOrder.push(n.id);
          const slot = computeIdeaSlotY(n.slotIndex, viewport);
          const targetX = seedLayout.ideaLaneX + slot.xOffset;
          n.x = targetX;
          n.y = slot.y;
        }
        nodesById.set(nodeId, n);
      } else if (action.type === "add_edge") {
        const ed = action.edge || {};
        const source = String(ed.source || "");
        const target = String(ed.target || "");
        if (!source || !target) return;
        edges.push({
          source,
          target,
          kind: String(ed.kind || "link"),
          addedAt: tNow,
        });
      } else if (action.type === "set_status") {
        const id = String(action.id || "");
        const n = nodesById.get(id);
        if (!n) return;
        n.status = String(action.status || n.status);
        n.statusChangedAt = tNow;
      } else if (action.type === "highlight") {
        const ids = safeArray(action.ids).map((x) => String(x));
        for (const id of ids) {
          const n = nodesById.get(id);
          if (!n) continue;
          n.highlight = {
            style: String(action.style || "glow"),
            until: tNow + stepDurationMs,
            startedAt: tNow,
          };
        }
      }
    }

    const steps = safeArray(timeline.steps);
    const maxI = clamp(uptoStepIndex, 0, Math.max(0, steps.length - 1));
    for (let i = 0; i <= maxI; i++) {
      const st = steps[i];
      const dur = Number(st.duration_ms || 0);
      const tNow = i === maxI ? nowMs() : 0;
      for (const action of safeArray(st.actions)) applyAction(action, dur, tNow);
    }

    return { nodesById, edges, ideaOrder, seedLayout };
  }

  function createRunner() {
    const ctx = UI.canvas.getContext("2d");
    let viewport = resetCanvasSize();
    let timeline = null;
    let nodesById = new Map();
    let edges = [];
    let ideaOrder = [];
    let seedLayout = null;
    let stepIndex = 0;
    let playing = false;
    let timer = null;
    let stepStart = 0;
    let stepDuration = 0;
    let speed = 1.0;

    function getStep() {
      if (!timeline) return null;
      return safeArray(timeline.steps)[stepIndex] || null;
    }

    function setUiStep(step) {
      UI.stepTitle.textContent = step?.title || "—";
      UI.stepSubtitle.textContent = step?.subtitle || "";
      UI.stepIndex.textContent = timeline ? `${stepIndex + 1}/${safeArray(timeline.steps).length}` : "—";
      UI.stepDuration.textContent = step ? formatMs((Number(step.duration_ms || 0) / speed) | 0) : "—";

      const notes = step?.notes || null;
      const primary = safeArray(notes?.primary_reasons).filter((x) => String(x).trim());
      const rev = notes?.revision_instructions ? String(notes.revision_instructions) : "";
      const noteLines = [];
      if (primary.length) {
        noteLines.push("Reviewer reasons:");
        for (const r of primary.slice(0, 6)) noteLines.push(`- ${r}`);
      }
      if (rev) {
        if (noteLines.length) noteLines.push("");
        noteLines.push("Revision instructions:");
        noteLines.push(rev);
      }
      UI.stepNotes.textContent =
        noteLines.join("\n") || step?.subtitle || "Load a `timeline.json` to begin.";
    }

    function clearTimer() {
      if (timer) window.clearTimeout(timer);
      timer = null;
    }

    function scheduleNext() {
      clearTimer();
      if (!playing) return;
      if (!timeline) return;
      const step = getStep();
      if (!step) return;
      const elapsed = nowMs() - stepStart;
      const remaining = Math.max(0, stepDuration - elapsed);
      timer = window.setTimeout(() => {
        nextStep();
      }, remaining);
    }

    function applyStepActions() {
      const step = getStep();
      if (!step) return;
      const tNow = nowMs();
      stepStart = tNow;
      stepDuration = Number(step.duration_ms || 0) / speed;

      const prevNodes = nodesById;
      const rebuild = rebuildGraphFromTimeline(timeline, stepIndex, viewport);
      const nextNodes = rebuild.nodesById;

      const dpr = viewport.dpr;
      const transitionMs = clamp(stepDuration * 0.55, 420, 980);
      const spawnX = 90 * dpr;
      const spawnY = 70 * dpr;

      for (const n of nextNodes.values()) {
        const prev = prevNodes.get(n.id);
        const fromX = prev ? prev.x : spawnX;
        const fromY = prev ? prev.y : spawnY;
        const toX = n.x;
        const toY = n.y;
        const dist = Math.hypot(toX - fromX, toY - fromY);
        if (dist > 1.0) {
          n.move = { fromX, fromY, toX, toY, start: tNow, duration: transitionMs };
          n.x = fromX;
          n.y = fromY;
        } else {
          n.move = null;
        }
      }

      nodesById = nextNodes;
      edges = rebuild.edges;
      ideaOrder = rebuild.ideaOrder;
      seedLayout = rebuild.seedLayout;

      setUiStep(step);
      scheduleNext();
    }

    function prevStep() {
      if (!timeline) return;
      stepIndex = clamp(stepIndex - 1, 0, safeArray(timeline.steps).length - 1);
      applyStepActions();
    }

    function nextStep() {
      if (!timeline) return;
      if (stepIndex >= safeArray(timeline.steps).length - 1) {
        playing = false;
        clearTimer();
        return;
      }
      stepIndex += 1;
      applyStepActions();
    }

    function play() {
      if (!timeline) return;
      playing = true;
      scheduleNext();
    }

    function pause() {
      playing = false;
      clearTimer();
    }

    function setSpeed(nextSpeed) {
      speed = clamp(nextSpeed, 0.5, 2.5);
      UI.speedVal.textContent = `${speed.toFixed(1)}×`;
      applyStepActions();
    }

    function load(nextTimeline) {
      timeline = nextTimeline;
      stepIndex = 0;
      applyStepActions();
    }

    function pickNodeAt(xCss, yCss) {
      const dpr = viewport.dpr;
      const x = xCss * dpr;
      const y = yCss * dpr;
      let best = null;
      let bestDist = Infinity;
      for (const n of nodesById.values()) {
        const r = (nodeRadius(n) + 5) * dpr;
        const dx = n.x - x;
        const dy = n.y - y;
        const d = Math.sqrt(dx * dx + dy * dy);
        if (d <= r && d < bestDist) {
          best = n;
          bestDist = d;
        }
      }
      return best;
    }

    function draw() {
      viewport = resetCanvasSize();
      const { dpr } = viewport;
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, UI.canvas.width, UI.canvas.height);

      const tNow = nowMs();
      // Apply node movements.
      for (const n of nodesById.values()) {
        if (!n.move) continue;
        const t = (tNow - n.move.start) / Math.max(1, n.move.duration);
        if (t >= 1) {
          n.x = n.move.toX;
          n.y = n.move.toY;
          n.move = null;
          continue;
        }
        const k = easeOutCubic(t);
        n.x = n.move.fromX + (n.move.toX - n.move.fromX) * k;
        n.y = n.move.fromY + (n.move.toY - n.move.fromY) * k;
      }

      // Background subtle grid.
      ctx.save();
      ctx.globalAlpha = 0.12;
      ctx.strokeStyle = "rgba(255,255,255,0.08)";
      const step = 28 * dpr;
      for (let x = 0; x < UI.canvas.width; x += step) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, UI.canvas.height);
        ctx.stroke();
      }
      for (let y = 0; y < UI.canvas.height; y += step) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(UI.canvas.width, y);
        ctx.stroke();
      }
      ctx.restore();

      // Empty state: no timeline loaded yet.
      if (!timeline || nodesById.size === 0) {
        ctx.save();
        const w = UI.canvas.width;
        const h = UI.canvas.height;
        ctx.fillStyle = "rgba(255,255,255,0.86)";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.font = `${22 * dpr}px ui-sans-serif, system-ui, -apple-system`;
        ctx.fillText("No timeline loaded", w / 2, h / 2 - 18 * dpr);
        ctx.fillStyle = "rgba(255,255,255,0.62)";
        ctx.font = `${13 * dpr}px ui-sans-serif, system-ui, -apple-system`;
        ctx.fillText("Click “Load timeline.json…” or regenerate one from the repo root.", w / 2, h / 2 + 10 * dpr);
        ctx.restore();
        requestAnimationFrame(draw);
        return;
      }

      // Lane divider between seed KG and idea lane.
      if (seedLayout && Number.isFinite(seedLayout.leftW)) {
        const x = seedLayout.leftW * dpr;
        ctx.save();
        ctx.strokeStyle = "rgba(255,255,255,0.12)";
        ctx.lineWidth = 1.0 * dpr;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, UI.canvas.height);
        ctx.stroke();
        ctx.restore();
      }

      // Edges.
      for (const e of edges) {
        const s = nodesById.get(e.source);
        const t = nodesById.get(e.target);
        if (!s || !t) continue;
        ctx.save();
        ctx.strokeStyle = edgeStroke(e);
        ctx.lineWidth = edgeWidth(e) * dpr;
        ctx.beginPath();
        ctx.moveTo(s.x, s.y);
        ctx.lineTo(t.x, t.y);
        ctx.stroke();
        ctx.restore();
      }

      // Nodes.
      const nodes = Array.from(nodesById.values()).slice();
      nodes.sort((a, b) => {
        const ta = isIdeaType(a.type) ? 1 : 0;
        const tb = isIdeaType(b.type) ? 1 : 0;
        return ta - tb;
      });

      for (const n of nodes) {
        const r = nodeRadius(n) * dpr;
        const fill = nodeFill(n);

        // highlight ring
        if (n.highlight && tNow <= n.highlight.until) {
          const phase = (tNow - n.highlight.startedAt) / 650;
          const pulse = 0.5 + 0.5 * Math.sin(phase * Math.PI * 2);
          const ring = (r + 10 * dpr) * (0.9 + 0.2 * pulse);
          ctx.save();
          ctx.globalAlpha = 0.55;
          ctx.strokeStyle = "rgba(255,255,255,0.22)";
          ctx.lineWidth = 2.2 * dpr;
          ctx.beginPath();
          ctx.arc(n.x, n.y, ring, 0, Math.PI * 2);
          ctx.stroke();
          ctx.restore();
        }

        // node
        ctx.save();
        ctx.fillStyle = fill;
        ctx.beginPath();
        ctx.arc(n.x, n.y, r, 0, Math.PI * 2);
        ctx.fill();

        // outline
        ctx.strokeStyle = "rgba(0,0,0,0.35)";
        ctx.lineWidth = (isIdeaType(n.type) ? 2.6 : 1.8) * dpr;
        ctx.stroke();
        ctx.restore();
      }

      // Labels (roots + ideas).
      ctx.save();
      ctx.font = `${12 * dpr}px ui-sans-serif, system-ui, -apple-system`;
      ctx.textBaseline = "middle";
      for (const n of nodes) {
        if (n.type !== "RootBox" && n.type !== "Idea") continue;
        const r = nodeRadius(n) * dpr;
        ctx.fillStyle = n.type === "Idea" ? COLORS.label : COLORS.labelMuted;
        ctx.fillText(String(n.label || n.id), n.x + r + 7 * dpr, n.y);
      }
      ctx.restore();

      requestAnimationFrame(draw);
    }

    window.addEventListener("resize", () => {
      viewport = resetCanvasSize();
      if (timeline) applyStepActions();
    });

    UI.canvas.addEventListener("mousemove", (ev) => {
      const rect = UI.canvas.getBoundingClientRect();
      const x = ev.clientX - rect.left;
      const y = ev.clientY - rect.top;
      const picked = pickNodeAt(x, y);
      if (!picked) {
        setTooltipVisible(false);
        return;
      }
      const title = picked.label || picked.id;
      const subtitle =
        (picked.meta && picked.meta.novelty_summary) ||
        (picked.meta && picked.meta.idea_id ? `idea_id: ${picked.meta.idea_id}` : picked.type);
      setTooltipContent(title, subtitle);
      setTooltipPosition(clamp(x + 14, 10, rect.width - 380), clamp(y + 14, 10, rect.height - 180));
      setTooltipVisible(true);
    });
    UI.canvas.addEventListener("mouseleave", () => setTooltipVisible(false));

    UI.btnPrev.addEventListener("click", () => prevStep());
    UI.btnNext.addEventListener("click", () => nextStep());
    UI.btnPlay.addEventListener("click", () => play());
    UI.btnPause.addEventListener("click", () => pause());
    UI.speed.addEventListener("input", () => setSpeed(Number(UI.speed.value || 1)));

    UI.btnDemo.addEventListener("click", () => load(BUILTIN_DEMO_TIMELINE));

    UI.btnLoad.addEventListener("click", () => {
      UI.fileInput.value = "";
      UI.fileInput.click();
    });
    UI.fileInput.addEventListener("change", async () => {
      const file = UI.fileInput.files && UI.fileInput.files[0];
      if (!file) return;
      UI.stepNotes.textContent = `Loading ${file.name}…`;
      try {
        const text = await readFileAsText(file);
        const obj = JSON.parse(text);
        if (!isTimelineObject(obj)) {
          const sv = obj && typeof obj === "object" ? obj.schema_version : null;
          UI.stepNotes.textContent =
            `That JSON doesn't look like a timeline.\n\n` +
            `Expected schema_version like:\n` +
            `  knowledge_graph.ideator_visual_timeline.v1\n\n` +
            `Got:\n` +
            `  ${sv ? String(sv) : "(missing schema_version)"}`;
          return;
        }
        load(obj);
        UI.stepNotes.textContent =
          `Loaded ${file.name}.\n` +
          `seed_nodes=${obj.seed.nodes.length}, seed_edges=${obj.seed.edges.length}, steps=${obj.steps.length}`;
      } catch (e) {
        UI.stepNotes.textContent = `Failed to load JSON: ${String(e)}`;
      }
    });

    // Initial speed label.
    UI.speedVal.textContent = `${speed.toFixed(1)}×`;

    requestAnimationFrame(draw);

    return {
      load,
      tryLoadDefault: async function () {
        if (window.__IDEATOR_TIMELINE__ && isTimelineObject(window.__IDEATOR_TIMELINE__)) {
          load(window.__IDEATOR_TIMELINE__);
          return;
        }
        try {
          const res = await fetch("./timeline.json", { cache: "no-store" });
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          const obj = await res.json();
          if (!isTimelineObject(obj)) throw new Error("timeline.json is not a valid timeline object");
          load(obj);
        } catch (_e) {
          // Last resort: always show something so you can test the visualizer.
          load(BUILTIN_DEMO_TIMELINE);
          UI.stepNotes.textContent =
            "Loaded built-in demo timeline.\n\n" +
            "To use real data, regenerate `timeline.json` and/or `timeline.inline.js` and reload.";
        }
      },
    };
  }

  const runner = createRunner();
  runner.tryLoadDefault();
})();
