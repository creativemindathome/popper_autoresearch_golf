"""Knowledge graph module for falsifier precedent checking and interpolation."""

from falsifier.graph.query import load_graph, save_graph, add_node
from falsifier.types import KnowledgeGraph

__all__ = ["KnowledgeGraph", "load_graph", "save_graph", "add_node"]
