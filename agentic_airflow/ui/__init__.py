"""Lightweight graph interface: JSON graph, self-contained HTML, optional HTTP server."""

from .graph import build_graph, layout_graph
from .html import render_embed, render_html

__all__ = ["build_graph", "layout_graph", "render_html", "render_embed"]
