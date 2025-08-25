from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import networkx as nx
import plotly.graph_objects as go


def build_graph(
    boxes: Sequence[Tuple[float, float, float, float]],
    labels: Sequence[int],
    relations: Sequence[Tuple[int, int, float]],
) -> nx.DiGraph:
    g = nx.DiGraph()
    for idx, (box, label) in enumerate(zip(boxes, labels)):
        g.add_node(idx, box=box, label=label)
    for s, o, score in relations:
        g.add_edge(s, o, score=score)
    return g


def to_plotly_figure(graph: nx.DiGraph, node_text: Optional[List[str]] = None) -> go.Figure:
    pos = nx.spring_layout(graph, seed=42)
    x_nodes = [pos[n][0] for n in graph.nodes]
    y_nodes = [pos[n][1] for n in graph.nodes]
    node_trace = go.Scatter(
        x=x_nodes,
        y=y_nodes,
        mode="markers+text",
        text=node_text if node_text else [str(n) for n in graph.nodes],
        textposition="top center",
        marker=dict(size=14, color="#1f77b4"),
    )
    edge_x = []
    edge_y = []
    for s, o in graph.edges:
        edge_x += [pos[s][0], pos[o][0], None]
        edge_y += [pos[s][1], pos[o][1], None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1, color="#888"))

    # Edge labels (scores)
    edge_label_x = []
    edge_label_y = []
    edge_labels = []
    for s, o, data in graph.edges(data=True):
        mx = (pos[s][0] + pos[o][0]) / 2
        my = (pos[s][1] + pos[o][1]) / 2
        edge_label_x.append(mx)
        edge_label_y.append(my)
        edge_labels.append(f"{data.get('score', 0):.2f}")
    edge_label_trace = go.Scatter(
        x=edge_label_x,
        y=edge_label_y,
        mode="text",
        text=edge_labels,
        textposition="middle center",
        textfont=dict(color="#444"),
        hoverinfo="skip",
    )
    fig = go.Figure(data=[edge_trace, node_trace, edge_label_trace])
    fig.update_layout(showlegend=False, margin=dict(l=20, r=20, t=20, b=20))
    return fig


