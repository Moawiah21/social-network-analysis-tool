import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def draw_graph(G, node_scores=None, title="Graph Visualization"):
    # Handle empty graph
    if G.number_of_nodes() == 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title(title)
        ax.axis("off")
        return fig

    components = list(nx.connected_components(G))

    if len(components) == 1:
        pos = nx.spring_layout(G, seed=42, k=1.5)
    else:
        # Layout each component individually then arrange them in a grid
        pos = {}
        component_layouts = []

        for component in sorted(components, key=len, reverse=True):
            subgraph = G.subgraph(component)
            if len(component) == 1:
                node = list(component)[0]
                sub_pos = {node: np.array([0.0, 0.0])}
            else:
                # Scale k by component size so nodes within a component aren't cramped
                k_val = 1.5 / max(1, np.sqrt(len(component) / 10))
                sub_pos = nx.spring_layout(subgraph, seed=42, k=k_val)

            # Normalise each component to fit in a 0-1 box
            coords = np.array(list(sub_pos.values()))
            min_xy = coords.min(axis=0)
            max_xy = coords.max(axis=0)
            span = max_xy - min_xy
            span[span == 0] = 1  # avoid division by zero for single nodes

            normalised = {}
            for node, p in sub_pos.items():
                normalised[node] = (p - min_xy) / span

            component_layouts.append((component, normalised))

        # Arrange components left to right with gap proportional to component size
        # Larger components get more horizontal space
        total_nodes = G.number_of_nodes()
        x_cursor = 0.0

        for component, normalised in component_layouts:
            # Width of this component's column proportional to its size
            width = max(0.3, len(component) / total_nodes) * 3.5
            gap = 0.4  # gap between components

            for node, p in normalised.items():
                pos[node] = np.array([x_cursor + p[0] * width, p[1]])

            x_cursor += width + gap

    if node_scores is None:
        node_sizes = [500 for _ in G.nodes()]
        node_colors = ["lightblue" for _ in G.nodes()]
    else:
        max_score = max(node_scores.values())
        min_score = min(node_scores.values())

        if max_score == min_score:
            normalized_scores = {node: 0.5 for node in G.nodes()}
        else:
            normalized_scores = {
                node: (node_scores[node] - min_score) / (max_score - min_score)
                for node in G.nodes()
            }

        node_sizes = [300 + normalized_scores[node] * 2000 for node in G.nodes()]
        node_colors = [normalized_scores[node] for node in G.nodes()]

    fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw(
        G,
        pos,
        ax=ax,
        with_labels=True,
        node_color=node_colors,
        cmap=plt.cm.Blues,
        edge_color="gray",
        node_size=node_sizes,
        font_size=8
    )

    ax.set_title(title)
    return fig


def draw_communities(G, communities, title="Community Detection"):
    pos = nx.spring_layout(G, seed=42)

    colors = [
        "red", "blue", "green", "orange", "purple",
        "brown", "pink", "cyan", "yellow"
    ]

    node_colors = []
    for node in G.nodes():
        for i, community in enumerate(communities):
            if node in community:
                node_colors.append(colors[i % len(colors)])
                break

    fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw(
        G,
        pos,
        ax=ax,
        with_labels=True,
        node_color=node_colors,
        edge_color="gray",
        node_size=600,
        font_size=8
    )

    ax.set_title(title)
    return fig