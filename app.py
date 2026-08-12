import streamlit as st
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

from community_detection import detect_communities_girvan_newman
from utils import draw_communities
from data_loader import load_karate_club, load_connected_random_graph
from utils import draw_graph
from degree_centrality import calculate_degree_centrality
from pagerank import calculate_pagerank
from betweenness_centrality import calculate_betweenness_centrality
from closeness_centrality import calculate_closeness_centrality
from evaluation import measure_runtime

st.set_page_config(page_title="Social Network Analysis", layout="wide")

st.title("Social Network Analysis Tool")
st.write(
    "This application analyses social networks using Degree Centrality, "
    "Closeness Centrality, PageRank, and Betweenness Centrality, and compares "
    "manual implementations with NetworkX."
)

# -------------------------
# Dataset selection
# -------------------------
dataset_option = st.selectbox(
    "Choose dataset",
    [
        "Karate Club",
        "Random Graph (30 nodes)",
        "Random Graph (50 nodes)",
        "Random Graph (100 nodes)",
        "Random Graph (200 nodes)"
    ]
)

if dataset_option == "Karate Club":
    G = load_karate_club()
elif dataset_option == "Random Graph (30 nodes)":
    G = load_connected_random_graph(num_nodes=30, probability=0.12, seed=42)
elif dataset_option == "Random Graph (50 nodes)":
    G = load_connected_random_graph(num_nodes=50, probability=0.10, seed=42)
elif dataset_option == "Random Graph (100 nodes)":
    G = load_connected_random_graph(num_nodes=100, probability=0.06, seed=42)
elif dataset_option == "Random Graph (200 nodes)":
    G = load_connected_random_graph(num_nodes=200, probability=0.04, seed=42)
else:
    G = load_karate_club()

# -------------------------
# Metric calculations
# -------------------------
manual_degree, manual_degree_time = measure_runtime(calculate_degree_centrality, G)
nx_degree, nx_degree_time = measure_runtime(nx.degree_centrality, G)

manual_pagerank, manual_pagerank_time = measure_runtime(calculate_pagerank, G)
nx_pagerank, nx_pagerank_time = measure_runtime(nx.pagerank, G)

manual_betweenness, manual_betweenness_time = measure_runtime(calculate_betweenness_centrality, G)
nx_betweenness, nx_betweenness_time = measure_runtime(nx.betweenness_centrality, G)

manual_closeness, manual_closeness_time = measure_runtime(calculate_closeness_centrality, G)
nx_closeness, nx_closeness_time = measure_runtime(nx.closeness_centrality, G)

manual_degree_sorted = sorted(manual_degree.items(), key=lambda x: x[1], reverse=True)
nx_degree_sorted = sorted(nx_degree.items(), key=lambda x: x[1], reverse=True)

manual_pagerank_sorted = sorted(manual_pagerank.items(), key=lambda x: x[1], reverse=True)
nx_pagerank_sorted = sorted(nx_pagerank.items(), key=lambda x: x[1], reverse=True)

manual_betweenness_sorted = sorted(manual_betweenness.items(), key=lambda x: x[1], reverse=True)
nx_betweenness_sorted = sorted(nx_betweenness.items(), key=lambda x: x[1], reverse=True)

manual_closeness_sorted = sorted(manual_closeness.items(), key=lambda x: x[1], reverse=True)
nx_closeness_sorted = sorted(nx_closeness.items(), key=lambda x: x[1], reverse=True)

# -------------------------
# Session state for interactive editor
# -------------------------
if "editor_graph" not in st.session_state:
    st.session_state.editor_graph = load_karate_club()

if "editor_history" not in st.session_state:
    st.session_state.editor_history = []

if "editor_future" not in st.session_state:
    st.session_state.editor_future = []

def save_history():
    st.session_state.editor_history.append(st.session_state.editor_graph.copy())
    st.session_state.editor_future = []
    if len(st.session_state.editor_history) > 30:
        st.session_state.editor_history.pop(0)

# -------------------------
# Tabs
# -------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "Overview", "Degree Centrality", "PageRank", "Betweenness", "Closeness",
    "Metric Comparison", "Evaluation", "Graph Editor", "Communities"
])

# -------------------------
# TAB 1: OVERVIEW
# -------------------------
with tab1:
    st.header("Overview")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Dataset Information")
        st.write(f"**Dataset:** {dataset_option}")
        st.write(f"**Nodes:** {G.number_of_nodes()}")
        st.write(f"**Edges:** {G.number_of_edges()}")
        st.write(f"**Connected:** {'Yes' if nx.is_connected(G) else 'No'}")

        metric_option = st.selectbox(
            "Choose visualization metric",
            ["None", "Degree Centrality", "PageRank", "Betweenness Centrality", "Closeness Centrality"],
            key="overview_metric"
        )

        if metric_option == "Degree Centrality":
            scores = manual_degree
            title = "Degree Centrality Visualization"
        elif metric_option == "PageRank":
            scores = manual_pagerank
            title = "PageRank Visualization"
        elif metric_option == "Betweenness Centrality":
            scores = manual_betweenness
            title = "Betweenness Centrality Visualization"
        elif metric_option == "Closeness Centrality":
            scores = manual_closeness
            title = "Closeness Centrality Visualization"
        else:
            scores = None
            title = "Basic Graph"

        st.subheader("What this tab shows")
        st.write(
            "This tab provides a general view of the selected network. "
            "You can switch between different metrics to see how node size "
            "and colour change depending on their importance."
        )

    with col2:
        st.subheader(title)
        fig = draw_graph(G, node_scores=scores, title=title)
        st.pyplot(fig)

# -------------------------
# TAB 2: DEGREE
# -------------------------
with tab2:
    st.header("Degree Centrality")

    st.write(
        "Degree Centrality measures how many direct connections a node has. "
        "Nodes with high degree centrality can be seen as highly connected or popular."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Visualization")
        fig = draw_graph(G, node_scores=manual_degree, title="Degree Centrality")
        st.pyplot(fig)

    with col2:
        st.subheader("Top 5 Nodes (Manual)")
        for node, score in manual_degree_sorted[:5]:
            st.write(f"Node {node}: {score:.4f}")

        st.subheader("Top 5 Nodes (NetworkX)")
        for node, score in nx_degree_sorted[:5]:
            st.write(f"Node {node}: {score:.4f}")

        degree_match = all(abs(manual_degree[n] - nx_degree[n]) < 1e-9 for n in G.nodes())
        if degree_match:
            st.success("Manual Degree Centrality matches NetworkX exactly.")
        else:
            st.error("Manual Degree Centrality does not match NetworkX.")

        st.write(f"**Manual runtime:** {manual_degree_time:.6f} seconds")
        st.write(f"**NetworkX runtime:** {nx_degree_time:.6f} seconds")

# -------------------------
# TAB 3: PAGERANK
# -------------------------
with tab3:
    st.header("PageRank")

    st.write(
        "PageRank measures node importance based not only on the number of connections, "
        "but also on the importance of the connected neighbours."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Visualization")
        fig = draw_graph(G, node_scores=manual_pagerank, title="PageRank")
        st.pyplot(fig)

    with col2:
        st.subheader("Top 5 Nodes (Manual)")
        for node, score in manual_pagerank_sorted[:5]:
            st.write(f"Node {node}: {score:.4f}")

        st.subheader("Top 5 Nodes (NetworkX)")
        for node, score in nx_pagerank_sorted[:5]:
            st.write(f"Node {node}: {score:.4f}")

        total_difference = sum(abs(manual_pagerank[n] - nx_pagerank[n]) for n in G.nodes())
        average_difference = total_difference / G.number_of_nodes()
        st.write(f"**Total difference:** {total_difference:.6f}")
        st.write(f"**Average difference per node:** {average_difference:.6f}")

        manual_top_node = manual_pagerank_sorted[0][0]
        nx_top_node = nx_pagerank_sorted[0][0]
        if manual_top_node == nx_top_node:
            st.success(f"Top PageRank node matches NetworkX: Node {manual_top_node}")
        else:
            st.warning(
                f"Top PageRank node differs. Manual: Node {manual_top_node}, "
                f"NetworkX: Node {nx_top_node}"
            )

        st.write(f"**Manual runtime:** {manual_pagerank_time:.6f} seconds")
        st.write(f"**NetworkX runtime:** {nx_pagerank_time:.6f} seconds")

# -------------------------
# TAB 4: BETWEENNESS
# -------------------------
with tab4:
    st.header("Betweenness Centrality")

    st.write(
        "Betweenness Centrality measures how often a node lies on shortest paths "
        "between other nodes. Nodes with high betweenness often act as bridges "
        "between different parts of the network."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Visualization")
        fig = draw_graph(G, node_scores=manual_betweenness, title="Betweenness Centrality")
        st.pyplot(fig)

    with col2:
        st.subheader("Top 5 Nodes (Manual)")
        for node, score in manual_betweenness_sorted[:5]:
            st.write(f"Node {node}: {score:.4f}")

        st.subheader("Top 5 Nodes (NetworkX)")
        for node, score in nx_betweenness_sorted[:5]:
            st.write(f"Node {node}: {score:.4f}")

        total_betweenness_difference = sum(
            abs(manual_betweenness[n] - nx_betweenness[n]) for n in G.nodes()
        )
        average_betweenness_difference = total_betweenness_difference / G.number_of_nodes()
        st.write(f"**Total difference:** {total_betweenness_difference:.6f}")
        st.write(f"**Average difference per node:** {average_betweenness_difference:.6f}")

        manual_betweenness_top = manual_betweenness_sorted[0][0]
        nx_betweenness_top = nx_betweenness_sorted[0][0]
        if manual_betweenness_top == nx_betweenness_top:
            st.success(f"Top Betweenness node matches NetworkX: Node {manual_betweenness_top}")
        else:
            st.warning(
                f"Top Betweenness node differs. Manual: Node {manual_betweenness_top}, "
                f"NetworkX: Node {nx_betweenness_top}"
            )

        st.write(f"**Manual runtime:** {manual_betweenness_time:.6f} seconds")
        st.write(f"**NetworkX runtime:** {nx_betweenness_time:.6f} seconds")

# -------------------------
# TAB 5: CLOSENESS
# -------------------------
with tab5:
    st.header("Closeness Centrality")

    st.write(
        "Closeness Centrality measures how quickly a node can reach all other nodes "
        "in the network. A high score means the node is close to everyone else on average, "
        "making it efficient at spreading information across the network."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Visualization")
        fig = draw_graph(G, node_scores=manual_closeness, title="Closeness Centrality")
        st.pyplot(fig)

    with col2:
        st.subheader("Top 5 Nodes (Manual)")
        for node, score in manual_closeness_sorted[:5]:
            st.write(f"Node {node}: {score:.4f}")

        st.subheader("Top 5 Nodes (NetworkX)")
        for node, score in nx_closeness_sorted[:5]:
            st.write(f"Node {node}: {score:.4f}")

        total_closeness_difference = sum(
            abs(manual_closeness[n] - nx_closeness[n]) for n in G.nodes()
        )
        average_closeness_difference = total_closeness_difference / G.number_of_nodes()
        st.write(f"**Total difference:** {total_closeness_difference:.6f}")
        st.write(f"**Average difference per node:** {average_closeness_difference:.6f}")

        manual_closeness_top = manual_closeness_sorted[0][0]
        nx_closeness_top = nx_closeness_sorted[0][0]
        if manual_closeness_top == nx_closeness_top:
            st.success(f"Top Closeness node matches NetworkX: Node {manual_closeness_top}")
        else:
            st.warning(
                f"Top Closeness node differs. Manual: Node {manual_closeness_top}, "
                f"NetworkX: Node {nx_closeness_top}"
            )

        st.write(f"**Manual runtime:** {manual_closeness_time:.6f} seconds")
        st.write(f"**NetworkX runtime:** {nx_closeness_time:.6f} seconds")

# -------------------------
# TAB 6: METRIC COMPARISON
# -------------------------
with tab6:
    st.header("Metric Comparison")

    st.write(
        "Different centrality metrics often identify different nodes as the most important. "
        "This tab compares how each metric ranks the top nodes and explains why they disagree."
    )

    all_nodes = list(G.nodes())
    comparison_data = []
    for node in all_nodes:
        comparison_data.append({
            "Node": node,
            "Degree": round(manual_degree[node], 4),
            "PageRank": round(manual_pagerank[node], 4),
            "Betweenness": round(manual_betweenness[node], 4),
            "Closeness": round(manual_closeness[node], 4),
        })
    comparison_df = pd.DataFrame(comparison_data)

    st.subheader("Top 10 Nodes by Each Metric")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("**Degree**")
        for node, score in manual_degree_sorted[:10]:
            st.write(f"Node {node}: {score:.4f}")

    with col2:
        st.markdown("**PageRank**")
        for node, score in manual_pagerank_sorted[:10]:
            st.write(f"Node {node}: {score:.4f}")

    with col3:
        st.markdown("**Betweenness**")
        for node, score in manual_betweenness_sorted[:10]:
            st.write(f"Node {node}: {score:.4f}")

    with col4:
        st.markdown("**Closeness**")
        for node, score in manual_closeness_sorted[:10]:
            st.write(f"Node {node}: {score:.4f}")

    st.subheader("Top Ranked Node per Metric")
    top_degree = manual_degree_sorted[0][0]
    top_pagerank = manual_pagerank_sorted[0][0]
    top_betweenness = manual_betweenness_sorted[0][0]
    top_closeness = manual_closeness_sorted[0][0]

    top_summary = pd.DataFrame({
        "Metric": ["Degree", "PageRank", "Betweenness", "Closeness"],
        "Top Node": [top_degree, top_pagerank, top_betweenness, top_closeness],
        "Score": [
            round(manual_degree_sorted[0][1], 4),
            round(manual_pagerank_sorted[0][1], 4),
            round(manual_betweenness_sorted[0][1], 4),
            round(manual_closeness_sorted[0][1], 4),
        ]
    })
    st.dataframe(top_summary, use_container_width=True, hide_index=True)

    if top_degree == top_pagerank == top_betweenness == top_closeness:
        st.success(
            f"All four metrics agree: Node {top_degree} is the most important node in this network."
        )
    else:
        st.info(
            "The metrics do not fully agree on the most important node. "
            "This is expected — each metric captures a different aspect of importance."
        )

    st.subheader("Full Scores Table")
    st.write("Scores for all nodes across all four metrics.")
    st.dataframe(
        comparison_df.sort_values("Degree", ascending=False),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Why Do the Metrics Disagree?")
    st.write(
        "**Degree Centrality** only counts direct connections. A node with many neighbours "
        "scores highly regardless of where those neighbours sit in the network."
    )
    st.write(
        "**PageRank** goes further — it weights connections by the importance of the neighbour. "
        "Being connected to a few highly influential nodes can outrank having many connections "
        "to less important ones."
    )
    st.write(
        "**Betweenness Centrality** identifies bridge nodes — those that sit on the shortest "
        "paths between many other pairs. A node can have low degree but high betweenness if it "
        "connects otherwise separate parts of the network."
    )
    st.write(
        "**Closeness Centrality** rewards nodes that are geometrically central — close to "
        "everyone else on average. These nodes are efficient at spreading information quickly "
        "across the whole network."
    )
    st.write(
        "Together, these metrics paint a richer picture of network structure than any single "
        "measure could provide on its own."
    )

# -------------------------
# TAB 7: EVALUATION
# -------------------------
with tab7:
    st.header("Evaluation")

    st.subheader("Runtime Comparison — Current Dataset")
    st.write("Comparing manual vs NetworkX runtimes on the currently selected dataset.")

    runtime_data = {
        "Metric": ["Degree", "PageRank", "Betweenness", "Closeness"],
        "Manual Time (s)": [
            round(manual_degree_time, 6),
            round(manual_pagerank_time, 6),
            round(manual_betweenness_time, 6),
            round(manual_closeness_time, 6)
        ],
        "NetworkX Time (s)": [
            round(nx_degree_time, 6),
            round(nx_pagerank_time, 6),
            round(nx_betweenness_time, 6),
            round(nx_closeness_time, 6)
        ]
    }
    runtime_df = pd.DataFrame(runtime_data)
    st.dataframe(runtime_df, use_container_width=True, hide_index=True)

    st.subheader("Runtime Scaling Across Graph Sizes")
    st.write(
        "This chart shows how the runtime of each manual algorithm grows "
        "as the graph size increases, illustrating the computational complexity of each approach."
    )

    @st.cache_data
    def run_scaling_experiment():
        sizes = [10, 20, 34, 50, 75, 100, 150, 200]
        results = {
            "nodes": sizes,
            "degree": [], "pagerank": [], "betweenness": [], "closeness": []
        }
        for n in sizes:
            if n == 34:
                g = nx.karate_club_graph()
            else:
                seed = 42
                while True:
                    g = nx.erdos_renyi_graph(n, 0.15, seed=seed)
                    if nx.is_connected(g):
                        break
                    seed += 1
            _, t = measure_runtime(calculate_degree_centrality, g)
            results["degree"].append(round(t, 6))
            _, t = measure_runtime(calculate_pagerank, g)
            results["pagerank"].append(round(t, 6))
            _, t = measure_runtime(calculate_betweenness_centrality, g)
            results["betweenness"].append(round(t, 6))
            _, t = measure_runtime(calculate_closeness_centrality, g)
            results["closeness"].append(round(t, 6))
        return results

    scaling_results = run_scaling_experiment()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(scaling_results["nodes"], scaling_results["degree"], marker="o", label="Degree Centrality")
    ax.plot(scaling_results["nodes"], scaling_results["pagerank"], marker="s", label="PageRank")
    ax.plot(scaling_results["nodes"], scaling_results["closeness"], marker="^", label="Closeness Centrality")
    ax.plot(scaling_results["nodes"], scaling_results["betweenness"], marker="D", label="Betweenness Centrality")
    ax.set_xlabel("Number of Nodes")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("Algorithm Runtime vs Graph Size (Manual Implementations)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig)

    st.subheader("Interpretation")
    st.write(
        "- **Degree Centrality** is O(n) — it only requires counting neighbours, "
        "so runtime grows linearly and stays very fast across all graph sizes. "
        "Manual and NetworkX runtimes are comparable throughout."
    )
    st.write(
        "- **Closeness Centrality** requires a BFS from every node, giving O(n(n+m)) complexity. "
        "NetworkX becomes noticeably faster at larger graph sizes due to its optimised internals."
    )
    st.write(
        "- **PageRank** is iterative and NetworkX becomes significantly faster at larger sizes. "
        "At 200 nodes, NetworkX is around 5x faster, reflecting its use of optimised "
        "sparse matrix operations compared to the manual Python loop approach."
    )
    st.write(
        "- **Betweenness Centrality** shows the most interesting behaviour. Despite being "
        "O(n²(n+m)) in complexity, the manual Brandes implementation remains competitive with "
        "NetworkX even at 200 nodes, avoiding NetworkX's general-purpose overhead "
        "for graphs of this size."
    )

    st.subheader("Raw Scaling Data")
    scaling_df = pd.DataFrame({
        "Nodes": scaling_results["nodes"],
        "Degree (s)": scaling_results["degree"],
        "PageRank (s)": scaling_results["pagerank"],
        "Closeness (s)": scaling_results["closeness"],
        "Betweenness (s)": scaling_results["betweenness"]
    })
    st.dataframe(
        scaling_df.sort_values("Nodes"),
        use_container_width=True,
        hide_index=True
    )

# -------------------------
# TAB 8: GRAPH EDITOR
# -------------------------
with tab8:
    st.header("Interactive Graph Editor")
    st.write(
        "Modify the Karate Club network by adding or removing nodes and edges. "
        "Both graphs show the same network — each with an independently chosen metric. "
        "Metrics and top 3 rankings update live after every change."
    )

    EG = st.session_state.editor_graph
    node_list = sorted(EG.nodes())

    # ── Helper: compute scores safely ────────────────────────────────────
    def get_editor_scores(graph, metric):
        if metric == "Degree Centrality":
            return calculate_degree_centrality(graph), "Degree Centrality"
        elif metric == "PageRank":
            if nx.is_connected(graph) and graph.number_of_nodes() > 1:
                return calculate_pagerank(graph), "PageRank"
            else:
                return calculate_degree_centrality(graph), "PageRank (disconnected — showing Degree)"
        elif metric == "Betweenness Centrality":
            return calculate_betweenness_centrality(graph), "Betweenness Centrality"
        elif metric == "Closeness Centrality":
            return calculate_closeness_centrality(graph), "Closeness Centrality"
        else:
            return None, "Graph Structure"

    # ── Layout: controls | graph1 | graph2 ───────────────────────────────
    col_controls, col_g1, col_g2 = st.columns([1, 1.6, 1.6])

    with col_controls:

        # Undo / Redo
        undo_col, redo_col = st.columns(2)
        with undo_col:
            if st.button("↩ Undo", disabled=len(st.session_state.editor_history) == 0):
                st.session_state.editor_future.append(st.session_state.editor_graph.copy())
                st.session_state.editor_graph = st.session_state.editor_history.pop()
                st.rerun()
        with redo_col:
            if st.button("↪ Redo", disabled=len(st.session_state.editor_future) == 0):
                st.session_state.editor_history.append(st.session_state.editor_graph.copy())
                st.session_state.editor_graph = st.session_state.editor_future.pop()
                st.rerun()

        st.write(f"**Nodes:** {EG.number_of_nodes()} | **Edges:** {EG.number_of_edges()}")
        st.caption(f"Undo steps: {len(st.session_state.editor_history)} | Redo steps: {len(st.session_state.editor_future)}")

        st.divider()

        # Add Node
        st.subheader("Add Node")
        existing_nodes = set(st.session_state.editor_graph.nodes())
        max_node = max(existing_nodes) if existing_nodes else -1
        removed_nodes = sorted([n for n in range(max_node + 1) if n not in existing_nodes])
        new_node_options = removed_nodes + [max_node + 1]
        node_to_add = st.selectbox("Select node ID to add", new_node_options, key="add_node_select")
        if st.button("Add selected node"):
            save_history()
            st.session_state.editor_graph.add_node(node_to_add)
            st.rerun()

        st.divider()

        # Remove Node
        st.subheader("Remove Node")
        if node_list:
            node_to_remove = st.selectbox("Select node to remove", node_list, key="remove_node_select")
            if st.button("Remove selected node"):
                save_history()
                st.session_state.editor_graph.remove_node(node_to_remove)
                st.rerun()
        else:
            st.write("No nodes to remove.")

        st.divider()

        # Add Edge
        st.subheader("Add Edge")
        if len(node_list) >= 2:
            edge_src = st.selectbox("From node", node_list, key="edge_src")
            edge_dst = st.selectbox("To node", node_list, key="edge_dst")
            if st.button("Add edge"):
                if edge_src == edge_dst:
                    st.warning("Cannot add a self-loop.")
                elif EG.has_edge(edge_src, edge_dst):
                    st.warning("Edge already exists.")
                else:
                    save_history()
                    st.session_state.editor_graph.add_edge(edge_src, edge_dst)
                    st.rerun()
        else:
            st.write("Need at least 2 nodes to add an edge.")

        st.divider()

        # Remove Edge
        st.subheader("Remove Edge")
        edge_list = list(EG.edges())
        if edge_list:
            edge_labels = [f"{u} — {v}" for u, v in edge_list]
            selected_edge_label = st.selectbox("Select edge to remove", edge_labels, key="remove_edge_select")
            if st.button("Remove selected edge"):
                idx = edge_labels.index(selected_edge_label)
                u, v = edge_list[idx]
                save_history()
                st.session_state.editor_graph.remove_edge(u, v)
                st.rerun()
        else:
            st.write("No edges to remove.")

        st.divider()

        if st.button("Reset to original Karate Club"):
            save_history()
            st.session_state.editor_graph = load_karate_club()
            st.session_state.editor_history = []
            st.session_state.editor_future = []
            st.rerun()

    # ── Graph 1 ───────────────────────────────────────────────────────────
    with col_g1:
        metric_options = ["None", "Degree Centrality", "PageRank", "Betweenness Centrality", "Closeness Centrality"]
        metric1 = st.selectbox("Graph 1 metric", metric_options, key="editor_metric_1")
        EG = st.session_state.editor_graph

        if EG.number_of_nodes() == 0:
            st.warning("No nodes in graph.")
        else:
            scores1, title1 = get_editor_scores(EG, metric1)
            fig1 = draw_graph(EG, node_scores=scores1, title=title1)
            st.pyplot(fig1)

        # Live Top 3 — Degree and PageRank
        st.markdown("**Live Top 3 — Degree &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; PageRank**")
        EG = st.session_state.editor_graph
        if EG.number_of_nodes() > 0:
            live_degree = calculate_degree_centrality(EG)
            if nx.is_connected(EG) and EG.number_of_nodes() > 1:
                live_pagerank = calculate_pagerank(EG)
            else:
                live_pagerank = live_degree
            sub1, sub2 = st.columns(2)
            with sub1:
                st.markdown("**Degree**")
                for rank, (node, score) in enumerate(sorted(live_degree.items(), key=lambda x: x[1], reverse=True)[:3], 1):
                    st.write(f"{rank}. Node {node} — {score:.4f}")
            with sub2:
                st.markdown("**PageRank**")
                for rank, (node, score) in enumerate(sorted(live_pagerank.items(), key=lambda x: x[1], reverse=True)[:3], 1):
                    st.write(f"{rank}. Node {node} — {score:.4f}")

    # ── Graph 2 ───────────────────────────────────────────────────────────
    with col_g2:
        metric2 = st.selectbox("Graph 2 metric", metric_options, index=1, key="editor_metric_2")
        EG = st.session_state.editor_graph

        if EG.number_of_nodes() == 0:
            st.warning("No nodes in graph.")
        else:
            scores2, title2 = get_editor_scores(EG, metric2)
            fig2 = draw_graph(EG, node_scores=scores2, title=title2)
            st.pyplot(fig2)

        # Live Top 3 — Betweenness and Closeness
        EG = st.session_state.editor_graph
        if EG.number_of_nodes() > 0:
            live_betweenness = calculate_betweenness_centrality(EG)
            live_closeness = calculate_closeness_centrality(EG)
            sub3, sub4 = st.columns(2)
            with sub3:
                st.markdown("**Betweenness**")
                for rank, (node, score) in enumerate(sorted(live_betweenness.items(), key=lambda x: x[1], reverse=True)[:3], 1):
                    st.write(f"{rank}. Node {node} — {score:.4f}")
            with sub4:
                st.markdown("**Closeness**")
                for rank, (node, score) in enumerate(sorted(live_closeness.items(), key=lambda x: x[1], reverse=True)[:3], 1):
                    st.write(f"{rank}. Node {node} — {score:.4f}")

# -------------------------
# TAB 9: COMMUNITY DETECTION
# -------------------------
with tab9:
    st.header("Community Detection (Girvan–Newman)")

    st.write(
        "Community detection identifies groups of nodes that are more densely connected "
        "to each other than to the rest of the network."
    )

    level = st.slider("Number of communities", 2, 5, 2)
    communities = detect_communities_girvan_newman(G, level=level)

    st.subheader(f"Detected {level} Communities")
    for i, community in enumerate(communities):
        st.write(f"Community {i+1}: {community}")

    fig = draw_communities(G, communities)
    st.pyplot(fig)