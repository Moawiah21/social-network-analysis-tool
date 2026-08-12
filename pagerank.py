def calculate_pagerank(G, damping=0.85, max_iterations=100, tolerance=1e-6):
    """
    Manual PageRank implementation for an undirected graph.
    We treat each edge as a two-way link.

    Parameters:
        G: NetworkX graph
        damping: damping factor, usually 0.85
        max_iterations: maximum number of update rounds
        tolerance: stop if total change is below this

    Returns:
        Dictionary of node -> PageRank score
    """
    nodes = list(G.nodes())
    n = len(nodes)

    if n == 0:
        return {}

    # Start with equal rank for all nodes
    ranks = {node: 1 / n for node in nodes}

    for _ in range(max_iterations):
        new_ranks = {}
        total_change = 0

        for node in nodes:
            rank_sum = 0

            for neighbor in G.neighbors(node):
                neighbor_degree = len(list(G.neighbors(neighbor)))
                if neighbor_degree > 0:
                    rank_sum += ranks[neighbor] / neighbor_degree

            new_rank = (1 - damping) / n + damping * rank_sum
            new_ranks[node] = new_rank
            total_change += abs(new_rank - ranks[node])

        ranks = new_ranks

        if total_change < tolerance:
            break

    return ranks