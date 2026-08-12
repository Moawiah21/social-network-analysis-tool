def calculate_degree_centrality(G):
    """
    Manual implementation of degree centrality.
    Degree centrality of a node = degree / (n - 1)
    """
    n = len(G.nodes())
    centrality = {}

    if n <= 1:
        for node in G.nodes():
            centrality[node] = 0
        return centrality

    for node in G.nodes():
        degree = len(list(G.neighbors(node)))
        centrality[node] = degree / (n - 1)

    return centrality