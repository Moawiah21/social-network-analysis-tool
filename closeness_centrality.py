def calculate_closeness_centrality(G):
    """
    Manual implementation of Closeness Centrality.
 
    Closeness centrality of a node = (n - 1) / sum of shortest path distances to all other nodes.
    A higher score means the node can reach all others more quickly.
 
    For disconnected graphs, only reachable nodes are counted (wasserman normalisation).
 
    Parameters:
        G: NetworkX graph
 
    Returns:
        Dictionary of node -> closeness centrality score
    """
    import collections
 
    centrality = {}
    n = len(G.nodes())
 
    for source in G.nodes():
 
        # BFS to find shortest distances from source to all reachable nodes
        visited = {source: 0}
        queue = collections.deque([source])
 
        while queue:
            current = queue.popleft()
            for neighbor in G.neighbors(current):
                if neighbor not in visited:
                    visited[neighbor] = visited[current] + 1
                    queue.append(neighbor)
 
        # Sum of distances to all reachable nodes (excluding source itself)
        reachable = len(visited) - 1  # excludes source
        total_distance = sum(visited.values())
 
        if reachable == 0 or total_distance == 0:
            centrality[source] = 0.0
        else:
            # Wasserman normalisation handles disconnected graphs fairly
            centrality[source] = (reachable / (n - 1)) * (reachable / total_distance)
 
    return centrality
 