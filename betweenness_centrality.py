import collections


def bfs_shortest_paths(G, source):
    """
    Perform BFS from source node to find:
    - The shortest distance from source to every other node
    - All predecessors on shortest paths (to reconstruct all shortest paths)
    - The number of shortest paths from source to every other node

    Parameters:
        G: NetworkX graph
        source: starting node

    Returns:
        distances   : dict of node -> shortest distance from source
        predecessors: dict of node -> list of nodes that precede it on shortest paths
        num_paths   : dict of node -> number of shortest paths from source to that node
        visit_order : list of nodes in the order they were finalised by BFS
    """
    distances = {source: 0}
    num_paths = {source: 1}
    predecessors = {node: [] for node in G.nodes()}

    queue = collections.deque([source])
    visit_order = []

    while queue:
        current = queue.popleft()
        visit_order.append(current)

        for neighbor in G.neighbors(current):
            # First time we reach this neighbor
            if neighbor not in distances:
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)

            # If this path is equally short, record it
            if distances[neighbor] == distances[current] + 1:
                num_paths[neighbor] = num_paths.get(neighbor, 0) + num_paths[current]
                predecessors[neighbor].append(current)

    return distances, predecessors, num_paths, visit_order


def calculate_betweenness_centrality(G):
    """
    Fully manual implementation of Betweenness Centrality using BFS.
    No NetworkX path functions are used internally.

    Uses Brandes' algorithm approach:
    1. BFS from each source node to find shortest path counts and predecessors
    2. Back-propagate dependency scores through the graph
    3. Accumulate betweenness scores and normalise

    Parameters:
        G: NetworkX graph

    Returns:
        Dictionary of node -> betweenness centrality score
    """
    nodes = list(G.nodes())
    n = len(nodes)
    betweenness = {node: 0.0 for node in nodes}

    for source in nodes:
        distances, predecessors, num_paths, visit_order = bfs_shortest_paths(G, source)

        # Dependency scores: how much of source's betweenness flows through each node
        dependency = {node: 0.0 for node in nodes}

        # Process nodes in reverse BFS order (furthest first)
        for node in reversed(visit_order):
            for predecessor in predecessors[node]:
                if num_paths[node] > 0:
                    # Fraction of shortest paths through predecessor -> node
                    fraction = (num_paths[predecessor] / num_paths[node])
                    dependency[predecessor] += fraction * (1 + dependency[node])

            if node != source:
                betweenness[node] += dependency[node]

    # Normalise for undirected graph
    # Each pair (s, t) is counted twice so divide by 2
    if n > 2:
        scale = 1 / ((n - 1) * (n - 2))
        for node in betweenness:
            betweenness[node] *= scale

    return betweenness