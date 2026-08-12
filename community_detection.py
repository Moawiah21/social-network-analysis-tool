import networkx as nx

def detect_communities_girvan_newman(G, level=2):
    """
    Detect communities using Girvan-Newman algorithm.

    level = number of communities to produce (minimum 2).
    Internally calls next() (level - 1) times on the generator.
    """
    comp = nx.community.girvan_newman(G)

    communities = None
    for _ in range(level - 1):
        communities = next(comp)

    if communities is None:
        communities = next(nx.community.girvan_newman(G))

    return [list(c) for c in communities]