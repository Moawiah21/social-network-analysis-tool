import networkx as nx


def load_karate_club():
    """
    Loads the famous Karate Club dataset.
    """
    return nx.karate_club_graph()


def load_random_graph(num_nodes=50, probability=0.1, seed=42):
    """
    Generates a random Erdős–Rényi graph.
    """
    return nx.erdos_renyi_graph(num_nodes, probability, seed=seed)


def load_connected_random_graph(num_nodes=50, probability=0.1, seed=42):
    """
    Generates a connected random graph by retrying until connected.
    """
    while True:
        G = nx.erdos_renyi_graph(num_nodes, probability, seed=seed)
        if nx.is_connected(G):
            return G
        seed += 1