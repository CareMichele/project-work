from tqdm import tqdm
import numpy as np
import networkx as nx

# prepares the basic data, to speed up GA assessments
def get_problem_data(p):
    n = p.graph.number_of_nodes()

    gold_dict = nx.get_node_attributes(p.graph, "gold")
    gold_values = np.array([float(gold_dict[i]) for i in range(n)], dtype=float)

    # Create a n x n matrix with the minimum distances between all pairs of nodes
    dist_matrix = np.zeros((n, n), dtype=float)
    for src, dist_map in tqdm(nx.all_pairs_dijkstra_path_length(p.graph, weight="dist"),
                              total=n, desc="Precompute dist", disable=(n < 300)):
        for dst, d in dist_map.items():
            dist_matrix[src][dst] = float(d)

    # pre-calculate and save all minimum paths between each pair of nodes
    path_map = dict(nx.all_pairs_dijkstra_path(p.graph, weight="dist"))

    return n, gold_values, dist_matrix, path_map

#calculate the total cost of a path using function cost
def check_solution_score(p, path):
    total = 0.0
    current = 0
    weight = 0.0
    for nxt, g in path:
        total += float(p.cost([current, nxt], weight))
        current = nxt
        if current == 0:
            weight = 0.0
        else:
            weight += float(g)
    return total

