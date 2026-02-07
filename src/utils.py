from tqdm import tqdm
import math
import numpy as np
import networkx as nx


def get_problem_data(p):
    n = p.graph.number_of_nodes()

    gold_dict = nx.get_node_attributes(p.graph, "gold")
    gold_values = np.array([float(gold_dict[i]) for i in range(n)], dtype=float)

    dist_matrix = np.zeros((n, n), dtype=float)
    for src, dist_map in tqdm(nx.all_pairs_dijkstra_path_length(p.graph, weight="dist"),
                              total=n, desc="Precompute dist", disable=(n < 300)):
        for dst, d in dist_map.items():
            dist_matrix[src][dst] = float(d)

    path_map = dict(nx.all_pairs_dijkstra_path(p.graph, weight="dist"))

    return n, gold_values, dist_matrix, path_map


def estimate_density(p):
    if hasattr(p, "density"):
        try:
            return float(p.density)
        except Exception:
            pass

    n = p.graph.number_of_nodes()
    if n <= 1:
        return 1.0
    e = p.graph.number_of_edges()
    return 2 * e / (n * (n - 1))


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

"""
def compute_pickup_cap(dist_to_base, alpha, beta):
    if alpha <= 0 or beta <= 1:
        return float("inf")
    if dist_to_base <= 0:
        return float("inf")
    ratio = 3.0
    cap = (ratio ** (1.0 / beta)) * (dist_to_base ** ((1.0 - beta) / beta)) / alpha
    return max(cap, 0.0)


def compute_chunk_plan(gold, cap):
    if not math.isfinite(cap) or cap <= 0:
        return 0, gold
    if gold <= cap:
        return 0, gold
    trips = int(math.ceil(gold / cap))
    extra_trips = max(0, trips - 1)
    last_chunk = gold - (extra_trips * cap)
    return extra_trips, last_chunk
"""