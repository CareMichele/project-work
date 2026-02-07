from Problem import Problem
import networkx as nx
import numpy as np
from tqdm import tqdm

PICKUP_FRACTION = 0.4


def precompute_base_paths(graph):
    # dist0: dizionario dist0[v] = distanza minima da 0 a v
    # path0: dizionario path0[v] = lista nodi del cammino minimo da 0 a v
    # path_to0: {node: [node, ..., 0]}
    dist0, path0 = nx.single_source_dijkstra(graph, source=0, weight="dist")
    path_to0 = {v: list(reversed(path)) for v, path in path0.items()}
    return dist0, path0, path_to0


def get_shortest_path(graph, u, v):
    if u == v:
        return 0.0, [u]

    if graph.has_edge(u, v):
        dist = float(graph[u][v]["dist"])
        path = [u, v]
    else:
        path = nx.shortest_path(graph, u, v, weight="dist")
        dist = float(nx.path_weight(graph, path, weight="dist"))
    return dist, path


def get_cached_path(graph, dist0, path0, path_to0, u, v):
    if u == 0:
        return float(dist0[v]), path0[v]
    if v == 0:
        return float(dist0[u]), path_to0[u]
    return get_shortest_path(graph, u, v)


def build_nn_tour(graph):
    # costruisco l'ordine in cui provo a visitare le città (tour nearest neighbor
    # basato su coordinate, non su grafo)
    n = graph.number_of_nodes()
    #insieme di città da visitare (tutte tranne la base 0)
    unvisited = set(range(1, n))
    tour = []
    current = 0

    #pos[i] = (x,y)
    pos = nx.get_node_attributes(graph, "pos") #coordinates 

    #calcola distanza euclidea al quadrato tra coordinate di città a e città b
    def euclid(a, b):
        pa = pos.get(a)
        pb = pos.get(b)
        if pa is None or pb is None:
            return float("inf")
        dx = float(pa[0]) - float(pb[0])
        dy = float(pa[1]) - float(pb[1])
        return dx * dx + dy * dy

    for _ in tqdm(range(len(unvisited)), desc="NN tour"):
        #scelgo la città più vicina geometricamente tra le città non visitate
        next_city = min(unvisited, key=lambda c: euclid(current, c))
        tour.append(int(next_city))
        unvisited.remove(next_city)
        current = next_city

    return tour


def solve_large_n(p: Problem):
    graph = p.graph
    dist0, path0, path_to0 = precompute_base_paths(graph)

    tour = build_nn_tour(graph)
    n = graph.number_of_nodes()
    gold_dict = nx.get_node_attributes(graph, "gold")
    gold_values = np.array([float(gold_dict[i]) for i in range(n)], dtype=float)
    pos = nx.get_node_attributes(graph, "pos") #for cleanup

    alpha, beta = float(p.alpha), float(p.beta)
    current = 0
    carried = 0.0
    remaining = gold_values.copy()
    full_path = []

    def path_cost(path_nodes, weight):
        if not path_nodes or len(path_nodes) < 2:
            return 0.0
        cost = 0.0
        for u, v in zip(path_nodes, path_nodes[1:]): #zip crea le coppie consecutive di nodi del path
            d = float(graph[u][v]["dist"])
            cost += d + (d * alpha * weight) ** beta
        return cost

    def path_distance(path_nodes):
        if not path_nodes or len(path_nodes) < 2:
            return 0.0
        dist = 0.0
        for u, v in zip(path_nodes, path_nodes[1:]):
            dist += float(graph[u][v]["dist"])
        return dist

    def euclid(a, b):
        pa = pos.get(a)
        pb = pos.get(b)
        if pa is None or pb is None:
            return float("inf")
        dx = float(pa[0]) - float(pb[0])
        dy = float(pa[1]) - float(pb[1])
        return dx * dx + dy * dy

    def append_path(path_nodes, gold_amount=0.0):
        end_node = path_nodes[-1] if path_nodes else None
        for node in path_nodes[1:]:
            g = gold_amount if end_node is not None and node == end_node else 0.0
            full_path.append((int(node), float(g)))

    def choose_pickup(g_rem):
        if alpha <= 0 or beta <= 1:
            return g_rem
        return g_rem * PICKUP_FRACTION

    for nxt in tqdm(tour, desc="Policy"):
        g_rem = float(remaining[nxt])
        if g_rem <= 0:
            continue
        g = choose_pickup(g_rem)
        remaining[nxt] = g_rem - g

        _, p_curr_nxt = get_cached_path(graph, dist0, path0, path_to0, current, nxt)
        _, p_curr_base = get_cached_path(graph, dist0, path0, path_to0, current, 0)
        _, p_base_nxt = get_cached_path(graph, dist0, path0, path_to0, 0, nxt)
        _, p_nxt_base = get_cached_path(graph, dist0, path0, path_to0, nxt, 0)

        cA_go = path_cost(p_curr_nxt, carried)
        cA_ret = path_cost(p_nxt_base, carried + g)
        scoreA = cA_go + cA_ret

        cB_ret_now = path_cost(p_curr_base, carried)
        cB_go = path_cost(p_base_nxt, 0.0)
        cB_ret = path_cost(p_nxt_base, g)
        scoreB = cB_ret_now + cB_go + cB_ret

        if scoreB < scoreA:
            if current != 0:
                append_path(p_curr_base, gold_amount=0.0)
            append_path(p_base_nxt, gold_amount=g)
            carried = g
        else:
            append_path(p_curr_nxt, gold_amount=g)
            carried += g

        current = int(nxt)

    remaining_cities = [i for i in range(1, len(gold_values)) if remaining[i] > 0]
    cleanup_iters = len(remaining_cities)
    cleanup_bar = None
    if cleanup_iters > 0:
        cleanup_bar = tqdm(range(cleanup_iters), desc="Cleanup", leave=False)

    while remaining_cities:
        nxt = min(remaining_cities, key=lambda c: euclid(current, c))
        g = float(remaining[nxt])

        _, p_curr_nxt = get_cached_path(graph, dist0, path0, path_to0, current, nxt)
        _, p_curr_base = get_cached_path(graph, dist0, path0, path_to0, current, 0)
        _, p_base_nxt = get_cached_path(graph, dist0, path0, path_to0, 0, nxt)
        _, p_nxt_base = get_cached_path(graph, dist0, path0, path_to0, nxt, 0)

        cA_go = path_cost(p_curr_nxt, carried)
        cA_ret = path_cost(p_nxt_base, carried + g)
        scoreA = cA_go + cA_ret

        cB_ret_now = path_cost(p_curr_base, carried)
        cB_go = path_cost(p_base_nxt, 0.0)
        cB_ret = path_cost(p_nxt_base, g)
        scoreB = cB_ret_now + cB_go + cB_ret

        if scoreB < scoreA:
            if current != 0:
                append_path(p_curr_base, gold_amount=0.0)
            append_path(p_base_nxt, gold_amount=g)
            carried = g
        else:
            append_path(p_curr_nxt, gold_amount=g)
            carried += g

        remaining[nxt] = 0.0
        remaining_cities.remove(nxt)
        current = int(nxt)

        if cleanup_bar is not None:
            cleanup_bar.update(1)

    if cleanup_bar is not None:
        cleanup_bar.close()

    if current != 0:
        _, p_curr_base = get_cached_path(graph, dist0, path0, path_to0, current, 0)
        append_path(p_curr_base, gold_amount=0.0)

    return full_path
