from Problem import Problem
import networkx as nx
import numpy as np
from tqdm import tqdm

PICKUP_FRACTION = 0.4

# This function calculates the minimum distances and paths 
# from the base (node 0) to all other nodes once, 
# so that I can reuse them quickly.
def precompute_base_paths(graph):
    dist0, path0 = nx.single_source_dijkstra(graph, source=0, weight="dist")
    path_to0 = {v: list(reversed(path)) for v, path in path0.items()}
    return dist0, path0, path_to0

# This function returns the minimum distance and path between u and v
def get_shortest_path(graph, u, v):
    if u == v:
        return 0.0, [u]

    if graph.has_edge(u, v):
        dist = float(graph[u][v]["dist"])
        path = [u, v]
    else:
        # calculate the shortest path 
        # using Dijkstra's algorithm and the total distance 
        path = nx.shortest_path(graph, u, v, weight="dist")
        dist = float(nx.path_weight(graph, path, weight="dist"))
    return dist, path

# returns the minimum distance and path 
# between u and v using the cache when possible
def get_cached_path(graph, dist0, path0, path_to0, u, v):
    if u == 0:
        return float(dist0[v]), path0[v]
    if v == 0:
        return float(dist0[u]), path_to0[u]
    return get_shortest_path(graph, u, v)

# this function builds a nearest neighbor tour
# based on euclidean coordinates (not on graph distances)
def build_nn_tour(graph):
    n = graph.number_of_nodes()
    # set of cities to visit (all except base 0)
    unvisited = set(range(1, n))
    tour = []
    current = 0

    # pos[i] = (x,y)
    pos = nx.get_node_attributes(graph, "pos") #coordinates 

    # calculate the Euclidean distance squared between 
    # the coordinates of city a and city b
    def euclid(a, b):
        pa = pos.get(a)
        pb = pos.get(b)
        if pa is None or pb is None:
            return float("inf")
        dx = float(pa[0]) - float(pb[0])
        dy = float(pa[1]) - float(pb[1])
        return dx * dx + dy * dy

    for _ in tqdm(range(len(unvisited)), desc="NN tour"):
        #choose the closest city geometrically among the cities I haven't visited
        next_city = min(unvisited, key=lambda c: euclid(current, c))
        tour.append(int(next_city))
        unvisited.remove(next_city)
        current = next_city

    return tour


def solve_large_n(p: Problem):
    graph = p.graph
    # dist0 = dict dist0[v] => min distance from 0 to v
    # path0 = dict path0[v] => nodes list of min path from 0 to v
    # path_to0 = inverse path to come back to base
    dist0, path0, path_to0 = precompute_base_paths(graph)

    #nodes list (order of visit)
    tour = build_nn_tour(graph)
    
    n = graph.number_of_nodes()
    
    #extract the attribute gold of each node of graph and put it in a dict
    gold_dict = nx.get_node_attributes(graph, "gold")
    #creates an array with the gold of each node in order of indices
    gold_values = np.array([float(gold_dict[i]) for i in range(n)], dtype=float)
    #used for cleanup phase for choosing the next city nearest for euclidian term
    pos = nx.get_node_attributes(graph, "pos") 

    alpha, beta = float(p.alpha), float(p.beta)
    current = 0
    carried = 0.0
    remaining = gold_values.copy()
    full_path = []

    # calculate the total cost of a path given the weight transported
    def path_cost(path_nodes, weight):
        if not path_nodes or len(path_nodes) < 2:
            return 0.0
        cost = 0.0
        for u, v in zip(path_nodes, path_nodes[1:]): #zip creates consecutive pairs of path nodes
            d = float(graph[u][v]["dist"])
            cost += d + (d * alpha * weight) ** beta
        return cost

    def euclid(a, b):
        pa = pos.get(a)
        pb = pos.get(b)
        if pa is None or pb is None:
            return float("inf")
        dx = float(pa[0]) - float(pb[0])
        dy = float(pa[1]) - float(pb[1])
        return dx * dx + dy * dy

    # used to record the complete path, 
    # including intermediate nodes with zero gold 
    # and gold only in the final node.
    def append_path(path_nodes, gold_amount=0.0):
        end_node = path_nodes[-1] if path_nodes else None
        for node in path_nodes[1:]:
            g = gold_amount if end_node is not None and node == end_node else 0.0
            full_path.append((int(node), float(g)))

    def choose_pickup(g_rem):
        #it is advisable to take everything
        if alpha <= 0 or beta <= 1:
            return g_rem
        #takes only a fixed fraction
        return g_rem * PICKUP_FRACTION

    for nxt in tqdm(tour, desc="Policy"):
        # takes the remained gold of the city nxt
        g_rem = float(remaining[nxt])
        # if <= 0 skip the city
        if g_rem <= 0:
            continue
        # decides how much gold take
        g = choose_pickup(g_rem)
        # update the remaining gold of this city
        remaining[nxt] = g_rem - g

        # this 4 lines recover the shortest paths
        # They are used to calculate the costs of alternatives A and B
        _, p_curr_nxt = get_cached_path(graph, dist0, path0, path_to0, current, nxt)
        _, p_curr_base = get_cached_path(graph, dist0, path0, path_to0, current, 0)
        _, p_base_nxt = get_cached_path(graph, dist0, path0, path_to0, 0, nxt)
        _, p_nxt_base = get_cached_path(graph, dist0, path0, path_to0, nxt, 0)

        # calculate the cost of option A
        cA_go = path_cost(p_curr_nxt, carried)
        cA_ret = path_cost(p_nxt_base, carried + g)
        scoreA = cA_go + cA_ret

        # calculate the cost of option B
        cB_ret_now = path_cost(p_curr_base, carried)
        cB_go = path_cost(p_base_nxt, 0.0)
        cB_ret = path_cost(p_nxt_base, g)
        scoreB = cB_ret_now + cB_go + cB_ret

        #decides which option to use
        if scoreB < scoreA:
            if current != 0:
                append_path(p_curr_base, gold_amount=0.0)
            append_path(p_base_nxt, gold_amount=g)
            carried = g
        else:
            append_path(p_curr_nxt, gold_amount=g)
            carried += g

        #updates the current position
        current = int(nxt)

    # list of cities who still have gold 
    remaining_cities = [i for i in range(1, len(gold_values)) if remaining[i] > 0]
    cleanup_iters = len(remaining_cities)
    cleanup_bar = None
    if cleanup_iters > 0:
        cleanup_bar = tqdm(range(cleanup_iters), desc="Cleanup", leave=False)

    # The cleanup repeats the same mechanism 
    # but only for cities that still have gold after the first round
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
