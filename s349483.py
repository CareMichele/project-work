from tqdm import tqdm
from Problem import Problem
import numpy as np
import networkx as nx
import random
import math

PICKUP_FRACTION = 0.4

# --------------------------
# Data extraction
# --------------------------
def get_problem_data(p, threshold=100):
    n = p.graph.number_of_nodes()

    gold_dict = nx.get_node_attributes(p.graph, "gold")
    gold_values = np.array([float(gold_dict[i]) for i in range(n)], dtype=float)

    dist_matrix = None
    path_map = None

    if n <= threshold:
        print(f"N={n} <= {threshold}: pre-calcolo dist_matrix + path_map per GA...")

        # distanze minime (numeri) — veloce
        dist_matrix = np.zeros((n, n), dtype=float)
        for src, dist_map in nx.all_pairs_dijkstra_path_length(p.graph, weight="dist"):
            for dst, d in dist_map.items():
                dist_matrix[src][dst] = float(d)

        # shortest paths (lista nodi) — serve per fitness corretta + reconstruct
        path_map = dict(nx.all_pairs_dijkstra_path(p.graph, weight="dist"))
    else:
        print(f"N={n} > {threshold}: modalità lightweight (solo gold).")

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

"""
def compute_trip_limit(alpha, beta, density):
    if alpha <= 0 or beta < 2:
        return float("inf")

    if beta >= 4:
        base = 1
    elif beta >= 2:
        base = 1 if density <= 0.3 else 2
    else:
        return float("inf")

    density_bonus = int(density * 3)
    return max(1, min(8, base + density_bonus))
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


# --------------------------
# GA helpers (solo N piccoli)
# --------------------------
def greedy_from_base(n, dist_matrix):
    unvisited = set(range(1, n))
    tour = []
    current = 0
    while unvisited:
        nxt = min(unvisited, key=lambda c: dist_matrix[current][c])
        tour.append(nxt)
        unvisited.remove(nxt)
        current = nxt
    return np.array(tour, dtype=int)


def greedy_nearest_neighbor(n, dist_matrix):
    unvisited = set(range(1, n))
    start = random.choice(list(unvisited))
    tour = [start]
    unvisited.remove(start)
    current = start
    while unvisited:
        nxt = min(unvisited, key=lambda c: dist_matrix[current][c])
        tour.append(nxt)
        unvisited.remove(nxt)
        current = nxt
    return np.array(tour, dtype=int)


def initialize_population(n, pop_size, dist_matrix):
    population = []
    base_tour = np.arange(1, n, dtype=int)

    n_greedy = int(pop_size * 0.5) if n < 50 else int(pop_size * 0.2)

    for i in range(pop_size):
        if i == 0:
            ind = greedy_from_base(n, dist_matrix)
        elif i < n_greedy:
            ind = greedy_nearest_neighbor(n, dist_matrix)
        else:
            ind = base_tour.copy()
            np.random.shuffle(ind)
        population.append(ind)

    return population


def calculate_fitness(ind, gold_values, graph, path_map, alpha, beta, density):
    current = 0
    carried = 0.0
    total = 0.0
    remaining = gold_values.copy()

    def path_cost(path_nodes, weight):
        if not path_nodes or len(path_nodes) < 2:
            return 0.0
        cost = 0.0
        for u, v in zip(path_nodes, path_nodes[1:]):
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

    for nxt in ind:
        g_rem = float(remaining[nxt])
        if g_rem <= 0:
            continue
        if alpha <= 0 or beta <= 1:
            g = g_rem
        else:
            g = g_rem * PICKUP_FRACTION
        remaining[nxt] = g_rem - g

        p_curr_nxt = path_map[current][nxt]  #da posizione attuale a città 
        p_curr_base = path_map[current][0]   #da posizione attuale a base
        p_base_nxt = path_map[0][nxt]        # da base a città
        p_nxt_base = path_map[nxt][0]        # da città a base

        # A: continuo senza scaricare 
        cA_go = path_cost(p_curr_nxt, carried)
        cA_ret = path_cost(p_nxt_base, carried + g)
        scoreA = cA_go + cA_ret

        # B: scarico in base prima
        cB_ret_now = path_cost(p_curr_base, carried)
        cB_go = path_cost(p_base_nxt, 0.0)
        cB_ret = path_cost(p_nxt_base, g)
        scoreB = cB_ret_now + cB_go + cB_ret

        if scoreB < scoreA:
            total += cB_ret_now + cB_go
            carried = g
        else:
            total += cA_go
            carried += g

        current = int(nxt)
    
    #potrebbero esserci ancora città on oro residuo
    remaining_cities = [i for i in range(1, len(gold_values)) if remaining[i] > 0]
    #finchè ci sono città con oro rimanente, scelgo la più vicina e ripeto lo stesso schema di prima (confronto A o B)
    while remaining_cities:
        nxt = min(remaining_cities, key=lambda c: path_distance(path_map[current][c]))
        g = float(remaining[nxt])

        p_curr_nxt = path_map[current][nxt]
        p_curr_base = path_map[current][0]
        p_base_nxt = path_map[0][nxt]
        p_nxt_base = path_map[nxt][0]

        cA_go = path_cost(p_curr_nxt, carried)
        cA_ret = path_cost(p_nxt_base, carried + g)
        scoreA = cA_go + cA_ret

        cB_ret_now = path_cost(p_curr_base, carried)
        cB_go = path_cost(p_base_nxt, 0.0)
        cB_ret = path_cost(p_nxt_base, g)
        scoreB = cB_ret_now + cB_go + cB_ret

        if scoreB < scoreA:
            total += cB_ret_now + cB_go
            carried = g
        else:
            total += cA_go
            carried += g

        remaining[nxt] = 0.0
        remaining_cities.remove(nxt)
        current = int(nxt)

    total += path_cost(path_map[current][0], carried)
    return total


def tournament_selection(population, tau=3):
    pool = random.sample(population, k=tau)
    return min(pool, key=lambda x: x[0])


def crossover(p1, p2):
    size = len(p1)
    l1 = np.random.randint(0, size)
    l2 = np.random.randint(0, size)
    if l1 > l2:
        l1, l2 = l2, l1

    child = np.full(size, -1, dtype=int)
    child[l1:l2+1] = p1[l1:l2+1]

    pos = 0
    for x in p2:
        if x not in child:
            while l1 <= pos <= l2:
                pos += 1
                if pos >= size:
                    pos = 0
            child[pos] = x
            pos += 1
            if pos >= size:
                pos = 0

    return child


def mutate(individual, mutation_rate):
    if np.random.random() < mutation_rate:
        size = len(individual)
        l1 = np.random.randint(0, size)
        l2 = np.random.randint(0, size)
        if l1 > l2:
            l1, l2 = l2, l1
        segment = individual[l1:l2+1].copy()
        np.random.shuffle(segment)
        individual[l1:l2+1] = segment
    return individual


# --------------------------
# Path reconstruction (per output)
# --------------------------
def reconstruct_path(order, gold_values, path_map, alpha, beta, graph, density):
    full_path = []
    current = 0
    carried = 0.0
    remaining = gold_values.copy()

    local_cache = {}

    def get_path(u, v):
        if u == v:
            return [u]
        if path_map is not None:
            return path_map[u][v]
        key = (u, v)
        if key in local_cache:
            return local_cache[key]
        path = nx.shortest_path(graph, u, v, weight="dist")
        local_cache[key] = path
        return path

    def path_cost(path_nodes, weight):
        if not path_nodes or len(path_nodes) < 2:
            return 0.0
        cost = 0.0
        for u, v in zip(path_nodes, path_nodes[1:]):
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

    def append_path(path_nodes, gold_amount=0.0):
        end_node = path_nodes[-1] if path_nodes else None
        for node in path_nodes[1:]:
            g = gold_amount if end_node is not None and node == end_node else 0.0
            full_path.append((int(node), float(g)))

    for nxt in order:
        nxt = int(nxt)
        g_rem = float(remaining[nxt])
        if g_rem <= 0:
            continue
        if alpha <= 0 or beta <= 1:
            g = g_rem
        else:
            g = g_rem * PICKUP_FRACTION
        remaining[nxt] = g_rem - g

        p_curr_nxt = get_path(current, nxt)
        p_curr_base = get_path(current, 0)
        p_base_nxt = get_path(0, nxt)
        p_nxt_base = get_path(nxt, 0)

        # scenario A
        cA_go = path_cost(p_curr_nxt, carried)
        cA_ret = path_cost(p_nxt_base, carried + g)
        scoreA = cA_go + cA_ret

        # scenario B
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

        current = nxt

    remaining_cities = [i for i in range(1, len(gold_values)) if remaining[i] > 0]
    while remaining_cities:
        nxt = min(remaining_cities, key=lambda c: path_distance(get_path(current, c)))
        g = float(remaining[nxt])

        p_curr_nxt = get_path(current, nxt)
        p_curr_base = get_path(current, 0)
        p_base_nxt = get_path(0, nxt)
        p_nxt_base = get_path(nxt, 0)

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
        current = nxt

    if current != 0:
        append_path(get_path(current, 0), gold_amount=0.0)

    return full_path


# --------------------------
# MAIN solution
# --------------------------
def solution(p: Problem):
    n, gold_values, dist_matrix, path_map = get_problem_data(p, threshold=100)
    density = estimate_density(p)
    alpha, beta = float(p.alpha), float(p.beta)

    # -------- N grande: greedy lazy (come tuo) --------
    if n > 100:
        """
        print(f"Lazy Greedy per N={n}...")

        dist_from_base = nx.single_source_dijkstra_path_length(p.graph, 0, weight="dist")
        dist_to_base = dist_from_base  # grafo non diretto

        current = 0
        carried = 0.0
        unvisited = set(range(1, n))
        order = []

        limit = compute_trip_limit(alpha, beta, density)
        items_in_trip = 0

        pbar = tqdm(total=len(unvisited), desc="Greedy Building", disable=(n < 300))

        while unvisited:
            local_dists = nx.single_source_dijkstra_path_length(p.graph, current, weight="dist")

            candidates = list(unvisited)
            if len(candidates) > 50:
                candidates = random.sample(candidates, 50)

            best_next = min(
                candidates,
                key=lambda c: local_dists.get(c, float("inf")) + (local_dists.get(c, float("inf")) * alpha * carried) ** beta
            )

            d_go = local_dists.get(best_next, 1e9)
            d_ret_fut = dist_to_base.get(best_next, 1e9)
            d_ret_now = dist_to_base.get(current, 1e9)
            d_leave = dist_from_base.get(best_next, 1e9)

            cost_cont = d_go + (d_go * alpha * carried) ** beta + (d_ret_fut * alpha * (carried + gold_values[best_next])) ** beta
            cost_unload = (d_ret_now + (d_ret_now * alpha * carried) ** beta) + d_leave + (d_ret_fut * alpha * gold_values[best_next]) ** beta

            if (cost_unload < cost_cont) or (items_in_trip >= limit):
                carried = 0.0
                items_in_trip = 0

            order.append(best_next)
            unvisited.remove(best_next)
            current = best_next
            carried += float(gold_values[best_next])
            items_in_trip += 1

            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        return reconstruct_path(order, gold_values, None, alpha, beta, p.graph, density)
        """
    # -------- N piccolo: GA --------
    POP_SIZE = max(100, int(10 * np.sqrt(n)))
    MAX_GENERATIONS = max(200, int(20 * np.sqrt(n)))
    MUTATION_RATE = 0.1 if n < 50 else 0.2
    OFFSPRING_SIZE = int(POP_SIZE * 0.5)

    raw_pop = initialize_population(n, POP_SIZE, dist_matrix)

    population = []
    for tour in raw_pop:
        fit = calculate_fitness(tour, gold_values, p.graph, path_map, alpha, beta, density)
        population.append((fit, tour))

    population.sort(key=lambda x: x[0])
    best_fit, best_tour = population[0]

    no_improv = 0
    for _ in tqdm(range(MAX_GENERATIONS), desc="GA"):
        offspring = []
        for _ in range(OFFSPRING_SIZE):
            p1 = tournament_selection(population)[1]
            p2 = tournament_selection(population)[1]

            child = crossover(p1, p2) if np.random.random() < 0.8 else p1.copy()
            child = mutate(child, MUTATION_RATE)

            child_fit = calculate_fitness(child, gold_values, p.graph, path_map, alpha, beta, density)
            offspring.append((child_fit, child))

        population.extend(offspring)
        population.sort(key=lambda x: x[0])
        population = population[:POP_SIZE]

        if population[0][0] < best_fit:
            best_fit, best_tour = population[0]
            no_improv = 0
            MUTATION_RATE = max(0.1, MUTATION_RATE * 0.9)
        else:
            no_improv += 1

        if no_improv > 30:
            MUTATION_RATE = min(0.6, MUTATION_RATE * 1.5)
            no_improv = 15

    return reconstruct_path(best_tour, gold_values, path_map, alpha, beta, p.graph, density)


def check_solution_score(p: Problem, path):
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


if __name__ == '__main__':
    """
    p = Problem(1000, density=0.2, alpha=1, beta=1, seed=42)
    
    print("-" * 50)
    print("CALCOLO BASELINE")
    baseline_cost = p.baseline()
    print(f"Costo Baseline: {baseline_cost:,.2f}")
    
    #print("-" * 50)
    print("CALCOLO GENETIC ALGORITHM")
    solution_path = solution(p)
    
    # Calcolo il costo della tua soluzione
    my_cost = float(check_solution_score(p, solution_path))
    print(f"Costo Tuo GA:   {my_cost:,.2f}")
    
    print("-" * 50)
    # CONFRONTO
    gap = baseline_cost - my_cost
    improvement = (gap / baseline_cost) * 100
    
    if my_cost < baseline_cost:
        print(f"✅ VITTORIA! Hai risparmiato {gap:,.2f} ({improvement:.2f}%)")
    else:
        print(f"❌ SCONFITTA. La baseline è migliore di {-gap:,.2f}")
        
    print(f"Lunghezza percorso: {len(solution_path)} tappe")
    """
    from itertools import product
    import pandas as pd
    
    results = []

    n_cities = [10]
    alpha_values = [1.0]
    beta_values = [0.5,1.0,2.0,4.0]
    density_values = [0.5]
    seed = 42

    param_list = list(product(n_cities, density_values, alpha_values, beta_values))

    for n, density, alpha, beta in tqdm(param_list, desc="grid"):
        try:
            p = Problem(n, density=density, alpha=alpha, beta=beta, seed=seed)

            # baseline prof
            baseline_cost = float(p.baseline())

            # tua soluzione
            path = solution(p)

            # validazione minima: deve finire a 0
            ends_at_base = (len(path) > 0 and path[-1][0] == 0)

            # costo tuo
            my_cost = float(check_solution_score(p, path)) if ends_at_base else np.nan

            # miglioramento
            improvement = (baseline_cost - my_cost) / baseline_cost * 100.0 if np.isfinite(my_cost) else np.nan

            density_used = density
            #trip_limit_used = compute_trip_limit(alpha, beta, density)
            
            results.append({
                "n_cities": n,
                "density": density,
                "density_used": density_used,
                "alpha": alpha,
                "beta": beta,
                "trip_limit_used": trip_limit_used,
                "seed": seed,
                "baseline_cost": baseline_cost,
                "my_cost": my_cost,
                "improvement_pct": improvement,
                "wins": (my_cost < baseline_cost) if np.isfinite(my_cost) else False,
                "path_len": len(path),
            })

        except Exception as e:
            results.append({
                "n_cities": n,
                "density": density,
                "density_used": density,
                "alpha": alpha,
                "beta": beta,
                "trip_limit_used": np.nan,
                "seed": seed,
                "baseline_cost": np.nan,
                "my_cost": np.nan,
                "improvement_pct": np.nan,
                "wins": False,
                "path_len": np.nan,
                "error": f"{type(e).__name__}: {e}",
            })

    df = pd.DataFrame(results)
    df.to_csv("results_grid_1000.csv", index=False)

    
