from tqdm import tqdm
import numpy as np
import random

from .utils import estimate_density, get_problem_data

PICKUP_FRACTION = 0.4


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

        current = int(nxt)

    remaining_cities = [i for i in range(1, len(gold_values)) if remaining[i] > 0]
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
    child[l1:l2 + 1] = p1[l1:l2 + 1]

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
        segment = individual[l1:l2 + 1].copy()
        np.random.shuffle(segment)
        individual[l1:l2 + 1] = segment
    return individual


def reconstruct_path(order, gold_values, path_map, alpha, beta, graph, density):
    full_path = []
    current = 0
    carried = 0.0
    remaining = gold_values.copy()

    def get_path(u, v):
        if u == v:
            return [u]
        return path_map[u][v]

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

    order_iter = tqdm(order, desc="Reconstruct", disable=(len(order) < 300), leave=False)
    for nxt in order_iter:
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

        current = nxt

    remaining_cities = [i for i in range(1, len(gold_values)) if remaining[i] > 0]
    cleanup_bar = None
    if len(remaining_cities) > 0:
        cleanup_bar = tqdm(total=len(remaining_cities), desc="Cleanup", disable=(len(remaining_cities) < 300), leave=False)
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
        if cleanup_bar is not None:
            cleanup_bar.update(1)

    if cleanup_bar is not None:
        cleanup_bar.close()

    if current != 0:
        append_path(get_path(current, 0), gold_amount=0.0)

    return full_path


def solve_ga(p):
    n, gold_values, dist_matrix, path_map = get_problem_data(p)
    density = estimate_density(p)
    alpha, beta = float(p.alpha), float(p.beta)

    pop_size = max(100, int(10 * np.sqrt(n)))
    max_generations = max(200, int(20 * np.sqrt(n)))
    mutation_rate = 0.1 if n < 50 else 0.2
    offspring_size = int(pop_size * 0.5)

    raw_pop = initialize_population(n, pop_size, dist_matrix)

    population = []
    for tour in raw_pop:
        fit = calculate_fitness(tour, gold_values, p.graph, path_map, alpha, beta, density)
        population.append((fit, tour))

    population.sort(key=lambda x: x[0])
    best_fit, best_tour = population[0]

    no_improv = 0
    for _ in tqdm(range(max_generations), desc="GA"):
        offspring = []
        for _ in range(offspring_size):
            p1 = tournament_selection(population)[1]
            p2 = tournament_selection(population)[1]

            child = crossover(p1, p2) if np.random.random() < 0.8 else p1.copy()
            child = mutate(child, mutation_rate)

            child_fit = calculate_fitness(child, gold_values, p.graph, path_map, alpha, beta, density)
            offspring.append((child_fit, child))

        population.extend(offspring)
        population.sort(key=lambda x: x[0])
        population = population[:pop_size]

        if population[0][0] < best_fit:
            best_fit, best_tour = population[0]
            no_improv = 0
            mutation_rate = max(0.1, mutation_rate * 0.9)
        else:
            no_improv += 1

        if no_improv > 30:
            mutation_rate = min(0.6, mutation_rate * 1.5)
            no_improv = 15

    return reconstruct_path(best_tour, gold_values, path_map, alpha, beta, p.graph, density)
