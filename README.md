# project-work

## Solution Overview

The entry point is [s349483.py](s349483.py). The `solution(p)` function selects the algorithm based on the instance size:

- **n <= 100**: Genetic Algorithm (GA) in [src/ga_solver.py](src/ga_solver.py).
- **n > 100**: Fast constructive heuristic in [src/large_n_solver.py](src/large_n_solver.py).

Both algorithms return a full path as a list of `(node, gold_picked)` pairs. The path always ends at base `(0, 0.0)`.

## Algorithm for n <= 100 (Genetic Algorithm)

Implementation: [src/ga_solver.py](src/ga_solver.py)

1. **Precompute shortest paths**: compute all-pairs shortest paths and distances once (`get_problem_data`). This avoids repeated Dijkstra calls during evaluation.
2. **Population initialization**:
    - One greedy tour starting from the base (nearest by distance).
    - Several nearest-neighbor tours starting from random cities.
    - The rest are random permutations.
    - This mix gives a good balance between quality and diversity.
3. **Fitness evaluation (cost simulation)**:
    - Each candidate is a permutation of cities (no base node inside the genome).
    - The simulation keeps track of carried gold and remaining gold per city.
    - At each city, it picks a fixed fraction `PICKUP_FRACTION = 0.4` of the remaining gold.
    - For the next move, it compares two options:
      - **A**: go directly to the next city and return to base afterward.
      - **B**: return to base first, then go to the next city and return.
    - The cheaper option is chosen locally and its cost is accumulated.
    - This approximates when it is convenient to unload without explicitly modeling multiple trips per city during the GA search.
4. **GA operators**:
    - **Selection**: tournament selection with small pool size.
    - **Crossover**: order crossover that preserves relative order and yields valid permutations.
    - **Mutation**: random segment shuffle, with a mutation rate that adapts based on progress.
5. **Reconstruction**:
    - The best permutation is converted into a full path by inserting the shortest paths between consecutive cities and the base.

Rationale: GA explores the order of visiting cities, while the local return decision handles the unload timing. This is effective for medium-size instances where a full combinatorial search is too expensive but GA can still explore a large portion of the space.

## Algorithm for n > 100 (Constructive Heuristic)

Implementation: [src/large_n_solver.py](src/large_n_solver.py)

1. **Nearest-neighbor tour** based on Euclidean coordinates (fast ordering on positions, not on graph distances).
2. **Greedy pickup policy** with fixed fraction `PICKUP_FRACTION = 0.4`.
3. **Local return decision** at each step:
    - Compute the cost of two alternatives using the current carried gold:
      - **A**: go to next city, then return to base.
      - **B**: return to base first, then go to the next city and return.
    - Select the cheaper alternative and append the corresponding shortest paths to the output path.
4. **Cleanup pass**:
    - After the main tour, if any city still has gold, visit remaining cities in nearest-neighbor order.
    - Apply the same return decision and append the resulting paths.

Rationale: for large `n`, exhaustive evaluation or population search is too slow. The heuristic is linear in the number of cities (plus shortest-path lookups) and still uses the cost model to decide when to unload at the base.


