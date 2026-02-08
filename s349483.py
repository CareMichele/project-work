from Problem import Problem
from src.ga_solver import solve_ga
from src.large_n_solver import solve_large_n


def solution(p: Problem):
    n = p.graph.number_of_nodes()
    if n > 100:
        path = solve_large_n(p)
    else:
        path = solve_ga(p)
    print(path)
    return path


