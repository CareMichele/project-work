from Problem import Problem
from src.ga_solver import solve_ga
from src.large_n_solver import solve_large_n


def solution(p: Problem):
    n = p.graph.number_of_nodes()
    if n > 100:
        return solve_large_n(p)
    return solve_ga(p)


