import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from itertools import product
import numpy as np
import pandas as pd
from tqdm import tqdm
from Problem import Problem
from s349483 import solution, check_solution_score

def compute_trip_limit(alpha, beta, density):
    if alpha <= 0 or beta < 2:
        return np.inf
    return max(1, int((6 - beta) * (0.5 + density / 2)))

def run_grid():
    results = []

    n_cities = [10, 50, 100]
    alpha_values = [0.0, 1.0, 2.0, 4.0]
    beta_values = [0.5, 1.0, 2.0, 4.0]
    density_values = [0.2, 0.5, 1.0]
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
            trip_limit_used = compute_trip_limit(alpha, beta, density)
            
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
    df.to_csv("results_grid_split_gold_fraction_0_4.csv", index=False)

    # summary
    ok = df[np.isfinite(df["my_cost"]) & np.isfinite(df["baseline_cost"])]
    win_rate = (ok["my_cost"] < ok["baseline_cost"]).mean() * 100 if len(ok) else 0.0

    print("\nSaved: results_grid_split_gold_fraction_0_4.csv")
    print(f"Valid runs: {len(ok)}/{len(df)} | Win rate: {win_rate:.1f}%")

if __name__ == "__main__":
    run_grid()