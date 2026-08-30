from __future__ import annotations

from learning_to_cut import generate_knapsack, run_cutting_plane, solve_milp


def main() -> None:
    problem = generate_knapsack(n_items=20, n_constraints=4, seed=7)
    optimum = solve_milp(problem).objective
    for policy in ("random", "efficacy", "hybrid", "oracle"):
        result = run_cutting_plane(
            problem,
            policy=policy,
            rounds=5,
            seed=7,
            integer_optimum=optimum,
        )
        print(
            f"{policy:8s} root={result.initial_bound:.3f} "
            f"final={result.final_bound:.3f} "
            f"gap_closure={result.gap_closure:.3f} cuts={result.cuts_added}"
        )


if __name__ == "__main__":
    main()
