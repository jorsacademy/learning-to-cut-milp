from __future__ import annotations

import argparse
import json

from .core import generate_knapsack, solve_milp
from .loop import run_cutting_plane
from .model import load_cut_scorer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a transparent root-node cut-selection comparison on one MILP instance."
    )
    parser.add_argument("--items", type=int, default=20)
    parser.add_argument("--constraints", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    problem = generate_knapsack(
        n_items=args.items,
        n_constraints=args.constraints,
        seed=args.seed,
    )
    optimum = solve_milp(problem).objective
    model = None
    policies = ["random", "efficacy", "hybrid", "oracle"]
    if args.checkpoint:
        model, _ = load_cut_scorer(args.checkpoint)
        policies.insert(3, "learned")

    payload: dict[str, object] = {
        "instance": {
            "seed": args.seed,
            "items": args.items,
            "constraints": args.constraints,
            "integer_optimum": optimum,
        },
        "policies": {},
    }
    policy_payload = payload["policies"]
    if not isinstance(policy_payload, dict):
        raise AssertionError("internal payload construction error")

    for policy in policies:
        result = run_cutting_plane(
            problem,
            policy=policy,  # type: ignore[arg-type]
            model=model if policy == "learned" else None,
            rounds=args.rounds,
            seed=args.seed,
            integer_optimum=optimum,
        )
        policy_payload[policy] = {
            "initial_bound": result.initial_bound,
            "final_bound": result.final_bound,
            "gap_closure": result.gap_closure,
            "cuts_added": result.cuts_added,
            "main_lp_solves": result.main_lp_solves,
            "selection_lp_solves": result.selection_lp_solves,
        }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
