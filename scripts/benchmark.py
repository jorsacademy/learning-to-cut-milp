import argparse

import numpy as np
import torch

from learning_to_cut import CutScorer, generate_knapsack, run_cutting_plane


def load_model(path: str) -> CutScorer:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = CutScorer(hidden_dim=int(checkpoint["hidden_dim"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--instances", type=int, default=20)
    parser.add_argument("--items", type=int, default=18)
    parser.add_argument("--rounds", type=int, default=8)
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    policies = ["random", "efficacy", "learned", "oracle"]
    results = {policy: [] for policy in policies}

    for seed in range(args.instances):
        problem = generate_knapsack(args.items, seed=1000 + seed)
        for policy in policies:
            result = run_cutting_plane(
                problem,
                policy=policy,
                model=model if policy == "learned" else None,
                rounds=args.rounds,
                seed=seed,
            )
            results[policy].append((result.bound_improvement, result.lp_solves))

    for policy in policies:
        improvements = np.asarray([value[0] for value in results[policy]])
        lp_solves = np.asarray([value[1] for value in results[policy]])
        print(
            f"{policy:10s} "
            f"mean_bound_improvement={improvements.mean():.4f} "
            f"mean_lp_solves={lp_solves.mean():.2f}"
        )


if __name__ == "__main__":
    main()
