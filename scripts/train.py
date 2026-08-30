from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import torch

from learning_to_cut import (
    CutScorer,
    collect_cut_dataset,
    evaluate_ranking,
    save_cut_scorer,
    train_cut_scorer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a cut-ranking model on sequential multidimensional-knapsack states."
    )
    parser.add_argument("--instances", type=int, default=80)
    parser.add_argument("--validation-instances", type=int, default=24)
    parser.add_argument("--items", type=int, default=24)
    parser.add_argument("--constraints", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", default="checkpoints/cut_scorer.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    training = collect_cut_dataset(
        n_instances=args.instances,
        n_items=args.items,
        n_constraints=args.constraints,
        rounds=args.rounds,
        seed=args.seed,
    )
    validation_seed = args.seed + 100_000
    validation = collect_cut_dataset(
        n_instances=args.validation_instances,
        n_items=args.items,
        n_constraints=args.constraints,
        rounds=args.rounds,
        seed=validation_seed,
    )

    model = CutScorer(hidden_dim=args.hidden_dim)
    report = train_cut_scorer(
        model,
        training,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    validation_metrics = evaluate_ranking(model, validation)
    metadata = {
        "training_seed": args.seed,
        "validation_seed": validation_seed,
        "n_items": args.items,
        "n_constraints": args.constraints,
        "rounds": args.rounds,
        "training_instances": args.instances,
        "validation_instances": args.validation_instances,
    }
    save_cut_scorer(model, args.checkpoint, metadata=metadata)

    payload = {
        "checkpoint": args.checkpoint,
        "training_candidates": training.n_candidates,
        "training_groups": training.n_groups,
        "validation_candidates": validation.n_candidates,
        "validation_groups": validation.n_groups,
        "training": asdict(report),
        "validation": asdict(validation_metrics),
        "metadata": metadata,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
