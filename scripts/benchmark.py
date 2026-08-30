from __future__ import annotations

import argparse
import json

from learning_to_cut import (
    benchmark_to_dict,
    load_cut_scorer,
    paired_gap_closure_comparison,
    run_benchmark,
    summarize_benchmark,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark learned and heuristic cut selectors on unseen MILP instances."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--instances", type=int, default=30)
    parser.add_argument("--items", type=int, default=24)
    parser.add_argument("--constraints", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--seed-start", type=int, default=200_000)
    parser.add_argument("--json", action="store_true", dest="json_output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, metadata = load_cut_scorer(args.checkpoint)
    rows = run_benchmark(
        model,
        n_instances=args.instances,
        n_items=args.items,
        n_constraints=args.constraints,
        rounds=args.rounds,
        seed_start=args.seed_start,
    )
    summaries = summarize_benchmark(rows)
    comparison = paired_gap_closure_comparison(rows, "learned", "efficacy")

    if args.json_output:
        payload = benchmark_to_dict(rows, summaries, comparison)
        payload["checkpoint_metadata"] = metadata
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print("policy      gap_closed  final_gap  cuts  main_lp  selection_lp  seconds")
    for summary in summaries:
        print(
            f"{summary.policy:10s} "
            f"{summary.mean_gap_closure:10.3f} "
            f"{summary.mean_final_gap:10.3f} "
            f"{summary.mean_cuts_added:5.2f} "
            f"{summary.mean_main_lp_solves:8.2f} "
            f"{summary.mean_selection_lp_solves:12.2f} "
            f"{summary.mean_elapsed_seconds:8.4f}"
        )
    print(
        "paired learned-efficacy gap-closure difference: "
        f"{comparison.mean_gap_closure_difference:+.4f} "
        f"(SE {comparison.standard_error:.4f}, n={comparison.n_instances})"
    )


if __name__ == "__main__":
    main()
