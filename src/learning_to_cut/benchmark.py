from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np

from .core import generate_knapsack, solve_milp
from .loop import Policy, run_cutting_plane
from .model import CutScorer


@dataclass(frozen=True)
class BenchmarkRow:
    instance_seed: int
    policy: str
    integer_optimum: float
    initial_bound: float
    final_bound: float
    gap_closure: float
    cuts_added: int
    main_lp_solves: int
    selection_lp_solves: int
    elapsed_seconds: float


@dataclass(frozen=True)
class BenchmarkSummary:
    policy: str
    n_instances: int
    mean_gap_closure: float
    median_gap_closure: float
    std_gap_closure: float
    mean_final_gap: float
    mean_cuts_added: float
    mean_main_lp_solves: float
    mean_selection_lp_solves: float
    mean_elapsed_seconds: float


@dataclass(frozen=True)
class PairedComparison:
    policy_a: str
    policy_b: str
    n_instances: int
    mean_gap_closure_difference: float
    standard_error: float


def run_benchmark(
    model: CutScorer | None,
    *,
    n_instances: int = 30,
    n_items: int = 24,
    n_constraints: int = 5,
    rounds: int = 6,
    seed_start: int = 10_000,
    policies: tuple[Policy, ...] = (
        "random",
        "efficacy",
        "hybrid",
        "learned",
        "oracle",
    ),
    randomized_orders: int = 8,
    max_candidates: int = 64,
) -> list[BenchmarkRow]:
    if n_instances < 1:
        raise ValueError("n_instances must be positive")
    if "learned" in policies and model is None:
        raise ValueError("a model is required when benchmarking the learned policy")

    rows: list[BenchmarkRow] = []
    for offset in range(n_instances):
        instance_seed = seed_start + offset
        problem = generate_knapsack(
            n_items=n_items,
            n_constraints=n_constraints,
            seed=instance_seed,
        )
        integer_optimum = solve_milp(problem).objective

        for policy in policies:
            start = perf_counter()
            result = run_cutting_plane(
                problem,
                policy=policy,
                model=model if policy == "learned" else None,
                rounds=rounds,
                seed=instance_seed,
                randomized_orders=randomized_orders,
                max_candidates=max_candidates,
                integer_optimum=integer_optimum,
            )
            elapsed = perf_counter() - start
            closure = result.gap_closure
            if closure is None:
                raise AssertionError("benchmark result unexpectedly lacks an integer optimum")
            rows.append(
                BenchmarkRow(
                    instance_seed=instance_seed,
                    policy=policy,
                    integer_optimum=integer_optimum,
                    initial_bound=result.initial_bound,
                    final_bound=result.final_bound,
                    gap_closure=closure,
                    cuts_added=result.cuts_added,
                    main_lp_solves=result.main_lp_solves,
                    selection_lp_solves=result.selection_lp_solves,
                    elapsed_seconds=elapsed,
                )
            )
    return rows


def summarize_benchmark(rows: list[BenchmarkRow]) -> list[BenchmarkSummary]:
    if not rows:
        raise ValueError("rows cannot be empty")
    policies = sorted({row.policy for row in rows})
    summaries: list[BenchmarkSummary] = []
    for policy in policies:
        selected = [row for row in rows if row.policy == policy]
        closures = np.asarray([row.gap_closure for row in selected], dtype=float)
        final_gaps = np.asarray(
            [row.final_bound - row.integer_optimum for row in selected], dtype=float
        )
        cuts = np.asarray([row.cuts_added for row in selected], dtype=float)
        main_solves = np.asarray([row.main_lp_solves for row in selected], dtype=float)
        selection_solves = np.asarray(
            [row.selection_lp_solves for row in selected], dtype=float
        )
        times = np.asarray([row.elapsed_seconds for row in selected], dtype=float)
        summaries.append(
            BenchmarkSummary(
                policy=policy,
                n_instances=len(selected),
                mean_gap_closure=float(closures.mean()),
                median_gap_closure=float(np.median(closures)),
                std_gap_closure=float(closures.std(ddof=1)) if len(selected) > 1 else 0.0,
                mean_final_gap=float(final_gaps.mean()),
                mean_cuts_added=float(cuts.mean()),
                mean_main_lp_solves=float(main_solves.mean()),
                mean_selection_lp_solves=float(selection_solves.mean()),
                mean_elapsed_seconds=float(times.mean()),
            )
        )
    return summaries


def paired_gap_closure_comparison(
    rows: list[BenchmarkRow], policy_a: str, policy_b: str
) -> PairedComparison:
    by_policy: dict[str, dict[int, float]] = {}
    for row in rows:
        by_policy.setdefault(row.policy, {})[row.instance_seed] = row.gap_closure
    if policy_a not in by_policy or policy_b not in by_policy:
        raise ValueError("both policies must appear in rows")
    common = sorted(set(by_policy[policy_a]) & set(by_policy[policy_b]))
    if not common:
        raise ValueError("policies have no common instance seeds")
    differences = np.asarray(
        [by_policy[policy_a][seed] - by_policy[policy_b][seed] for seed in common],
        dtype=float,
    )
    standard_error = (
        float(differences.std(ddof=1) / np.sqrt(len(differences)))
        if len(differences) > 1
        else 0.0
    )
    return PairedComparison(
        policy_a=policy_a,
        policy_b=policy_b,
        n_instances=len(common),
        mean_gap_closure_difference=float(differences.mean()),
        standard_error=standard_error,
    )


def benchmark_to_dict(
    rows: list[BenchmarkRow],
    summaries: list[BenchmarkSummary],
    comparison: PairedComparison | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "rows": [asdict(row) for row in rows],
        "summaries": [asdict(summary) for summary in summaries],
    }
    if comparison is not None:
        payload["paired_comparison"] = asdict(comparison)
    return payload
