from __future__ import annotations

import torch

from learning_to_cut import (
    CutScorer,
    collect_cut_dataset,
    generate_knapsack,
    run_cutting_plane,
    solve_milp,
    train_cut_scorer,
)


def test_oracle_is_best_one_step_selector_in_same_candidate_pool() -> None:
    problem = generate_knapsack(n_items=18, n_constraints=4, seed=101)
    optimum = solve_milp(problem).objective
    efficacy = run_cutting_plane(
        problem,
        policy="efficacy",
        rounds=1,
        seed=11,
        integer_optimum=optimum,
    )
    oracle = run_cutting_plane(
        problem,
        policy="oracle",
        rounds=1,
        seed=11,
        integer_optimum=optimum,
    )
    assert oracle.final_bound <= efficacy.final_bound + 1e-8
    assert oracle.selection_lp_solves >= 1
    assert oracle.gap_closure is not None
    assert 0.0 <= oracle.gap_closure <= 1.0


def test_cutting_trace_is_monotone() -> None:
    problem = generate_knapsack(n_items=18, n_constraints=4, seed=102)
    result = run_cutting_plane(problem, policy="hybrid", rounds=5, seed=22)
    assert result.final_bound <= result.initial_bound + 1e-8
    for record in result.trace:
        assert record.bound_after <= record.bound_before + 1e-8
        assert record.candidate_count >= 1


def test_learned_policy_smoke() -> None:
    torch.manual_seed(0)
    dataset = collect_cut_dataset(
        n_instances=4,
        n_items=14,
        n_constraints=3,
        rounds=2,
        seed=120,
        randomized_orders=3,
        max_candidates=24,
    )
    model = CutScorer(hidden_dim=16)
    train_cut_scorer(model, dataset, epochs=3, seed=0)
    problem = generate_knapsack(n_items=14, n_constraints=3, seed=999)
    result = run_cutting_plane(problem, policy="learned", model=model, rounds=3, seed=999)
    assert result.final_bound <= result.initial_bound + 1e-8
