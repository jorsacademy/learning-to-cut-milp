from __future__ import annotations

import numpy as np

from learning_to_cut import (
    cut_features,
    cut_violation,
    generate_cover_cuts,
    generate_knapsack,
    integrality_gap_closure,
    is_minimal_cover,
    is_valid_cover_cut,
    solve_lp,
    solve_milp,
)


def test_problem_generation_is_reproducible() -> None:
    left = generate_knapsack(n_items=16, n_constraints=4, seed=12)
    right = generate_knapsack(n_items=16, n_constraints=4, seed=12)
    assert np.array_equal(left.profits, right.profits)
    assert np.array_equal(left.weights, right.weights)
    assert np.array_equal(left.capacities, right.capacities)


def test_lp_relaxation_is_an_upper_bound_on_binary_optimum() -> None:
    problem = generate_knapsack(n_items=14, n_constraints=3, seed=4)
    lp = solve_lp(problem)
    mip = solve_milp(problem)
    assert lp.objective + 1e-8 >= mip.objective


def test_generated_cuts_are_violated_minimal_valid_covers() -> None:
    problem = generate_knapsack(n_items=18, n_constraints=4, seed=9)
    lp = solve_lp(problem)
    cuts = generate_cover_cuts(problem, lp.x, seed=91)
    assert cuts
    for cut in cuts:
        assert is_valid_cover_cut(problem, cut)
        assert is_minimal_cover(problem, cut)
        assert cut_violation(lp.x, cut) > 1e-8
        features = cut_features(problem, lp.x, cut)
        assert np.all(np.isfinite(features))


def test_valid_cover_cuts_preserve_integer_optimum() -> None:
    problem = generate_knapsack(n_items=12, n_constraints=3, seed=15)
    root = solve_lp(problem)
    optimum = solve_milp(problem).objective
    cuts = generate_cover_cuts(problem, root.x, seed=151)
    assert cuts
    for cut in cuts[:5]:
        cut_mip = solve_milp(problem, [cut])
        cut_lp = solve_lp(problem, [cut])
        assert abs(cut_mip.objective - optimum) <= 1e-7
        assert cut_lp.objective <= root.objective + 1e-8


def test_candidate_generation_is_deterministic_under_fixed_seed() -> None:
    problem = generate_knapsack(n_items=18, n_constraints=4, seed=21)
    lp = solve_lp(problem)
    first = generate_cover_cuts(problem, lp.x, seed=123)
    second = generate_cover_cuts(problem, lp.x, seed=123)
    assert first == second


def test_gap_closure_definition() -> None:
    assert integrality_gap_closure(120.0, 110.0, 100.0) == 0.5
    assert integrality_gap_closure(100.0, 100.0, 100.0) == 1.0
