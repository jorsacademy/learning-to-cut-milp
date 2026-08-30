import numpy as np
import torch

from learning_to_cut import (
    CutScorer,
    collect_root_dataset,
    generate_cover_cuts,
    generate_knapsack,
    run_cutting_plane,
    solve_lp,
    train_cut_scorer,
)


def test_generated_cover_cuts_are_valid_and_violated_at_root() -> None:
    problem = generate_knapsack(n_items=18, seed=3)
    lp = solve_lp(problem)
    cuts = generate_cover_cuts(problem, lp.x)
    assert cuts
    for cut in cuts:
        idx = np.asarray(cut.indices, dtype=int)
        assert problem.weights[idx].sum() > problem.capacity
        assert lp.x[idx].sum() > cut.rhs


def test_cutting_loop_never_weakens_lp_bound() -> None:
    problem = generate_knapsack(n_items=18, seed=8)
    result = run_cutting_plane(problem, policy="efficacy", rounds=4)
    assert result.final_bound <= result.initial_bound + 1e-9
    assert result.lp_solves >= 1


def test_training_and_learned_policy_smoke() -> None:
    torch.manual_seed(0)
    dataset = collect_root_dataset(n_instances=8, n_items=14, seed=20)
    model = CutScorer(hidden_dim=16)
    loss = train_cut_scorer(model, dataset, epochs=3, learning_rate=1e-3)
    assert np.isfinite(loss)
    problem = generate_knapsack(n_items=14, seed=99)
    result = run_cutting_plane(problem, policy="learned", model=model, rounds=2)
    assert result.final_bound <= result.initial_bound + 1e-9
