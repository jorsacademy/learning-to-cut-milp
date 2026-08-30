from __future__ import annotations

import numpy as np
import torch

from learning_to_cut import (
    CutScorer,
    collect_cut_dataset,
    evaluate_ranking,
    load_cut_scorer,
    save_cut_scorer,
    train_cut_scorer,
)


def test_sequential_dataset_has_consistent_groups() -> None:
    dataset = collect_cut_dataset(
        n_instances=5,
        n_items=14,
        n_constraints=3,
        rounds=2,
        seed=30,
        randomized_orders=3,
        max_candidates=24,
    )
    assert dataset.n_groups >= 1
    assert dataset.n_candidates >= 2 * dataset.n_groups
    assert dataset.group_offsets[-1] == dataset.n_candidates
    assert torch.isfinite(dataset.features).all()
    assert torch.isfinite(dataset.expert_scores).all()
    assert torch.all(dataset.expert_scores >= -1e-9)


def test_training_produces_finite_ranking_metrics() -> None:
    torch.manual_seed(0)
    dataset = collect_cut_dataset(
        n_instances=6,
        n_items=14,
        n_constraints=3,
        rounds=2,
        seed=50,
        randomized_orders=3,
        max_candidates=24,
    )
    model = CutScorer(hidden_dim=16)
    report = train_cut_scorer(
        model,
        dataset,
        epochs=6,
        learning_rate=3e-3,
        seed=0,
    )
    metrics = evaluate_ranking(model, dataset)
    assert np.isfinite(report.initial_loss)
    assert np.isfinite(report.final_loss)
    assert report.final_loss <= report.initial_loss + 1e-6
    assert 0.0 <= metrics.top1_accuracy <= 1.0
    assert 0.0 <= metrics.mean_normalized_regret <= 1.0 + 1e-7


def test_checkpoint_round_trip_preserves_scores(tmp_path) -> None:
    dataset = collect_cut_dataset(
        n_instances=3,
        n_items=12,
        n_constraints=3,
        rounds=1,
        seed=70,
        randomized_orders=2,
        max_candidates=16,
    )
    model = CutScorer(hidden_dim=12)
    train_cut_scorer(model, dataset, epochs=2, seed=2)
    path = tmp_path / "cut_scorer.pt"
    save_cut_scorer(model, path, metadata={"purpose": "test"})
    restored, metadata = load_cut_scorer(path)
    with torch.no_grad():
        expected = model(dataset.features)
        actual = restored(dataset.features)
    assert torch.allclose(expected, actual)
    assert metadata == {"purpose": "test"}
