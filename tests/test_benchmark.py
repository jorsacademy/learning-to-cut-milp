from __future__ import annotations

from learning_to_cut import (
    CutScorer,
    benchmark_to_dict,
    paired_gap_closure_comparison,
    run_benchmark,
    summarize_benchmark,
)


def test_benchmark_summaries_and_paired_comparison() -> None:
    model = CutScorer(hidden_dim=8)
    rows = run_benchmark(
        model,
        n_instances=2,
        n_items=12,
        n_constraints=3,
        rounds=2,
        seed_start=800,
        policies=("efficacy", "oracle"),
        randomized_orders=2,
        max_candidates=16,
    )
    summaries = summarize_benchmark(rows)
    comparison = paired_gap_closure_comparison(rows, "oracle", "efficacy")
    payload = benchmark_to_dict(rows, summaries, comparison)
    assert len(rows) == 4
    assert len(summaries) == 2
    assert comparison.n_instances == 2
    assert comparison.mean_gap_closure_difference >= -1e-8
    assert "paired_comparison" in payload
