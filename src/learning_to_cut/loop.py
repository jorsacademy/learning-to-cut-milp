from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from .core import (
    FEATURE_NAMES,
    KnapsackMILP,
    LPCut,
    LPResult,
    cut_features,
    generate_cover_cuts,
    integrality_gap_closure,
    rank_candidates_by_expert,
    solve_lp,
    solve_milp,
)
from .model import CutScorer

Policy = Literal["random", "efficacy", "hybrid", "oracle", "learned"]


@dataclass(frozen=True)
class RoundRecord:
    round_index: int
    bound_before: float
    bound_after: float
    candidate_count: int
    selected_cut: LPCut

    @property
    def bound_improvement(self) -> float:
        return self.bound_before - self.bound_after


@dataclass(frozen=True)
class CuttingResult:
    initial_bound: float
    final_bound: float
    integer_optimum: float | None
    cuts_added: int
    main_lp_solves: int
    selection_lp_solves: int
    trace: tuple[RoundRecord, ...]

    @property
    def lp_solves(self) -> int:
        return self.main_lp_solves + self.selection_lp_solves

    @property
    def bound_improvement(self) -> float:
        return self.initial_bound - self.final_bound

    @property
    def final_gap(self) -> float | None:
        if self.integer_optimum is None:
            return None
        return max(0.0, self.final_bound - self.integer_optimum)

    @property
    def gap_closure(self) -> float | None:
        if self.integer_optimum is None:
            return None
        return integrality_gap_closure(
            self.initial_bound,
            self.final_bound,
            self.integer_optimum,
        )


def _minmax(values: np.ndarray) -> np.ndarray:
    lo = float(values.min())
    hi = float(values.max())
    if hi - lo <= 1e-12:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def _hybrid_scores(features: np.ndarray) -> np.ndarray:
    """A transparent, non-SCIP heuristic combining several cut features."""

    efficacy = _minmax(features[:, FEATURE_NAMES.index("efficacy")])
    violation = _minmax(features[:, FEATURE_NAMES.index("violation")])
    objective_parallelism = _minmax(
        features[:, FEATURE_NAMES.index("objective_parallelism")]
    )
    density = _minmax(features[:, FEATURE_NAMES.index("density")])
    return 0.55 * efficacy + 0.20 * violation + 0.20 * objective_parallelism + 0.05 * (
        1.0 - density
    )


def _select_index(
    problem: KnapsackMILP,
    cuts: list[LPCut],
    lp: LPResult,
    candidates: list[LPCut],
    policy: Policy,
    rng: np.random.Generator,
    model: CutScorer | None,
) -> tuple[int, int]:
    if policy == "random":
        return int(rng.integers(len(candidates))), 0

    features = np.stack([cut_features(problem, lp.x, cut) for cut in candidates])
    if policy == "efficacy":
        return int(np.argmax(features[:, FEATURE_NAMES.index("efficacy")])), 0
    if policy == "hybrid":
        return int(np.argmax(_hybrid_scores(features))), 0
    if policy == "oracle":
        scores = rank_candidates_by_expert(problem, cuts, lp, candidates)
        return int(np.argmax(scores)), len(candidates)
    if policy == "learned":
        if model is None:
            raise ValueError("learned policy requires a model")
        model.eval()
        with torch.no_grad():
            tensor = torch.as_tensor(features, dtype=torch.float32)
            scores = model(tensor).cpu().numpy()
        return int(np.argmax(scores)), 0
    raise ValueError(f"unknown policy: {policy}")


def run_cutting_plane(
    problem: KnapsackMILP,
    policy: Policy = "efficacy",
    model: CutScorer | None = None,
    rounds: int = 8,
    seed: int = 0,
    *,
    randomized_orders: int = 8,
    max_candidates: int = 64,
    integer_optimum: float | None = None,
    compute_integer_optimum: bool = False,
) -> CuttingResult:
    """Run a root-node cutting-plane loop with a fixed cut-selection policy."""

    if rounds < 0:
        raise ValueError("rounds must be nonnegative")
    if integer_optimum is not None and compute_integer_optimum:
        raise ValueError("provide integer_optimum or request its computation, not both")
    if compute_integer_optimum:
        integer_optimum = solve_milp(problem).objective

    rng = np.random.default_rng(seed)
    cuts: list[LPCut] = []
    root = solve_lp(problem)
    main_lp_solves = 1
    selection_lp_solves = 0
    current = root
    trace: list[RoundRecord] = []

    for round_index in range(rounds):
        candidates = generate_cover_cuts(
            problem,
            current.x,
            seed=seed + 10_007 * (round_index + 1),
            randomized_orders=randomized_orders,
            max_candidates=max_candidates,
        )
        existing = {cut.indices for cut in cuts}
        candidates = [cut for cut in candidates if cut.indices not in existing]
        if not candidates:
            break

        selected_index, extra_lp_solves = _select_index(
            problem,
            cuts,
            current,
            candidates,
            policy,
            rng,
            model,
        )
        selection_lp_solves += extra_lp_solves
        selected_cut = candidates[selected_index]
        bound_before = current.objective
        cuts.append(selected_cut)
        current = solve_lp(problem, cuts)
        main_lp_solves += 1
        trace.append(
            RoundRecord(
                round_index=round_index,
                bound_before=bound_before,
                bound_after=current.objective,
                candidate_count=len(candidates),
                selected_cut=selected_cut,
            )
        )

    return CuttingResult(
        initial_bound=root.objective,
        final_bound=current.objective,
        integer_optimum=integer_optimum,
        cuts_added=len(cuts),
        main_lp_solves=main_lp_solves,
        selection_lp_solves=selection_lp_solves,
        trace=tuple(trace),
    )
