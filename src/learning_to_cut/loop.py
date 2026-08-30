from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from .core import (
    LPCut,
    KnapsackMILP,
    LPResult,
    cut_features,
    generate_cover_cuts,
    rank_candidates_by_expert,
    solve_lp,
)
from .model import CutScorer


Policy = Literal["random", "efficacy", "oracle", "learned"]


@dataclass(frozen=True)
class CuttingResult:
    initial_bound: float
    final_bound: float
    cuts_added: int
    lp_solves: int

    @property
    def bound_improvement(self) -> float:
        return self.initial_bound - self.final_bound


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
        return int(np.argmax(features[:, 1])), 0
    if policy == "oracle":
        scores = rank_candidates_by_expert(problem, cuts, lp, candidates)
        return int(np.argmax(scores)), len(candidates)
    if policy == "learned":
        if model is None:
            raise ValueError("learned policy requires a model")
        with torch.no_grad():
            scores = model(torch.tensor(features, dtype=torch.float32)).cpu().numpy()
        return int(np.argmax(scores)), 0
    raise ValueError(f"unknown policy: {policy}")


def run_cutting_plane(
    problem: KnapsackMILP,
    policy: Policy = "efficacy",
    model: CutScorer | None = None,
    rounds: int = 8,
    seed: int = 0,
) -> CuttingResult:
    if rounds < 0:
        raise ValueError("rounds must be nonnegative")
    rng = np.random.default_rng(seed)
    cuts: list[LPCut] = []
    root = solve_lp(problem)
    lp_solves = 1
    current = root

    for _ in range(rounds):
        candidates = generate_cover_cuts(problem, current.x)
        existing = {cut.indices for cut in cuts}
        candidates = [cut for cut in candidates if cut.indices not in existing]
        if not candidates:
            break
        selected, selection_lp_solves = _select_index(
            problem,
            cuts,
            current,
            candidates,
            policy,
            rng,
            model,
        )
        lp_solves += selection_lp_solves
        cuts.append(candidates[selected])
        current = solve_lp(problem, cuts)
        lp_solves += 1

    return CuttingResult(
        initial_bound=root.objective,
        final_bound=current.objective,
        cuts_added=len(cuts),
        lp_solves=lp_solves,
    )
