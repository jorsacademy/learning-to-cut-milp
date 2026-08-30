from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog


@dataclass(frozen=True)
class KnapsackMILP:
    profits: np.ndarray
    weights: np.ndarray
    capacity: float

    def __post_init__(self) -> None:
        n = self.profits.size
        if self.profits.shape != (n,) or self.weights.shape != (n,):
            raise ValueError("profits and weights must be one-dimensional with equal length")
        if n < 2 or np.any(self.weights <= 0) or self.capacity <= 0:
            raise ValueError("invalid knapsack data")


@dataclass(frozen=True)
class LPCut:
    indices: tuple[int, ...]

    @property
    def rhs(self) -> float:
        return float(len(self.indices) - 1)


@dataclass(frozen=True)
class LPResult:
    objective: float
    x: np.ndarray


def generate_knapsack(n_items: int = 18, seed: int = 0) -> KnapsackMILP:
    if n_items < 2:
        raise ValueError("n_items must be at least two")
    rng = np.random.default_rng(seed)
    weights = rng.integers(2, 20, size=n_items).astype(float)
    profits = rng.integers(3, 30, size=n_items).astype(float)
    capacity = float(0.42 * weights.sum())
    return KnapsackMILP(profits=profits, weights=weights, capacity=capacity)


def solve_lp(problem: KnapsackMILP, cuts: list[LPCut] | None = None) -> LPResult:
    cuts = [] if cuts is None else cuts
    n = problem.profits.size
    rows = [problem.weights.copy()]
    rhs = [problem.capacity]
    for cut in cuts:
        row = np.zeros(n)
        row[list(cut.indices)] = 1.0
        rows.append(row)
        rhs.append(cut.rhs)
    result = linprog(
        -problem.profits,
        A_ub=np.asarray(rows),
        b_ub=np.asarray(rhs),
        bounds=[(0.0, 1.0)] * n,
        method="highs",
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"LP relaxation failed: {result.message}")
    return LPResult(objective=float(-result.fun), x=np.asarray(result.x, dtype=float))


def _minimal_cover(problem: KnapsackMILP, order: np.ndarray) -> tuple[int, ...] | None:
    selected: list[int] = []
    total_weight = 0.0
    for raw_index in order:
        index = int(raw_index)
        selected.append(index)
        total_weight += float(problem.weights[index])
        if total_weight > problem.capacity:
            break
    if total_weight <= problem.capacity:
        return None

    # Remove any item whose removal still leaves a cover. This preserves validity and
    # produces a minimal cover without enumerating exponentially many subsets.
    changed = True
    while changed:
        changed = False
        for index in selected.copy():
            remaining_weight = sum(
                float(problem.weights[other]) for other in selected if other != index
            )
            if remaining_weight > problem.capacity:
                selected.remove(index)
                changed = True
    return tuple(sorted(selected))


def generate_cover_cuts(
    problem: KnapsackMILP,
    x: np.ndarray,
    violation_tol: float = 1e-8,
) -> list[LPCut]:
    if x.shape != problem.profits.shape:
        raise ValueError("x has wrong shape")
    n = problem.profits.size
    index_jitter = np.arange(n, dtype=float) * 1e-6
    score_vectors = [
        x,
        x * problem.weights,
        x * problem.profits,
        problem.profits / problem.weights,
        problem.weights,
        problem.profits,
        x + index_jitter,
        x - index_jitter,
    ]

    candidates: list[LPCut] = []
    seen: set[tuple[int, ...]] = set()
    for scores in score_vectors:
        order = np.argsort(scores)[::-1]
        subset = _minimal_cover(problem, order)
        if subset is None or subset in seen:
            continue
        violation = float(x[list(subset)].sum() - (len(subset) - 1))
        if violation > violation_tol:
            candidates.append(LPCut(indices=subset))
            seen.add(subset)
    return candidates


def cut_features(problem: KnapsackMILP, x: np.ndarray, cut: LPCut) -> np.ndarray:
    idx = np.asarray(cut.indices, dtype=int)
    violation = float(x[idx].sum() - cut.rhs)
    norm = float(np.sqrt(idx.size))
    efficacy = violation / max(norm, 1e-12)
    support = idx.size / problem.profits.size
    cover_weight = float(problem.weights[idx].sum() / problem.capacity)
    objective_alignment = float(
        problem.profits[idx].sum() / max(float(problem.profits.sum()), 1e-12)
    )
    mean_fractionality = float(np.minimum(x[idx], 1.0 - x[idx]).mean())
    return np.asarray(
        [violation, efficacy, support, cover_weight, objective_alignment, mean_fractionality],
        dtype=np.float32,
    )


def one_step_bound_improvement(
    problem: KnapsackMILP,
    base_cuts: list[LPCut],
    base_objective: float,
    candidate: LPCut,
) -> float:
    child = solve_lp(problem, [*base_cuts, candidate])
    return max(0.0, base_objective - child.objective)


def rank_candidates_by_expert(
    problem: KnapsackMILP,
    base_cuts: list[LPCut],
    lp: LPResult,
    candidates: list[LPCut],
) -> np.ndarray:
    return np.asarray(
        [
            one_step_bound_improvement(problem, base_cuts, lp.objective, candidate)
            for candidate in candidates
        ],
        dtype=np.float32,
    )
