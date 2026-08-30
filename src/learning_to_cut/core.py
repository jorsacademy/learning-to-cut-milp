from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

FEATURE_NAMES = (
    "violation",
    "efficacy",
    "density",
    "cover_excess_ratio",
    "objective_parallelism",
    "mean_fractionality",
    "mean_lp_value",
    "source_row_activity_ratio",
    "support_profit_share",
)


@dataclass(frozen=True)
class KnapsackMILP:
    """A transparent multidimensional 0-1 knapsack MILP."""

    profits: np.ndarray
    weights: np.ndarray
    capacities: np.ndarray

    def __post_init__(self) -> None:
        profits = np.asarray(self.profits, dtype=float)
        weights = np.asarray(self.weights, dtype=float)
        capacities = np.asarray(self.capacities, dtype=float)
        if profits.ndim != 1:
            raise ValueError("profits must be one-dimensional")
        if weights.ndim != 2 or weights.shape[1] != profits.size:
            raise ValueError("weights must have shape (n_constraints, n_items)")
        if capacities.shape != (weights.shape[0],):
            raise ValueError("capacities must have one entry per resource row")
        if profits.size < 2 or weights.shape[0] < 1:
            raise ValueError("problem must contain at least two items and one constraint")
        if np.any(profits < 0) or not np.any(profits > 0):
            raise ValueError("profits must be nonnegative with at least one positive entry")
        if np.any(weights <= 0) or np.any(capacities <= 0):
            raise ValueError("weights and capacities must be strictly positive")
        if np.any(capacities >= weights.sum(axis=1)):
            raise ValueError("every resource row must exclude at least one all-items solution")
        object.__setattr__(self, "profits", profits)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "capacities", capacities)

    @property
    def n_items(self) -> int:
        return int(self.profits.size)

    @property
    def n_constraints(self) -> int:
        return int(self.weights.shape[0])


@dataclass(frozen=True, order=True)
class LPCut:
    """A cover inequality sum(i in C) x_i <= |C| - 1."""

    indices: tuple[int, ...]
    source_row: int

    def __post_init__(self) -> None:
        canonical = tuple(sorted(set(int(i) for i in self.indices)))
        if not canonical:
            raise ValueError("a cover cut must contain at least one item")
        if canonical != self.indices:
            object.__setattr__(self, "indices", canonical)
        if self.source_row < 0:
            raise ValueError("source_row must be nonnegative")

    @property
    def rhs(self) -> float:
        return float(len(self.indices) - 1)


@dataclass(frozen=True)
class LPResult:
    objective: float
    x: np.ndarray


@dataclass(frozen=True)
class MILPResult:
    objective: float
    x: np.ndarray


def generate_knapsack(
    n_items: int = 24,
    n_constraints: int = 5,
    seed: int = 0,
    capacity_ratio_range: tuple[float, float] = (0.36, 0.48),
) -> KnapsackMILP:
    """Generate a reproducible synthetic multidimensional 0-1 knapsack."""

    if n_items < 4:
        raise ValueError("n_items must be at least four")
    if n_constraints < 1:
        raise ValueError("n_constraints must be positive")
    low, high = capacity_ratio_range
    if not (0.05 < low <= high < 0.95):
        raise ValueError("capacity ratios must lie strictly between 0.05 and 0.95")

    rng = np.random.default_rng(seed)
    weights = rng.integers(2, 21, size=(n_constraints, n_items)).astype(float)
    base_profit = rng.integers(6, 35, size=n_items).astype(float)
    resource_signal = weights.mean(axis=0)
    profits = np.round(base_profit + 0.65 * resource_signal, 3)

    row_totals = weights.sum(axis=1)
    ratios = rng.uniform(low, high, size=n_constraints)
    capacities = np.floor(row_totals * ratios)
    capacities = np.maximum(capacities, weights.max(axis=1))
    capacities = np.minimum(capacities, row_totals - 1.0)
    return KnapsackMILP(profits=profits, weights=weights, capacities=capacities)


def _cut_row(problem: KnapsackMILP, cut: LPCut) -> np.ndarray:
    if cut.source_row >= problem.n_constraints:
        raise ValueError("cut source_row is outside the problem")
    if cut.indices[-1] >= problem.n_items or cut.indices[0] < 0:
        raise ValueError("cut contains an item index outside the problem")
    row = np.zeros(problem.n_items, dtype=float)
    row[list(cut.indices)] = 1.0
    return row


def is_valid_cover_cut(problem: KnapsackMILP, cut: LPCut, tol: float = 1e-12) -> bool:
    """Return whether the cut follows from its declared knapsack resource row."""

    if cut.source_row >= problem.n_constraints:
        return False
    if cut.indices[-1] >= problem.n_items or cut.indices[0] < 0:
        return False
    idx = np.asarray(cut.indices, dtype=int)
    return bool(
        problem.weights[cut.source_row, idx].sum()
        > problem.capacities[cut.source_row] + tol
    )


def is_minimal_cover(problem: KnapsackMILP, cut: LPCut, tol: float = 1e-12) -> bool:
    if not is_valid_cover_cut(problem, cut, tol=tol):
        return False
    idx = list(cut.indices)
    weights = problem.weights[cut.source_row]
    capacity = problem.capacities[cut.source_row]
    return all(sum(weights[j] for j in idx if j != i) <= capacity + tol for i in idx)


def _constraint_matrix(
    problem: KnapsackMILP, cuts: list[LPCut] | tuple[LPCut, ...] | None
) -> tuple[np.ndarray, np.ndarray]:
    rows = [row.copy() for row in problem.weights]
    rhs = list(problem.capacities.astype(float))
    for cut in () if cuts is None else cuts:
        if not is_valid_cover_cut(problem, cut):
            raise ValueError(f"invalid cover cut: {cut}")
        rows.append(_cut_row(problem, cut))
        rhs.append(cut.rhs)
    return np.asarray(rows, dtype=float), np.asarray(rhs, dtype=float)


def solve_lp(
    problem: KnapsackMILP, cuts: list[LPCut] | tuple[LPCut, ...] | None = None
) -> LPResult:
    """Solve the LP relaxation and return its maximization upper bound."""

    a_ub, b_ub = _constraint_matrix(problem, cuts)
    result = linprog(
        -problem.profits,
        A_ub=a_ub,
        b_ub=b_ub,
        bounds=[(0.0, 1.0)] * problem.n_items,
        method="highs",
    )
    if not result.success or result.x is None or result.fun is None:
        raise RuntimeError(f"LP relaxation failed: {result.message}")
    return LPResult(objective=float(-result.fun), x=np.asarray(result.x, dtype=float))


def solve_milp(
    problem: KnapsackMILP, cuts: list[LPCut] | tuple[LPCut, ...] | None = None
) -> MILPResult:
    """Solve the binary MILP exactly enough for small-instance benchmarking."""

    a_ub, b_ub = _constraint_matrix(problem, cuts)
    constraints = LinearConstraint(a_ub, -np.inf, b_ub)
    result = milp(
        c=-problem.profits,
        integrality=np.ones(problem.n_items, dtype=int),
        bounds=Bounds(np.zeros(problem.n_items), np.ones(problem.n_items)),
        constraints=constraints,
        options={"presolve": True},
    )
    if result.status != 0 or result.x is None or result.fun is None:
        raise RuntimeError(f"MILP solve failed with status {result.status}: {result.message}")
    return MILPResult(objective=float(-result.fun), x=np.rint(result.x).astype(float))


def _minimal_cover(
    weights: np.ndarray, capacity: float, order: np.ndarray
) -> tuple[int, ...] | None:
    selected: list[int] = []
    total_weight = 0.0
    for raw_index in order:
        index = int(raw_index)
        selected.append(index)
        total_weight += float(weights[index])
        if total_weight > capacity:
            break
    if total_weight <= capacity:
        return None

    changed = True
    while changed:
        changed = False
        for index in selected.copy():
            remaining_weight = sum(float(weights[j]) for j in selected if j != index)
            if remaining_weight > capacity:
                selected.remove(index)
                changed = True
    return tuple(sorted(selected))


def cut_violation(x: np.ndarray, cut: LPCut) -> float:
    idx = np.asarray(cut.indices, dtype=int)
    return float(x[idx].sum() - cut.rhs)


def generate_cover_cuts(
    problem: KnapsackMILP,
    x: np.ndarray,
    *,
    seed: int = 0,
    randomized_orders: int = 8,
    max_candidates: int = 64,
    violation_tol: float = 1e-8,
) -> list[LPCut]:
    """Generate a diverse heuristic pool of violated minimal cover inequalities."""

    x = np.asarray(x, dtype=float)
    if x.shape != (problem.n_items,):
        raise ValueError("x has wrong shape")
    if randomized_orders < 0 or max_candidates < 1:
        raise ValueError("invalid candidate-generation configuration")

    rng = np.random.default_rng(seed)
    fractionality = 0.5 - np.abs(x - 0.5)
    profit_scale = problem.profits / max(float(problem.profits.max()), 1e-12)
    candidates: list[LPCut] = []
    seen: set[tuple[int, ...]] = set()

    for row_index in range(problem.n_constraints):
        row_weights = problem.weights[row_index]
        capacity = float(problem.capacities[row_index])
        score_vectors = [
            x,
            x * row_weights,
            x * problem.profits,
            fractionality,
            problem.profits / row_weights,
            row_weights,
            problem.profits,
            x + 0.20 * profit_scale + 0.10 * fractionality,
        ]
        for _ in range(randomized_orders):
            score_vectors.append(
                x + 0.16 * rng.standard_normal(problem.n_items) + 0.08 * profit_scale
            )

        for scores in score_vectors:
            base_order = np.argsort(scores, kind="stable")[::-1]
            for shift in range(min(3, problem.n_items)):
                order = np.concatenate((base_order[shift:], base_order[:shift]))
                subset = _minimal_cover(row_weights, capacity, order)
                if subset is None or subset in seen:
                    continue
                cut = LPCut(indices=subset, source_row=row_index)
                if cut_violation(x, cut) <= violation_tol:
                    continue
                if not is_minimal_cover(problem, cut):
                    raise AssertionError("candidate-generation bug produced a non-minimal cover")
                candidates.append(cut)
                seen.add(subset)
                if len(candidates) >= max_candidates:
                    return candidates
    return candidates


def cut_features(problem: KnapsackMILP, x: np.ndarray, cut: LPCut) -> np.ndarray:
    """Compute transparent cut-level features used by heuristic and learned selectors."""

    if not is_valid_cover_cut(problem, cut):
        raise ValueError("features requested for an invalid cover cut")
    idx = np.asarray(cut.indices, dtype=int)
    row = problem.weights[cut.source_row]
    capacity = float(problem.capacities[cut.source_row])
    violation = cut_violation(x, cut)
    support_norm = float(np.sqrt(idx.size))
    efficacy = violation / max(support_norm, 1e-12)
    density = idx.size / problem.n_items
    cover_excess_ratio = float((row[idx].sum() - capacity) / capacity)
    objective_parallelism = float(
        problem.profits[idx].sum()
        / max(support_norm * np.linalg.norm(problem.profits), 1e-12)
    )
    mean_fractionality = float(np.minimum(x[idx], 1.0 - x[idx]).mean())
    mean_lp_value = float(x[idx].mean())
    source_row_activity_ratio = float((row @ x) / capacity)
    support_profit_share = float(problem.profits[idx].sum() / problem.profits.sum())
    values = np.asarray(
        [
            violation,
            efficacy,
            density,
            cover_excess_ratio,
            objective_parallelism,
            mean_fractionality,
            mean_lp_value,
            source_row_activity_ratio,
            support_profit_share,
        ],
        dtype=np.float32,
    )
    if not np.all(np.isfinite(values)):
        raise RuntimeError("non-finite cut feature encountered")
    return values


def one_step_bound_improvement(
    problem: KnapsackMILP,
    base_cuts: list[LPCut] | tuple[LPCut, ...],
    base_objective: float,
    candidate: LPCut,
) -> float:
    """Return the one-step decrease in the maximization LP upper bound."""

    child = solve_lp(problem, [*base_cuts, candidate])
    return max(0.0, float(base_objective - child.objective))


def rank_candidates_by_expert(
    problem: KnapsackMILP,
    base_cuts: list[LPCut] | tuple[LPCut, ...],
    lp: LPResult,
    candidates: list[LPCut],
) -> np.ndarray:
    """Score candidates using expensive one-step strong branching-style lookahead."""

    return np.asarray(
        [
            one_step_bound_improvement(problem, base_cuts, lp.objective, candidate)
            for candidate in candidates
        ],
        dtype=np.float32,
    )


def integrality_gap_closure(
    initial_upper_bound: float,
    final_upper_bound: float,
    integer_optimum: float,
    tol: float = 1e-9,
) -> float:
    """Fraction of the root LP integrality gap closed by added cuts."""

    initial_gap = max(0.0, float(initial_upper_bound - integer_optimum))
    if initial_gap <= tol:
        return 1.0
    improvement = float(initial_upper_bound - final_upper_bound)
    closure = improvement / initial_gap
    return float(np.clip(closure, 0.0, 1.0))
