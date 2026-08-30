from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .core import (
    FEATURE_NAMES,
    LPCut,
    cut_features,
    generate_cover_cuts,
    generate_knapsack,
    rank_candidates_by_expert,
    solve_lp,
)


class CutScorer(nn.Module):
    """Small MLP that scores one valid cut at a time."""

    def __init__(self, hidden_dim: int = 48) -> None:
        super().__init__()
        if hidden_dim < 4:
            raise ValueError("hidden_dim must be at least four")
        self.hidden_dim = int(hidden_dim)
        n_features = len(FEATURE_NAMES)
        self.register_buffer("feature_mean", torch.zeros(n_features))
        self.register_buffer("feature_std", torch.ones(n_features))
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def set_normalizer(self, features: torch.Tensor) -> None:
        if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
            raise ValueError("features have the wrong shape")
        mean = features.mean(dim=0)
        std = features.std(dim=0, unbiased=False).clamp_min(1e-6)
        self.feature_mean.copy_(mean)
        self.feature_std.copy_(std)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = (features - self.feature_mean) / self.feature_std
        return self.net(normalized).squeeze(-1)


@dataclass(frozen=True)
class CutDataset:
    """Candidate-cut features grouped by the LP state in which they were generated."""

    features: torch.Tensor
    expert_scores: torch.Tensor
    group_offsets: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.features.ndim != 2 or self.features.shape[1] != len(FEATURE_NAMES):
            raise ValueError("features have the wrong shape")
        if self.expert_scores.shape != (self.features.shape[0],):
            raise ValueError("expert_scores must have one value per candidate")
        if len(self.group_offsets) < 2 or self.group_offsets[0] != 0:
            raise ValueError("group_offsets must start at zero and contain at least one group")
        if self.group_offsets[-1] != self.features.shape[0]:
            raise ValueError("last group offset must equal the number of candidates")
        if any(
            b <= a
            for a, b in zip(self.group_offsets, self.group_offsets[1:], strict=False)
        ):
            raise ValueError("every dataset group must contain at least one candidate")

    @property
    def n_groups(self) -> int:
        return len(self.group_offsets) - 1

    @property
    def n_candidates(self) -> int:
        return int(self.features.shape[0])


@dataclass(frozen=True)
class RankingMetrics:
    top1_accuracy: float
    mean_normalized_regret: float


@dataclass(frozen=True)
class TrainingReport:
    initial_loss: float
    final_loss: float
    epochs: int
    training_metrics: RankingMetrics


def _choose_rollout_cut(
    rollout_code: int,
    candidates: list[LPCut],
    features: np.ndarray,
    expert_scores: np.ndarray,
    rng: np.random.Generator,
) -> int:
    if rollout_code == 0:
        return int(np.argmax(expert_scores))
    if rollout_code == 1:
        return int(np.argmax(features[:, FEATURE_NAMES.index("efficacy")]))
    return int(rng.integers(len(candidates)))


def collect_cut_dataset(
    n_instances: int = 80,
    n_items: int = 24,
    n_constraints: int = 5,
    rounds: int = 4,
    seed: int = 0,
    *,
    randomized_orders: int = 8,
    max_candidates: int = 64,
) -> CutDataset:
    """Collect sequential cut-ranking states under mixed expert/heuristic/random rollouts."""

    if n_instances < 1 or rounds < 1:
        raise ValueError("n_instances and rounds must be positive")
    feature_groups: list[np.ndarray] = []
    score_groups: list[np.ndarray] = []
    offsets = [0]

    for instance_offset in range(n_instances):
        problem_seed = seed + instance_offset
        problem = generate_knapsack(
            n_items=n_items,
            n_constraints=n_constraints,
            seed=problem_seed,
        )
        current = solve_lp(problem)
        cuts: list[LPCut] = []
        rng = np.random.default_rng(seed + 1_000_003 + instance_offset)

        for round_index in range(rounds):
            candidates = generate_cover_cuts(
                problem,
                current.x,
                seed=seed + 97_409 * (instance_offset + 1) + round_index,
                randomized_orders=randomized_orders,
                max_candidates=max_candidates,
            )
            existing = {cut.indices for cut in cuts}
            candidates = [cut for cut in candidates if cut.indices not in existing]
            if not candidates:
                break

            features = np.stack([cut_features(problem, current.x, cut) for cut in candidates])
            expert_scores = rank_candidates_by_expert(problem, cuts, current, candidates)

            if len(candidates) >= 2 and float(expert_scores.max()) > 1e-10:
                feature_groups.append(features)
                score_groups.append(expert_scores)
                offsets.append(offsets[-1] + len(candidates))

            rollout_code = (instance_offset + round_index) % 3
            selected = _choose_rollout_cut(
                rollout_code,
                candidates,
                features,
                expert_scores,
                rng,
            )
            cuts.append(candidates[selected])
            current = solve_lp(problem, cuts)

    if not feature_groups:
        raise RuntimeError("no informative candidate groups were generated")
    return CutDataset(
        features=torch.as_tensor(np.concatenate(feature_groups), dtype=torch.float32),
        expert_scores=torch.as_tensor(np.concatenate(score_groups), dtype=torch.float32),
        group_offsets=tuple(offsets),
    )


def _ranking_loss(model: CutScorer, dataset: CutDataset) -> torch.Tensor:
    scores = model(dataset.features)
    losses: list[torch.Tensor] = []
    for start, end in zip(dataset.group_offsets, dataset.group_offsets[1:], strict=False):
        expert = dataset.expert_scores[start:end]
        best = int(torch.argmax(expert).item())
        losses.append(
            nn.functional.cross_entropy(
                scores[start:end].unsqueeze(0),
                torch.tensor([best], dtype=torch.long, device=scores.device),
            )
        )
    return torch.stack(losses).mean()


def evaluate_ranking(model: CutScorer, dataset: CutDataset) -> RankingMetrics:
    model.eval()
    correct = 0
    regrets: list[float] = []
    with torch.no_grad():
        predictions = model(dataset.features)
    for start, end in zip(dataset.group_offsets, dataset.group_offsets[1:], strict=False):
        expert = dataset.expert_scores[start:end]
        predicted_index = int(torch.argmax(predictions[start:end]).item())
        selected = float(expert[predicted_index])
        oracle = float(expert.max())
        if oracle - selected <= 1e-8:
            correct += 1
        regrets.append((oracle - selected) / max(oracle, 1e-9))
    return RankingMetrics(
        top1_accuracy=correct / dataset.n_groups,
        mean_normalized_regret=float(np.mean(regrets)),
    )


def train_cut_scorer(
    model: CutScorer,
    dataset: CutDataset,
    epochs: int = 120,
    learning_rate: float = 2e-3,
    weight_decay: float = 1e-5,
    seed: int = 0,
) -> TrainingReport:
    """Train the scorer with groupwise cross-entropy on the expert-best candidate."""

    if epochs < 1 or learning_rate <= 0 or weight_decay < 0:
        raise ValueError("invalid training configuration")
    torch.manual_seed(seed)
    model.set_normalizer(dataset.features)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    model.train()
    initial_loss = float(_ranking_loss(model, dataset).detach())
    final_loss = initial_loss
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = _ranking_loss(model, dataset)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())

    metrics = evaluate_ranking(model, dataset)
    return TrainingReport(
        initial_loss=initial_loss,
        final_loss=final_loss,
        epochs=epochs,
        training_metrics=metrics,
    )


def save_cut_scorer(
    model: CutScorer,
    path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "hidden_dim": model.hidden_dim,
            "feature_names": list(FEATURE_NAMES),
            "metadata": {} if metadata is None else metadata,
        },
        path,
    )


def load_cut_scorer(path: str | Path) -> tuple[CutScorer, dict[str, Any]]:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if tuple(checkpoint["feature_names"]) != FEATURE_NAMES:
        raise ValueError("checkpoint feature schema does not match this package")
    model = CutScorer(hidden_dim=int(checkpoint["hidden_dim"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, dict(checkpoint.get("metadata", {}))
