from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .core import (
    cut_features,
    generate_cover_cuts,
    generate_knapsack,
    rank_candidates_by_expert,
    solve_lp,
)


class CutScorer(nn.Module):
    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


@dataclass(frozen=True)
class CutDataset:
    features: torch.Tensor
    targets: torch.Tensor


def collect_root_dataset(
    n_instances: int = 80,
    n_items: int = 18,
    seed: int = 0,
) -> CutDataset:
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for offset in range(n_instances):
        problem = generate_knapsack(n_items=n_items, seed=seed + offset)
        lp = solve_lp(problem)
        candidates = generate_cover_cuts(problem, lp.x)
        if not candidates:
            continue
        features.append(np.stack([cut_features(problem, lp.x, cut) for cut in candidates]))
        targets.append(rank_candidates_by_expert(problem, [], lp, candidates))
    if not features:
        raise RuntimeError("no violated cover cuts were generated")
    return CutDataset(
        features=torch.tensor(np.concatenate(features), dtype=torch.float32),
        targets=torch.tensor(np.concatenate(targets), dtype=torch.float32),
    )


def train_cut_scorer(
    model: CutScorer,
    dataset: CutDataset,
    epochs: int = 100,
    learning_rate: float = 1e-3,
) -> float:
    if epochs < 1 or learning_rate <= 0:
        raise ValueError("invalid training configuration")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_value = float("nan")
    for _ in range(epochs):
        optimizer.zero_grad()
        predictions = model(dataset.features)
        loss = nn.functional.mse_loss(predictions, dataset.targets)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach())
    return loss_value
