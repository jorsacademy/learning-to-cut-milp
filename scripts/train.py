from pathlib import Path

import torch

from learning_to_cut import CutScorer, collect_root_dataset, train_cut_scorer


def main() -> None:
    torch.manual_seed(0)
    dataset = collect_root_dataset(n_instances=80, n_items=18, seed=0)
    model = CutScorer(hidden_dim=32)
    loss = train_cut_scorer(model, dataset, epochs=120, learning_rate=1e-3)
    Path("checkpoints").mkdir(exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict(), "hidden_dim": 32},
        "checkpoints/cut_scorer.pt",
    )
    print(f"samples={len(dataset.targets)} loss={loss:.6f}")


if __name__ == "__main__":
    main()
