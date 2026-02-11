"""
Training pipeline for game-by-game PPG prediction (next 3 games).
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from data.nba_data import fetch_league_game_logs, prepare_game_dataset
from models.player_predictor import NextGamePPGPredictor

logger = logging.getLogger(__name__)


def train(
    epochs: int = 80,
    batch_size: int = 128,
    lr: float = 1e-3,
    window: int = 10,
    device: str | None = None,
    cache_path: Path | str | None = None,
    output_dir: Path | str | None = None,
    seasons: list[str] | None = None,
    force_refresh: bool = False,
) -> tuple[NextGamePPGPredictor, list[float], dict]:
    """
    Fetch game-by-game data, train model for next 3 games PPG.
    Uses position for related players. Saves model + meta.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cache_path = Path(cache_path) if cache_path else Path("data/cache/league_game_logs.csv")
    output_dir = Path(output_dir) if output_dir else Path("output")

    logger.info("Fetching game-by-game logs...")
    game_logs = fetch_league_game_logs(seasons=seasons, cache_path=cache_path, force_refresh=force_refresh)
    logger.info("Loaded %d game records, %d players", len(game_logs), game_logs["PLAYER_ID"].nunique())

    X, y, meta = prepare_game_dataset(game_logs, window=window)
    logger.info("Training samples: %d, Features: %d, Target: next 3 games PPG", len(X), X.shape[1])

    X_norm = (X - meta["mean"]) / meta["std"]

    indices = np.random.permutation(len(X))
    split = int(0.9 * len(X))
    train_idx, val_idx = indices[:split], indices[split:]

    X_train = torch.from_numpy(X_norm[train_idx]).float()
    y_train = torch.from_numpy(y[train_idx]).float()
    X_val = torch.from_numpy(X_norm[val_idx]).float()
    y_val = torch.from_numpy(y[val_idx]).float()

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = NextGamePPGPredictor(
        input_dim=X.shape[1],
        hidden_dims=[256, 128, 64],
        dropout=0.2,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = torch.nn.MSELoss()

    history = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        history.append(avg_loss)

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val.to(device))
            val_loss = criterion(val_pred, y_val.to(device)).item()
            val_mae = (val_pred - y_val.to(device)).abs().mean().item()
        model.train()

        if (epoch + 1) % 15 == 0 or epoch == 0:
            logger.info(
                "Epoch %4d | Train MSE: %.4f | Val MSE: %.4f | Val MAE: %.2f ppg",
                epoch + 1,
                avg_loss,
                val_loss,
                val_mae,
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_meta = {
        "mean": meta["mean"].tolist(),
        "std": meta["std"].tolist(),
        "feature_cols": meta["feature_cols"],
        "window": meta["window"],
    }
    with open(output_dir / "meta.json", "w") as f:
        json.dump(save_meta, f, indent=2)

    torch.save(model.state_dict(), output_dir / "model.pt")
    logger.info("Saved model to %s", output_dir)

    return model, history, meta
