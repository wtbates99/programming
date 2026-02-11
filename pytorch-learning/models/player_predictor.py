"""
PyTorch model for single-game PPG prediction.
Input: rolling stats + position + opponent + home_away.
Output: predicted PTS for that game.
"""

import torch
import torch.nn as nn


class NextGamePPGPredictor(nn.Module):
    """
    Predicts PPG for one game given player rolling stats + opponent + home/away.
    Call 3 times for next 3 games with different opponent/home inputs.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.dropout_p = dropout
        hidden_dims = hidden_dims or [256, 128, 64]

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim) - rolling stats + position + opponent_idx + home_away

        Returns:
            (batch, 1) - predicted PTS
        """
        h = self.encoder(x)
        return self.head(h)

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 50,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.train()
        preds = [self.forward(x) for _ in range(n_samples)]
        preds = torch.stack(preds, dim=0)
        mean = preds.mean(dim=0)
        std = preds.std(dim=0) + 1e-6
        self.eval()
        return mean, std


# Aliases
Next3GamesPPGPredictor = NextGamePPGPredictor
PPGPredictor = NextGamePPGPredictor
PlayerStatPredictor = NextGamePPGPredictor
