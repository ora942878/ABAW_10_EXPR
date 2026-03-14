from __future__ import annotations

import torch
import torch.nn as nn


class MLPHead(nn.Module):
    def __init__(
        self,
        vis_dim: int = 1024,
        aud_dim: int = 1024,
        num_classes: int = 8,
        hidden: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()
        in_dim = int(vis_dim + aud_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x_v: torch.Tensor, x_a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x_v, x_a], dim=-1)
        return self.net(x)