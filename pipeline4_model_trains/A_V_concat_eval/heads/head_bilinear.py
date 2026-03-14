from __future__ import annotations

import torch
import torch.nn as nn


class BilinearFusionHead(nn.Module):
    def __init__(
        self,
        vis_dim: int = 1024,
        aud_dim: int = 1024,
        hidden_dim: int = 512,
        num_classes: int = 8,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.vis_proj = nn.Linear(vis_dim, hidden_dim)
        self.aud_proj = nn.Linear(aud_dim, hidden_dim)

        self.post = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x_v: torch.Tensor, x_a: torch.Tensor) -> torch.Tensor:
        z_v = self.vis_proj(x_v)   # [B, H]
        z_a = self.aud_proj(x_a)   # [B, H]

        z = z_v * z_a              # Hadamard product
        return self.post(z)