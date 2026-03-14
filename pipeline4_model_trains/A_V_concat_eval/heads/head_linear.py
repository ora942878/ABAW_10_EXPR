from __future__ import annotations

import torch
import torch.nn as nn


class LinearHead(nn.Module):
    def __init__(self, vis_dim: int = 1024, aud_dim: int = 1024, num_classes: int = 8):
        super().__init__()
        self.fc = nn.Linear(int(vis_dim + aud_dim), int(num_classes))

    def forward(self, x_v: torch.Tensor, x_a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x_v, x_a], dim=-1)
        return self.fc(x)