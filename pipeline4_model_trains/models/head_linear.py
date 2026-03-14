from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 8):
        super().__init__()
        self.fc = nn.Linear(int(in_dim), int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)