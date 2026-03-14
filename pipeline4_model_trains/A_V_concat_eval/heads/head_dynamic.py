from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicFusionHead(nn.Module):
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

        self.weight_net = nn.Sequential(
            nn.Linear(vis_dim + aud_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

        self.cls = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x_v: torch.Tensor, x_a: torch.Tensor, return_aux: bool = False):
        z_v = self.vis_proj(x_v)  # [B, H]
        z_a = self.aud_proj(x_a)  # [B, H]

        alpha = F.softmax(self.weight_net(torch.cat([x_v, x_a], dim=-1)), dim=-1)  # [B, 2]
        z = alpha[:, 0:1] * z_v + alpha[:, 1:2] * z_a

        logits = self.cls(z)

        if return_aux:
            return logits, {"modality_weights": alpha}
        return logits