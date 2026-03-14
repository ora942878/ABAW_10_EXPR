import torch
import torch.nn.functional as F
import torch.nn as nn
"""
from pipeline4_model_trains.models.MoE import GatedMoEMLPHead
"""
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or (not self.training):
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


class SwiGLU(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(in_dim, hidden_dim, bias=True)
        self.proj = nn.Linear(hidden_dim, in_dim, bias=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.drop(self.fc1(x) * F.silu(self.fc2(x))))


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float, dropout: float, drop_path: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = SwiGLU(dim, int(dim * mlp_ratio), dropout=dropout)
        self.dp = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dp(self.mlp(self.norm(x)))


class GatedMoEMLPHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        num_experts: int = 4,
        depth: int = 3,
        mlp_ratio: float = 2.0,
        dropout: float = 0.3,
        drop_path: float = 0.1,
    ):
        super().__init__()
        self.in_norm = nn.LayerNorm(in_dim)

        self.router = nn.Sequential(
            nn.Linear(in_dim, in_dim // 4),
            nn.LayerNorm(in_dim // 4),
            nn.GELU(),
            nn.Linear(in_dim // 4, num_experts),
        )

        dpr = torch.linspace(0, drop_path, steps=depth).tolist()
        self.experts = nn.ModuleList(
            [
                nn.Sequential(*[ResidualMLPBlock(in_dim, mlp_ratio, dropout, dpr[i]) for i in range(depth)])
                for _ in range(num_experts)
            ]
        )

        self.out_norm = nn.LayerNorm(in_dim)
        self.head_drop = nn.Dropout(dropout)
        self.cls = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.in_norm(x)
        gate_logits = self.router(x_norm)
        routing_weights = F.softmax(gate_logits, dim=-1)

        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x_norm))

        stacked_outputs = torch.stack(expert_outputs, dim=1)
        fused_features = (stacked_outputs * routing_weights.unsqueeze(-1)).sum(dim=1)
        return self.cls(self.head_drop(self.out_norm(fused_features)))
