import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(in_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, in_dim)
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


class GatedMoEHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_classes: int = 8,
        num_experts: int = 4,
        depth: int = 1,
        mlp_ratio: float = 1.5,
        dropout: float = 0.2,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.in_norm = nn.LayerNorm(in_dim)

        router_hidden = max(in_dim // 4, 64)
        self.router = nn.Sequential(
            nn.Linear(in_dim, router_hidden),
            nn.LayerNorm(router_hidden),
            nn.GELU(),
            nn.Linear(router_hidden, num_experts),
        )

        dpr = torch.linspace(0, drop_path, steps=depth).tolist() if depth > 0 else []
        self.experts = nn.ModuleList([
            nn.Sequential(*[
                ResidualMLPBlock(in_dim, mlp_ratio, dropout, dpr[i])
                for i in range(depth)
            ])
            for _ in range(num_experts)
        ])

        self.out_norm = nn.LayerNorm(in_dim)
        self.head_drop = nn.Dropout(dropout)
        self.cls = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        x = self.in_norm(x)

        gate_logits = self.router(x)
        routing_weights = F.softmax(gate_logits, dim=-1)   # [B, E]

        expert_outputs = [expert(x) for expert in self.experts]
        stacked_outputs = torch.stack(expert_outputs, dim=1)  # [B, E, D]
        fused = (stacked_outputs * routing_weights.unsqueeze(-1)).sum(dim=1)

        logits = self.cls(self.head_drop(self.out_norm(fused)))

        if return_aux:
            return logits, {"routing_weights": routing_weights}
        return logits