from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionFusionHead(nn.Module):
    def __init__(
        self,
        vis_dim: int = 1024,
        aud_dim: int = 1024,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_classes: int = 8,
        dropout: float = 0.2,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # modality projection
        self.v_in = nn.Sequential(
            nn.Linear(vis_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.a_in = nn.Sequential(
            nn.Linear(aud_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # V <- A
        self.q_v = nn.Linear(hidden_dim, hidden_dim)
        self.k_a = nn.Linear(hidden_dim, hidden_dim)
        self.v_a = nn.Linear(hidden_dim, hidden_dim)

        # A <- V
        self.q_a = nn.Linear(hidden_dim, hidden_dim)
        self.k_v = nn.Linear(hidden_dim, hidden_dim)
        self.v_v = nn.Linear(hidden_dim, hidden_dim)

        # post attention projection
        self.proj_v = nn.Linear(hidden_dim, hidden_dim)
        self.proj_a = nn.Linear(hidden_dim, hidden_dim)

        self.drop = nn.Dropout(dropout)

        # FFN blocks
        self.norm_v1 = nn.LayerNorm(hidden_dim)
        self.norm_v2 = nn.LayerNorm(hidden_dim)
        self.norm_a1 = nn.LayerNorm(hidden_dim)
        self.norm_a2 = nn.LayerNorm(hidden_dim)

        self.ffn_v = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.ffn_a = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # final fusion gate
        self.fuse_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.cls = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B, H] -> [B, n_head, d_head]
        B = x.size(0)
        return x.view(B, self.num_heads, self.head_dim)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B, n_head, d_head] -> [B, H]
        B = x.size(0)
        return x.reshape(B, self.hidden_dim)

    def _cross_head_mix(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        单向多头“向量交互”：
        q,k,v: [B, H]
        返回: [B, H]
        """
        qh = self._reshape_heads(q)   # [B, nh, dh]
        kh = self._reshape_heads(k)   # [B, nh, dh]
        vh = self._reshape_heads(v)   # [B, nh, dh]

        # 每个 head 一个标量权重，但已经比单头强很多
        attn = torch.sigmoid((qh * kh).sum(dim=-1, keepdim=True) * self.scale)  # [B, nh, 1]
        out = attn * vh + (1.0 - attn) * qh
        return self._merge_heads(out)

    def forward(self, x_v: torch.Tensor, x_a: torch.Tensor) -> torch.Tensor:
        # input projection
        hv = self.v_in(x_v)   # [B, H]
        ha = self.a_in(x_a)   # [B, H]

        # -------------------------
        # block 1: V <- A
        # -------------------------
        v_res = hv
        v_update = self._cross_head_mix(
            self.q_v(hv),
            self.k_a(ha),
            self.v_a(ha),
        )
        hv = self.norm_v1(v_res + self.drop(self.proj_v(v_update)))
        hv = self.norm_v2(hv + self.drop(self.ffn_v(hv)))

        # -------------------------
        # block 2: A <- V
        # -------------------------
        a_res = ha
        a_update = self._cross_head_mix(
            self.q_a(ha),
            self.k_v(hv),
            self.v_v(hv),
        )
        ha = self.norm_a1(a_res + self.drop(self.proj_a(a_update)))
        ha = self.norm_a2(ha + self.drop(self.ffn_a(ha)))

        # -------------------------
        # final gated fusion
        # -------------------------
        gate = self.fuse_gate(torch.cat([hv, ha], dim=-1))   # [B, H]
        z = gate * hv + (1.0 - gate) * ha

        z = self.out_proj(z)
        return self.cls(z)