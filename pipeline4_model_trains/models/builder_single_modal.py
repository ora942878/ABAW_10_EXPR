from __future__ import annotations

from typing import Any, Dict, Optional

from pipeline4_model_trains.models.head_linear import LinearHead
from pipeline4_model_trains.models.head_mlp import MLPHead
from pipeline4_model_trains.models.head_MoE import GatedMoEHead


def build_single_modal_head(
    head_type: str,
    in_dim: int = 1024,
    num_classes: int = 8,
    hidden_dim: int = 512,
    dropout: float = 0.3,
    moe_num_experts: int = 4,
    moe_depth: int = 1,
    moe_mlp_ratio: float = 1.5,
    moe_drop_path: float = 0.0,
    **kwargs: Any,
):
    head_type = str(head_type).strip().lower()

    if head_type == "linear":
        return LinearHead(
            in_dim=in_dim,
            num_classes=num_classes,
        )

    if head_type == "mlp":
        return MLPHead(
            in_dim=in_dim,
            num_classes=num_classes,
            hidden=hidden_dim,
            dropout=dropout,
        )

    if head_type == "moe":
        return GatedMoEHead(
            in_dim=in_dim,
            num_classes=num_classes,
            num_experts=moe_num_experts,
            depth=moe_depth,
            mlp_ratio=moe_mlp_ratio,
            dropout=dropout,
            drop_path=moe_drop_path,
        )

    raise ValueError(
        f"Unknown head_type: {head_type}. "
        f"Supported head types are: ['linear', 'mlp', 'moe']"
    )


def build_single_modal_head_from_cfg(cfg) -> object:
    return build_single_modal_head(
        head_type=getattr(cfg, "HEAD_TYPE", "linear"),
        in_dim=getattr(cfg, "IN_DIM", 1024),
        num_classes=getattr(cfg, "NUM_CLASSES", 8),
        hidden_dim=getattr(cfg, "HIDDEN_DIM", 512),
        dropout=getattr(cfg, "DROPOUT", 0.3),
        moe_num_experts=getattr(cfg, "MOE_NUM_EXPERTS", 4),
        moe_depth=getattr(cfg, "MOE_DEPTH", 1),
        moe_mlp_ratio=getattr(cfg, "MOE_MLP_RATIO", 1.5),
        moe_drop_path=getattr(cfg, "MOE_DROP_PATH", 0.0),
    )