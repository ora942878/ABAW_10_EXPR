from __future__ import annotations

import importlib
import torch
import torch.nn as nn

from pipeline4_model_trains.A_V_concat_eval.heads.head_linear import LinearHead
from pipeline4_model_trains.A_V_concat_eval.heads.head_mlp import MLPHead
from pipeline4_model_trains.A_V_concat_eval.heads.head_gate import GatedFusionHead
from pipeline4_model_trains.A_V_concat_eval.heads.head_dynamic import DynamicFusionHead
from pipeline4_model_trains.A_V_concat_eval.heads.head_bilinear import BilinearFusionHead
from pipeline4_model_trains.A_V_concat_eval.heads.head_crossattn import CrossAttentionFusionHead
from pipeline4_model_trains.A_V_concat_eval.heads.head_MoE import GatedMoEHead


CFG_MAP = {
    "linear": "pipeline4_model_trains.A_V_concat_eval.cfg.cfg_linear",
    "mlp": "pipeline4_model_trains.A_V_concat_eval.cfg.cfg_mlp",
    "gate": "pipeline4_model_trains.A_V_concat_eval.cfg.cfg_gate",
    "dynamic": "pipeline4_model_trains.A_V_concat_eval.cfg.cfg_dynamic",
    "bilinear": "pipeline4_model_trains.A_V_concat_eval.cfg.cfg_bilinear",
    "crossattn": "pipeline4_model_trains.A_V_concat_eval.cfg.cfg_crossattn",
    "moe": "pipeline4_model_trains.A_V_concat_eval.cfg.cfg_moe",
}


def load_cfg(model_name: str):
    model_name = model_name.lower()
    if model_name not in CFG_MAP:
        raise ValueError(f"Unknown model {model_name}")
    module = importlib.import_module(CFG_MAP[model_name])
    return module.CFG()


def apply_runtime_defaults(cfg):
    if not hasattr(cfg, "USE_INPUT_DROPOUT"):
        cfg.USE_INPUT_DROPOUT = True
    if not hasattr(cfg, "INPUT_DROPOUT_VIS"):
        cfg.INPUT_DROPOUT_VIS = 0.20
    if not hasattr(cfg, "INPUT_DROPOUT_AUD"):
        cfg.INPUT_DROPOUT_AUD = 0.25
    if not hasattr(cfg, "INPUT_DROPOUT_MODE"):
        cfg.INPUT_DROPOUT_MODE = "element"   # element / sample

    if not hasattr(cfg, "DYNAMIC_AUX_WEIGHT"):
        cfg.DYNAMIC_AUX_WEIGHT = 0.0

    if str(cfg.HEAD_TYPE).lower() == "moe":
        if not hasattr(cfg, "MOE_AUX_WEIGHT"):
            cfg.MOE_AUX_WEIGHT = 0.05

        if not hasattr(cfg, "ROUTER_LR"):
            cfg.ROUTER_LR = float(cfg.LR) * 0.5
        if not hasattr(cfg, "EXPERT_LR"):
            cfg.EXPERT_LR = float(cfg.LR)
        if not hasattr(cfg, "OTHER_LR"):
            cfg.OTHER_LR = float(cfg.LR)

        if not hasattr(cfg, "ROUTER_WD"):
            cfg.ROUTER_WD = float(cfg.WEIGHT_DECAY)
        if not hasattr(cfg, "EXPERT_WD"):
            cfg.EXPERT_WD = float(cfg.WEIGHT_DECAY)
        if not hasattr(cfg, "OTHER_WD"):
            cfg.OTHER_WD = float(cfg.WEIGHT_DECAY)

    return cfg


def build_model(cfg):
    t = cfg.HEAD_TYPE.lower()

    if t == "linear":
        model = LinearHead(
            vis_dim=cfg.VIS_DIM,
            aud_dim=cfg.AUD_DIM,
            num_classes=cfg.NUM_CLASSES,
        )

    elif t == "mlp":
        model = MLPHead(
            vis_dim=cfg.VIS_DIM,
            aud_dim=cfg.AUD_DIM,
            hidden=cfg.HIDDEN_DIM,
            num_classes=cfg.NUM_CLASSES,
            dropout=cfg.DROPOUT,
        )

    elif t == "gate":
        model = GatedFusionHead(
            vis_dim=cfg.VIS_DIM,
            aud_dim=cfg.AUD_DIM,
            hidden_dim=cfg.HIDDEN_DIM,
            num_classes=cfg.NUM_CLASSES,
            dropout=cfg.DROPOUT,
        )

    elif t == "dynamic":
        model = DynamicFusionHead(
            vis_dim=cfg.VIS_DIM,
            aud_dim=cfg.AUD_DIM,
            hidden_dim=cfg.HIDDEN_DIM,
            num_classes=cfg.NUM_CLASSES,
            dropout=cfg.DROPOUT,
        )

    elif t == "bilinear":
        model = BilinearFusionHead(
            vis_dim=cfg.VIS_DIM,
            aud_dim=cfg.AUD_DIM,
            hidden_dim=cfg.HIDDEN_DIM,
            num_classes=cfg.NUM_CLASSES,
            dropout=cfg.DROPOUT,
        )

    elif t == "crossattn":
        model = CrossAttentionFusionHead(
            vis_dim=cfg.VIS_DIM,
            aud_dim=cfg.AUD_DIM,
            hidden_dim=cfg.HIDDEN_DIM,
            num_heads=cfg.NUM_HEADS,
            num_classes=cfg.NUM_CLASSES,
            dropout=cfg.DROPOUT,
        )

    elif t == "moe":
        model = GatedMoEHead(
            vis_dim=cfg.VIS_DIM,
            aud_dim=cfg.AUD_DIM,
            num_classes=cfg.NUM_CLASSES,
            num_experts=cfg.NUM_EXPERTS,
            depth=cfg.MOE_DEPTH,
            mlp_ratio=cfg.MOE_MLP_RATIO,
            dropout=cfg.MOE_DROPOUT,
            drop_path=cfg.MOE_DROP_PATH,
        )

    else:
        raise ValueError(f"Unknown HEAD_TYPE {cfg.HEAD_TYPE}")

    return model


def build_optimizer(cfg, model):
    t = cfg.HEAD_TYPE.lower()

    if t == "moe":
        router_params = []
        expert_params = []
        other_params = []

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if "router" in name:
                router_params.append(p)
            elif "experts" in name:
                expert_params.append(p)
            else:
                other_params.append(p)

        return torch.optim.AdamW(
            [
                {
                    "params": router_params,
                    "lr": float(cfg.ROUTER_LR),
                    "weight_decay": float(cfg.ROUTER_WD),
                },
                {
                    "params": expert_params,
                    "lr": float(cfg.EXPERT_LR),
                    "weight_decay": float(cfg.EXPERT_WD),
                },
                {
                    "params": other_params,
                    "lr": float(cfg.OTHER_LR),
                    "weight_decay": float(cfg.OTHER_WD),
                },
            ]
        )

    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg.LR,
        weight_decay=cfg.WEIGHT_DECAY,
    )


def build_scheduler(cfg, optimizer):
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.EPOCHS,
        eta_min=cfg.LR * 0.1,
    )


def moe_load_balance_loss(routing_weights: torch.Tensor) -> torch.Tensor:
    usage = routing_weights.mean(dim=0)
    target = torch.full_like(usage, 1.0 / usage.numel())
    return torch.sum((usage - target) ** 2)


def dynamic_balance_loss(modality_weights: torch.Tensor) -> torch.Tensor:
    usage = modality_weights.mean(dim=0)
    target = torch.full_like(usage, 0.5)
    return torch.sum((usage - target) ** 2)


def build_criterion(cfg, class_weight=None):
    return nn.CrossEntropyLoss(weight=class_weight)


def build_all(model_name: str):
    cfg = load_cfg(model_name)
    cfg = apply_runtime_defaults(cfg)

    model = build_model(cfg)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    criterion = build_criterion(cfg)

    return cfg, model, optimizer, scheduler, criterion