import torch
import torch.nn as nn
from typing import Dict, List, Tuple

def _is_no_wd_param(name: str, p: torch.Tensor) -> bool:
    if name.endswith(".bias") or p.ndim == 1:
        return True
    n = name.lower()
    if "norm" in n or "ln" in n:
        return True
    return False

def build_param_groups(
    model: nn.Module,
    weight_decay: float,
    lr_backbone: float,
    lr_head: float,
    layer_decay: float = 1.0
) -> List[dict]:
    if layer_decay is None or layer_decay >= 1.0:
        return [{"params": [p for p in model.parameters() if p.requires_grad],
                 "lr": lr_backbone, "weight_decay": weight_decay}]

    groups: Dict[Tuple[float, float], List[torch.nn.Parameter]] = {}

    def add_param(lr: float, wd: float, p: torch.nn.Parameter):
        groups.setdefault((float(lr), float(wd)), []).append(p)

    for n, p in model.head.named_parameters():
        if p.requires_grad:
            wd = 0.0 if _is_no_wd_param("head." + n, p) else weight_decay
            add_param(lr_head, wd, p)

    num_blocks = len(model.backbone.blocks)
    for i, block in enumerate(model.backbone.blocks):
        block_lr = lr_backbone * (layer_decay ** (num_blocks - 1 - i))
        for n, p in block.named_parameters():
            if p.requires_grad:
                wd = 0.0 if _is_no_wd_param(f"backbone.blocks.{i}." + n, p) else weight_decay
                add_param(block_lr, wd, p)

    for n, p in model.backbone.named_parameters():
        if p.requires_grad and not (n.startswith("blocks.") or n.startswith("patch_embed.")):
            wd = 0.0 if _is_no_wd_param("backbone." + n, p) else weight_decay
            add_param(lr_backbone, wd, p)

    param_groups = [{"params": ps, "lr": lr, "weight_decay": wd} for (lr, wd), ps in groups.items()]
    return sorted(param_groups, key=lambda g: (g["lr"], g["weight_decay"]))

