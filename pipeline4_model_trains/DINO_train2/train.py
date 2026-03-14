from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

from configs.paths import PATH
from pipeline4_model_trains.DINO_train2.CFG_trainDINOv2 import CFG_DINOV2

from pipeline4_model_trains.utils.data_utils import collect_imgset_pairs, collect_abaw_uniform_pairs, RawImageDataset
from pipeline4_model_trains.common.set_seed import set_seed
from pipeline4_model_trains.common.compute_class_weights import compute_class_weights
from pipeline4_model_trains.common.evaluate_classification import evaluate_classification

from pipeline4_model_trains.DINO_augment.random_blackpad import RandomBlackPadShift
from pipeline4_model_trains.DINO_augment.soft_label import smooth_onehot, soft_ce_with_class_weight
from pipeline4_model_trains.DINO_augment.mixup import mixup_batch
from pipeline4_model_trains.DINO_augment.llrd import build_param_groups

from pipeline4_model_trains.models.MoE_for_dinotrain import GatedMoEMLPHead

import sys
from pathlib import Path as _Path

_DINO_ROOT = _Path(__file__).resolve().parents[2] / "pipeline3_feature_extract" / "lib" / "dinov2"
if str(_DINO_ROOT) not in sys.path:
    sys.path.insert(0, str(_DINO_ROOT))

from dinov2.hub.backbones import dinov2_vitl14


C = 8
CLASS_NAMES = ["Neutral", "Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise", "Other"]


class UnifiedDinoFER(nn.Module):
    def __init__(self, weight_path: Path, cfg: CFG_DINOV2):
        super().__init__()
        self.cfg = cfg

        self.backbone = dinov2_vitl14(pretrained=False)
        sd = torch.load(weight_path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        self.backbone.load_state_dict(sd, strict=False)

        for p in self.backbone.patch_embed.parameters():
            p.requires_grad = False
        for i, block in enumerate(self.backbone.blocks):
            if i < cfg.freeze_blocks:
                for p in block.parameters():
                    p.requires_grad = False

        if cfg.use_moe:
            self.head = GatedMoEMLPHead(
                in_dim=1024,
                num_classes=C,
                num_experts=cfg.moe_num_experts,
                depth=cfg.moe_depth,
                mlp_ratio=cfg.moe_mlp_ratio,
                dropout=cfg.moe_dropout,
                drop_path=cfg.moe_drop_path,
            )
        else:
            self.head = nn.Sequential(nn.Dropout(0.5), nn.Linear(1024, C))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone.forward_features(x)["x_norm_clstoken"]
        return self.head(feat)

def build_train_transform(cfg: CFG_DINOV2):
    norm = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    if cfg.use_augment:
        tfms: list[Callable[..., Any]] = [
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.02)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.10),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))
            ], p=0.10),
        ]

        if getattr(cfg, "use_pad_aug", False):
            tfms.append(
                RandomBlackPadShift(
                    p=cfg.pad_prob,
                    max_area_frac=cfg.pad_max_area,
                    min_bar_frac=cfg.pad_min_bar,
                    max_shift_frac=cfg.pad_shift_frac,
                    allow_L=True,
                )
            )

        tfms.extend([
            transforms.ToTensor(),
            norm,
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3), value=0),
        ])

        return transforms.Compose(tfms)

def build_val_transform():
    norm = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        norm,
    ])


def main():
    set_seed(CFG_DINOV2.seed)
    device = CFG_DINOV2.device
    cfg = CFG_DINOV2()

    print(f"[INFO] Initializing Unified Training Pipeline in Mode: {cfg.mode.upper()}")

    train_tfm = build_train_transform(cfg)
    val_tfm = build_val_transform()

    print("[INFO] Collecting Datasets...")
    tr_pairs = collect_imgset_pairs(PATH.Dataset_IMG, is_train=True)
    val1_pairs = collect_imgset_pairs(PATH.Dataset_IMG, is_train=False)

    official_roots = [PATH.IMG_batch1_ABAW10th, PATH.IMG_batch2_ABAW10th]
    val2_pairs = collect_abaw_uniform_pairs(
        [PATH.EXPR_VALID_ABAW10th, PATH.EXPR_TRAIN_ABAW10th],
        official_roots,
        cfg.K_samples,
    )

    processed_abaw_roots = [PATH.ABAW_FACE09_ROOT, PATH.ABAW_FACE12_ROOT, PATH.ABAW_FACE15_ROOT]
    processed_abaw_roots = [r for r in processed_abaw_roots if isinstance(r, Path) and r.exists()]
    val3_pairs = collect_abaw_uniform_pairs(
        [PATH.EXPR_VALID_ABAW10th, PATH.EXPR_TRAIN_ABAW10th],
        processed_abaw_roots,
        cfg.K_samples,
    )

    print(f"  Train IMGset: {len(tr_pairs)} imgs")
    print(f"  Val 1 (IMGset Valid): {len(val1_pairs)} imgs")
    print(f"  Val 2 (ABAW Official): {len(val2_pairs)} imgs")
    print(f"  Val 3 (ABAW Face09/12/15): {len(val3_pairs)} imgs")

    dl_tr = DataLoader(
        RawImageDataset(tr_pairs, train_tfm),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    dl_val1 = DataLoader(
        RawImageDataset(val1_pairs, val_tfm),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    dl_val2 = DataLoader(
        RawImageDataset(val2_pairs, val_tfm),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    dl_val3 = DataLoader(
        RawImageDataset(val3_pairs, val_tfm),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    train_labels = [y for _, y in tr_pairs]
    class_w = compute_class_weights(train_labels, num_classes=C).to(device)

    weight_path = PATH.VIT_WEIGHTS_ROOT / "dinov2_vitl14.pth"
    model = UnifiedDinoFER(weight_path, cfg).to(device)

    param_groups = build_param_groups(
        model=model,
        weight_decay=cfg.weight_decay,
        lr_backbone=cfg.lr_backbone,
        lr_head=cfg.lr_head,
        layer_decay=cfg.layer_decay,
    )
    opt = torch.optim.AdamW(param_groups)
    scheduler = CosineAnnealingLR(opt, T_max=cfg.epoch, eta_min=1e-7)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    run_dir = Path(PATH.RUN_ROOT) / f"unified_{cfg.mode}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    best_v1_mf1 = -1.0

    val_criterion = lambda logits, y: F.cross_entropy(logits, y, weight=class_w)

    for ep in range(1, cfg.epoch + 1):
        model.train()
        t0 = time.time()
        tr_loss, tr_corr, total = 0.0, 0, 0

        pbar = tqdm(dl_tr, desc=f"Ep {ep}/{cfg.epoch}", leave=False)
        opt.zero_grad(set_to_none=True)

        for X, y in pbar:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if cfg.use_augment:
                y_target = smooth_onehot(y, C, cfg.label_smoothing)
                X, y_target = mixup_batch(X, y_target, alpha=cfg.mixup_alpha, p=cfg.mixup_prob)
            else:
                y_target = y

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(X)
                if cfg.use_augment:
                    loss = soft_ce_with_class_weight(logits, y_target, class_w)
                else:
                    loss = F.cross_entropy(logits, y_target, weight=class_w)

            scaler.scale(loss).backward()

            if cfg.grad_clip is not None:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

            with torch.no_grad():
                pred = logits.argmax(-1)
                tr_corr += (pred == y).sum().item()
                total += y.numel()
                tr_loss += float(loss) * y.numel()

            pbar.set_postfix(loss=float(loss), acc=(tr_corr / max(1, total)))

        scheduler.step()

        v1 = evaluate_classification(model, dl_val1, val_criterion, device, num_classes=C)
        v2 = evaluate_classification(model, dl_val2, val_criterion, device, num_classes=C)
        v3 = evaluate_classification(model, dl_val3, val_criterion, device, num_classes=C)

        dt = time.time() - t0
        line = (
            f"Ep {ep:02d} | Mode: {cfg.mode} | LR: {opt.param_groups[-1]['lr']:.2e} | "
            f"TR loss: {tr_loss / max(1, total):.4f} | TR acc: {tr_corr / max(1, total):.4f} | "
            f"V1 MF1: {v1['mf1']:.4f} | V2 MF1: {v2['mf1']:.4f} | V3 MF1: {v3['mf1']:.4f} | {dt:.1f}s"
        )
        print(line)
        with (run_dir / "metrics.txt").open("a", encoding="utf-8") as f:
            f.write(line + "\n")

        if v1["mf1"] > best_v1_mf1:
            best_v1_mf1 = v1["mf1"]
            save_dict = {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": ep,
                "best_v1_mf1": best_v1_mf1,
                "cfg_mode": cfg.mode,
            }
            torch.save(save_dict, run_dir / "best.pt")
            print(f"  --> [BEST] saved (V1_MF1: {best_v1_mf1:.4f})")


if __name__ == "__main__":
    main()
