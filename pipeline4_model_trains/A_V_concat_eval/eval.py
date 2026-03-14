# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from pipeline4_model_trains.A_V_concat_eval.build import (
    build_all,
    build_criterion,
    moe_load_balance_loss,
    dynamic_balance_loss,
)

from pipeline4_model_trains.common.set_seed import set_seed
from pipeline4_model_trains.common.compute_class_weights import compute_class_weights
from pipeline4_model_trains.common.macro_f1_from_cm import macro_f1_from_cm
from pipeline4_model_trains.common.confusion_matrix_np import confusion_matrix_np

from pipeline4_model_trains.utils.Dataset_AVFrameLevel import AVFrameConcatDataset


# =========================================================
MODEL_NAME = "linear"   # linear / mlp / gate / dynamic / bilinear / crossattn / moe


# =========================================================
def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_txt(path: str | Path, text: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def append_jsonl(obj, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def list_txtids(expr_dir: str | Path) -> List[str]:
    expr_dir = Path(expr_dir)
    return sorted([p.stem for p in expr_dir.glob("*.txt")])


def format_cm(cm: np.ndarray) -> str:
    lines = []
    for r in cm:
        lines.append(" ".join(f"{int(x):7d}" for x in r))
    return "\n".join(lines)


def acc_from_cm(cm: np.ndarray, eps: float = 1e-12) -> float:
    correct = float(np.diag(cm).sum())
    total = float(cm.sum())
    return correct / max(total, eps)


def cfg_to_dict(cfg) -> dict:
    if is_dataclass(cfg):
        return asdict(cfg)
    if hasattr(cfg, "__dict__"):
        return dict(cfg.__dict__)
    return {"cfg": str(cfg)}


# =========================================================
class AVFrameDatasetWithNorm(torch.utils.data.Dataset):
    def __init__(
        self,
        base_ds,
        vis_mean: np.ndarray | None = None,
        vis_std: np.ndarray | None = None,
        aud_mean: np.ndarray | None = None,
        aud_std: np.ndarray | None = None,
        do_zscore: bool = False,
        eps: float = 1e-6,
    ):
        self.base_ds = base_ds
        self.do_zscore = bool(do_zscore)
        self.eps = float(eps)

        self.V = base_ds.V.clone().float()
        self.A = base_ds.A.clone().float()
        self.Y = base_ds.Y.clone().long()

        self.vis_dim = int(self.V.shape[1]) if self.V.ndim == 2 and self.V.shape[0] > 0 else 0
        self.aud_dim = int(self.A.shape[1]) if self.A.ndim == 2 and self.A.shape[0] > 0 else 0

        self.vis_mean = None if vis_mean is None else np.asarray(vis_mean, dtype=np.float32)
        self.vis_std = None if vis_std is None else np.asarray(vis_std, dtype=np.float32)
        self.aud_mean = None if aud_mean is None else np.asarray(aud_mean, dtype=np.float32)
        self.aud_std = None if aud_std is None else np.asarray(aud_std, dtype=np.float32)

        if self.do_zscore:
            if self.vis_mean is None or self.vis_std is None or self.aud_mean is None or self.aud_std is None:
                raise ValueError("vis/aud mean/std must be provided when do_zscore=True")

            vis_mean_t = torch.from_numpy(self.vis_mean).float()
            vis_std_t = torch.from_numpy(np.maximum(self.vis_std, self.eps)).float()
            aud_mean_t = torch.from_numpy(self.aud_mean).float()
            aud_std_t = torch.from_numpy(np.maximum(self.aud_std, self.eps)).float()

            self.V = (self.V - vis_mean_t) / vis_std_t
            self.A = (self.A - aud_mean_t) / aud_std_t

    def __len__(self):
        return int(self.Y.shape[0])

    def __getitem__(self, idx: int):
        return self.V[idx], self.A[idx], self.Y[idx]


def compute_mean_std_from_tensor(X: torch.Tensor, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError(f"bad X shape: {tuple(X.shape)}")
    mean = X.mean(dim=0).cpu().numpy().astype(np.float32)
    std = X.std(dim=0, unbiased=False).cpu().numpy().astype(np.float32)
    std = np.maximum(std, eps).astype(np.float32)
    return mean, std


# =========================================================
def apply_input_dropout(vis, aud, cfg, is_train: bool):
    if (not is_train) or (not getattr(cfg, "USE_INPUT_DROPOUT", False)):
        return vis, aud

    p_vis = float(getattr(cfg, "INPUT_DROPOUT_VIS", 0.0))
    p_aud = float(getattr(cfg, "INPUT_DROPOUT_AUD", 0.0))
    mode = str(getattr(cfg, "INPUT_DROPOUT_MODE", "element")).lower()

    if mode == "sample":
        if p_vis > 0:
            keep_v = (torch.rand(vis.shape[0], 1, device=vis.device) > p_vis).float()
            vis = vis * keep_v / max(1.0 - p_vis, 1e-6)
        if p_aud > 0:
            keep_a = (torch.rand(aud.shape[0], 1, device=aud.device) > p_aud).float()
            aud = aud * keep_a / max(1.0 - p_aud, 1e-6)
    else:
        if p_vis > 0:
            vis = torch.nn.functional.dropout(vis, p=p_vis, training=True)
        if p_aud > 0:
            aud = torch.nn.functional.dropout(aud, p=p_aud, training=True)

    return vis, aud


# =========================================================
def forward_model(cfg, model, vis: torch.Tensor, aud: torch.Tensor, is_train: bool):
    t = str(cfg.HEAD_TYPE).lower()

    if t == "moe":
        if is_train:
            return model(vis, aud, return_aux=True)
        return model(vis, aud, return_aux=False)

    if t == "dynamic":
        if is_train:
            return model(vis, aud, return_aux=True)
        return model(vis, aud, return_aux=False)

    return model(vis, aud)


def compute_total_loss(cfg, criterion, outputs, targets):
    t = str(cfg.HEAD_TYPE).lower()

    if t == "moe":
        logits, aux = outputs
        loss_cls = criterion(logits, targets)
        loss_aux = moe_load_balance_loss(aux["routing_weights"])
        total_loss = loss_cls + float(getattr(cfg, "MOE_AUX_WEIGHT", 0.0)) * loss_aux
        return total_loss, logits, {
            "loss_cls": float(loss_cls.item()),
            "loss_aux": float(loss_aux.item()),
        }

    if t == "dynamic":
        if isinstance(outputs, tuple):
            logits, aux = outputs
            loss_cls = criterion(logits, targets)
            aux_w = float(getattr(cfg, "DYNAMIC_AUX_WEIGHT", 0.0))
            if aux_w > 0:
                loss_aux = dynamic_balance_loss(aux["modality_weights"])
                total_loss = loss_cls + aux_w * loss_aux
                return total_loss, logits, {
                    "loss_cls": float(loss_cls.item()),
                    "loss_aux": float(loss_aux.item()),
                }
            return loss_cls, logits, {
                "loss_cls": float(loss_cls.item()),
                "loss_aux": 0.0,
            }

    logits = outputs
    loss = criterion(logits, targets)
    return loss, logits, {
        "loss_cls": float(loss.item()),
        "loss_aux": 0.0,
    }


# =========================================================
@torch.no_grad()
def evaluate_av_classification(
    cfg,
    model,
    loader,
    criterion,
    device,
    num_classes: int,
):
    model.eval()

    total_loss = 0.0
    total_n = 0

    all_pred = []
    all_true = []

    use_amp = bool(cfg.AMP and device.type == "cuda")

    for vis, aud, y in tqdm(loader, desc="eval", ncols=120, leave=False):
        vis = vis.to(device, non_blocking=True).float()
        aud = aud.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).long()

        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = forward_model(cfg, model, vis, aud, is_train=False)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            loss = criterion(logits, y)

        bs = int(y.shape[0])
        total_loss += float(loss.item()) * bs
        total_n += bs

        pred = logits.argmax(dim=1)
        all_pred.append(pred.detach().cpu().numpy())
        all_true.append(y.detach().cpu().numpy())

    y_pred = np.concatenate(all_pred, axis=0) if len(all_pred) > 0 else np.empty((0,), dtype=np.int64)
    y_true = np.concatenate(all_true, axis=0) if len(all_true) > 0 else np.empty((0,), dtype=np.int64)

    cm = confusion_matrix_np(y_true, y_pred, num_classes=num_classes)
    mf1 = macro_f1_from_cm(cm)
    acc = acc_from_cm(cm)

    return {
        "loss": total_loss / max(total_n, 1),
        "acc": float(acc),
        "mf1": float(mf1),
        "cm": cm,
    }


# =========================================================
def main():
    cfg, model, optimizer, scheduler, _ = build_all(MODEL_NAME)

    set_seed(cfg.SEED)
    device = torch.device(cfg.DEVICE)
    model = model.to(device)

    run_name = f"{cfg.EXP_NAME}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = ensure_dir(Path(cfg.SAVE_ROOT) / run_name)

    print("=" * 100)
    print(f"[RUN] {run_name}")
    print(f"[MODEL_NAME]          = {MODEL_NAME}")
    print(f"[HEAD_TYPE]           = {cfg.HEAD_TYPE}")
    print(f"[PATH] TRAIN_EXPR_DIR = {cfg.TRAIN_EXPR_DIR}")
    print(f"[PATH] VALID_EXPR_DIR = {cfg.VALID_EXPR_DIR}")
    print(f"[CFG] LR              = {cfg.LR}")
    print(f"[CFG] WEIGHT_DECAY    = {cfg.WEIGHT_DECAY}")
    print(f"[CFG] BATCH_SIZE      = {cfg.BATCH_SIZE}")
    print(f"[CFG] EPOCHS          = {cfg.EPOCHS}")
    print(f"[CFG] DEVICE          = {cfg.DEVICE}")
    print(f"[CFG] USE_INPUT_DROPOUT = {getattr(cfg, 'USE_INPUT_DROPOUT', False)}")
    print(f"[CFG] INPUT_DROP_VIS    = {getattr(cfg, 'INPUT_DROPOUT_VIS', 0.0)}")
    print(f"[CFG] INPUT_DROP_AUD    = {getattr(cfg, 'INPUT_DROPOUT_AUD', 0.0)}")
    print(f"[CFG] INPUT_DROP_MODE   = {getattr(cfg, 'INPUT_DROPOUT_MODE', 'element')}")
    print("=" * 100)

    save_json(cfg_to_dict(cfg), run_dir / "config.json")

    train_txtids = list_txtids(cfg.TRAIN_EXPR_DIR)
    valid_txtids = list_txtids(cfg.VALID_EXPR_DIR)

    print("[INFO] Building raw train dataset for statistics ...")
    ds_tr_raw_base = AVFrameConcatDataset(
        txtids=train_txtids,
        txt_dir=cfg.TRAIN_EXPR_DIR,
        do_l2_norm=cfg.DO_BRANCH_L2_NORM,
        label_min=0,
        label_max=cfg.NUM_CLASSES - 1,
    )
    if len(ds_tr_raw_base) == 0:
        raise RuntimeError("Train dataset is empty.")

    vis_mean, vis_std, aud_mean, aud_std = None, None, None, None
    if cfg.DO_ZSCORE:
        vis_mean, vis_std = compute_mean_std_from_tensor(ds_tr_raw_base.V, eps=cfg.ZSCORE_EPS)
        aud_mean, aud_std = compute_mean_std_from_tensor(ds_tr_raw_base.A, eps=cfg.ZSCORE_EPS)

        np.save(run_dir / "train_vis_mean.npy", vis_mean)
        np.save(run_dir / "train_vis_std.npy", vis_std)
        np.save(run_dir / "train_aud_mean.npy", aud_mean)
        np.save(run_dir / "train_aud_std.npy", aud_std)
        print("[INFO] train vis/aud mean/std saved.")

    print("[INFO] Building normalized train dataset ...")
    ds_tr = AVFrameDatasetWithNorm(
        base_ds=ds_tr_raw_base,
        vis_mean=vis_mean,
        vis_std=vis_std,
        aud_mean=aud_mean,
        aud_std=aud_std,
        do_zscore=cfg.DO_ZSCORE,
        eps=cfg.ZSCORE_EPS,
    )

    print("[INFO] Building normalized val dataset ...")
    ds_va_base = AVFrameConcatDataset(
        txtids=valid_txtids,
        txt_dir=cfg.VALID_EXPR_DIR,
        do_l2_norm=cfg.DO_BRANCH_L2_NORM,
        label_min=0,
        label_max=cfg.NUM_CLASSES - 1,
    )
    if len(ds_va_base) == 0:
        raise RuntimeError("Val dataset is empty.")

    ds_va = AVFrameDatasetWithNorm(
        base_ds=ds_va_base,
        vis_mean=vis_mean,
        vis_std=vis_std,
        aud_mean=aud_mean,
        aud_std=aud_std,
        do_zscore=cfg.DO_ZSCORE,
        eps=cfg.ZSCORE_EPS,
    )

    print(f"[DATA] Train: N={len(ds_tr)}, vis_dim={ds_tr.vis_dim}, aud_dim={ds_tr.aud_dim}")
    print(f"[DATA] Val  : N={len(ds_va)}, vis_dim={ds_va.vis_dim}, aud_dim={ds_va.aud_dim}")

    class_weight = None
    if cfg.USE_CLASS_WEIGHT:
        class_weight = compute_class_weights(ds_tr.Y.cpu().numpy(), num_classes=cfg.NUM_CLASSES).to(device)
        print(f"[INFO] class_weight = {class_weight.detach().cpu().numpy()}")

    criterion = build_criterion(cfg, class_weight=class_weight)

    ld_tr = DataLoader(
        ds_tr,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        drop_last=False,
        persistent_workers=(cfg.PERSISTENT_WORKERS and cfg.NUM_WORKERS > 0),
    )
    ld_va = DataLoader(
        ds_va,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        drop_last=False,
        persistent_workers=(cfg.PERSISTENT_WORKERS and cfg.NUM_WORKERS > 0),
    )

    use_amp = bool(cfg.AMP and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_mf1 = -1.0
    best_epoch = -1
    history_lines = []

    print("[INFO] Start training ...")
    for ep in range(1, cfg.EPOCHS + 1):
        t0 = time.time()
        model.train()

        tr_loss = 0.0
        tr_n = 0
        tr_correct = 0

        tr_loss_cls = 0.0
        tr_loss_aux = 0.0

        pbar = tqdm(ld_tr, desc=f"ep{ep:02d} train", ncols=140)
        for vis, aud, y in pbar:
            vis = vis.to(device, non_blocking=True).float()
            aud = aud.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True).long()

            vis, aud = apply_input_dropout(vis, aud, cfg, is_train=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = forward_model(cfg, model, vis, aud, is_train=True)
                loss, logits, loss_info = compute_total_loss(cfg, criterion, outputs, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = int(y.shape[0])
            tr_loss += float(loss.item()) * bs
            tr_loss_cls += float(loss_info["loss_cls"]) * bs
            tr_loss_aux += float(loss_info["loss_aux"]) * bs
            tr_n += bs

            pred = logits.argmax(dim=1)
            tr_correct += int((pred == y).sum().item())

            pbar.set_postfix(
                loss=f"{tr_loss / max(tr_n, 1):.4f}",
                cls=f"{tr_loss_cls / max(tr_n, 1):.4f}",
                aux=f"{tr_loss_aux / max(tr_n, 1):.4f}",
                acc=f"{tr_correct / max(tr_n, 1):.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        scheduler.step()

        tr_loss_avg = tr_loss / max(tr_n, 1)
        tr_loss_cls_avg = tr_loss_cls / max(tr_n, 1)
        tr_loss_aux_avg = tr_loss_aux / max(tr_n, 1)
        tr_acc = tr_correct / max(tr_n, 1)

        va_stats = evaluate_av_classification(
            cfg=cfg,
            model=model,
            loader=ld_va,
            criterion=criterion,
            device=device,
            num_classes=cfg.NUM_CLASSES,
        )

        msg = {
            "epoch": ep,
            "train_loss": float(tr_loss_avg),
            "train_loss_cls": float(tr_loss_cls_avg),
            "train_loss_aux": float(tr_loss_aux_avg),
            "train_acc": float(tr_acc),
            "val_loss": float(va_stats["loss"]),
            "val_acc": float(va_stats["acc"]),
            "val_mf1": float(va_stats["mf1"]),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "time_sec": float(time.time() - t0),
        }

        line = (
            f"[ep {ep:02d}] "
            f"tr_loss={msg['train_loss']:.4f} "
            f"tr_cls={msg['train_loss_cls']:.4f} "
            f"tr_aux={msg['train_loss_aux']:.4f} "
            f"tr_acc={msg['train_acc']:.4f} | "
            f"va_loss={msg['val_loss']:.4f} "
            f"va_acc={msg['val_acc']:.4f} "
            f"va_mf1={msg['val_mf1']:.4f} | "
            f"time={msg['time_sec']:.1f}s"
        )
        print(line)
        history_lines.append(line)

        append_jsonl(msg, run_dir / "metrics.jsonl")
        save_txt(run_dir / "history.txt", "\n".join(history_lines))

        if msg["val_mf1"] > best_mf1:
            best_mf1 = msg["val_mf1"]
            best_epoch = ep

            ckpt = {
                "cfg": cfg_to_dict(cfg),
                "model_name": MODEL_NAME,
                "epoch": ep,
                "best_mf1": float(best_mf1),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "train_vis_mean": vis_mean,
                "train_vis_std": vis_std,
                "train_aud_mean": aud_mean,
                "train_aud_std": aud_std,
                "val_cm": va_stats["cm"],
            }
            torch.save(ckpt, run_dir / "best.pt")
            np.save(run_dir / "best_cm.npy", va_stats["cm"])

            best_txt = []
            best_txt.append(f"[BEST] epoch={ep}")
            best_txt.append(f"model_name={MODEL_NAME}")
            best_txt.append(f"val_loss={va_stats['loss']:.6f}")
            best_txt.append(f"val_acc={va_stats['acc']:.6f}")
            best_txt.append(f"val_mf1={va_stats['mf1']:.6f}")
            best_txt.append("")
            best_txt.append("[CONFUSION MATRIX]")
            best_txt.append(format_cm(va_stats["cm"]))
            save_txt(run_dir / "best_metrics.txt", "\n".join(best_txt))

            save_json(
                {
                    "best_epoch": int(best_epoch),
                    "best_mf1": float(best_mf1),
                    "train_size": int(len(ds_tr)),
                    "val_size": int(len(ds_va)),
                    "vis_dim": int(ds_tr.vis_dim),
                    "aud_dim": int(ds_tr.aud_dim),
                    "model_name": MODEL_NAME,
                },
                run_dir / "best_summary.json",
            )

            print(f"[SAVE] best updated -> {run_dir / 'best.pt'} (mf1={best_mf1:.4f})")

    print("=" * 100)
    print(f"[DONE] best_epoch={best_epoch}, best_mf1={best_mf1:.4f}")
    print(f"[RUN_DIR] {run_dir}")
    print("=" * 100)


if __name__ == "__main__":
    main()