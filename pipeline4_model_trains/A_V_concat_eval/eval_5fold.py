# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.paths import PATH
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


# =========================
MODEL_NAME = "linear"   # linear / mlp / gate / dynamic / bilinear / crossattn / moe
FOLD_CSV = PATH.EXPR_5FOLD
NUM_FOLDS = 5


# =========================
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


def format_cm(cm: np.ndarray) -> str:
    return "\n".join(" ".join(f"{int(x):7d}" for x in r) for r in cm)


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


def summarize(vals: List[float]) -> dict:
    x = np.asarray(vals, dtype=np.float64)
    return {"mean": float(x.mean()), "std": float(x.std(ddof=0))}


# =========================
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


class SimpleTensorDataset:
    def __init__(self, V: torch.Tensor, A: torch.Tensor, Y: torch.Tensor):
        self.V = V.float()
        self.A = A.float()
        self.Y = Y.long()


def concat_base_datasets(ds_list: List[AVFrameConcatDataset]) -> SimpleTensorDataset:
    if len(ds_list) == 0:
        return SimpleTensorDataset(
            torch.empty((0, 1024), dtype=torch.float32),
            torch.empty((0, 1024), dtype=torch.float32),
            torch.empty((0,), dtype=torch.long),
        )

    V = torch.cat([ds.V for ds in ds_list], dim=0).float()
    A = torch.cat([ds.A for ds in ds_list], dim=0).float()
    Y = torch.cat([ds.Y for ds in ds_list], dim=0).long()
    return SimpleTensorDataset(V, A, Y)


def compute_mean_std_from_tensor(X: torch.Tensor, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError(f"bad X shape: {tuple(X.shape)}")
    mean = X.mean(dim=0).cpu().numpy().astype(np.float32)
    std = X.std(dim=0, unbiased=False).cpu().numpy().astype(np.float32)
    std = np.maximum(std, eps).astype(np.float32)
    return mean, std


# =========================
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


# =========================
def forward_model(cfg, model, vis: torch.Tensor, aud: torch.Tensor, is_train: bool):
    t = str(cfg.HEAD_TYPE).lower()

    if t == "moe":
        return model(vis, aud, return_aux=is_train)
    if t == "dynamic":
        return model(vis, aud, return_aux=is_train)
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

    if t == "dynamic" and isinstance(outputs, tuple):
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


# =========================
@torch.no_grad()
def evaluate_av_classification(cfg, model, loader, criterion, device, num_classes: int):
    model.eval()

    total_loss = 0.0
    total_n = 0
    all_pred, all_true = [], []

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

    y_pred = np.concatenate(all_pred, axis=0) if all_pred else np.empty((0,), dtype=np.int64)
    y_true = np.concatenate(all_true, axis=0) if all_true else np.empty((0,), dtype=np.int64)

    cm = confusion_matrix_np(y_true, y_pred, num_classes=num_classes)
    mf1 = macro_f1_from_cm(cm)
    acc = acc_from_cm(cm)

    return {
        "loss": total_loss / max(total_n, 1),
        "acc": float(acc),
        "mf1": float(mf1),
        "cm": cm,
    }


# =========================
def load_fold_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"fold csv not found: {path}")

    df = pd.read_csv(path)
    need = {"txtid", "fold"}
    if not need.issubset(set(df.columns)):
        raise RuntimeError(f"fold csv must contain columns: {need}, got {list(df.columns)}")
    if "source_dir" not in df.columns:
        raise RuntimeError("fold csv must contain column: source_dir")

    df["txtid"] = df["txtid"].astype(str)
    df["source_dir"] = df["source_dir"].astype(str)
    df["fold"] = df["fold"].astype(int)
    return df


def split_fold_txtids(df: pd.DataFrame, fold_id: int):
    tr = df[df["fold"] != fold_id].copy()
    va = df[df["fold"] == fold_id].copy()
    return tr, va


def rows_to_groups(rows: pd.DataFrame) -> List[Tuple[Path, List[str]]]:
    groups = []
    for src_name, part in rows.groupby("source_dir"):
        src_low = str(src_name).lower()
        if "train" in src_low:
            txt_dir = PATH.EXPR_TRAIN_ABAW10th
        elif "valid" in src_low or "val" in src_low:
            txt_dir = PATH.EXPR_VALID_ABAW10th
        else:
            raise RuntimeError(f"unknown source_dir: {src_name}")
        txtids = sorted(part["txtid"].astype(str).tolist())
        groups.append((Path(txt_dir), txtids))
    return groups


# =========================
def build_base_dataset_from_groups(cfg, groups: List[Tuple[Path, List[str]]]) -> SimpleTensorDataset:
    ds_list = []
    for txt_dir, txtids in groups:
        if len(txtids) == 0:
            continue
        ds = AVFrameConcatDataset(
            txtids=txtids,
            txt_dir=txt_dir,
            do_l2_norm=cfg.DO_BRANCH_L2_NORM,
            label_min=0,
            label_max=cfg.NUM_CLASSES - 1,
        )
        if len(ds) > 0:
            ds_list.append(ds)
    base = concat_base_datasets(ds_list)
    if base.Y.numel() == 0:
        raise RuntimeError("dataset is empty after grouping")
    return base


def build_norm_dataset(base_ds, cfg, vis_mean, vis_std, aud_mean, aud_std):
    return AVFrameDatasetWithNorm(
        base_ds=base_ds,
        vis_mean=vis_mean,
        vis_std=vis_std,
        aud_mean=aud_mean,
        aud_std=aud_std,
        do_zscore=cfg.DO_ZSCORE,
        eps=cfg.ZSCORE_EPS,
    )


# =========================
def train_one_fold(root_run_dir: Path, fold_id: int, fold_df: pd.DataFrame):
    cfg, model, optimizer, scheduler, _ = build_all(MODEL_NAME)

    set_seed(int(cfg.SEED) + fold_id)
    device = torch.device(cfg.DEVICE)
    model = model.to(device)

    fold_dir = ensure_dir(root_run_dir / f"fold_{fold_id}")
    save_json(cfg_to_dict(cfg), fold_dir / "config.json")

    tr_rows, va_rows = split_fold_txtids(fold_df, fold_id)
    tr_groups = rows_to_groups(tr_rows)
    va_groups = rows_to_groups(va_rows)

    print("=" * 100)
    print(f"[FOLD] {fold_id}")
    print(f"[MODEL_NAME] = {MODEL_NAME}")
    print(f"[HEAD_TYPE]  = {cfg.HEAD_TYPE}")
    print(f"[TRAIN_TXT]  = {len(tr_rows)}")
    print(f"[VAL_TXT]    = {len(va_rows)}")
    print("=" * 100)

    print("[INFO] Building raw train dataset ...")
    ds_tr_raw = build_base_dataset_from_groups(cfg, tr_groups)

    vis_mean, vis_std, aud_mean, aud_std = None, None, None, None
    if cfg.DO_ZSCORE:
        vis_mean, vis_std = compute_mean_std_from_tensor(ds_tr_raw.V, eps=cfg.ZSCORE_EPS)
        aud_mean, aud_std = compute_mean_std_from_tensor(ds_tr_raw.A, eps=cfg.ZSCORE_EPS)

        np.save(fold_dir / "train_vis_mean.npy", vis_mean)
        np.save(fold_dir / "train_vis_std.npy", vis_std)
        np.save(fold_dir / "train_aud_mean.npy", aud_mean)
        np.save(fold_dir / "train_aud_std.npy", aud_std)

    print("[INFO] Building normalized train dataset ...")
    ds_tr = build_norm_dataset(ds_tr_raw, cfg, vis_mean, vis_std, aud_mean, aud_std)

    print("[INFO] Building raw val dataset ...")
    ds_va_raw = build_base_dataset_from_groups(cfg, va_groups)

    print("[INFO] Building normalized val dataset ...")
    ds_va = build_norm_dataset(ds_va_raw, cfg, vis_mean, vis_std, aud_mean, aud_std)

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
    best_acc = 0.0
    best_loss = 0.0
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

        pbar = tqdm(ld_tr, desc=f"fold{fold_id} ep{ep:02d} train", ncols=140)
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

        va_stats = evaluate_av_classification(
            cfg=cfg,
            model=model,
            loader=ld_va,
            criterion=criterion,
            device=device,
            num_classes=cfg.NUM_CLASSES,
        )

        msg = {
            "fold": int(fold_id),
            "epoch": ep,
            "train_loss": float(tr_loss / max(tr_n, 1)),
            "train_loss_cls": float(tr_loss_cls / max(tr_n, 1)),
            "train_loss_aux": float(tr_loss_aux / max(tr_n, 1)),
            "train_acc": float(tr_correct / max(tr_n, 1)),
            "val_loss": float(va_stats["loss"]),
            "val_acc": float(va_stats["acc"]),
            "val_mf1": float(va_stats["mf1"]),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "time_sec": float(time.time() - t0),
        }

        line = (
            f"[fold {fold_id} | ep {ep:02d}] "
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

        append_jsonl(msg, fold_dir / "metrics.jsonl")
        save_txt(fold_dir / "history.txt", "\n".join(history_lines))

        if msg["val_mf1"] > best_mf1:
            best_mf1 = msg["val_mf1"]
            best_epoch = ep
            best_acc = msg["val_acc"]
            best_loss = msg["val_loss"]

            ckpt = {
                "cfg": cfg_to_dict(cfg),
                "model_name": MODEL_NAME,
                "fold": int(fold_id),
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
            torch.save(ckpt, fold_dir / "best.pt")
            np.save(fold_dir / "best_cm.npy", va_stats["cm"])

            best_txt = [
                f"[BEST] fold={fold_id}, epoch={ep}",
                f"model_name={MODEL_NAME}",
                f"val_loss={va_stats['loss']:.6f}",
                f"val_acc={va_stats['acc']:.6f}",
                f"val_mf1={va_stats['mf1']:.6f}",
                "",
                "[CONFUSION MATRIX]",
                format_cm(va_stats["cm"]),
            ]
            save_txt(fold_dir / "best_metrics.txt", "\n".join(best_txt))

            save_json(
                {
                    "fold": int(fold_id),
                    "best_epoch": int(best_epoch),
                    "best_mf1": float(best_mf1),
                    "best_acc": float(best_acc),
                    "best_loss": float(best_loss),
                    "train_size": int(len(ds_tr)),
                    "val_size": int(len(ds_va)),
                    "vis_dim": int(ds_tr.vis_dim),
                    "aud_dim": int(ds_tr.aud_dim),
                    "model_name": MODEL_NAME,
                },
                fold_dir / "best_summary.json",
            )

            print(f"[SAVE] fold {fold_id} best -> {fold_dir / 'best.pt'} (mf1={best_mf1:.4f})")

    print("=" * 100)
    print(f"[FOLD DONE] fold={fold_id}, best_epoch={best_epoch}, best_mf1={best_mf1:.4f}")
    print("=" * 100)

    return {
        "fold": int(fold_id),
        "best_epoch": int(best_epoch),
        "best_mf1": float(best_mf1),
        "best_acc": float(best_acc),
        "best_loss": float(best_loss),
    }


# =========================
def main():
    cfg0, _, _, _, _ = build_all(MODEL_NAME)
    fold_df = load_fold_csv(FOLD_CSV)

    run_name = f"{cfg0.EXP_NAME}_cv5_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = ensure_dir(Path(cfg0.SAVE_ROOT) / run_name)

    print("=" * 100)
    print(f"[RUN] {run_name}")
    print(f"[MODEL_NAME] = {MODEL_NAME}")
    print(f"[HEAD_TYPE]  = {cfg0.HEAD_TYPE}")
    print(f"[FOLD_CSV]   = {FOLD_CSV}")
    print("=" * 100)

    save_json(cfg_to_dict(cfg0), run_dir / "config.json")

    results = []
    mf1_list, acc_list, loss_list, epoch_list = [], [], [], []

    for fold_id in range(1, NUM_FOLDS + 1):
        res = train_one_fold(run_dir, fold_id, fold_df)
        results.append(res)
        mf1_list.append(res["best_mf1"])
        acc_list.append(res["best_acc"])
        loss_list.append(res["best_loss"])
        epoch_list.append(float(res["best_epoch"]))

    mf1_stat = summarize(mf1_list)
    acc_stat = summarize(acc_list)
    loss_stat = summarize(loss_list)
    epoch_stat = summarize(epoch_list)

    summary = {
        "model_name": MODEL_NAME,
        "num_folds": NUM_FOLDS,
        "fold_csv": str(FOLD_CSV),
        "fold_results": results,
        "best_mf1": mf1_stat,
        "best_acc": acc_stat,
        "best_loss": loss_stat,
        "best_epoch": epoch_stat,
    }
    save_json(summary, run_dir / "cv5_summary.json")

    lines = [
        f"[MODEL] {MODEL_NAME}",
        f"[FOLDS] {NUM_FOLDS}",
        f"[FOLD_CSV] {FOLD_CSV}",
        "",
    ]
    for r in results:
        lines.append(
            f"fold={r['fold']} | best_epoch={r['best_epoch']} | "
            f"best_mf1={r['best_mf1']:.6f} | best_acc={r['best_acc']:.6f} | "
            f"best_loss={r['best_loss']:.6f}"
        )

    lines += [
        "",
        f"CV{NUM_FOLDS} best_mf1 : mean={mf1_stat['mean']:.6f}, std={mf1_stat['std']:.6f}",
        f"CV{NUM_FOLDS} best_acc  : mean={acc_stat['mean']:.6f}, std={acc_stat['std']:.6f}",
        f"CV{NUM_FOLDS} best_loss : mean={loss_stat['mean']:.6f}, std={loss_stat['std']:.6f}",
        f"CV{NUM_FOLDS} best_epoch: mean={epoch_stat['mean']:.2f}, std={epoch_stat['std']:.2f}",
    ]
    save_txt(run_dir / "cv5_summary.txt", "\n".join(lines))

    print("=" * 100)
    print("\n".join(lines))
    print(f"[RUN_DIR] {run_dir}")
    print("=" * 100)


if __name__ == "__main__":
    main()