

from __future__ import annotations

import os
import gc
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from configs.paths import PATH

from pipeline4_model_trains.common.set_seed import set_seed
from pipeline4_model_trains.common.macro_f1_from_cm import macro_f1_from_cm
from pipeline4_model_trains.utils.utils_id_mapper import IDMapper
from pipeline4_model_trains.utils.utils_read_expr_txt import read_expr_txt


# ========================= configs =========================
SEED = 3407
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# choose one aligned audio root here
AUDIO_ALIGNED_ROOT = Path(PATH.OFFAUDIO_WAV2VEC2_PT_aligned_window50)
# AUDIO_ALIGNED_ROOT = Path(PATH.OFFAUDIO_WAV2VEC2_PT_aligned_nearest)
# AUDIO_ALIGNED_ROOT = Path(PATH.OFFAUDIO_WAV2VEC2_PT_aligned_window25)
# AUDIO_ALIGNED_ROOT = Path(PATH.OFFAUDIO_WAV2VEC2_PT_aligned_window75)

RUN_TAG = f"audio_linear_{AUDIO_ALIGNED_ROOT.name}_{time.strftime('%Y%m%d_%H%M%S')}"
RUN_ROOT = Path(PATH.PROJECT_ROOT) / "pipeline4_model_trains" / "wav2vec2_eval" / "runs" / RUN_TAG

NUM_CLASSES = 8
FEAT_DIM = 1024

BATCH_SIZE = 4096
NUM_WORKERS = 4
EPOCHS = 20
LR = 1e-3
WEIGHT_DECAY = 1e-4
AMP = True

DO_ZSCORE = True
ZSCORE_EPS = 1e-6
USE_CLASS_WEIGHT = True

SAVE_BEST_BY = "mf1"   # "mf1" or "loss"

MAX_TRAIN_SAMPLES = None
MAX_VAL_SAMPLES = None

PIN_MEMORY = True
PERSISTENT_WORKERS = True

# ========================= io utils =========================
def save_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_txt(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def list_txtids(expr_dir: Path) -> List[str]:
    return sorted([p.stem for p in expr_dir.glob("*.txt")])


def load_audio_pt(audio_root: Path, videoid: str) -> Optional[dict]:
    p = audio_root / f"{videoid}.pt"
    if not p.exists():
        return None
    try:
        obj = torch.load(p, map_location="cpu")
        if not isinstance(obj, dict):
            return None
        return obj
    except Exception:
        return None


def parse_expr_seq(txt_path: Path) -> np.ndarray:
    txt_path = Path(txt_path)
    obj = read_expr_txt(txt_path)

    #  util returns: (seq, fmap)
    if isinstance(obj, tuple) and len(obj) == 2:
        seq, fmap = obj
        return np.asarray(seq, dtype=np.int64)

    if isinstance(obj, dict):
        if "seq" in obj and obj["seq"] is not None:
            return np.asarray(obj["seq"], dtype=np.int64)
        if "labels" in obj and obj["labels"] is not None:
            return np.asarray(obj["labels"], dtype=np.int64)
        if "y" in obj and obj["y"] is not None:
            return np.asarray(obj["y"], dtype=np.int64)

    if isinstance(obj, (list, np.ndarray)):
        return np.asarray(obj, dtype=np.int64)

    raise ValueError(f"Unsupported read_expr_txt output for: {txt_path}, type={type(obj)}")

def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    idx = y_true.astype(np.int64) * num_classes + y_pred.astype(np.int64)
    cm = np.bincount(idx, minlength=num_classes * num_classes)
    return cm.reshape(num_classes, num_classes).astype(np.int64)


def accuracy_from_cm(cm: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.trace(cm) / (cm.sum() + eps))


def classwise_f1_from_cm(cm: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    tp = np.diag(cm).astype(np.float64)
    p = tp / (cm.sum(axis=0) + eps)
    r = tp / (cm.sum(axis=1) + eps)
    f1 = 2.0 * p * r / (p + r + eps)
    return f1.astype(np.float64)


def format_cm(cm: np.ndarray) -> str:
    lines = []
    for r in cm:
        lines.append(" ".join(f"{int(x):7d}" for x in r))
    return "\n".join(lines)


# ========================= dataset build =========================
def build_frame_dataset(
    expr_dir: Path,
    audio_root: Path,
    mapper: IDMapper,
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    txtids = list_txtids(expr_dir)

    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []

    stats = {
        "expr_dir": str(expr_dir),
        "audio_root": str(audio_root),
        "n_txtids_total": len(txtids),
        "n_txtids_used": 0,
        "n_txtids_missing_videoid": 0,
        "n_txtids_missing_audio": 0,
        "n_txtids_bad_audio": 0,
        "n_frames_all_labels": 0,
        "n_frames_valid_labels": 0,
        "n_frames_used": 0,
        "n_frames_clipped_absdiff": 0,
    }

    for txtid in tqdm(txtids, desc=f"Build samples: {expr_dir.name}"):
        videoid = mapper.get_videoid(txtid)
        if videoid is None or len(str(videoid).strip()) == 0:
            stats["n_txtids_missing_videoid"] += 1
            continue

        audio_obj = load_audio_pt(audio_root, str(videoid))
        if audio_obj is None:
            stats["n_txtids_missing_audio"] += 1
            continue

        feats = audio_obj.get("feats", None)
        if (not torch.is_tensor(feats)) or feats.ndim != 2 or int(feats.shape[1]) != FEAT_DIM:
            stats["n_txtids_bad_audio"] += 1
            continue

        labels = parse_expr_seq(expr_dir / f"{txtid}.txt")
        feats_np = feats.detach().cpu().float().numpy()

        stats["n_frames_all_labels"] += int(labels.shape[0])

        T = min(int(labels.shape[0]), int(feats_np.shape[0]))
        stats["n_frames_clipped_absdiff"] += abs(int(labels.shape[0]) - int(feats_np.shape[0]))

        labels = labels[:T]
        feats_np = feats_np[:T]

        valid_mask = (labels >= 0) & (labels < NUM_CLASSES)
        n_valid = int(valid_mask.sum())
        stats["n_frames_valid_labels"] += n_valid

        if n_valid <= 0:
            continue

        xs.append(feats_np[valid_mask].astype(np.float32, copy=False))
        ys.append(labels[valid_mask].astype(np.int64, copy=False))

        stats["n_txtids_used"] += 1
        stats["n_frames_used"] += n_valid

        if max_samples is not None and stats["n_frames_used"] >= int(max_samples):
            break

    if len(xs) == 0:
        X = np.empty((0, FEAT_DIM), dtype=np.float32)
        y = np.empty((0,), dtype=np.int64)
    else:
        X = np.concatenate(xs, axis=0).astype(np.float32, copy=False)
        y = np.concatenate(ys, axis=0).astype(np.int64, copy=False)

    if max_samples is not None and X.shape[0] > int(max_samples):
        X = X[: int(max_samples)]
        y = y[: int(max_samples)]
        stats["n_frames_used"] = int(X.shape[0])

    return X, y, stats


class FrameFeatDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        do_zscore: bool = True,
    ):
        assert X.ndim == 2 and X.shape[1] == FEAT_DIM
        assert y.ndim == 1 and X.shape[0] == y.shape[0]

        self.X = X.astype(np.float32, copy=False)
        self.y = y.astype(np.int64, copy=False)
        self.mean = None if mean is None else mean.astype(np.float32, copy=False)
        self.std = None if std is None else std.astype(np.float32, copy=False)
        self.do_zscore = bool(do_zscore)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        x = self.X[idx]
        if self.do_zscore and self.mean is not None and self.std is not None:
            x = (x - self.mean) / self.std
        y = self.y[idx]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


# ========================= model =========================
class LinearHead(nn.Module):
    def __init__(self, in_dim: int = FEAT_DIM, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# ========================= train / eval =========================
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion, device: str) -> Dict:
    model.eval()

    total_loss = 0.0
    total_n = 0
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    autocast_enabled = AMP and device.startswith("cuda")

    for x, y in tqdm(loader, desc="Eval", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=autocast_enabled):
            logits = model(x)
            loss = criterion(logits, y)

        bs = int(y.size(0))
        total_loss += float(loss.item()) * bs
        total_n += bs

        pred = logits.argmax(dim=1).detach().cpu().numpy()
        gt = y.detach().cpu().numpy()
        cm += confusion_matrix_np(gt, pred, NUM_CLASSES)

    avg_loss = total_loss / max(total_n, 1)
    acc = accuracy_from_cm(cm)
    mf1 = macro_f1_from_cm(cm)
    cf1 = classwise_f1_from_cm(cm)

    return {
        "loss": float(avg_loss),
        "acc": float(acc),
        "mf1": float(mf1),
        "cf1": cf1,
        "cm": cm,
        "n": int(total_n),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    optimizer,
    scaler,
    device: str,
) -> float:
    model.train()

    total_loss = 0.0
    total_n = 0

    autocast_enabled = AMP and device.startswith("cuda")

    for x, y in tqdm(loader, desc="Train", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=autocast_enabled):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = int(y.size(0))
        total_loss += float(loss.item()) * bs
        total_n += bs

    return float(total_loss / max(total_n, 1))


# ========================= main =========================
def main():
    set_seed(SEED)
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    train_expr_dir = Path(PATH.EXPR_TRAIN_ABAW10th)
    val_expr_dir = Path(PATH.EXPR_VALID_ABAW10th)

    print("=" * 100)
    print(f"[RUN] {RUN_TAG}")
    print(f"[PATH] AUDIO_ALIGNED_ROOT = {AUDIO_ALIGNED_ROOT}")
    print(f"[PATH] TRAIN_EXPR_DIR    = {train_expr_dir}")
    print(f"[PATH] VALID_EXPR_DIR    = {val_expr_dir}")
    print(f"[CFG] DEVICE            = {DEVICE}")
    print(f"[CFG] BATCH_SIZE        = {BATCH_SIZE}")
    print(f"[CFG] EPOCHS            = {EPOCHS}")
    print(f"[CFG] LR                = {LR}")
    print(f"[CFG] WEIGHT_DECAY      = {WEIGHT_DECAY}")
    print(f"[CFG] DO_ZSCORE         = {DO_ZSCORE}")
    print(f"[CFG] USE_CLASS_WEIGHT  = {USE_CLASS_WEIGHT}")
    print("=" * 100)

    if not AUDIO_ALIGNED_ROOT.exists():
        raise FileNotFoundError(f"Missing AUDIO_ALIGNED_ROOT: {AUDIO_ALIGNED_ROOT}")

    mapper = IDMapper(Path(PATH.EXPR_VIDEO_INDEX_CSV))

    print("[INFO] Building train dataset ...")
    X_train, y_train, train_stats = build_frame_dataset(
        expr_dir=train_expr_dir,
        audio_root=AUDIO_ALIGNED_ROOT,
        mapper=mapper,
        max_samples=MAX_TRAIN_SAMPLES,
    )

    print("[INFO] Building val dataset ...")
    X_val, y_val, val_stats = build_frame_dataset(
        expr_dir=val_expr_dir,
        audio_root=AUDIO_ALIGNED_ROOT,
        mapper=mapper,
        max_samples=MAX_VAL_SAMPLES,
    )

    if X_train.shape[0] == 0:
        raise RuntimeError("Train samples are empty.")
    if X_val.shape[0] == 0:
        raise RuntimeError("Val samples are empty.")

    print(f"[DATA] Train: N={X_train.shape[0]}, dim={X_train.shape[1]}")
    print(f"[DATA] Val  : N={X_val.shape[0]}, dim={X_val.shape[1]}")

    train_mean = X_train.mean(axis=0).astype(np.float32)
    train_std = X_train.std(axis=0).astype(np.float32)
    train_std = np.maximum(train_std, ZSCORE_EPS).astype(np.float32)

    class_count = np.bincount(y_train, minlength=NUM_CLASSES).astype(np.float64)
    class_weight = None
    if USE_CLASS_WEIGHT:
        w = class_count.sum() / np.maximum(class_count, 1.0)
        w = w / w.mean()
        class_weight = torch.tensor(w, dtype=torch.float32, device=DEVICE)

    ds_train = FrameFeatDataset(
        X=X_train,
        y=y_train,
        mean=train_mean,
        std=train_std,
        do_zscore=DO_ZSCORE,
    )
    ds_val = FrameFeatDataset(
        X=X_val,
        y=y_val,
        mean=train_mean,
        std=train_std,
        do_zscore=DO_ZSCORE,
    )

    loader_train = DataLoader(
        ds_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
        persistent_workers=(PERSISTENT_WORKERS and NUM_WORKERS > 0),
    )
    loader_val = DataLoader(
        ds_val,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
        persistent_workers=(PERSISTENT_WORKERS and NUM_WORKERS > 0),
    )

    model = LinearHead(in_dim=FEAT_DIM, num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=(AMP and DEVICE.startswith("cuda")))

    save_json(RUN_ROOT / "train_stats.json", train_stats)
    save_json(RUN_ROOT / "val_stats.json", val_stats)

    cfg = {
        "RUN_TAG": RUN_TAG,
        "AUDIO_ALIGNED_ROOT": str(AUDIO_ALIGNED_ROOT),
        "TRAIN_EXPR_DIR": str(train_expr_dir),
        "VALID_EXPR_DIR": str(val_expr_dir),
        "DEVICE": DEVICE,
        "NUM_CLASSES": NUM_CLASSES,
        "FEAT_DIM": FEAT_DIM,
        "BATCH_SIZE": BATCH_SIZE,
        "NUM_WORKERS": NUM_WORKERS,
        "EPOCHS": EPOCHS,
        "LR": LR,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "DO_ZSCORE": DO_ZSCORE,
        "USE_CLASS_WEIGHT": USE_CLASS_WEIGHT,
        "AMP": AMP,
    }
    save_json(RUN_ROOT / "config.json", cfg)

    best_score = -1e18 if SAVE_BEST_BY == "mf1" else 1e18
    best_epoch = -1
    history_lines: List[str] = []

    for ep in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model=model,
            loader=loader_train,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=DEVICE,
        )

        val_res = evaluate(
            model=model,
            loader=loader_val,
            criterion=criterion,
            device=DEVICE,
        )

        line = (
            f"ep {ep:02d} | "
            f"tr_loss {train_loss:.6f} | "
            f"va_loss {val_res['loss']:.6f} | "
            f"va_acc {val_res['acc']:.6f} | "
            f"va_mf1 {val_res['mf1']:.6f} | "
            f"time {time.time() - t0:.1f}s"
        )
        print(line)
        history_lines.append(line)

        cur_score = val_res["mf1"] if SAVE_BEST_BY == "mf1" else (-val_res["loss"])
        is_best = cur_score > best_score

        if is_best:
            best_score = cur_score
            best_epoch = ep

            ckpt = {
                "epoch": ep,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": cfg,
                "val_loss": float(val_res["loss"]),
                "val_acc": float(val_res["acc"]),
                "val_mf1": float(val_res["mf1"]),
                "val_cm": val_res["cm"],
                "train_mean": train_mean,
                "train_std": train_std,
            }
            torch.save(ckpt, RUN_ROOT / "best.pt")

            best_txt = []
            best_txt.append(f"[BEST] epoch={ep}")
            best_txt.append(f"val_loss={val_res['loss']:.6f}")
            best_txt.append(f"val_acc={val_res['acc']:.6f}")
            best_txt.append(f"val_mf1={val_res['mf1']:.6f}")
            best_txt.append("")
            best_txt.append("[CONFUSION MATRIX]")
            best_txt.append(format_cm(val_res["cm"]))
            best_txt.append("")
            best_txt.append("[CLASSWISE F1]")
            best_txt.append(" ".join([f"{x:.6f}" for x in val_res["cf1"]]))
            save_txt(RUN_ROOT / "best_metrics.txt", "\n".join(best_txt))

        save_txt(RUN_ROOT / "history.txt", "\n".join(history_lines))

        gc.collect()
        if DEVICE.startswith("cuda"):
            torch.cuda.empty_cache()

    print("=" * 100)
    print(f"[DONE] best_epoch={best_epoch}")
    if (RUN_ROOT / "best.pt").exists():
        best_ckpt = torch.load(RUN_ROOT / "best.pt", map_location="cpu")
        print(f"[BEST] val_loss={best_ckpt['val_loss']:.6f}")
        print(f"[BEST] val_acc ={best_ckpt['val_acc']:.6f}")
        print(f"[BEST] val_mf1 ={best_ckpt['val_mf1']:.6f}")
    print(f"[RUN_ROOT] {RUN_ROOT}")
    print("=" * 100)


if __name__ == "__main__":
    main()