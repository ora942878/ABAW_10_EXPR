from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from configs.paths import PATH

# =========================
# New dependency set only
# =========================
from pipeline4_model_trains.common.set_seed import set_seed
from pipeline4_model_trains.common.evaluate_classification import evaluate_classification
from pipeline4_model_trains.models.builder_single_modal import build_single_modal_head
from pipeline4_model_trains.utils.utils_read_expr_txt import read_expr_txt


# =========================
@dataclass
class CFG:
    TRAIN_EXPR_DIR: str = str(PATH.EXPR_TRAIN_ABAW10th)
    VALID_EXPR_DIR: str = str(PATH.EXPR_VALID_ABAW10th)
    FEATURE_ROOT: str = str(PATH.FACE09_V_DINOV2_FT2)

    USE_MEAN_091215: bool = False
    FEATURE_ROOT_09: str = str(PATH.FACE09_V_DINOV2_FT1)
    FEATURE_ROOT_12: str = str(PATH.FACE12_V_DINOV2_FT1)
    FEATURE_ROOT_15: str = str(PATH.FACE15_V_DINOV2_FT1)

    EXP_NAME: str = "dino_eval_default"
    HEAD_TYPE: str = "linear"
    NUM_CLASSES: int = 8
    BATCH_SIZE: int = 4096
    NUM_WORKERS: int = 8
    EPOCHS: int = 20
    LR: float = 1e-3
    WEIGHT_DECAY: float = 1e-4

    HIDDEN_DIM: int = 512
    DROPOUT: float = 0.2


    DO_ZSCORE: bool = True
    DO_L2: bool = True

    SEED: int = 3407
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    SAVE_ROOT: str = str(PATH.RUN_ROOT / "dino_eval")


cfg = CFG()

def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(obj, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def acc_from_cm(cm: np.ndarray, eps: float = 1e-12) -> float:
    correct = float(np.diag(cm).sum())
    total = float(cm.sum())
    return correct / max(total, eps)


def list_txtids_from_expr_dir(expr_dir: str | Path) -> List[str]:
    expr_dir = Path(expr_dir)
    txts = sorted(expr_dir.glob("*.txt"))
    return [p.stem for p in txts]


def read_expr_txt_labels(txt_path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    seq, fmap = read_expr_txt(Path(txt_path))

    if fmap:
        frames = np.asarray(sorted(fmap.keys()), dtype=np.int64)
        labels = np.asarray([int(fmap[int(fr)]) for fr in frames], dtype=np.int64)
        return frames, labels

    if len(seq) == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)

    labels = np.asarray(seq, dtype=np.int64)
    frames = np.arange(1, len(labels) + 1, dtype=np.int64)
    return frames, labels


def load_feat_pt(pt_path: str | Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    pt_path = Path(pt_path)
    if not pt_path.exists():
        return None

    try:
        obj = torch.load(pt_path, map_location="cpu")
    except Exception:
        return None

    if not isinstance(obj, dict):
        return None
    if "frames" not in obj or "feats" not in obj:
        return None

    frames = obj["frames"]
    feats = obj["feats"]

    if not isinstance(frames, list):
        return None
    if not torch.is_tensor(feats) or feats.ndim != 2:
        return None

    frames_np = np.asarray(frames, dtype=np.int64)
    feats_np = feats.detach().cpu().float().numpy()

    if len(frames_np) != feats_np.shape[0]:
        return None
    return frames_np, feats_np


def intersect_frames(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    common = sorted(set(map(int, a.tolist())) & set(map(int, b.tolist())))
    return np.asarray(common, dtype=np.int64)


def gather_by_common_frames(
    frames_src: np.ndarray,
    values_src: np.ndarray,
    common_frames: np.ndarray,
) -> np.ndarray:
    pos = {int(fr): i for i, fr in enumerate(frames_src.tolist())}
    idx = np.asarray([pos[int(fr)] for fr in common_frames], dtype=np.int64)
    return values_src[idx]


def compute_mean_std_from_numpy_list(xs: List[np.ndarray], eps: float = 1e-12):
    if len(xs) == 0:
        raise ValueError("xs is empty")

    dim = xs[0].shape[1]
    sum_x = np.zeros((dim,), dtype=np.float64)
    sum_x2 = np.zeros((dim,), dtype=np.float64)
    n = 0

    for x in xs:
        if x.ndim != 2 or x.shape[1] != dim:
            raise ValueError("inconsistent feature shape in xs")
        x64 = x.astype(np.float64)
        sum_x += x64.sum(axis=0)
        sum_x2 += (x64 * x64).sum(axis=0)
        n += x.shape[0]

    mean = sum_x / max(n, 1)
    var = sum_x2 / max(n, 1) - mean ** 2
    var = np.clip(var, a_min=eps, a_max=None)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


def normalize_features(
    x: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
    do_zscore: bool = False,
    do_l2: bool = False,
    eps: float = 1e-12,
) -> np.ndarray:
    out = x.astype(np.float32, copy=True)
    if do_zscore:
        if mean is None or std is None:
            raise ValueError("mean/std must be provided when do_zscore=True")
        out = (out - mean) / (std + eps)
    if do_l2:
        norm = np.linalg.norm(out, ord=2, axis=1, keepdims=True)
        out = out / (norm + eps)
    return out.astype(np.float32)


# =========================
class VisualExprEvalDataset(Dataset):
    def __init__(
        self,
        expr_dir: str | Path,
        feat_root: str | Path,
        num_classes: int = 8,
        do_zscore: bool = False,
        do_l2: bool = False,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        verbose: bool = True,
        use_mean_091215: bool = False,
        feat_root_09: Optional[str | Path] = None,
        feat_root_12: Optional[str | Path] = None,
        feat_root_15: Optional[str | Path] = None,
    ):
        self.expr_dir = Path(expr_dir)
        self.feat_root = Path(feat_root)
        self.num_classes = int(num_classes)

        self.do_zscore = bool(do_zscore)
        self.do_l2 = bool(do_l2)
        self.mean = mean
        self.std = std

        self.use_mean_091215 = bool(use_mean_091215)
        self.feat_root_09 = Path(feat_root_09) if feat_root_09 is not None else None
        self.feat_root_12 = Path(feat_root_12) if feat_root_12 is not None else None
        self.feat_root_15 = Path(feat_root_15) if feat_root_15 is not None else None

        self.x_list: List[np.ndarray] = []
        self.y_list: List[np.ndarray] = []

        self.feat_dim: Optional[int] = None
        self.total_videos = 0
        self.used_videos = 0
        self.total_frames = 0
        self.miss_feat = 0
        self.no_valid_label = 0
        self.no_overlap = 0
        self.bad_dim = 0

        self._build(verbose=verbose)

        if len(self.x_list) == 0:
            raise RuntimeError(f"No usable samples found. expr_dir={self.expr_dir}, feat_root={self.feat_root}")

        self.X = np.concatenate(self.x_list, axis=0).astype(np.float32)
        self.Y = np.concatenate(self.y_list, axis=0).astype(np.int64)

        if self.do_zscore:
            if self.mean is None or self.std is None:
                raise ValueError("mean/std must be provided when do_zscore=True")
            self.X = normalize_features(self.X, mean=self.mean, std=self.std, do_zscore=True, do_l2=False)

        if self.do_l2:
            self.X = normalize_features(self.X, do_zscore=False, do_l2=True)

    def _load_single_stream(self, feat_root: Path, txtid: str):
        return load_feat_pt(feat_root / f"{txtid}.pt")

    def _load_mean_091215(self, txtid: str):
        if self.feat_root_09 is None or self.feat_root_12 is None or self.feat_root_15 is None:
            return None

        pack09 = self._load_single_stream(self.feat_root_09, txtid)
        pack12 = self._load_single_stream(self.feat_root_12, txtid)
        pack15 = self._load_single_stream(self.feat_root_15, txtid)
        if pack09 is None or pack12 is None or pack15 is None:
            return None

        f09, x09 = pack09
        f12, x12 = pack12
        f15, x15 = pack15

        common = intersect_frames(intersect_frames(f09, f12), f15)
        if len(common) == 0:
            return None

        x09c = gather_by_common_frames(f09, x09, common)
        x12c = gather_by_common_frames(f12, x12, common)
        x15c = gather_by_common_frames(f15, x15, common)

        if not (x09c.shape == x12c.shape == x15c.shape):
            return None

        x = (x09c + x12c + x15c) / 3.0
        return common, x.astype(np.float32)

    def _build(self, verbose: bool = True):
        txtids = list_txtids_from_expr_dir(self.expr_dir)
        iterator = tqdm(txtids, desc=f"Collect {self.expr_dir.name}") if verbose else txtids

        for txtid in iterator:
            self.total_videos += 1

            label_frames, label_values = read_expr_txt_labels(self.expr_dir / f"{txtid}.txt")
            keep = (label_values >= 0) & (label_values < self.num_classes)
            label_frames = label_frames[keep]
            label_values = label_values[keep]

            if len(label_frames) == 0:
                self.no_valid_label += 1
                continue

            if self.use_mean_091215:
                feat_pack = self._load_mean_091215(txtid)
            else:
                feat_pack = self._load_single_stream(self.feat_root, txtid)

            if feat_pack is None:
                self.miss_feat += 1
                continue

            feat_frames, feats = feat_pack
            common = intersect_frames(label_frames, feat_frames)
            if len(common) == 0:
                self.no_overlap += 1
                continue

            y = gather_by_common_frames(label_frames, label_values, common).astype(np.int64)
            x = gather_by_common_frames(feat_frames, feats, common).astype(np.float32)

            if x.ndim != 2:
                self.bad_dim += 1
                continue

            if self.feat_dim is None:
                self.feat_dim = int(x.shape[1])
            elif int(x.shape[1]) != self.feat_dim:
                self.bad_dim += 1
                continue

            self.used_videos += 1
            self.total_frames += int(x.shape[0])
            self.x_list.append(x)
            self.y_list.append(y)

        if verbose:
            print(
                f"[DATA] {self.expr_dir.name}: videos={self.used_videos}/{self.total_videos}, "
                f"frames={self.total_frames}, miss_feat={self.miss_feat}, "
                f"no_valid_label={self.no_valid_label}, no_overlap={self.no_overlap}, bad_dim={self.bad_dim}"
            )

    def __len__(self):
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.X[idx])
        y = torch.tensor(int(self.Y[idx]), dtype=torch.long)
        return x, y


@torch.no_grad()
def compute_dataset_mean_std(ds: VisualExprEvalDataset):
    return compute_mean_std_from_numpy_list(ds.x_list)




# =========================
def main():
    set_seed(cfg.SEED)
    device = torch.device(cfg.DEVICE)

    run_name = f"{cfg.EXP_NAME}_{Path(cfg.FEATURE_ROOT).name}_{cfg.HEAD_TYPE}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = ensure_dir(Path(cfg.SAVE_ROOT) / run_name)

    print("=" * 80)
    print(f"[RUN] {run_name}")
    print(f"[PATH] FEATURE_ROOT   = {cfg.FEATURE_ROOT}")
    print(f"[PATH] TRAIN_EXPR_DIR = {cfg.TRAIN_EXPR_DIR}")
    print(f"[PATH] VALID_EXPR_DIR = {cfg.VALID_EXPR_DIR}")
    print(f"[CFG] HEAD_TYPE       = {cfg.HEAD_TYPE}")
    print(f"[CFG] DO_ZSCORE       = {cfg.DO_ZSCORE}")
    print(f"[CFG] DO_L2           = {cfg.DO_L2}")
    print("=" * 80)

    save_json(asdict(cfg), run_dir / "config.json")

    print("[INFO] Building raw train dataset for statistics ...")
    ds_tr_raw = VisualExprEvalDataset(
        expr_dir=cfg.TRAIN_EXPR_DIR,
        feat_root=cfg.FEATURE_ROOT,
        num_classes=cfg.NUM_CLASSES,
        do_zscore=False,
        do_l2=False,
        mean=None,
        std=None,
        verbose=True,
        use_mean_091215=cfg.USE_MEAN_091215,
        feat_root_09=cfg.FEATURE_ROOT_09,
        feat_root_12=cfg.FEATURE_ROOT_12,
        feat_root_15=cfg.FEATURE_ROOT_15,
    )

    mean = None
    std = None
    if cfg.DO_ZSCORE:
        mean, std = compute_dataset_mean_std(ds_tr_raw)
        np.save(run_dir / "train_mean.npy", mean)
        np.save(run_dir / "train_std.npy", std)
        print("[INFO] train mean/std saved.")

    print("[INFO] Building normalized train/val datasets ...")
    ds_tr = VisualExprEvalDataset(
        expr_dir=cfg.TRAIN_EXPR_DIR,
        feat_root=cfg.FEATURE_ROOT,
        num_classes=cfg.NUM_CLASSES,
        do_zscore=cfg.DO_ZSCORE,
        do_l2=cfg.DO_L2,
        mean=mean,
        std=std,
        verbose=True,
        use_mean_091215=cfg.USE_MEAN_091215,
        feat_root_09=cfg.FEATURE_ROOT_09,
        feat_root_12=cfg.FEATURE_ROOT_12,
        feat_root_15=cfg.FEATURE_ROOT_15,
    )
    ds_va = VisualExprEvalDataset(
        expr_dir=cfg.VALID_EXPR_DIR,
        feat_root=cfg.FEATURE_ROOT,
        num_classes=cfg.NUM_CLASSES,
        do_zscore=cfg.DO_ZSCORE,
        do_l2=cfg.DO_L2,
        mean=mean,
        std=std,
        verbose=True,
        use_mean_091215=cfg.USE_MEAN_091215,
        feat_root_09=cfg.FEATURE_ROOT_09,
        feat_root_12=cfg.FEATURE_ROOT_12,
        feat_root_15=cfg.FEATURE_ROOT_15,
    )

    ld_tr = DataLoader(
        ds_tr,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )
    ld_va = DataLoader(
        ds_va,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    model = build_single_modal_head(
        head_type=cfg.HEAD_TYPE,
        in_dim=ds_tr.feat_dim,
        num_classes=cfg.NUM_CLASSES,
        hidden_dim=cfg.HIDDEN_DIM,
        dropout=cfg.DROPOUT,
    ).to(device)
    train_criterion = torch.nn.CrossEntropyLoss()
    val_criterion = lambda logits, y: F.cross_entropy(logits, y)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.EPOCHS,
        eta_min=cfg.LR * 0.1,
    )

    best_mf1 = -1.0
    best_epoch = -1

    print("[INFO] Start training ...")
    for ep in range(1, cfg.EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        tr_n = 0
        tr_correct = 0

        pbar = tqdm(ld_tr, desc=f"ep{ep:02d} train", ncols=120)
        for x, y in pbar:
            x = x.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True).long()

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = train_criterion(logits, y)
            loss.backward()
            optimizer.step()

            bs = int(x.shape[0])
            tr_loss += float(loss.item()) * bs
            tr_n += bs
            pred = logits.argmax(dim=1)
            tr_correct += int((pred == y).sum().item())

            pbar.set_postfix(
                loss=f"{tr_loss / max(tr_n, 1):.4f}",
                acc=f"{tr_correct / max(tr_n, 1):.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        scheduler.step()

        tr_loss_avg = tr_loss / max(tr_n, 1)
        tr_acc = tr_correct / max(tr_n, 1)
        va_stats = evaluate_classification(model, ld_va, val_criterion, device, num_classes=cfg.NUM_CLASSES)
        va_acc_check = acc_from_cm(va_stats["cm"])

        msg = {
            "epoch": ep,
            "train_loss": float(tr_loss_avg),
            "train_acc": float(tr_acc),
            "val_loss": float(va_stats["loss"]),
            "val_acc": float(va_stats["acc"]),
            "val_acc_from_cm": float(va_acc_check),
            "val_mf1": float(va_stats["mf1"]),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }

        print(
            f"[ep {ep:02d}] "
            f"tr_loss={msg['train_loss']:.4f} "
            f"tr_acc={msg['train_acc']:.4f} | "
            f"va_loss={msg['val_loss']:.4f} "
            f"va_acc={msg['val_acc']:.4f} "
            f"va_mf1={msg['val_mf1']:.4f}"
        )

        append_jsonl(msg, run_dir / "metrics.jsonl")

        if msg["val_mf1"] > best_mf1:
            best_mf1 = msg["val_mf1"]
            best_epoch = ep
            ckpt = {
                "cfg": asdict(cfg),
                "epoch": ep,
                "best_mf1": float(best_mf1),
                "model": model.state_dict(),
                "feat_dim": int(ds_tr.feat_dim),
            }
            torch.save(ckpt, run_dir / "best.pt")
            save_json(
                {
                    "best_epoch": int(best_epoch),
                    "best_mf1": float(best_mf1),
                    "feature_root": cfg.FEATURE_ROOT,
                    "head_type": cfg.HEAD_TYPE,
                },
                run_dir / "best_summary.json",
            )
            np.save(run_dir / "best_cm.npy", va_stats["cm"])
            print(f"[SAVE] best updated -> {run_dir / 'best.pt'} (mf1={best_mf1:.4f})")

    print("=" * 80)
    print(f"[DONE] best_epoch={best_epoch}, best_mf1={best_mf1:.4f}")
    print(f"[RUN_DIR] {run_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
