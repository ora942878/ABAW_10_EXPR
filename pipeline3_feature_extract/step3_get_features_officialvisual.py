from __future__ import annotations
import os
import re
import sys
import gc
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from configs.paths import PATH

# --------- OMP_NUM_THREADS
v = os.environ.get("OMP_NUM_THREADS", "")
if (not v.isdigit()) or int(v) <= 0:
    os.environ["OMP_NUM_THREADS"] = "8"

# --------- load local dinov2
sys.path.insert(0, str(PATH.PROJECT_ROOT))
sys.path.insert(0, str(PATH.LIB_ROOT / "dinov2"))  # dinov2 repo root

try:
    from dinov2.hub.backbones import dinov2_vitl14
except Exception:
    from pipeline3_feature_extract.lib.dinov2.dinov2.hub.backbones import dinov2_vitl14


# =========================== config ===========================
BATCH_SIZE = 512
NUM_WORKERS = 1
STORE_FP16 = True
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
"""
weight_name : 
    dinov2_vitl14.pth                      --- PT ---- PATH.OFFIMG_V_DINOV2_PT
    finetune_dinov2_normal_weights_only.pt --- FT1 --- PATH.OFFIMG_V_DINOV2_FT1
    finetune_dinov2_auged_weights_only.pt  --- FT2 --- PATH.OFFIMG_V_DINOV2_FT2
"""
WEIGHT_PATH = PATH.VIT_WEIGHTS_ROOT / "finetune_dinov2_normal_weights_only.pt"
OUT_ROOT = PATH.OFFIMG_V_DINOV2_FT1
ROOTS = [PATH.IMG_batch1_ABAW10th, PATH.IMG_batch2_ABAW10th]

MODEL_NAME = "finetune_dinov2_normal"


# =========================== weightloader ===========================
def unwrap_to_state_dict(ckpt: Any) -> Optional[Dict[str, torch.Tensor]]:
    if isinstance(ckpt, dict):
        for k in ["model", "state_dict", "model_state", "backbone"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
        for v in ckpt.values():
            if torch.is_tensor(v):
                return ckpt
    return None


def auto_fix_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not sd:
        return sd
    prefixes = ["backbone.", "module.", "student.backbone."]
    for pref in prefixes:
        keys = list(sd.keys())
        if keys and all(k.startswith(pref) for k in keys[:20]):
            print(f"  [INFO] 自动剥离前缀: {pref}")
            sd = {k[len(pref):]: v for k, v in sd.items()}
    return sd


# =========================== common ===========================
def parse_frame_no(p: Path) -> int:
    m = re.search(r"(\d+)$", p.stem)
    return int(m.group(1)) if m else -1


def checkpoint_is_valid(pt_path: Path, expect_dim: int = 1024) -> bool:
    if not pt_path.exists():
        return False
    try:
        obj = torch.load(pt_path, map_location="cpu")
        required = {"folderid", "frames", "feats", "model", "fp16"}
        if not required.issubset(obj.keys()):
            return False
        if obj["feats"].ndim != 2 or obj["feats"].shape[1] != expect_dim:
            return False
        return True
    except Exception:
        return False


class ImageListDataset(Dataset):
    def __init__(self, items: List[Tuple[Path, int]], tfm):
        self.items = items
        self.tfm = tfm

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        p, frame_no = self.items[i]
        img = Image.open(p).convert("RGB")
        return self.tfm(img), frame_no


def collect_labeled_ids() -> set[str]:
    ids = set()
    for d in [Path(PATH.EXPR_TRAIN_ABAW10th), Path(PATH.EXPR_VALID_ABAW10th)]:
        if not d.exists():
            continue
        for p in d.glob("*.txt"):
            ids.add(p.stem)
    return ids


def collect_folderids_only_labeled(roots: list[Path], labeled: set[str]) -> list[Path]:
    # roots: IMG_batch1_ABAW10th / IMG_batch2_ABAW10th
    mapping = {}
    for r in roots:
        r = Path(r)
        if not r.exists():
            continue
        for sub in r.iterdir():
            if not sub.is_dir():
                continue
            name = sub.name
            if name in labeled:
                mapping[name] = sub

    return [mapping[k] for k in sorted(mapping.keys())]

@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    tfm = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    if not WEIGHT_PATH.exists():
        raise FileNotFoundError(f"weight not found: {WEIGHT_PATH}")

    print("=" * 70)
    print(f"[INFO] weight : {WEIGHT_PATH}")
    print(f"[INFO] out    : {OUT_ROOT}")
    print(f"[INFO] roots  : {ROOTS}")
    print("=" * 70)

    ckpt = torch.load(WEIGHT_PATH, map_location="cpu")
    sd = unwrap_to_state_dict(ckpt)
    if sd is None:
        raise RuntimeError(f"unrecognized checkpoint format: {WEIGHT_PATH}")
    sd = auto_fix_state_dict(sd)

    model = dinov2_vitl14(pretrained=False)
    incompatible = model.load_state_dict(sd, strict=False)
    print(f"[LOAD] missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}")

    model.eval().to(device)
    if device.type == "cuda":
        model.half()

    del ckpt, sd
    gc.collect()
    labeled = collect_labeled_ids()
    print(f"[INFO] labeled ids: {len(labeled)} (train+valid)")

    folder_dirs = collect_folderids_only_labeled(ROOTS, labeled)
    print(f"[INFO] will extract folderids: {len(folder_dirs)} (only labeled)")
    print(f"[INFO] found folderids: {len(folder_dirs)}")

    for vdir in tqdm(folder_dirs, desc=f"Total [{MODEL_NAME}]", ncols=110):
        folderid = vdir.name
        out_pt = OUT_ROOT / f"{folderid}.pt"
        if checkpoint_is_valid(out_pt):
            continue

        img_paths = [p for p in vdir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
        if not img_paths:
            continue

        items = [(p, parse_frame_no(p)) for p in img_paths]
        items.sort(key=lambda x: (x[1] < 0, x[1] if x[1] >= 0 else 0, x[0].name))

        dl = DataLoader(
            ImageListDataset(items, tfm),
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            pin_memory=(device.type == "cuda"),
        )

        all_feats = []
        all_frames: List[int] = []

        for batch_x, batch_fr in tqdm(dl, desc=f"Video [{folderid}]", leave=False, ncols=110):
            batch_x = batch_x.to(device, non_blocking=True)
            if device.type == "cuda":
                batch_x = batch_x.half()

            feats = model.forward_features(batch_x)["x_norm_clstoken"]  # [B,1024]
            if STORE_FP16 and feats.dtype != torch.float16:
                feats = feats.half()

            all_feats.append(feats.cpu())
            all_frames.extend([int(x) for x in batch_fr.tolist()])

        final_feats = torch.cat(all_feats, dim=0)


        if any(f < 0 for f in all_frames):
            all_frames = list(range(1, len(all_frames) + 1))

        torch.save(
            {
                "folderid": folderid,
                "frames": all_frames,      # list[int]
                "feats": final_feats,      # tensor [N,1024]
                "model": MODEL_NAME,
                "fp16": bool(STORE_FP16),
            },
            out_pt,
        )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    print(f"\n[DONE] saved to: {OUT_ROOT}")


if __name__ == "__main__":
    main()