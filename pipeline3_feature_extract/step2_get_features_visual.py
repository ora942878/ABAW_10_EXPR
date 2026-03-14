from __future__ import annotations
import os
import re
import sys
import time
import gc
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm

# =========================== 路径与环境配置 ===========================
from configs.paths import PATH

sys.path.insert(0, str(PATH.LIB_ROOT / "dinov2"))

from pipeline3_feature_extract.lib.dinov2.dinov2.hub.backbones import dinov2_vitl14

# =========================== Global Config ===========================
BATCH_SIZE = 512
NUM_WORKERS = 1 # CPU data loading speed
STORE_FP16 = True
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


TASKS = [
    {
        "model_name": "finetune_dinov2_auged_withoutPadAug",
        "weight_path": PATH.VIT_WEIGHTS_ROOT / 'finetune_dinov2_auged_withoutPadAug.pt',
        "out_root": PATH.FACE09_V_DINOV2_FT3,
        "roots": [PATH.ABAW_FACE09_ROOT]
    },
    {
        "model_name": "finetune_dinov2_auged_withoutPadAug",
        "weight_path": PATH.VIT_WEIGHTS_ROOT / 'finetune_dinov2_auged_withoutPadAug.pt',
        "out_root": PATH.FACE12_V_DINOV2_FT3,
        "roots": [PATH.ABAW_FACE12_ROOT]
    },
    {
        "model_name": "finetune_dinov2_auged_withoutPadAug",
        "weight_path": PATH.VIT_WEIGHTS_ROOT / 'finetune_dinov2_auged_withoutPadAug.pt',
        "out_root": PATH.FACE15_V_DINOV2_FT3,
        "roots": [PATH.ABAW_FACE15_ROOT]
    },
    # {
    #     "model_name": "dinov2_vitl14_pretrained",
    #     "weight_path": PATH.VIT_WEIGHTS_ROOT / 'dinov2_vitl14.pth',
    #     "out_root": PATH.FACE09_V_DINOV2_PT,
    #     "roots": [PATH.ABAW_FACE09_ROOT]
    # },
    # {
    #     "model_name": "finetune_dinov2_normal_weights_only",
    #     "weight_path": PATH.VIT_WEIGHTS_ROOT / 'finetune_dinov2_normal_weights_only.pt',
    #     "out_root": PATH.FACE09_V_DINOV2_FT1,
    #     "roots": [PATH.ABAW_FACE09_ROOT]
    # },
    # {
    #     "model_name": "finetune_dinov2_auged",
    #     "weight_path": PATH.VIT_WEIGHTS_ROOT / 'finetune_dinov2_auged_weights_only.pt',
    #     "out_root": PATH.FACE09_V_DINOV2_FT2,
    #     "roots": [PATH.ABAW_FACE09_ROOT]
    # },
# ===========================
#     {
#         "model_name": "finetune_dinov2_auged",
#         "weight_path": PATH.VIT_WEIGHTS_ROOT / 'finetune_dinov2_auged_weights_only.pt',
#         "out_root": PATH.FACE12_V_DINOV2_FT2,
#         "roots": [PATH.ABAW_FACE12_ROOT]
#     },
#     {
#         "model_name": "finetune_dinov2_auged",
#         "weight_path": PATH.VIT_WEIGHTS_ROOT / 'finetune_dinov2_auged_weights_only.pt',
#         "out_root": PATH.FACE15_V_DINOV2_FT2,
#         "roots": [PATH.ABAW_FACE15_ROOT]
#     },
#
#
#
#     {
#         "model_name": "dinov2_vitl14_pretrained",
#         "weight_path": PATH.VIT_WEIGHTS_ROOT / 'dinov2_vitl14.pth',
#         "out_root": PATH.FACE12_V_DINOV2_PT,
#         "roots": [PATH.ABAW_FACE12_ROOT]
#     },
#     {
#         "model_name": "finetune_dinov2_normal_weights_only",
#         "weight_path": PATH.VIT_WEIGHTS_ROOT / 'finetune_dinov2_normal_weights_only.pt',
#         "out_root": PATH.FACE12_V_DINOV2_FT1,
#         "roots": [PATH.ABAW_FACE12_ROOT]
#     },

# ===========================
#     {
#         "model_name": "dinov2_vitl14_pretrained",
#         "weight_path": PATH.VIT_WEIGHTS_ROOT / 'dinov2_vitl14.pth',
#         "out_root": PATH.FACE15_V_DINOV2_PT,
#         "roots": [PATH.ABAW_FACE15_ROOT]
#     },
#     {
#         "model_name": "finetune_dinov2_normal_weights_only",
#         "weight_path": PATH.VIT_WEIGHTS_ROOT / 'finetune_dinov2_normal_weights_only.pt',
#         "out_root": PATH.FACE15_V_DINOV2_FT1,
#         "roots": [PATH.ABAW_FACE15_ROOT]
#     },

]

# =========================== Weight Loading Utils ===========================

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
    if not sd: return sd

    prefixes = ["backbone.", "module.", "student.backbone."]
    for pref in prefixes:
        keys = list(sd.keys())
        if all(k.startswith(pref) for k in keys[:20]):
            print(f"  [INFO] 自动剥离前缀: {pref}")
            sd = {k[len(pref):]: v for k, v in sd.items()}
    return sd


# =========================== General Utils ===========================

def parse_frame_no(p: Path) -> int:
    m = re.search(r"(\d+)$", p.stem)
    return int(m.group(1)) if m else -1


def checkpoint_is_valid(pt_path: Path, expect_dim: int = 1024) -> bool:
    if not pt_path.exists(): return False
    try:
        obj = torch.load(pt_path, map_location="cpu")
        required = {"folderid", "frames", "feats", "model", "fp16"}
        if not required.issubset(obj.keys()): return False
        if obj["feats"].shape[1] != expect_dim: return False
        return True
    except:
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


# =========================== Main ===========================

@torch.no_grad()
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tfm = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    for task_idx, task in enumerate(TASKS, 1):
        weight_path = task["weight_path"]
        out_root = task["out_root"]
        model_name = task["model_name"]
        roots = task["roots"]

        print(f"\n" + "=" * 60)
        print(f"Starting Task {task_idx}/{len(TASKS)}")
        print(f"Weights: {weight_path.name}")
        print(f"Output: {out_root}")
        print("=" * 60)
        if not weight_path.exists():
            print(f"[ERROR] Weight file not found: {weight_path}. Skipping.")
            continue

        out_root.mkdir(parents=True, exist_ok=True)
        # 1. weight loading
        print(f"[INFO] Loading model weights...")
        ckpt = torch.load(weight_path, map_location="cpu")
        sd = unwrap_to_state_dict(ckpt)
        if sd is None:
            print(f"[ERROR] Unrecognized weight format:{weight_path}")
            continue
        sd = auto_fix_state_dict(sd)
        # 2. load model
        model = dinov2_vitl14(pretrained=False)
        incompatible = model.load_state_dict(sd, strict=False)
        print(f"  [LOAD] Missing: {len(incompatible.missing_keys)}, Unexpected: {len(incompatible.unexpected_keys)}")

        model.eval().to(device)
        if device.type == "cuda":
            model.half()  #FP16

        del ckpt, sd
        gc.collect()

        # 3. Collect video directories
        video_dirs = []
        for r in roots:
            r = Path(r)
            if r.exists():
                video_dirs.extend([d for d in r.iterdir() if d.is_dir()])

        print(f"[INFO] Processing {len(video_dirs)} video folders")

        for vdir in tqdm(video_dirs, desc=f"Total [{model_name}]"):
            folderid = vdir.name
            out_pt = out_root / f"{folderid}.pt"

            if checkpoint_is_valid(out_pt): continue

            img_paths = [p for p in vdir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
            if not img_paths: continue

            items = [(p, parse_frame_no(p)) for p in img_paths]
            items.sort(key=lambda x: (x[1] < 0, x[1] if x[1] >= 0 else 0, x[0].name))

            dl = DataLoader(ImageListDataset(items, tfm), batch_size=BATCH_SIZE,
                            num_workers=NUM_WORKERS, pin_memory=True)

            all_feats, all_frames = [], []

            inner_pbar = tqdm(dl, desc=f"Video [{folderid}]", leave=False)

            for batch_x, batch_fr in inner_pbar:
                batch_x = batch_x.to(device, non_blocking=True)
                if device.type == "cuda":
                    batch_x = batch_x.half()

                feats = model.forward_features(batch_x)["x_norm_clstoken"]

                if STORE_FP16 and feats.dtype != torch.float16:
                    feats = feats.half()

                all_feats.append(feats.cpu())
                all_frames.extend(batch_fr.tolist())

            # 4. Assemble and save results
            final_feats = torch.cat(all_feats, dim=0)

            if any(f < 0 for f in all_frames):
                all_frames = list(range(1, len(all_frames) + 1))

            torch.save({
                "folderid": folderid,
                "frames": all_frames,
                "feats": final_feats,
                "model": model_name,
                "fp16": bool(STORE_FP16),
            }, out_pt)

        print(f"task {task_idx} complete. features saved to: {out_root}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    print("\n feature extraction tasks finished!")


if __name__ == "__main__":
    main()