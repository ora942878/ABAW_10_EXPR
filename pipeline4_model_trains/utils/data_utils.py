from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Union, Sequence
import numpy as np

"""
from pipeline4_model_trains.utils.data_utils import collect_imgset_pairs, collect_abaw_uniform_pairs, RawImageDataset
"""

from configs.paths import PATH
sys.path.insert(0, str(PATH.LIB_ROOT / "dinov2"))
C = 8
CLASS_NAMES = ["Neutral", "Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise", "Other"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# =========================== read data tools ===========================
def parse_class_to_label(class_dirname: str) -> int:
    s = class_dirname.strip().lower()
    if s.isdigit(): return int(s)
    if "_" in s and s.split("_")[0].isdigit(): return int(s.split("_")[0])
    name_map = {"neutral": 0, "anger": 1, "disgust": 2, "fear": 3, "happiness": 4, "sadness": 5, "surprise": 6,
                "other": 7}
    for k, v in name_map.items():
        if k in s: return v
    return -1

def read_expr_txt(txt_path: Path) -> Tuple[List[int], Dict[int, int]]:
    seq, fmap = [], {}
    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s: continue
            parts = [p.strip() for p in s.split(",") if p.strip() != ""]
            try:
                ints = [int(p) for p in parts]
            except ValueError:
                continue
            if len(ints) >= 2 and (ints[-1] in range(-1, 8)) and (ints[0] > 7):
                fmap[int(ints[0])] = int(ints[-1])
            else:
                seq.append(int(ints[-1]))
    return seq, fmap

def parse_frame_no(stem: str) -> int:
    digits = ""
    for ch in reversed(stem):
        if ch.isdigit():
            digits = ch + digits
        else:
            if digits: break
    return int(digits) if digits else -1

# =========================== build data ===========================
# return (Path, int(label))
def collect_imgset_pairs(dataset_root: Path, is_train: bool) -> List[Tuple[Path, int]]:
    pairs = []
    keywords = ["train", "tr"] if is_train else ["valid", "val", "va"]

    if not dataset_root.exists(): return pairs

    for batch_dir in dataset_root.iterdir():
        if not batch_dir.is_dir() or batch_dir.name.startswith("."): continue
        if not any(k in batch_dir.name.lower() for k in keywords): continue

        for cls_dir in batch_dir.iterdir():
            if not cls_dir.is_dir(): continue
            y = parse_class_to_label(cls_dir.name)
            if not (0 <= y < C): continue

            for img_p in cls_dir.iterdir():
                if img_p.suffix.lower() in IMG_EXTS:
                    pairs.append((img_p, y))
    return pairs

# return (Path, int(label))
def collect_abaw_uniform_pairs(
    txt_dir: Union[Path, str, Sequence[Union[Path, str]]],
    image_roots: List[Path],
    K_samples: int
) -> List[Tuple[Path, int]]:

    # list
    if isinstance(txt_dir, (Path, str)):
        txt_dirs = [Path(txt_dir)]
    else:
        txt_dirs = [Path(d) for d in txt_dir]

    pairs: List[Tuple[Path, int]] = []

    for d in txt_dirs:
        if not d.exists():
            continue

        for txt in sorted(d.glob("*.txt")):
            vid = txt.stem

            vid_dir = None
            for root in image_roots:
                cand = root / vid
                if cand.exists():
                    vid_dir = cand
                    break
            if not vid_dir:
                continue

            # fno -> image path
            img_map: Dict[int, Path] = {}
            for p in vid_dir.iterdir():
                if p.suffix.lower() in IMG_EXTS:
                    fno = parse_frame_no(p.stem)
                    if fno >= 0:
                        img_map[fno] = p

            if not img_map:
                continue

            seq, fmap = read_expr_txt(txt)

            valid_fnos: List[int] = []
            valid_labels: List[int] = []

            for fno in sorted(img_map.keys()):
                if fmap and (fno in fmap):
                    lb = fmap[fno]
                else:
                    lb = seq[fno - 1] if (1 <= fno <= len(seq)) else -1

                if 0 <= lb < C:
                    valid_fnos.append(fno)
                    valid_labels.append(lb)

            if not valid_fnos:
                continue

            # uniform K sampling
            if len(valid_fnos) <= K_samples:
                sampled_indices = range(len(valid_fnos))
            else:
                sampled_indices = np.linspace(
                    0, len(valid_fnos) - 1, K_samples, dtype=int
                )

            for idx in sampled_indices:
                fno = valid_fnos[int(idx)]
                pairs.append((img_map[fno], valid_labels[int(idx)]))

    return pairs



class RawImageDataset(Dataset):
    def __init__(self, pairs: List[Tuple[Path, int]], transform):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        p, y = self.pairs[i]
        try:
            img = Image.open(p).convert("RGB")
            x = self.transform(img)
        except Exception:
            x = torch.zeros((3, 224, 224))
        return x, torch.tensor(y, dtype=torch.long)

if __name__ == "__main__":

    print("[Test] collect_imgset_pairs")
    img_pairs = collect_imgset_pairs(PATH.Dataset_IMG, is_train=True)
    print(f"Collected IMG train pairs: {len(img_pairs)}")
    if len(img_pairs) > 0:
        print("Sample:", img_pairs[0])
    print()


    print("[Test] collect_abaw_uniform_pairs")
    abaw_pairs = collect_abaw_uniform_pairs(
        txt_dir=[PATH.EXPR_TRAIN_ABAW10th,PATH.EXPR_VALID_ABAW10th],
        image_roots=[PATH.ABAW_FACE09_ROOT],
        K_samples=5
    )
    print(f"Collected ABAW uniform pairs: {len(abaw_pairs)}")
