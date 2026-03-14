import os
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Literal

import cv2
import numpy as np
from tqdm import tqdm
import insightface

from configs.paths import PATH

# ==========================================================
#  txt/imgfolderid，not videoid
TXT_IDS: List[str] = [
    "314",
    # "16-30-1920x1080",
]


STRATEGY: Literal["id_keep", "largest", "left", "right", "top", "size_filter", "second_right"] \
    = "largest"

GPU_ID = 0

SCALES = [0.9, 1.2, 1.5]
SCALE_OUT_ROOT = {
    0.9: Path(PATH.ABAW_FACE09_ROOT_TEST),
    1.2: Path(PATH.ABAW_FACE12_ROOT_TEST),
    1.5: Path(PATH.ABAW_FACE15_ROOT_TEST),
}

INDEX_CSV = Path(PATH.EXPR_VIDEO_INDEX_CSV_TEST)

DET_SCORE_THR = 0.5
MIN_FACE_SIZE = 30
MIN_FACE_SIZE_SIZE_FILTER = 80
COS_THR = 0.25
FEAT_EMA = 0.90
JPG_QUALITY = 95
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI"}

Box = Tuple[int, int, int, int]
Det = Tuple[int, int, int, int, float]

# ==========================================================
def iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return float(inter) / float(area_a + area_b - inter + 1e-6)


def l2_normalize(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-12)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def crop_square_with_pad(img: np.ndarray, box: Box, scale: float) -> np.ndarray:
    H, W = img.shape[:2]
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) * float(scale)

    sx1 = int(round(cx - side / 2))
    sy1 = int(round(cy - side / 2))
    sx2 = int(round(cx + side / 2))
    sy2 = int(round(cy + side / 2))

    pad_l = max(0, -sx1)
    pad_t = max(0, -sy1)
    pad_r = max(0, sx2 - W)
    pad_b = max(0, sy2 - H)

    if pad_l or pad_t or pad_r or pad_b:
        img = cv2.copyMakeBorder(
            img, pad_t, pad_b, pad_l, pad_r,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

    return img[sy1 + pad_t: sy2 + pad_t, sx1 + pad_l: sx2 + pad_l]


# ==========================================================
_app = None

def init_app():
    global _app
    os.environ["OMP_NUM_THREADS"] = "1"
    _app = insightface.app.FaceAnalysis(
        name="buffalo_l",
        root=str(Path(PATH.INSIGHTFACE_ROOT))
    )
    _app.prepare(ctx_id=GPU_ID, det_size=(640, 640))


# ==========================================================
def last_saved_frame(out_dir: Path) -> int:
    if not out_dir.exists():
        return 0
    indices = [int(p.stem) for p in out_dir.glob("*.jpg") if p.stem.isdigit()]
    return max(indices) if indices else 0


def build_video_index() -> Dict[str, Path]:
    video_idx = {}
    roots = [
        Path(PATH.VIDEO_batch1_ABAW10th),
        Path(PATH.VIDEO_batch2_ABAW10th),
        Path(PATH.VIDEO_batch3_ABAW10th),
    ]
    valid_exts = {x.lower() for x in VIDEO_EXTS}

    for root in roots:
        if root.exists():
            for p in root.rglob("*"):
                if p.is_file() and p.suffix.lower() in valid_exts:
                    video_idx[p.stem] = p
    return video_idx


def load_txtid_to_videoid(index_csv: Path) -> Dict[str, str]:
    if not index_csv.exists():
        raise FileNotFoundError(f"index csv not found: {index_csv}")

    mapping = {}
    with open(index_csv, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            txtid = (row.get("txtid") or "").strip().replace("\ufeff", "")
            videoid = (row.get("videoid") or row.get("video_id") or "").strip().replace("\ufeff", "")
            if txtid and videoid:
                mapping[txtid] = videoid
    return mapping


def pick_largest(cands) -> Optional[int]:
    if not cands:
        return None
    areas = [(d[2] - d[0]) * (d[3] - d[1]) for d, _ in cands]
    return int(np.argmax(areas))


def pick_left(cands) -> Optional[int]:
    if not cands:
        return None
    xs = [d[0] for d, _ in cands]
    return int(np.argmin(xs))


def pick_right(cands) -> Optional[int]:
    if not cands:
        return None
    xs = [d[2] for d, _ in cands]
    return int(np.argmax(xs))


def pick_top(cands) -> Optional[int]:
    if not cands:
        return None
    ys = [d[1] for d, _ in cands]
    return int(np.argmin(ys))


def pick_second_right(cands) -> Optional[int]:
    if len(cands) < 2:
        return None
    order = []
    for i, (det, _) in enumerate(cands):
        x1, y1, x2, y2, _ = det
        cx = (x1 + x2) / 2.0
        order.append((cx, i))
    order.sort(key=lambda x: x[0], reverse=True)
    return order[1][1]


# ==========================================================
def process_single_video(txtid: str, videoid: str, video_path: Path) -> Tuple[str, str]:
    global _app
    try:
        out_dirs = {sc: (SCALE_OUT_ROOT[sc] / txtid) for sc in SCALES}
        for d in out_dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        resume_from = min([last_saved_frame(d) for d in out_dirs.values()])

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return txtid, f"ERROR: cannot open video ({videoid})"

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0 and resume_from >= total_frames:
            cap.release()
            return txtid, "SKIP"

        if resume_from > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(resume_from))

        target_feat: Optional[np.ndarray] = None
        prev_box: Optional[Box] = None
        frame_no = resume_from
        saved_count = 0

        with tqdm(
            total=total_frames if total_frames > 0 else None,
            desc=f"{txtid} [{STRATEGY}]",
            unit="f",
            dynamic_ncols=True
        ) as pbar:

            if resume_from > 0:
                pbar.update(resume_from)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_no += 1
                pbar.update(1)

                faces = _app.get(frame)
                cands = []

                for f in faces:
                    x1, y1, x2, y2 = f.bbox.astype(int)
                    s = float(getattr(f, "det_score", 1.0))
                    min_size_thr = MIN_FACE_SIZE_SIZE_FILTER if STRATEGY == "size_filter" else MIN_FACE_SIZE

                    if s >= DET_SCORE_THR and max(x2 - x1, y2 - y1) >= min_size_thr:
                        feat = getattr(f, "embedding", None)
                        feat_n = l2_normalize(np.asarray(feat)) if feat is not None else None
                        cands.append(((x1, y1, x2, y2, s), feat_n))

                if not cands:
                    continue

                best_idx = None

                if STRATEGY == "largest":
                    best_idx = pick_largest(cands)

                elif STRATEGY == "left":
                    best_idx = pick_left(cands)

                elif STRATEGY == "right":
                    best_idx = pick_right(cands)

                elif STRATEGY == "top":
                    best_idx = pick_top(cands)

                elif STRATEGY == "size_filter":
                    best_idx = pick_largest(cands)

                elif STRATEGY == "second_right":
                    best_idx = pick_second_right(cands)

                elif STRATEGY == "id_keep":
                    if target_feat is None:
                        best_idx = pick_largest(cands)
                    else:
                        best_sim, best_idx = -1.0, None
                        for i, (_, feat) in enumerate(cands):
                            if feat is not None:
                                sim = cosine(target_feat, feat)
                                if sim > best_sim:
                                    best_sim, best_idx = sim, i

                        if best_idx is None or best_sim < COS_THR:
                            if prev_box is not None:
                                best_iou, best_idx = -1.0, None
                                for i, (det, _) in enumerate(cands):
                                    v = iou(prev_box, det[:4])
                                    if v > best_iou:
                                        best_iou, best_idx = v, i

                        if best_idx is None:
                            best_idx = pick_largest(cands)

                else:
                    raise ValueError(f"Unknown STRATEGY: {STRATEGY}")

                if best_idx is None:
                    continue

                det_best, feat_best = cands[best_idx]
                prev_box = det_best[:4]

                if STRATEGY == "id_keep" and feat_best is not None:
                    if target_feat is None:
                        target_feat = feat_best
                    else:
                        target_feat = l2_normalize(FEAT_EMA * target_feat + (1.0 - FEAT_EMA) * feat_best)

                for sc, out_dir in out_dirs.items():
                    out_path = out_dir / f"{frame_no:05d}.jpg"
                    if not out_path.exists():
                        crop = crop_square_with_pad(frame, prev_box, scale=sc)
                        cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY])

                saved_count += 1

        cap.release()
        return txtid, f"DONE(saved_frames={saved_count})"

    except Exception as e:
        return txtid, f"ERROR: {str(e)}"


# ==========================================================
def main():
    if len(TXT_IDS) == 0:
        print("[WARN] TXT_IDS is empty.")
        return

    init_app()

    txt2vid = load_txtid_to_videoid(INDEX_CSV)
    video_idx = build_video_index()

    print(f"[INFO] INDEX_CSV: {INDEX_CSV}")
    print(f"[INFO] STRATEGY : {STRATEGY}")
    print(f"[INFO] total TXT_IDS: {len(TXT_IDS)}")

    for txtid in TXT_IDS:
        txtid = str(txtid).strip().replace("\ufeff", "")
        if not txtid:
            print("[WARN] empty txtid, skip")
            continue

        videoid = txt2vid.get(txtid)
        if videoid is None:
            print(f"[WARN] txtid not found in mapping csv: {txtid}")
            continue

        video_path = video_idx.get(videoid)
        if video_path is None:
            print(f"[WARN] video not found for txtid={txtid}, videoid={videoid}")
            continue

        print(f"[INFO] Processing txtid={txtid} -> videoid={videoid} | strategy={STRATEGY}")
        tid, status = process_single_video(txtid, videoid, video_path)

        if "ERROR" in status:
            print(f"[WARN] Failed: {tid} | {status}")
        else:
            print(f"[INFO] {tid} | {status}")


if __name__ == "__main__":
    main()