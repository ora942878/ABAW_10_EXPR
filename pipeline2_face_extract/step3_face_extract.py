import os
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
from tqdm import tqdm
import insightface


from configs.paths import PATH
# ==========================================================
NUM_WORKERS = 4
GPU_ID = 0

SCALES = [0.9, 1.2, 1.5]  # 来自 test3
SCALE_OUT_ROOT = {
    0.9: Path(PATH.ABAW_FACE09_ROOT),
    1.2: Path(PATH.ABAW_FACE12_ROOT),
    1.5: Path(PATH.ABAW_FACE15_ROOT),
}

CSV_PATH = Path(PATH.EXPR_VIDEO_INDEX_CSV)

DET_SCORE_THR = 0.5
MIN_FACE_SIZE = 30
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
    sx1, sy1 = int(round(cx - side / 2)), int(round(cy - side / 2))
    sx2, sy2 = int(round(cx + side / 2)), int(round(cy + side / 2))
    pad_l, pad_t = max(0, -sx1), max(0, -sy1)
    pad_r, pad_b = max(0, sx2 - W), max(0, sy2 - H)
    if pad_l or pad_t or pad_r or pad_b:
        img = cv2.copyMakeBorder(img, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return img[sy1 + pad_t: sy2 + pad_t, sx1 + pad_l: sx2 + pad_l]


# ==========================================================
_worker_app = None

def init_worker():
    global _worker_app
    os.environ["OMP_NUM_THREADS"] = "1"
    _worker_app = insightface.app.FaceAnalysis(
        name="buffalo_l",
        root=str(Path(PATH.INSIGHTFACE_ROOT))
    )
    _worker_app.prepare(ctx_id=GPU_ID, det_size=(640, 640))

# ==========================================================
def last_saved_frame(out_dir: Path) -> int:
    if not out_dir.exists(): return 0
    indices = [int(p.stem) for p in out_dir.glob("*.jpg") if p.stem.isdigit()]
    return max(indices) if indices else 0


def process_single_video(video_id: str, video_path: Path) -> Tuple[str, str]:
    global _worker_app
    try:
        out_dirs = {sc: (SCALE_OUT_ROOT[sc] / video_id) for sc in SCALES}
        for d in out_dirs.values(): d.mkdir(parents=True, exist_ok=True)

        resume_from = min([last_saved_frame(d) for d in out_dirs.values()])
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames > 0 and resume_from >= total_frames:
            cap.release()
            return video_id, "SKIP"

        if resume_from > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(resume_from))

        target_feat: Optional[np.ndarray] = None
        prev_box: Optional[Box] = None
        frame_no = resume_from

        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_no += 1

            faces = _worker_app.get(frame)
            cands = []
            for f in faces:
                x1, y1, x2, y2 = f.bbox.astype(int)
                s = float(getattr(f, "det_score", 1.0))
                if s >= DET_SCORE_THR and max(x2 - x1, y2 - y1) >= MIN_FACE_SIZE:
                    feat = getattr(f, "embedding", None)
                    feat_n = l2_normalize(np.asarray(feat)) if feat is not None else None
                    cands.append(((x1, y1, x2, y2, s), feat_n))

            if cands:
                best_idx = None
                if target_feat is None:
                    areas = [(d[2] - d[0]) * (d[3] - d[1]) for d, _ in cands]
                    best_idx = int(np.argmax(areas))
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

                if best_idx is not None:
                    det_best, feat_best = cands[best_idx]
                    prev_box = det_best[:4]
                    if feat_best is not None:
                        if target_feat is None:
                            target_feat = feat_best
                        else:
                            target_feat = l2_normalize(FEAT_EMA * target_feat + (1.0 - FEAT_EMA) * feat_best)

                    for sc, out_dir in out_dirs.items():
                        out_path = out_dir / f"{frame_no:05d}.jpg"
                        if not out_path.exists():
                            crop = crop_square_with_pad(frame, prev_box, scale=sc)
                            cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY])

        cap.release()
        return video_id, "DONE"
    except Exception as e:
        return video_id, f"ERROR: {str(e)}"


# ==========================================================

def main():
    video_order = []
    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = (row.get('videoid') or row.get('video_id', '')).strip()
            if vid: video_order.append(vid)

    unique_vids = list(dict.fromkeys(video_order))
    video_idx = {}
    roots = [PATH.VIDEO_batch1_ABAW10th, PATH.VIDEO_batch2_ABAW10th, PATH.VIDEO_batch3_ABAW10th]
    for r_path in roots:
        root = Path(r_path)
        if root.exists():
            for p in root.rglob("*"):
                if p.suffix.lower() in VIDEO_EXTS: video_idx[p.stem] = p

    final_tasks = [(vid, video_idx[vid]) for vid in unique_vids if vid in video_idx]

    print(f"[INFO] Processing {len(final_tasks)} videos using {NUM_WORKERS} workers...")

    with ProcessPoolExecutor(max_workers=NUM_WORKERS, initializer=init_worker) as executor:
        futures = {executor.submit(process_single_video, vid, vpath): vid for vid, vpath in final_tasks}

        with tqdm(total=len(futures), desc="Total Progress", unit="vid", dynamic_ncols=True) as pbar:
            for future in as_completed(futures):
                vid, status = future.result()
                if "ERROR" in status:
                    tqdm.write(f"[WARN] Failed: {vid} | {status}")
                pbar.update(1)


if __name__ == "__main__":
    main()