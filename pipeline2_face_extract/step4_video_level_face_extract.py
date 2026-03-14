from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Literal

import cv2
import numpy as np
from tqdm import tqdm
import insightface


from configs.paths import PATH

# ==========================================================
VIDEO_IDS = [



'347',

]

STRATEGY: Literal["id_keep", "largest", "left", "right", "top", "size_filter"] = "largest"

# ==========================================================
SCALES = [0.9, 1.2, 1.5]

SCALE_OUT_ROOT = {
    0.9: Path(PATH.ABAW_FACE09_ROOT),
    1.2: Path(PATH.ABAW_FACE12_ROOT),
    1.5: Path(PATH.ABAW_FACE15_ROOT),
}
# ==========================================================
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI"}
DET_SCORE_THR = 0.5
JPG_QUALITY = 95

MIN_FACE_SIZE = 30

# --- for id_keep  ---
IDKEEP_MIN_FACE_SIZE = 60
COS_THR = 0.35
FEAT_EMA = 0.90

Box = Tuple[int, int, int, int]
Det = Tuple[int, int, int, int, float]  # x1,y1,x2,y2,score



# ==========================================================
def build_video_index() -> Dict[str, Path]:
    idx: Dict[str, Path] = {}
    for root in [PATH.VIDEO_batch1_ABAW10th, PATH.VIDEO_batch2_ABAW10th, PATH.VIDEO_batch3_ABAW10th]:
        root = Path(root)
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                idx.setdefault(p.stem, p)
    return idx


def init_detector():
    model_root = Path(PATH.INSIGHTFACE_ROOT)
    app = insightface.app.FaceAnalysis(name="buffalo_l", root=str(model_root))
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def crop_square_with_pad(img, box: Box, scale: float):
    H, W = img.shape[:2]
    x1, y1, x2, y2 = box

    w = max(1, x2 - x1)
    h = max(1, y2 - y1)

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    side = max(w, h) * float(scale)

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

    sx1 += pad_l
    sx2 += pad_l
    sy1 += pad_t
    sy2 += pad_t

    return img[sy1:sy2, sx1:sx2]


def _area(d: Det) -> float:
    return float(max(1, d[2] - d[0]) * max(1, d[3] - d[1]))


def _center_x(d: Det) -> float:
    return (d[0] + d[2]) * 0.5


def _top_y1(d: Det) -> int:
    return int(d[1])


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-12)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))

# ==========================================================

def _collect_dets(app, frame, min_face_size: int) -> List[Det]:
    faces = app.get(frame)
    dets: List[Det] = []
    for f in faces:
        x1, y1, x2, y2 = f.bbox.astype(int)
        s = float(getattr(f, "det_score", 1.0))
        if s < DET_SCORE_THR: continue
        if max(1, x2 - x1) < min_face_size or max(1, y2 - y1) < min_face_size: continue
        dets.append((x1, y1, x2, y2, s))
    return dets


def _collect_dets_and_feats(app, frame, min_face_size: int) -> List[Tuple[Det, Optional[np.ndarray]]]:
    faces = app.get(frame)
    out = []
    for f in faces:
        x1, y1, x2, y2 = f.bbox.astype(int)
        s = float(getattr(f, "det_score", 1.0))
        if s < DET_SCORE_THR: continue
        if max(1, x2 - x1) < min_face_size or max(1, y2 - y1) < min_face_size: continue
        feat = getattr(f, "embedding", None)
        feat_n = _l2_normalize(np.asarray(feat)) if feat is not None else None
        out.append(((x1, y1, x2, y2, s), feat_n))
    return out


def pick_face_generic(app, frame) -> Optional[Det]:
    if STRATEGY == "size_filter":
        dets = _collect_dets(app, frame, min_face_size=50)
        return max(dets, key=_area) if dets else None

    dets = _collect_dets(app, frame, min_face_size=MIN_FACE_SIZE)
    if not dets: return None
    if len(dets) == 1: return dets[0]

    if STRATEGY == "largest": return max(dets, key=_area)
    if STRATEGY == "left": return min(dets, key=_center_x)
    if STRATEGY == "right": return max(dets, key=_center_x)
    if STRATEGY == "top": return min(dets, key=_top_y1)

    return max(dets, key=_area)


def pick_face_id_keep(app, frame, target_feat: Optional[np.ndarray]) -> Tuple[Optional[Det], Optional[np.ndarray]]:
    cands = _collect_dets_and_feats(app, frame, min_face_size=IDKEEP_MIN_FACE_SIZE)
    if not cands: return None, target_feat

    if target_feat is None:
        det0, feat0 = max(cands, key=lambda x: _area(x[0]))
        return det0, (feat0.copy() if feat0 is not None else None)

    best_sim, best_idx = -1e9, None
    for i, (_, feat) in enumerate(cands):
        if feat is not None:
            sim = _cosine(target_feat, feat)
            if sim > best_sim:
                best_sim, best_idx = sim, i

    if best_idx is None or best_sim < COS_THR:
        det1, feat1 = max(cands, key=lambda x: _area(x[0]))
        if feat1 is not None:
            target_feat = _l2_normalize(FEAT_EMA * target_feat + (1.0 - FEAT_EMA) * feat1)
        return det1, target_feat

    det_best, feat_best = cands[best_idx]
    if feat_best is not None:
        target_feat = _l2_normalize(FEAT_EMA * target_feat + (1.0 - FEAT_EMA) * feat_best)
    return det_best, target_feat



# ==========================================================

def process_video(video_id: str, video_path: Path, app):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] cannot open: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    out_dirs = {sc: (SCALE_OUT_ROOT[sc] / video_id) for sc in SCALES}
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    target_feat: Optional[np.ndarray] = None
    pbar = tqdm(total=total_frames, desc=f"Target: {video_id} | Mode: {STRATEGY}", leave=False)
    frame_no = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_no += 1

        det: Optional[Det] = None
        if STRATEGY == "id_keep":
            det, target_feat = pick_face_id_keep(app, frame, target_feat)
        else:
            det = pick_face_generic(app, frame)

        if det is not None:
            x1, y1, x2, y2, _ = det
            for sc, out_dir in out_dirs.items():
                crop = crop_square_with_pad(frame, (x1, y1, x2, y2), scale=sc)
                out_path = out_dir / f"{frame_no:05d}.jpg"
                cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY])

        pbar.update(1)

    pbar.close()
    cap.release()



# ==========================================================

def main():
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    video_idx = build_video_index()
    app = init_detector()

    for vid in VIDEO_IDS:
        vp = video_idx.get(vid)
        if vp is None:
            print(f"[WARN] not found in dataset: {vid}")
            continue
        process_video(vid, vp, app)
        print(f"[INFO] Successfully patched video: {vid} (Scales: 0.9, 1.2, 1.5)")


if __name__ == "__main__":
    main()