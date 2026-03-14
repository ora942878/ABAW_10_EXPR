import os
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm
import insightface

from configs.paths import PATH
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# configs
GPU_ID = 0

SCALES = [0.9, 1.2, 1.5]
SCALE_OUT_ROOT = {
    0.9: Path(PATH.ABAW_FACE09_ROOT_TEST),
    1.2: Path(PATH.ABAW_FACE12_ROOT_TEST),
    1.5: Path(PATH.ABAW_FACE15_ROOT_TEST),
}

INDEX_CSV = Path(PATH.EXPR_VIDEO_INDEX_CSV_TEST)
STRATEGY_CSV = Path(PATH.METADATA_ROOT_TEST) / "face_extract_strategy_TESTSET1.csv"

DET_SCORE_THR = 0.5
MIN_FACE_SIZE = 30
MIN_FACE_SIZE_SIZE_FILTER = 80
COS_THR = 0.25
FEAT_EMA = 0.90
JPG_QUALITY = 95
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI"}

WRITE_BACK_EACH_ROW = True

ALLOW_MARK_SKIP_AS_SUCCESS = True

Box = Tuple[int, int, int, int]
Det = Tuple[int, int, int, int, float]

_app = None


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


def init_app():
    global _app
    os.environ["OMP_NUM_THREADS"] = "1"
    _app = insightface.app.FaceAnalysis(
        name="buffalo_l",
        root=str(Path(PATH.INSIGHTFACE_ROOT))
    )
    _app.prepare(ctx_id=GPU_ID, det_size=(640, 640))


def last_saved_frame(out_dir: Path) -> int:
    if not out_dir.exists():
        return 0
    indices = [int(p.stem) for p in out_dir.glob("*.jpg") if p.stem.isdigit()]
    return max(indices) if indices else 0


def count_saved_frames(out_dir: Path) -> int:
    if not out_dir.exists():
        return 0
    return sum(1 for p in out_dir.glob("*.jpg") if p.is_file())


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

# csv ###########
import csv
from pathlib import Path


def read_strategy_csv(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(f"strategy csv not found: {csv_path}")

    encodings_to_try = [
        "utf-8-sig",
        "utf-8",
        "gb18030",
        "gbk",
        "cp936",
    ]

    last_err = None
    for enc in encodings_to_try:
        try:
            with open(csv_path, "r", encoding=enc, newline="") as f:
                rows = list(csv.reader(f))
            if len(rows) == 0:
                raise RuntimeError(f"empty csv: {csv_path}")
            print(f"[INFO] strategy csv loaded with encoding: {enc}")
            header = rows[0]
            data_rows = rows[1:]
            return header, data_rows, enc
        except Exception as e:
            last_err = e

    raise RuntimeError(
        f"failed to read csv with tried encodings={encodings_to_try}, "
        f"path={csv_path}, last_err={last_err}"
    )


def write_strategy_csv(csv_path: Path, header, data_rows, encoding_used="utf-8-sig"):
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data_rows)

def ensure_min_cols(row: List[str], n: int = 8) -> List[str]:
    if len(row) < n:
        row = row + [""] * (n - len(row))
    return row


def row_get_txtid(row: List[str]) -> str:
    row = ensure_min_cols(row, 8)
    return str(row[0]).strip().replace("\ufeff", "")


def row_get_videoid(row: List[str]) -> str:
    row = ensure_min_cols(row, 8)
    return str(row[1]).strip().replace("\ufeff", "")


def row_get_strategy(row: List[str]) -> str:
    row = ensure_min_cols(row, 8)
    return str(row[2]).strip()


def row_get_ok(row: List[str]) -> str:
    row = ensure_min_cols(row, 8)
    return str(row[4]).strip()


def row_set_ok(row: List[str], value: str) -> List[str]:
    row = ensure_min_cols(row, 8)
    row[4] = value
    return row



def process_single_video(
    txtid: str,
    videoid: str,
    video_path: Path,
    strategy: str,
) -> Tuple[bool, str]:
    global _app

    strategy = str(strategy).strip()

    try:
        out_dirs = {sc: (SCALE_OUT_ROOT[sc] / txtid) for sc in SCALES}
        for d in out_dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        resume_from = min([last_saved_frame(d) for d in out_dirs.values()])

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False, f"ERROR: cannot open video ({videoid})"

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames > 0 and resume_from >= total_frames:
            cap.release()
            if ALLOW_MARK_SKIP_AS_SUCCESS:
                saved09 = count_saved_frames(out_dirs[0.9])
                saved12 = count_saved_frames(out_dirs[1.2])
                saved15 = count_saved_frames(out_dirs[1.5])
                if min(saved09, saved12, saved15) > 0:
                    return True, f"SKIP_ALREADY_DONE(09={saved09},12={saved12},15={saved15})"
            return False, "SKIP_BUT_EMPTY"

        if resume_from > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(resume_from))

        target_feat: Optional[np.ndarray] = None
        prev_box: Optional[Box] = None
        frame_no = resume_from
        saved_count = 0

        with tqdm(
            total=total_frames if total_frames > 0 else None,
            desc=f"{txtid} [{strategy}]",
            unit="f",
            dynamic_ncols=True,
            mininterval=5,
            miniters=20,
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
                    min_size_thr = MIN_FACE_SIZE_SIZE_FILTER if strategy == "size_filter" else MIN_FACE_SIZE

                    if s >= DET_SCORE_THR and max(x2 - x1, y2 - y1) >= min_size_thr:
                        feat = getattr(f, "embedding", None)
                        feat_n = l2_normalize(np.asarray(feat)) if feat is not None else None
                        cands.append(((x1, y1, x2, y2, s), feat_n))

                if not cands:
                    continue

                best_idx = None

                if strategy == "largest":
                    best_idx = pick_largest(cands)

                elif strategy == "left":
                    best_idx = pick_left(cands)

                elif strategy == "right":
                    best_idx = pick_right(cands)

                elif strategy == "top":
                    best_idx = pick_top(cands)

                elif strategy == "size_filter":
                    best_idx = pick_largest(cands)

                elif strategy == "second_right":
                    best_idx = pick_second_right(cands)

                elif strategy == "id_keep":
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
                    cap.release()
                    return False, f"ERROR: unknown strategy: {strategy}"

                if best_idx is None:
                    continue

                det_best, feat_best = cands[best_idx]
                prev_box = det_best[:4]

                if strategy == "id_keep" and feat_best is not None:
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

        if saved_count > 0:
            return True, f"DONE(saved_frames={saved_count})"

        saved09 = count_saved_frames(out_dirs[0.9])
        saved12 = count_saved_frames(out_dirs[1.2])
        saved15 = count_saved_frames(out_dirs[1.5])
        if min(saved09, saved12, saved15) > 0:
            return True, f"DONE_EXISTING_ONLY(09={saved09},12={saved12},15={saved15})"

        return False, "FAILED_NO_FRAME_SAVED"

    except Exception as e:
        return False, f"ERROR: {str(e)}"



def main():
    print(f"[INFO] STRATEGY_CSV = {STRATEGY_CSV}")
    print(f"[INFO] INDEX_CSV    = {INDEX_CSV}")

    init_app()
    video_idx = build_video_index()
    print(f"[INFO] total indexed videos: {len(video_idx)}")

    header, data_rows, csv_encoding = read_strategy_csv(STRATEGY_CSV)

    total_rows = len(data_rows)
    need_rows_idx = []

    for i, row in enumerate(data_rows):
        row = ensure_min_cols(row, 8)
        strategy = row_get_strategy(row)
        ok_flag = row_get_ok(row)

        if strategy != "" and ok_flag != "1":
            need_rows_idx.append(i)

    print(f"[INFO] total rows          : {total_rows}")
    print(f"[INFO] rows need process  : {len(need_rows_idx)}")

    if len(need_rows_idx) == 0:
        print("[DONE] no pending rows.")
        return

    success_cnt = 0
    fail_cnt = 0

    for k, row_idx in enumerate(need_rows_idx, start=1):
        row = ensure_min_cols(data_rows[row_idx], 8)

        txtid = row_get_txtid(row)
        videoid = row_get_videoid(row)
        strategy = row_get_strategy(row)

        print("=" * 80)
        print(f"[{k}/{len(need_rows_idx)}] txtid={txtid} | videoid={videoid} | strategy={strategy}")

        if txtid == "":
            print("[WARN] empty txtid, skip")
            fail_cnt += 1
            continue

        if videoid == "":
            print("[WARN] empty videoid, skip")
            fail_cnt += 1
            continue

        video_path = video_idx.get(videoid)
        if video_path is None:
            print(f"[WARN] video not found for videoid={videoid}")
            fail_cnt += 1
            continue

        ok, status = process_single_video(
            txtid=txtid,
            videoid=videoid,
            video_path=video_path,
            strategy=strategy,
        )

        print(f"[INFO] status: {status}")

        if ok:
            row = row_set_ok(row, "1")
            data_rows[row_idx] = row
            success_cnt += 1

            if WRITE_BACK_EACH_ROW:
                write_strategy_csv(STRATEGY_CSV, header, data_rows, csv_encoding)
                print("[INFO] csv updated: OK=1 written back")
        else:
            fail_cnt += 1

    if not WRITE_BACK_EACH_ROW:
        write_strategy_csv(STRATEGY_CSV, header, data_rows, csv_encoding)
        print("[INFO] csv updated at end")

    print("=" * 80)
    print("[SUMMARY]")
    print(f"success: {success_cnt}")
    print(f"failed : {fail_cnt}")
    print("[DONE]")


if __name__ == "__main__":
    main()