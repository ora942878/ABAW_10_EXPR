# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from configs.paths import PATH


NUM_FOLDS = 5
NUM_TRIALS = 5000
SEED = 3407
NUM_CLASSES = 8
LABEL_MIN = 0
LABEL_MAX = 7

INPUT_TXT_DIRS = [
    PATH.EXPR_TRAIN_ABAW10th,
    PATH.EXPR_VALID_ABAW10th,
]
MAP_TXTID2VIDEOID_CSV = PATH.EXPR_VIDEO_INDEX_CSV
OUTPUT_DIR = PATH.METADATA_ROOT / "expr_cv5_txtid_randomsearch"


def ensure_dir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_hist(hist: np.ndarray) -> np.ndarray:
    s = float(hist.sum())
    if s <= 0:
        return np.zeros_like(hist, dtype=np.float64)
    return hist.astype(np.float64) / s


def read_expr_txt_labels(txt_path: str | Path) -> List[int]:
    txt_path = Path(txt_path)
    labels = []
    with txt_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            low = line.lower()
            if low in {"label", "labels", "frame,label", "frame label"}:
                continue
            parts = [x.strip() for x in line.replace(",", " ").split() if x.strip()]
            if not parts:
                continue
            try:
                y = int(parts[-1])
                labels.append(y)
            except Exception:
                continue
    return labels


def read_txtid_to_videoid_map(csv_path: str | Path) -> Dict[str, str]:
    df = pd.read_csv(csv_path)
    cols = {str(c).strip().lower(): c for c in df.columns}

    txt_col, vid_col = None, None
    for k in ["txtid", "txt_id", "txt", "id"]:
        if k in cols:
            txt_col = cols[k]
            break
    for k in ["videoid", "video_id", "video", "vid"]:
        if k in cols:
            vid_col = cols[k]
            break

    if txt_col is None or vid_col is None:
        raise RuntimeError(f"Cannot find txtid/videoid columns in {csv_path}, columns={list(df.columns)}")

    out = {}
    for _, row in df.iterrows():
        txtid = str(row[txt_col]).strip()
        videoid = str(row[vid_col]).strip()
        if txtid:
            out[txtid] = videoid if videoid else txtid
    return out


@dataclass
class TxtStat:
    txtid: str
    videoid: str
    source_dir: str
    total_frames_in_txt: int
    valid_frames: int
    invalid_frames: int
    class_hist: List[int]


def build_txt_stat(txt_path: Path, txt2vid: Dict[str, str]) -> TxtStat:
    txtid = txt_path.stem
    videoid = txt2vid.get(txtid, txtid)
    labels = read_expr_txt_labels(txt_path)

    hist = np.zeros(NUM_CLASSES, dtype=np.int64)
    valid_n = 0
    for y in labels:
        if LABEL_MIN <= y <= LABEL_MAX:
            hist[int(y)] += 1
            valid_n += 1

    return TxtStat(
        txtid=txtid,
        videoid=videoid,
        source_dir=txt_path.parent.name,
        total_frames_in_txt=len(labels),
        valid_frames=valid_n,
        invalid_frames=len(labels) - valid_n,
        class_hist=hist.tolist(),
    )


def collect_txt_stats(input_dirs, txt2vid):
    rows = []
    for d in input_dirs:
        d = Path(d)
        for p in sorted(d.glob("*.txt")):
            rows.append(build_txt_stat(p, txt2vid))
    if not rows:
        raise RuntimeError("No txt files found.")
    return rows


def evaluate_split(txt_stats: List[TxtStat], txt2fold: Dict[str, int]):
    global_hist = np.zeros(NUM_CLASSES, dtype=np.int64)
    for r in txt_stats:
        global_hist += np.asarray(r.class_hist, dtype=np.int64)
    global_ratio = normalize_hist(global_hist)

    fold_ratios = []
    fold_sizes = []
    divergences = []

    for fold in range(1, NUM_FOLDS + 1):
        hist = np.zeros(NUM_CLASSES, dtype=np.int64)
        n = 0
        for r in txt_stats:
            if txt2fold[r.txtid] == fold:
                hist += np.asarray(r.class_hist, dtype=np.int64)
                n += 1
        ratio = normalize_hist(hist)
        div = float(np.sqrt(np.sum((ratio - global_ratio) ** 2)))
        fold_ratios.append(ratio)
        fold_sizes.append(n)
        divergences.append(div)

    size_std = float(np.std(fold_sizes))
    mean_div = float(np.mean(divergences))
    max_div = float(np.max(divergences))

    score = 5.0 * size_std + 3.0 * max_div + 1.0 * mean_div
    return {
        "score": score,
        "global_ratio": global_ratio,
        "fold_ratios": fold_ratios,
        "fold_sizes": fold_sizes,
        "divergences": divergences,
        "size_std": size_std,
        "mean_div": mean_div,
        "max_div": max_div,
    }


def random_search_best_split(txt_stats: List[TxtStat], num_trials: int, seed: int):
    rng = np.random.default_rng(seed)
    txtids = [r.txtid for r in txt_stats]
    n = len(txtids)

    best_assign = None
    best_eval = None

    base_sizes = [n // NUM_FOLDS] * NUM_FOLDS
    for i in range(n % NUM_FOLDS):
        base_sizes[i] += 1

    for t in range(num_trials):
        perm = rng.permutation(n)
        shuffled = [txtids[i] for i in perm]

        txt2fold = {}
        st = 0
        for fold, sz in enumerate(base_sizes, start=1):
            ed = st + sz
            for txtid in shuffled[st:ed]:
                txt2fold[txtid] = fold
            st = ed

        ev = evaluate_split(txt_stats, txt2fold)

        if best_eval is None or ev["score"] < best_eval["score"]:
            best_eval = ev
            best_assign = txt2fold.copy()
            print(
                f"[trial {t+1:04d}] "
                f"score={ev['score']:.6f} "
                f"size_std={ev['size_std']:.4f} "
                f"mean_div={ev['mean_div']:.6f} "
                f"max_div={ev['max_div']:.6f}"
            )

    return best_assign, best_eval


def build_tables(txt_stats: List[TxtStat], txt2fold: Dict[str, int]):
    rows = []
    for r in txt_stats:
        row = {
            "txtid": r.txtid,
            "videoid": r.videoid,
            "fold": int(txt2fold[r.txtid]),
            "source_dir": r.source_dir,
            "total_frames_in_txt": int(r.total_frames_in_txt),
            "valid_frames": int(r.valid_frames),
            "invalid_frames": int(r.invalid_frames),
        }
        for c in range(NUM_CLASSES):
            row[f"class_{c}"] = int(r.class_hist[c])
        rows.append(row)

    df_txt = pd.DataFrame(rows).sort_values(["fold", "videoid", "txtid"]).reset_index(drop=True)

    stat_rows = []
    hist_rows = []

    global_hist = np.zeros(NUM_CLASSES, dtype=np.int64)
    for r in txt_stats:
        global_hist += np.asarray(r.class_hist, dtype=np.int64)
    global_ratio = normalize_hist(global_hist)

    for fold in range(1, NUM_FOLDS + 1):
        part = [r for r in txt_stats if txt2fold[r.txtid] == fold]
        hist = np.zeros(NUM_CLASSES, dtype=np.int64)
        valid_frames = 0
        invalid_frames = 0
        videos = set()

        for r in part:
            hist += np.asarray(r.class_hist, dtype=np.int64)
            valid_frames += int(r.valid_frames)
            invalid_frames += int(r.invalid_frames)
            videos.add(r.videoid)

        ratio = normalize_hist(hist)
        div = float(np.sqrt(np.sum((ratio - global_ratio) ** 2)))

        stat_rows.append({
            "fold": fold,
            "num_videos": int(len(videos)),
            "num_txtids": int(len(part)),
            "valid_frames": int(valid_frames),
            "invalid_frames": int(invalid_frames),
            "l2_to_global": div,
        })

        row_hist = {"fold": fold}
        for c in range(NUM_CLASSES):
            row_hist[f"class_{c}_count"] = int(hist[c])
            row_hist[f"class_{c}_ratio"] = float(ratio[c])
        hist_rows.append(row_hist)

    df_stats = pd.DataFrame(stat_rows).sort_values("fold").reset_index(drop=True)
    df_hist = pd.DataFrame(hist_rows).sort_values("fold").reset_index(drop=True)
    return df_txt, df_stats, df_hist


def main():
    out_dir = ensure_dir(OUTPUT_DIR)

    txt2vid = read_txtid_to_videoid_map(MAP_TXTID2VIDEOID_CSV)
    txt_stats = collect_txt_stats(INPUT_TXT_DIRS, txt2vid)

    best_assign, best_eval = random_search_best_split(
        txt_stats=txt_stats,
        num_trials=NUM_TRIALS,
        seed=SEED,
    )

    df_txt, df_stats, df_hist = build_tables(txt_stats, best_assign)

    global_hist = np.zeros(NUM_CLASSES, dtype=np.int64)
    total_valid = 0
    total_invalid = 0
    for r in txt_stats:
        global_hist += np.asarray(r.class_hist, dtype=np.int64)
        total_valid += int(r.valid_frames)
        total_invalid += int(r.invalid_frames)

    summary = {
        "num_folds": NUM_FOLDS,
        "num_trials": NUM_TRIALS,
        "num_txtids": len(txt_stats),
        "num_videos": len(set(r.videoid for r in txt_stats)),
        "total_valid_frames": int(total_valid),
        "total_invalid_frames": int(total_invalid),
        "global_class_hist": [int(x) for x in global_hist.tolist()],
        "global_class_ratio": [float(x) for x in normalize_hist(global_hist).tolist()],
        "best_score": float(best_eval["score"]),
        "best_size_std": float(best_eval["size_std"]),
        "best_mean_div": float(best_eval["mean_div"]),
        "best_max_div": float(best_eval["max_div"]),
        "fold_distribution_divergence": {
            f"fold_{i+1}_l2_to_global": float(best_eval["divergences"][i])
            for i in range(NUM_FOLDS)
        },
    }

    df_txt.to_csv(out_dir / "expr_cv5_folds.csv", index=False, encoding="utf-8-sig")
    df_stats.to_csv(out_dir / "expr_cv5_fold_stats.csv", index=False, encoding="utf-8-sig")
    df_hist.to_csv(out_dir / "expr_cv5_fold_class_hist.csv", index=False, encoding="utf-8-sig")
    save_json(summary, out_dir / "expr_cv5_summary.json")

    print("=" * 80)
    print(f"[DONE] output_dir = {out_dir}")
    print(f"[TXT]   num_txtids  = {summary['num_txtids']}")
    print(f"[VIDEO] num_videos  = {summary['num_videos']}")
    print(f"[FRAME] valid       = {summary['total_valid_frames']}")
    print(f"[FRAME] invalid     = {summary['total_invalid_frames']}")
    print("[BEST]")
    print(f"  score    : {summary['best_score']:.6f}")
    print(f"  size_std : {summary['best_size_std']:.6f}")
    print(f"  mean_div : {summary['best_mean_div']:.6f}")
    print(f"  max_div  : {summary['best_max_div']:.6f}")
    print("[FOLD_STATS]")
    print(df_stats.to_string(index=False))
    print("[FOLD_DIVERGENCE]")
    for k, v in summary["fold_distribution_divergence"].items():
        print(f"  {k}: {v:.6f}")
    print("=" * 80)

if __name__ == "__main__":
    main()