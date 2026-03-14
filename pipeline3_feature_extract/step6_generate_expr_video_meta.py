from pathlib import Path
import cv2
import pandas as pd
import torch
from tqdm import tqdm
from configs.paths import PATH

VIDEO_EXTS = {".mp4", ".avi"}
AUDIO_EXTS = {".pt"}


def find_video(videoid: str) -> Path | None:
    roots = [
        PATH.VIDEO_batch1_ABAW10th,
        PATH.VIDEO_batch2_ABAW10th,
        PATH.VIDEO_batch3_ABAW10th,
    ]
    for r in roots:
        if not r.exists():
            continue
        for ext in VIDEO_EXTS:
            p = r / f"{videoid}{ext}"
            if p.exists():
                return p
    return None


def get_video_meta(vpath: Path):
    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        return None, None
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if fps <= 1e-6:
        fps = None
    if n_frames < 0:
        n_frames = None

    return fps, n_frames


def find_audio_pt(videoid: str) -> Path | None:
    root = PATH.OFFAUDIO_WAV2VEC2_PT
    if not root.exists():
        return None

    p = root / f"{videoid}.pt"
    if p.exists():
        return p

    for ext in AUDIO_EXTS:
        q = root / f"{videoid}{ext}"
        if q.exists():
            return q

    return None


def get_audio_duration_sec(audio_pt: Path):
    obj = torch.load(audio_pt, map_location="cpu")

    if isinstance(obj, dict):
        if "duration_sec" in obj and obj["duration_sec"] is not None:
            try:
                return float(obj["duration_sec"])
            except Exception:
                pass

        if "t_sec" in obj and obj["t_sec"] is not None:
            t = obj["t_sec"]
            try:
                if hasattr(t, "numel") and t.numel() > 0:
                    return float(t[-1])
                if isinstance(t, (list, tuple)) and len(t) > 0:
                    return float(t[-1])
            except Exception:
                pass

    return None


def main():
    in_csv = PATH.EXPR_VIDEO_INDEX_CSV
    out_csv = PATH.METADATA_ROOT / "expr_video_meta.csv"

    print("[INFO] reading:", in_csv)
    df = pd.read_csv(in_csv)
    videoids = sorted(df["videoid"].unique())

    rows = []
    for vid in tqdm(videoids, desc="Scanning videos"):
        vpath = find_video(vid)
        if vpath is None:
            print("[WARN] video not found:", vid)
            continue

        fps, n_frames = get_video_meta(vpath)
        if fps is None or n_frames is None:
            print("[WARN] failed to read video meta:", vid)
            continue

        video_duration_sec = float(n_frames) / float(fps)

        audio_duration_sec = None
        apt = find_audio_pt(vid)
        if apt is None:
            pass
        else:
            audio_duration_sec = get_audio_duration_sec(apt)

        rows.append(
            dict(
                videoid=vid,
                fps=float(fps),
                n_frames=int(n_frames),
                video_duration_sec=float(video_duration_sec),
                audio_duration_sec=(None if audio_duration_sec is None else float(audio_duration_sec)),
            )
        )

    meta_df = pd.DataFrame(rows).sort_values("videoid").reset_index(drop=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    meta_df.to_csv(out_csv, index=False)

    print()
    print("[OK] saved:", out_csv)
    print("[INFO] videos:", len(meta_df))
    if "audio_duration_sec" in meta_df.columns:
        miss = int(meta_df["audio_duration_sec"].isna().sum())
        print("[INFO] missing audio_duration_sec:", miss)


if __name__ == "__main__":
    main()
    import pandas as pd
    from configs.paths import PATH

    csv_path = PATH.METADATA_ROOT / "expr_video_meta.csv"
    df = pd.read_csv(csv_path)

    miss = df[df["audio_duration_sec"].isna()]
    print("[MISS] n =", len(miss))
    print(miss[["videoid", "fps", "n_frames", "video_duration_sec"]].to_string(index=False))