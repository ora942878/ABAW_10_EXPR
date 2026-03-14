from pathlib import Path
import re
import pandas as pd
from configs.paths import PATH

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def base_id(txtid: str) -> str:
    return re.sub(r"_(left|right)$", "", txtid)


def load_test_ids(txt_path: Path):
    if not txt_path.exists():
        raise FileNotFoundError(f"test id txt not found: {txt_path}")

    ids = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            ids.append(s)
    return ids


def collect_video_stems(video_roots):
    video_stems = set()
    for root in video_roots:
        if root is None:
            continue
        if not root.exists():
            print(f"[WARN] video root not exists: {root}")
            continue

        for p in root.glob("*"):
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                video_stems.add(p.stem)

    return video_stems


def main():
    txt_ids = load_test_ids(PATH.test_ids_txt)
    if hasattr(PATH, "VIDEO_TEST_ABAW10th"):
        video_roots = [PATH.VIDEO_TEST_ABAW10th]
    else:
        video_roots = [
            PATH.VIDEO_batch1_ABAW10th,
            PATH.VIDEO_batch2_ABAW10th,
            PATH.VIDEO_batch3_ABAW10th,
        ]

    video_stems = collect_video_stems(video_roots)

    # 3) 对齐
    rows = []
    missing = []

    for txtid in txt_ids:
        bid = base_id(txtid)
        if bid in video_stems:
            rows.append({
                "txtid": txtid,
                "videoid": bid,
            })
        else:
            missing.append({
                "txtid": txtid,
                "base_videoid": bid,
            })

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values(["videoid", "txtid"]).reset_index(drop=True)
    else:
        df = pd.DataFrame(columns=["txtid", "videoid"])

    miss_df = pd.DataFrame(missing)
    if len(miss_df) == 0:
        miss_df = pd.DataFrame(columns=["txtid", "base_videoid"])

    out_csv = PATH.EXPR_VIDEO_INDEX_CSV_TEST

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("[OK] saved matched csv:", out_csv)
    print("[INFO] total txt ids :", len(txt_ids))
    print("[INFO] matched       :", len(df))
    print("[INFO] missing       :", len(miss_df))

    if len(miss_df) > 0:
        print("\n[INFO] first 20 missing:")
        print(miss_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()