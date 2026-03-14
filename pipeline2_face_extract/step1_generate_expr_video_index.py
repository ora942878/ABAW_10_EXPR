from pathlib import Path
import re
import pandas as pd
from configs.paths import PATH

VIDEO_EXTS = {".mp4", ".avi"}

def base_id(txtid: str) -> str:
    return re.sub(r"_(left|right)$", "", txtid)

def main():
    # 1. collect txt ids
    expr_roots = [PATH.EXPR_TRAIN_ABAW10th, PATH.EXPR_VALID_ABAW10th]
    txt_ids = []
    for r in expr_roots:
        if r.exists():
            txt_ids.extend([p.stem for p in r.glob("*.txt")])

    # 2. collect video stems
    video_roots = [PATH.VIDEO_batch1_ABAW10th, PATH.VIDEO_batch2_ABAW10th, PATH.VIDEO_batch3_ABAW10th]
    video_stems = set()
    for root in video_roots:
        if root.exists():
            for p in root.glob("*"):
                if p.suffix.lower() in VIDEO_EXTS:
                    video_stems.add(p.stem)

    # 3. match
    rows = []
    for txtid in txt_ids:
        bid = base_id(txtid)
        if bid in video_stems:
            rows.append({"txtid": txtid, "videoid": bid})

    df = pd.DataFrame(rows).sort_values("videoid").reset_index(drop=True)

    out_path = PATH.EXPR_VIDEO_INDEX_CSV
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print("[OK] saved:", out_path)
    print("[INFO] rows:", len(df))

if __name__ == "__main__":
    main()