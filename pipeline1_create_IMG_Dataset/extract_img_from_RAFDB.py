import pandas as pd
import shutil
from pathlib import Path
from tqdm import tqdm

from configs.paths import PATH

TARGET_ROOT = PATH.Dataset_IMG

CLASS_NAMES = {
    0: "0_Neutral",
    1: "4_Happiness",
    2: "5_Sadness",
    3: "6_Surprise",
    4: "3_Fear",
    5: "2_Disgust",
    6: "1_Anger",
    7: "7_Other",
}

RAF_DIGIT_TO_ABAW_CLASSFOLDER = {
    0: CLASS_NAMES[3],  # Surprise -> 6_Surprise
    1: CLASS_NAMES[4],  # Fear     -> 3_Fear
    2: CLASS_NAMES[5],  # Disgust  -> 2_Disgust
    3: CLASS_NAMES[1],  # Happiness-> 4_Happiness
    4: CLASS_NAMES[2],  # Sadness  -> 5_Sadness
    5: CLASS_NAMES[6],  # Anger    -> 1_Anger
    6: CLASS_NAMES[0],  # Neutral  -> 0_Neutral
}

def map_raf_class_folder(folder_name: str) -> str:
    name = folder_name.strip()
    low = name.lower()
    if low.isdigit():
        idx = int(low)
        if idx in RAF_DIGIT_TO_ABAW_CLASSFOLDER:
            return RAF_DIGIT_TO_ABAW_CLASSFOLDER[idx]
        return CLASS_NAMES[7]
    if "neutral" in low:
        return CLASS_NAMES[0]
    if "happy" in low or "happiness" in low:
        return CLASS_NAMES[1]
    if "sad" in low or "sadness" in low:
        return CLASS_NAMES[2]
    if "surprise" in low:
        return CLASS_NAMES[3]
    if "fear" in low:
        return CLASS_NAMES[4]
    if "disgust" in low:
        return CLASS_NAMES[5]
    if "anger" in low or "angry" in low:
        return CLASS_NAMES[6]
    return CLASS_NAMES[7]


def create_dirs(dest_root: Path):
    for cls in CLASS_NAMES.values():
        (dest_root / cls).mkdir(parents=True, exist_ok=True)


def process_raf_subset(src_root: Path, dest_folder_name: str, mapping_filename: str, original_key: str):
    print(f"\n{'=' * 50}")
    print(f"[INFO] Starting RAFDB Extraction: {dest_folder_name}")
    print(f"[INFO] Source: {src_root}")

    if not src_root.exists():
        print(f"[ERROR] Source folder not found: {src_root}")
        return

    dest_root = TARGET_ROOT / dest_folder_name
    create_dirs(dest_root)

    mapping_path = TARGET_ROOT / mapping_filename

    mapping_data = []
    start_index = 1
    processed_originals = set()

    if mapping_path.exists():
        try:
            existing = pd.read_csv(mapping_path)
            if not existing.empty:
                mapping_data = existing.to_dict("records")
                start_index = len(existing) + 1
                processed_originals = {row[original_key] for row in mapping_data if original_key in row}
                print(f"[INFO] Resuming from existing mapping. Starting at image #{start_index}")
        except Exception as e:
            print(f"[WARNING] Could not read existing mapping file: {e}")

    class_dirs = [p for p in src_root.iterdir() if p.is_dir()]
    if not class_dirs:
        print(f"[WARNING] No class subfolders found under: {src_root}")
        return

    all_imgs = []
    for cdir in class_dirs:
        for img in cdir.rglob("*"):
            if img.is_file() and img.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                all_imgs.append(img)

    total_imgs = len(all_imgs)
    print(f"[INFO] Total images discovered: {total_imgs}")

    success_count = 0
    missing_count = 0
    current_idx = start_index

    pbar = tqdm(total=total_imgs, desc=f"Copying {dest_folder_name}")

    for img_path in all_imgs:
        rel_path = img_path.relative_to(src_root).as_posix()

        if rel_path in processed_originals:
            pbar.update(1)
            continue

        src_class_folder = img_path.parent.name
        mapped_class_folder = map_raf_class_folder(src_class_folder)

        # 000001.jpg
        new_img_name = f"{current_idx:06d}.jpg"
        target_path = dest_root / mapped_class_folder / new_img_name

        if img_path.exists():
            if not target_path.exists():
                try:
                    shutil.copy2(img_path, target_path)
                    new_rel_path = f"{dest_folder_name}/{mapped_class_folder}/{new_img_name}"
                    mapping_data.append({
                        "new_path": new_rel_path,
                        original_key: rel_path
                    })
                    success_count += 1
                    current_idx += 1
                except Exception:
                    pass
            else:
                success_count += 1
                current_idx += 1
        else:
            missing_count += 1

        pbar.update(1)

        if len(mapping_data) % 5000 == 0 and len(mapping_data) > 0:
            pd.DataFrame(mapping_data).to_csv(mapping_path, index=False)

    pbar.close()

    if mapping_data:
        pd.DataFrame(mapping_data).to_csv(mapping_path, index=False)
        print(f"[INFO] Saved {len(mapping_data)} mapping records to: {mapping_path}")

    print(f"\n[DONE] {dest_folder_name} Complete.")
    print(f"       Successfully processed/verified: {success_count} images.")
    print(f"       Missing/invalid: {missing_count} images.")


def main():
    TARGET_ROOT.mkdir(parents=True, exist_ok=True)

    # Train（不拆 10 折）
    process_raf_subset(
        src_root=PATH.RAF_TRAIN,
        dest_folder_name="RAFDB_train",
        mapping_filename="RAFDB_train_mapping.csv",
        original_key="original_raf_train_path",
    )

    # Valid
    process_raf_subset(
        src_root=PATH.RAF_VALID,
        dest_folder_name="RAFDB_valid",
        mapping_filename="RAFDB_valid_mapping.csv",
        original_key="original_raf_valid_path",
    )

    print("\n[ALL DONE] RAFDB extracted and organized successfully!")


if __name__ == "__main__":
    main()