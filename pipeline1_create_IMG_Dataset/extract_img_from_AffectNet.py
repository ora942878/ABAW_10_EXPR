import pandas as pd
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm

from configs.paths import PATH


SOURCE_IMG_ROOT = PATH.IMG_AffectNet
TARGET_ROOT = PATH.Dataset_IMG

NUM_TRAIN_SPLITS = 10

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


def process_subset(csv_path, is_train=False):
    subset_name = "Training" if is_train else "Validation"
    print(f"\n{'=' * 50}")
    print(f"[INFO] Starting {subset_name} Extraction")
    print(f"[INFO] Reading CSV from: {csv_path}")

    if not csv_path.exists():
        print(f"[ERROR] CSV file not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    df = df[df['expression'] <= 7].copy()

    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    total_imgs = len(df)
    print(f"[INFO] Total valid images to process: {total_imgs}")

    df['fold'] = 0
    if is_train:
        print(f"[INFO] Calculating stratified splits for {NUM_TRAIN_SPLITS} folds...")
        for expr_label in range(8):
            class_indices = df[df['expression'] == expr_label].index
            splits = np.array_split(class_indices, NUM_TRAIN_SPLITS)
            for fold_idx, split_indices in enumerate(splits):
                df.loc[split_indices, 'fold'] = fold_idx + 1

    print("[INFO] Creating directory structures...")
    if is_train:
        for fold in range(1, NUM_TRAIN_SPLITS + 1):
            for cls_name in CLASS_NAMES.values():
                (TARGET_ROOT / f"Affnet_train{fold}" / cls_name).mkdir(parents=True, exist_ok=True)
    else:
        for cls_name in CLASS_NAMES.values():
            (TARGET_ROOT / "Affnet_valid" / cls_name).mkdir(parents=True, exist_ok=True)

    success_count = 0
    missing_count = 0
    mapping_data = []

    mapping_filename = "Affnet_train_mapping.csv" if is_train else "Affnet_valid_mapping.csv"
    mapping_path = TARGET_ROOT / mapping_filename

    if mapping_path.exists():
        try:
            existing_mapping = pd.read_csv(mapping_path)
            mapping_data = existing_mapping.to_dict('records')
            success_count = len(existing_mapping)
            print(f"[INFO] Resuming from existing mapping. Already processed: {success_count}")
        except Exception as e:
            print(f"[WARNING] Could not read existing mapping file: {e}")

    processed_originals = {item['original_affectnet_path'] for item in mapping_data}
    current_idx = success_count + 1

    pbar = tqdm(total=total_imgs, desc=f"Copying {subset_name}")

    for i in range(total_imgs):
        row = df.iloc[i]
        expr_label = int(row['expression'])
        rel_path = row['subDirectory_filePath']  # 例如: "683/4a0...jpg"
        fold = int(row['fold'])

        if rel_path in processed_originals:
            pbar.update(1)
            continue

        src_path = SOURCE_IMG_ROOT / rel_path

        class_folder = CLASS_NAMES[expr_label]
        dest_folder_name = f"Affnet_train{fold}" if is_train else "Affnet_valid"
        new_img_name = f"{current_idx:06d}.jpg"
        target_path = TARGET_ROOT / dest_folder_name / class_folder / new_img_name

        if src_path.exists():
            try:
                if not target_path.exists():
                    shutil.copy2(src_path, target_path)

                new_rel_path = f"{dest_folder_name}/{class_folder}/{new_img_name}"
                mapping_data.append({
                    "new_path": new_rel_path,
                    "original_affectnet_path": rel_path
                })
                success_count += 1
                current_idx += 1
            except Exception:
                pass
        else:
            missing_count += 1

        pbar.update(1)

        if len(mapping_data) % 5000 == 0:
            pd.DataFrame(mapping_data).to_csv(mapping_path, index=False)

    pbar.close()
    if mapping_data:
        pd.DataFrame(mapping_data).to_csv(mapping_path, index=False)

    print(f"\n[DONE] {subset_name} Complete.")
    print(f"       Successfully processed: {success_count}")
    print(f"       Missing in source: {missing_count}")


def main():
    TARGET_ROOT.mkdir(parents=True, exist_ok=True)
    process_subset(PATH.valid_csv_AffectNet, is_train=False)
    process_subset(PATH.train_csv_AffectNet, is_train=True)
    print("\n[ALL DONE] All datasets organized!")


if __name__ == "__main__":
    main()