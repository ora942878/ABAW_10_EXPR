# Pipeline2 — Face Extraction for ABAW10th EXPR (AffWild2)

This folder contains the face extraction pipeline used in our ABAW10th EXPR solution. It turns the official EXPR videos (batch1/2/3) into per-video face frame folders, exported at three aligned crop scales (0.9 / 1.2 / 1.5). All paths are centralized in `configs/paths.py`.

## Face detector / model

We use **InsightFace** via `insightface.app.FaceAnalysis` with the **`buffalo_l`** model pack (downloaded as ONNX files). The same model pack is used for:
- face detection (per-frame bounding boxes),
- identity features (embeddings) to stabilize tracking of the main subject across frames.

## What each step does (high level)

- `step1_generate_expr_video_index.py`  
  Builds an index `expr_video_index.csv` that maps EXPR annotation txt IDs to the corresponding video IDs found under batch1/2/3.

- `step2_download_insightface_models.py`  
  Downloads and extracts the InsightFace `buffalo_l` model pack into the local `INSIGHTFACE_ROOT` (used by step3/4).

- `step3_face_extract.py`  
  Main pipeline. Processes all indexed EXPR videos and exports face crops at 3 scales (0.9 / 1.2 / 1.5). It is designed for large-scale processing (multiprocessing + resume from checkpoints).

- `step4_video_level_face_extract.py`  
  Video-level “surgical” re-processing tool for hard cases. You can specify a list of video IDs and apply targeted selection strategies (e.g., keep identity, always pick left/right/top face, etc.) while still exporting synchronized 0.9 / 1.2 / 1.5 outputs.

## Recommended usage

From the project root:

```bash
# 1) build EXPR txt->video index
python pipeline2_face_extract/step1_generate_expr_video_index.py

# 2) download InsightFace models (buffalo_l)
python pipeline2_face_extract/step2_download_insightface_models.py

# 3) run full extraction (multi-scale)
python pipeline2_face_extract/step3_face_extract.py

# 4) (optional) re-run selected hard videos with tailored strategies
python pipeline2_face_extract/step4_video_level_face_extract.py
```

## Outputs

The pipeline produces three synchronized folders (one per crop scale), organized by `video_id`:

```
Data/Dataset_ABAW_processed/
  ABAW_face09/<video_id>/*.jpg
  ABAW_face12/<video_id>/*.jpg
  ABAW_face15/<video_id>/*.jpg
```

Each `<video_id>` folder contains face crops named by frame index (zero-padded). Missing detections are skipped; downstream loaders handle this during training.

## Notes (important)

After careful inspection, we found it is difficult to define a single “universal” configuration that can robustly and consistently detect faces across all poses, lighting conditions, and face sizes in AffWild2. Even in many single-target videos, the protagonist may look at screens containing other faces and react, which frequently introduces competing face candidates.

Therefore, our final dataset was not produced purely by automatic detection. After running **Step3** on all EXPR-annotated videos, we manually inspected the extracted faces and then re-ran approximately **one third** of the videos using **Step4**, leveraging video-level, multi-condition, targeted strategies to improve stability and correctness.

For certain multi-person dialogue scenarios where multiple identities appear simultaneously within the screen, we designed customized cropping strategies on a per-video basis to ensure accurate identity separation and stable tracking. We also documented the specific Step4 face-cropping strategy adopted for each individual video to guarantee reproducibility and transparency.

The detailed records of these per-video configurations are provided in the CSV file located at metadata/face_extract_strategy.csv.
