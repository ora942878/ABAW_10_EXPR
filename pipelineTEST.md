# pipelineTEST

`pipelineTEST` is the final test-time inference pipeline for the ABAW EXPR task.

Before running the pipeline, please ensure that the official test list files provided by the organizers are placed in the correct metadata directory. The pipeline scripts will read from the following exact paths:
METADATA_ROOT_TEST / 'Expression_Recognition_Challenge_test_set_release.txt' (This file contains the list of official test video IDs.)
METADATA_ROOT_TEST / 'ABAW_Expr_test_set_sample.txt' (This file contains the specific IDs/frames required for the final submission.)

## Important Note on Environments

Please note again that, due to environment conflicts, **face extraction** and **the other stages** are executed in **two different environments**.  
Both environment descriptions are provided in the project directory.

In practice:

- the **face extraction** stage should be run in the face-extraction environment;
- the **feature extraction / alignment / inference** stages should be run in the main inference environment.

---

## Overall Workflow

```text
Official test frame list
        ↓
Step 1  Generate txtid ↔ videoid index
        ↓
Step 2  Extract raw audio features with timestamps
        ↓
Step 3  Generate video metadata
        ↓
Step 4  Align audio features to video frames
        ↓
Step 5  Extract test faces
        ↓
Step 6  Extract visual features
        ↓
Inference  Fusion + median101 smoothing
        ↓
Final prediction txt
```

---

## Step 1. Generate EXPR test video index

Script:
- `TESTstep1_generate_expr_video_index.py`

This step reads the official test ids and matches each `txtid` to the corresponding raw test video.

Output:
- `PATH.EXPR_VIDEO_INDEX_CSV_TEST`


---

## Step 2. Extract raw audio features with timestamps

Script:
- `TESTstep2_get_features_audio_raw_last4mean_ts.py`

This step:

- reads the matched test videos,
- extracts `wav` audio with `ffmpeg`,
- uses Wav2Vec2 to obtain raw audio features,
- saves both feature tensors and physical timestamps.

Output:
- raw audio feature files under the test audio feature directory

---

## Step 3. Generate video metadata

Script:
- `TESTstep3_generate_expr_video_meta.py`

This step scans each raw video and records metadata such as:

- `fps`
- `n_frames`
- `video_duration_sec`

Output:
- `PATH.EXPR_VIDEO_META_CSV_TEST`

---

## Step 4. Align audio features to video frames

Script:
- `TESTstep4_align_audio_to_video_frames.py`

This step aligns raw audio features to frame-level video time, so that each video frame can obtain its corresponding audio representation.

Output:
- aligned audio feature files for test inference

---

## Step 5. Extract test faces

This stage is closely related to the multi-strategy face extraction design in `pipeline2_face_extract`.

For the test set, **we directly provide our own face extraction strategy file**, which records the human-selected extraction strategy for these test videos.  
Its role is similar to the multi-strategy face selection used in `pipeline2`: once this strategy file is placed correctly, the user can directly run the **one-step face extraction** script for the whole test set.

### Step 5.1 One-step face extraction

Script:
- `TESTstep5_face_extract_onestep.py`

This script reads the provided test face extraction strategy file and performs face extraction for the full test set.

Outputs:
- `PATH.ABAW_FACE09_ROOT_TEST`
- `PATH.ABAW_FACE12_ROOT_TEST`
- `PATH.ABAW_FACE15_ROOT_TEST`


### Step 5.2 Video-id-level face extraction

Script:
- `TESTstep5_video_level_face_extract.py`

We also provide a **video-id-level face extraction script**.  
This script is intended for manual debugging, targeted repair, or reprocessing of individual videos.

---

## Step 6. Extract visual features

Script:
- `TESTstep6_get_features_visual.py`

This step extracts visual features from the three face crop roots generated in Step 5.

Outputs:
- `PATH.FACE09_V_DINOV2_FT2_TEST`
- `PATH.FACE12_V_DINOV2_FT2_TEST`
- `PATH.FACE15_V_DINOV2_FT2_TEST`

---

## Final Inference

Two inference scripts are provided.

### Option A. Single-model inference

Script:
- `Inference_median101_train_on_trainset.py`

This script performs final prediction with the trained fusion model and fixed-window median smoothing.


### Option B. 5-fold ensemble inference

Script:
- `Inference_median101_5fold_ensamble.py`

This script loads the 5-fold models, averages their logits, and then applies the same `median101` smoothing to produce the final submission file.

---