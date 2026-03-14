# Pipeline 3: Feature Extraction and Temporal Alignment

### Summary
Pipeline 3 is a comprehensive framework designed for multimodal feature extraction and precise temporal synchronization. It utilizes self-supervised visual and acoustic backbone models to process multi-scale facial inputs and raw audio signals into frame-aligned feature matrices, providing a standardized data foundation for downstream emotion recognition tasks.

---

### Step-by-Step Functional Description

#### 1. Visual Feature Backbone (Steps 1-3)
* **Step 1: Weights Initialization**: Downloads the DINOv2 (ViT-L/14) backbone weights, which were pre-trained on the LVD-142M dataset.
* **Step 2: Multi-Scale Facial Extraction**: Extracts features from facial images cropped at three different scales (0.9, 1.2, and 1.5) generated in Pipeline 2. The features are saved as `<folder_id>.pt`.
* **Step 3: Official Baseline Extraction**: Extracts baseline features from the 112x112 aligned facial images provided by the ABAW organizers.

#### 2. Visual Feature Matrix (16 Paths)
To ensure comprehensive spatial representation, the pipeline applies four distinct weight configurations across the four visual input sources, resulting in **16 specific feature paths** defined in `configs/paths`:

> **Note: Weights Generated from Pipeline 4**
> In Pipeline 4, we designed three additional sets of finetuned weights based on DINOv2 (ViT-L/14). To ensure they can be correctly used by Pipeline 3 to extract video features, they MUST be named as follows:
> * `finetune_dinov2_normal.pt`: Corresponds to the standard finetuning method mentioned in our paper (FT1).
> * `finetune_dinov2_auged.pt`: Corresponds to the method combining the high-capacity MoE with the Padding strategy tailored for the Affwild2 dataset (FT2).
> * `finetune_dinov2_auged_withoutPadAug.pt`: Corresponds to the finetuning method utilizing a high-capacity Mixture-of-Experts (MoE) as the task head (FT3).

* **Weight Configurations Summary**:
  1. `PT`: Original DINOv2 pre-trained weights.
  2. `FT1`: `finetune_dinov2_normal.pt`
  3. `FT2`: `finetune_dinov2_auged.pt`
  4. `FT3`: `finetune_dinov2_auged_withoutPadAug.pt`
* **16 Feature Paths**: 
  * `OFFIMG_V_DINOV2_PT`, `FT1`, `FT2`, `FT3`
  * `FACE09_V_DINOV2_PT`, `FT1`, `FT2`, `FT3`
  * `FACE12_V_DINOV2_PT`, `FT1`, `FT2`, `FT3`
  * `FACE15_V_DINOV2_PT`, `FT1`, `FT2`, `FT3`

#### 3. Acoustic Feature Backbone (Steps 4-5)
* **Step 4: Acoustic Model Acquisition**: Downloads the Wav2Vec 2.0 Large (LV-60K + 960h) model, pre-trained on 60,000 hours of unlabeled speech and finetuned on 960 hours of Librispeech for ASR.
* **Step 5: Deep Semantic Extraction**: Computes the mean of the **last 4 Transformer layers** to capture stable acoustic prosody. It natively records precise physical timestamps (`t_sec`) for each audio step to facilitate high-accuracy temporal alignment.

#### 4. Metadata and Temporal Synchronization (Steps 6-7)
* **Step 6: Metadata Generation**: Utilizes OpenCV to parse the actual FPS and total frame counts of the videos, generating a centralized `expr_video_meta.csv` index file.
* **Step 7: Cross-Modal Alignment**: Maps asynchronous audio features to video frames using a sliding window mean approach. If an audio window is empty, it automatically falls back to nearest-neighbor matching.
* **4 Aligned Audio Paths**: This step generates four distinct temporal alignment variants:
  1. `OFFAUDIO_WAV2VEC2_PT_aligned_nearest` (Nearest-neighbor fallback)
  2. `OFFAUDIO_WAV2VEC2_PT_aligned_window25` (0.25s window mean)
  3. `OFFAUDIO_WAV2VEC2_PT_aligned_window50` (0.50s window mean)
  4. `OFFAUDIO_WAV2VEC2_PT_aligned_window75` (0.75s window mean)

---