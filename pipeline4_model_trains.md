# Pipeline 4: Model Training, Evaluation, and Temporal Smoothing

### Summary
Pipeline 4 encompasses the core training and evaluation framework of our project. It handles the dataset splitting, fine-tuning of the DINOv2 visual backbone on auxiliary image datasets, linear probing for single-modality feature validation, multi-modal audio-visual fusion, and finally, the implementation of our zero-shot temporal smoothing strategies.

---

### Module-by-Module Functional Description

All scripts described below are located under the `pipeline4_model_trains/` directory. 

#### 1. Data Splitting & Cross-Validation Setup
* **`make_video_balanced_folds.py`**: Generates 5 cross-validation folds with consistent class distributions using a random search approach. It outputs the fold configuration to `metadata/expr_cv5_txtid_randomsearch/`. *(Note: To ensure exact reproducibility, this generated folder has already been included in our uploaded codebase.)*

#### 2. DINOv2 Visual Finetuning
This module finetunes the DINOv2 backbone directly on the `IMGset` (a combined dataset of AffectNet and RAF-DB).
* **`DINO_augment/`**: Contains various data augmentation strategies applied during DINO finetuning.
* **`DINO_train2/`**: The main finetuning module. By modifying `CFG_trainDINOv2.py` and running `train.py`, we train the models using three distinct strategies defined in the config:
  1. `'base'` -> Corresponds to `finetune_dinov2_normal.pt` (Standard finetuning).
  2. `'auged1_withoutpadding'` -> Corresponds to `finetune_dinov2_auged_withoutPadAug.pt` (High-capacity MoE task head).
  3. `'auged1'` -> Corresponds to `finetune_dinov2_auged.pt` (MoE + Affwild2 Padding strategy).
  > **NOTE:**
  > After training, users MUST manually rename the resulting `best.pt` file to the corresponding name above and move it to `pipeline3_feature_extract/VIT_weights`. This manual step prevents accidental overwriting of existing baseline models.

#### 3. Single-Modality Linear Probing (Evaluation)
* **`DINO_eval/train_eval1.py`**: Validates the linear separability of visual features using a single linear layer. It evaluates **20 distinct feature sources** generated in Pipeline 3 (combining the 4 DINO weight variants with 5 facial crop types: official face / face09 / face12 / face15 / 091215mean).
* **`Wav2Vec2_eval/train_audio_linear.py`**: Validates the linear separability of audio features across the different temporal window scales defined in Pipeline 3, using a single linear layer.

#### 4. Audio-Visual Fusion & Task Heads
* **`A_V_concat_eval/`**: Contains the evaluation code for audio-visual fusion and concatenation strategies.
  * `cfg/`: Parameter configurations for various small-scale task heads.
  * `heads/`: Model architecture definitions for these task heads.
  * `build.py`: Utility functions to quickly instantiate models based on `cfg` and `heads`.
  * `eval.py`: Trains and evaluates the task heads on the official EXPR-valid set.
  * `eval_5fold.py`: Trains and evaluates the task heads across the aforementioned 5-fold cross-validation setup.
  * `model_param_summary.py`: Outputs parameter statistics for the different task heads.

#### 5. Temporal Smoothing & Inference
* **`A_V_concat_eval/inference_with_temporalSmooth/`**: Contains the inference code incorporating our zero-shot temporal smoothing methods. 
  > **Note:** Before running, you must place the trained weights (specifically those based on the Gated structure) and their configuration folders into the following paths:
  > * `PATH.PROJECT_ROOT / "pipeline4_model_trains" / "A_V_concat_eval" / "runs" / "av_concat_gate"`
  > * `PATH.PROJECT_ROOT / "pipeline4_model_trains" / "A_V_concat_eval" / "runs" / "av_concat_gate_cv5"`
  * **`eval_window_ablation.py`**: Conducts ablation studies on four basic smoothing methods, adjusting window lengths and strides to measure performance gains. It outputs a visualization plot of the results.
  * **`inference.py`**: Executes inference on both the official EXPR-valid set and the 5-fold split using optimized window lengths. It applies the trained weights and reports the performance improvements yielded by different temporal smoothing methods.

#### 6. Core Utilities
* **`common/`, `models/`, `utils/`**: Shared directories containing foundational tools, base model classes, and general-purpose utility functions utilized across Pipeline 4.

---