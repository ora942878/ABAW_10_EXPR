# README

This repository is organized into five parts: `pipeline1` for data preparation, `pipeline2` for face extraction and multi-strategy face selection, `pipeline3` for audio/visual feature extraction, `pipeline4` for model training and evaluation, and `pipelineTEST` for final test-time inference and submission generation.

## Requirements

Because of environment conflicts, this project uses two separate requirement files.

- `requirements_of_pipeline2_face_extract.txt` is used for `pipeline2` and for the face-extraction stage in `pipelineTEST`. This environment mainly includes the InsightFace / ONNXRuntime / OpenCV stack for face detection, tracking, and cropping.
- `requirements_torch28.txt` is used for the remaining stages, including feature extraction, training, evaluation, and final inference.

In other words, face extraction and the other stages should be run in different environments.

## Running Environment

Our main experiments, feature extraction, training, and inference were conducted on AutoDL with RTX 5090 GPU instances.

## Datasets

We use the official Aff-Wild2 / ABAW EXPR data as the main benchmark, and additionally use AffectNet and RAF-DB as auxiliary image datasets for improving generalization. In our pipeline, the auxiliary datasets are remapped into the unified 8-class EXPR label space before training.

## Pretrained Backbones

### Visual backbone
The visual branch is based on DINOv2 ViT-L/14.  
The pretrained checkpoint downloaded in the project is `dinov2_vitl14_pretrain.pth`, and the test-time visual feature extraction stage further uses our finetuned weight `finetune_dinov2_auged_weights_only.pt`.

### Audio backbone
The audio branch is based on wav2vec 2.0 Large LV-60K ASR.  
The checkpoint used in this project is `wav2vec2_fairseq_large_lv60k_asr_ls960.pth`. In our pipeline, raw audio is first encoded by wav2vec 2.0, and then the mean of the last 4 Transformer layers is used as the final frame-aligned audio representation.

- DINOv2 ViT-L/14 pretrained weight
https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth

- DINOv2 official repository
https://github.com/facebookresearch/dinov2

- wav2vec 2.0 Large LV-60K ASR (LS960) weight
https://download.pytorch.org/torchaudio/models/wav2vec2_fairseq_large_lv60k_asr_ls960.pth

- torchaudio official bundle page
https://docs.pytorch.org/audio/main/generated/torchaudio.pipelines.WAV2VEC2_ASR_LARGE_LV60K_960H.html

## Project Weights and Placement

To facilitate reproducing our results, we have provided all the necessary custom model weights via Google Drive:
- **[Download Custom Project Weights (Google Drive)](https://drive.google.com/drive/folders/1uPbV_mfSECWAINm71ng7K5iN1Zn_ACgF?usp=sharing)**

This folder contains:
1. The specific DINOv2 pretrained weight used in our project.
2. The three fine-tuned DINOv2 weights.
3. The final gate models trained on the `trainset` and via 5-fold cross-validation (`cv5`).

You can download these weights and place them directly in the corresponding directories to perform forward inference in the `pipelineTEST` stage.

---

The final models used in this repository are trained in `pipeline4`.

For the visual branch, the pretrained and finetuned DINOv2 weights should be placed under `PATH.VIT_WEIGHTS_ROOT` and renamed as follows:

- pretrained DINOv2 ViT-L/14: `dinov2_vitl14.pth`
- standard finetuned version: `finetune_dinov2_normal.pt`
- MoE-head finetuned version without PadAug: `finetune_dinov2_auged_withoutPadAug.pt`
- final finetuned version with PadAug: `finetune_dinov2_auged.pt`

In the visual feature extraction stage, we use our own finetuned DINOv2 weights rather than only the original pretrained checkpoint.

For the audio-visual fusion stage, the trained gate weights and their derived files should be placed under:

- `PATH.PROJECT_ROOT / "pipeline4_model_trains" / "A_V_concat_eval" / "runs" / "av_concat_gate_cv5"`
- `PATH.PROJECT_ROOT / "pipeline4_model_trains" / "A_V_concat_eval" / "runs" / "av_concat_gate"`

These directories store the final gate models used by the inference scripts in `pipelineTEST`.