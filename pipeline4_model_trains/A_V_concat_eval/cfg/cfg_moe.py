from dataclasses import dataclass
from pathlib import Path
import torch
from configs.paths import PATH


@dataclass
class CFG:
    EXP_NAME: str = "av_concat_moe"
    HEAD_TYPE: str = "moe"

    TRAIN_EXPR_DIR: str = str(PATH.EXPR_TRAIN_ABAW10th)
    VALID_EXPR_DIR: str = str(PATH.EXPR_VALID_ABAW10th)
    SAVE_ROOT: str = str(Path(PATH.PROJECT_ROOT) / "pipeline4_model_trains" / "A_V_concat_eval" / "runs")

    VIS_DIM: int = 1024
    AUD_DIM: int = 1024
    NUM_CLASSES: int = 8

    NUM_EXPERTS: int = 4
    MOE_DEPTH: int = 3
    MOE_MLP_RATIO: float = 1.5
    MOE_DROPOUT: float = 0.6
    MOE_DROP_PATH: float = 0.1
    MOE_AUX_WEIGHT: float = 0.01

    BATCH_SIZE: int = 2048
    NUM_WORKERS: int = 8
    EPOCHS: int = 10

    LR: float = 1e-5
    WEIGHT_DECAY: float = 1e-5

    INPUT_DROPOUT_VIS = 0.5
    # INPUT_DROPOUT_AUD = 0.4

    DO_BRANCH_L2_NORM: bool = True
    DO_ZSCORE: bool = True
    ZSCORE_EPS: float = 1e-6
    USE_CLASS_WEIGHT: bool = True

    AMP: bool = True
    PIN_MEMORY: bool = True
    PERSISTENT_WORKERS: bool = True

    SEED: int = 3407
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"