import numpy as np
import torch
"""
from pipeline4_model_trains.common.macro_f1_from_cm import macro_f1_from_cm  


"""
@torch.no_grad()
def macro_f1_from_cm(cm, eps=1e-12):
    tp = np.diag(cm).astype(np.float64)
    p = tp / (cm.sum(axis=0) + eps)
    r = tp / (cm.sum(axis=1) + eps)
    return float(np.mean(2 * p * r / (p + r + eps)))