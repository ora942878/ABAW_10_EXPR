import torch
import numpy as np
from typing import Dict, List, Tuple
import torch.nn.functional as F
import random

def mixup_batch(x: torch.Tensor, y_soft: torch.Tensor, alpha: float, p: float) -> Tuple[torch.Tensor, torch.Tensor]:
    if alpha <= 0 or p <= 0 or (random.random() > p):
        return x, y_soft
    lam = np.random.beta(alpha, alpha)
    lam = float(lam)
    b = x.size(0)
    idx = torch.randperm(b, device=x.device)
    x = lam * x + (1.0 - lam) * x[idx]
    y_soft = lam * y_soft + (1.0 - lam) * y_soft[idx]
    return x, y_soft