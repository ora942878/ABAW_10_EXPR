
import numpy as np
import torch
import random
"""
from pipeline4_model_trains.common.set_seed import set_seed  


"""
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False