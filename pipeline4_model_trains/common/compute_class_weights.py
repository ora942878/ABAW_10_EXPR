
import numpy as np
import torch


def compute_class_weights(labels, num_classes: int = 8) -> torch.Tensor:
    labels = np.asarray(labels).astype(np.int64)

    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)

    counts = np.maximum(counts, 1.0)

    weights = counts.sum() / (num_classes * counts)

    weights = weights / weights.mean()

    return torch.tensor(weights, dtype=torch.float32)