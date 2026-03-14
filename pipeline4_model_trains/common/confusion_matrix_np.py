import numpy as np

def confusion_matrix_np(y_true, y_pred, num_classes: int = 8):
    idx = y_true.astype(np.int64) * num_classes + y_pred.astype(np.int64)
    return np.bincount(
        idx,
        minlength=num_classes * num_classes
    ).reshape(num_classes, num_classes).astype(np.int64)