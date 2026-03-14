import numpy as np
import torch

from pipeline4_model_trains.common.confusion_matrix_np import confusion_matrix_np
from pipeline4_model_trains.common.macro_f1_from_cm import macro_f1_from_cm


@torch.no_grad()
def evaluate_classification(model, loader, criterion, device, num_classes: int = 8):
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    total_loss = 0.0
    total_n = 0
    use_amp = (device.type == "cuda")

    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(X)
            loss = criterion(logits, y)

        bs = y.numel()
        total_loss += loss.item() * bs
        total_n += bs

        pred = logits.argmax(dim=-1)
        cm += confusion_matrix_np(
            y.detach().cpu().numpy(),
            pred.detach().cpu().numpy(),
            num_classes,
        )

    acc = float(np.diag(cm).sum() / max(1, cm.sum()))
    mf1 = macro_f1_from_cm(cm)

    return {
        "loss": total_loss / max(1, total_n),
        "acc": acc,
        "mf1": mf1,
        "cm": cm,
    }