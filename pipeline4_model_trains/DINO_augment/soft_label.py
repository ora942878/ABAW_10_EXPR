import torch
import torch.nn.functional as F


def smooth_onehot(y: torch.Tensor, num_classes: int, eps: float) -> torch.Tensor:
    y_oh = F.one_hot(y, num_classes=num_classes).float()
    if eps <= 0:
        return y_oh
    return y_oh * (1.0 - eps) + (eps / num_classes)

def soft_ce_with_class_weight(logits: torch.Tensor, y_soft: torch.Tensor, class_w: torch.Tensor | None) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=-1)
    if class_w is not None:
        logp = logp * class_w.view(1, -1)
    return -(y_soft * logp).sum(dim=-1).mean()