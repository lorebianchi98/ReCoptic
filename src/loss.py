import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class UnidirectionalInfonce(nn.Module):
    def __init__(self, logit_scale_temp: float = 0.07):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / logit_scale_temp))
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        scale = self.logit_scale.exp()
        logits_per_image = scale * y_pred
        loss = F.cross_entropy(logits_per_image, y_true)
        return loss



def soft_f1_loss(y_pred: torch.Tensor, y_true: torch.Tensor, epsilon: float = 1e-7) -> torch.Tensor:
    """
    Computes the soft F1 score for binary classification.
    
    Args:
        y_pred (torch.Tensor): Predicted probabilities, shape (BS,), values in [0, 1]
        y_true (torch.Tensor): Ground truth binary labels, shape (BS,), values in {0, 1}
        epsilon (float): Small value to avoid division by zero

    Returns:
        torch.Tensor: Soft F1 score (scalar)
    """
    y_pred = y_pred.float()
    y_true = y_true.float()

    # True positives (soft)
    tp = torch.sum(y_pred * y_true)

    # Precision and recall (soft versions)
    precision = tp / (torch.sum(y_pred) + epsilon)
    recall = tp / (torch.sum(y_true) + epsilon)

    # Soft F1 score
    f1 = 2 * precision * recall / (precision + recall + epsilon)
    return 1 - f1

def norm_loss(embeddings):
    norms = torch.norm(embeddings, dim=1)
    return torch.mean((1.0 - norms) ** 2)  # encourages norm ~1