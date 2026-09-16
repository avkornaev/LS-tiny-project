"""Explicit label smoothing implementation."""

import torch
from torch.nn import functional as F


def smoothed_targets(
    labels: torch.Tensor, epsilon: float, num_classes: int = 2
) -> torch.Tensor:
    """Apply (1-epsilon)*one_hot + epsilon/num_classes."""
    if not 0.0 <= epsilon < 1.0:
        raise ValueError("epsilon must be in [0, 1)")
    one_hot = F.one_hot(labels, num_classes=num_classes).to(torch.float32)
    return (1.0 - epsilon) * one_hot + epsilon / num_classes


def label_smoothed_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor, epsilon: float
) -> torch.Tensor:
    """Compute mean cross-entropy against explicitly smoothed targets."""
    targets = smoothed_targets(labels, epsilon, logits.shape[1])
    return -(targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()

